import cv2
import os
import numpy as np
import argparse
from ultralytics import YOLO
from dotenv import load_dotenv
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


#TO RUN: video-data-extraction/extract-keypoints --pitcher <pitcher>


# Program that takes in video from data directory (for now) 
# Current capacity is one pitcher, will have to figure how to store/use multiple pitcher's data 
# Converts the video into a numpy array of these elements at each frame in the video
        # Nose
        # Eyes (left, right)
        # Ears (left, right)
        # Shoulders (left, right)
        # Elbows (left, right)
        # Wrists (left, right)
        # Hips (left, right)
        # Knees (left, right)
        # Ankles (left, right)
# Numpy array easier for CNN to process for the most part I believe, might have to do something 
# else if getting precise with wrist rotation, or clothes or something
# Finally, outputs into data/keypoints





#input and output directories from .env
load_dotenv()
DATA_DIR = os.getenv("EXTRACTION_DATA_DIR")
if DATA_DIR is None:
    raise ValueError("Missing EXTRACTION_DATA_DIR in .env")

#get pitcher name from arguments
parser = argparse.ArgumentParser(description="Extract keypoints from videos.")
parser.add_argument("--pitcher", required=True, help="Pitcher name")
args = parser.parse_args()
PITCHER_NAME = args.pitcher

#final directories to a per pitcher basis
VIDEO_DIR = DATA_DIR + PITCHER_NAME
OUTPUT_DIR = DATA_DIR + PITCHER_NAME + "/keypoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#sample model for now
POSE_MODEL = "yolov8x-pose.pt"

#change based on frame rate, sample for now
FRAME_SKIP = 1

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)


#load YOLO model ------------------------- might change!
model = YOLO(POSE_MODEL)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# extract keypoints from a single video
def process_video(video_path):

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    keypoint_sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        print(frame_idx)
        # frame skip
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        # run pose detection
        results = model(frame, verbose=False)

        if results[0].keypoints is not None:
            # results[0].keypoints.xy is a list of arrays, one per person
            persons = results[0].keypoints.xy

            if persons is None or len(persons) == 0:
                print(f"[WARN] No person detected in frame {frame_idx}, skipping…")
                frame_idx+=1
                continue

        # pick the correct pitcher person
        idx = pick_pitcher_person(persons, frame.shape)

        # store their keypoints
        person_kpts = persons[idx]
        person_kpts = person_kpts.detach().cpu().numpy() if hasattr(person_kpts, "detach") else np.array(person_kpts)

        #pitcher bounding box
        x_min, y_min = np.min(person_kpts, axis=0)
        x_max, y_max = np.max(person_kpts, axis=0)
        pitcher_box = [x_min, y_min, x_max, y_max]

        # run glove detection 
        glove_box = detect_pitcher_glove(frame, pitcher_box)
        if glove_box is None:
            glove_box = [-1, -1, -1, -1]
            print(f"[WARN] No glove detected at frame {frame_idx}")


        keypoint_sequence.append({
            "keypoints": person_kpts,
            "glove_box": glove_box
        })

        frame_idx += 1

    cap.release()

    #convert pose data to numpy array: (num_frames, num_keypoints, 2)
    #num_frames is total frames in video
    #num keypoints contain x and y coordinates of each body part in each frame
    keypoint_sequence = np.array(keypoint_sequence)

    #create output file to store with correct name as npy
    base = os.path.basename(video_path)
    name = os.path.splitext(base)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{name}.npy")
    np.save(out_path, keypoint_sequence)

    #confirmation message
    print(f"Saved keypoints: {out_path}  shape={keypoint_sequence.shape}")

#returns index of player that is most likely pitcher, closest to camera, for MLB video fitting
def pick_pitcher_person(kpts_xy, frame_shape):
    if len(kpts_xy) == 1:
        return 0

    h, w = frame_shape[:2]

    expected_x = 0.35 * w   # tune per camera
    x_weight = 0.4
    y_weight = 0.6  # pitcher being lowest is most important

    best_idx = 0
    best_score = -1e18  # now maximizing score

    for i, pts in enumerate(kpts_xy):
        xs = pts[:, 0]
        ys = pts[:, 1]

        center_x = np.nanmean(xs)
        center_y = np.nanmean(ys)

        # normalize into 0–1 scale
        nx = 1 - abs(center_x - expected_x) / w        # higher = better
        ny = center_y / h                              # higher = better

        score = x_weight * nx + y_weight * ny

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


def detect_pitcher_glove(frame, pitcher_box=None):
    # Detect glove near pitcher using Detectron2.
    outputs = predictor(frame)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()

    # Keep only baseball gloves (class 35)
    glove_candidates = [b for b, c in zip(boxes, classes) if c == 35]
    if not glove_candidates:
        return None

    # If pitcher box provided, choose glove closest to pitcher
    if pitcher_box is not None:
        px1, py1, px2, py2 = pitcher_box
        pitcher_center = ((px1 + px2)/2, (py1 + py2)/2)

        def distance_to_pitcher(box):
            x1, y1, x2, y2 = box
            glove_center = ((x1 + x2)/2, (y1 + y2)/2)
            return np.linalg.norm(np.array(glove_center) - np.array(pitcher_center))

        best_glove = min(glove_candidates, key=distance_to_pitcher)
    else:
        # fallback: largest glove box
        best_glove = max(glove_candidates, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

    return [int(coord) for coord in best_glove]


def main():
    #loop through each VIDEO_DIR subfolder of different pitch types
    if not os.path.exists(VIDEO_DIR):
        raise FileNotFoundError(f"Data directory not found: {VIDEO_DIR}")
    for label in os.listdir(VIDEO_DIR):
        label_dir = os.path.join(VIDEO_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        #status message
        print(f"\nProcessing label: {label}")

        #for each video in each subfolder
        for filename in os.listdir(label_dir):

            #check that is valid video file
            if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue

            #get correct video path
            video_path = os.path.join(label_dir, filename)

            #status message
            print(f"  - {filename}")

            #process the video, should produce output file of data
            process_video(video_path)


if __name__ == "__main__":
    main()
