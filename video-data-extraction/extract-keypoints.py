import cv2
import os
import numpy as np
import argparse
from ultralytics import YOLO
from dotenv import load_dotenv


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

        # frame skip
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        # run pose detection
        results = model(frame, verbose=False)

        if results[0].keypoints is not None:
            persons = results[0].keypoints.xy.cpu().numpy()

            # pick the correct pitcher person
            idx = pick_pitcher_person(persons, frame.shape)

            # store their keypoints
            person_kpts = persons[idx]
            keypoint_sequence.append(person_kpts)

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
