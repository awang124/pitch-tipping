import cv2
import os
import numpy as np
import argparse
from ultralytics import YOLO
from dotenv import load_dotenv

# TO RUN:
# python extract-keypoints.py --pitcher <pitcher>

load_dotenv()
DATA_DIR = os.getenv("EXTRACTION_DATA_DIR")
if DATA_DIR is None:
    raise ValueError("Missing EXTRACTION_DATA_DIR in .env")

# argument
parser = argparse.ArgumentParser(description="Extract keypoints from videos.")
parser.add_argument("--pitcher", required=True, help="Pitcher name")
args = parser.parse_args()
PITCHER_NAME = args.pitcher

VIDEO_DIR = os.path.join(DATA_DIR, PITCHER_NAME)
OUTPUT_DIR = os.path.join(DATA_DIR, PITCHER_NAME, "keypoints")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# YOLO MODELS ----------------------------
POSE_MODEL = "yolov8x-pose.pt"
GLOVE_MODEL = "runs/detect/train4/weights/best.pt"   # trained glove model

pose_model = YOLO(POSE_MODEL)
glove_model = YOLO(GLOVE_MODEL)

FRAME_PRINT = 100


# Pick the pitcher (same as before)
def pick_pitcher_person(kpts_xy, frame_shape):
    if len(kpts_xy) == 1:
        return 0

    h, w = frame_shape[:2]

    expected_x = 0.35 * w
    x_weight = 0.4
    y_weight = 0.6

    best_idx = 0
    best_score = -1e18

    for i, pts in enumerate(kpts_xy):
        xs = pts[:, 0]
        ys = pts[:, 1]

        center_x = np.nanmean(xs)
        center_y = np.nanmean(ys)

        nx = 1 - abs(center_x - expected_x) / w
        ny = center_y / h

        score = x_weight * nx + y_weight * ny
        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


# detect glove using our YOLO model
def detect_glove_yolo(frame, pitcher_box=None):
    results = glove_model(frame, verbose=False)[0]

    if results.boxes is None or len(results.boxes) == 0:
        return None

    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    # Class 0 = glove
    glove_boxes = [b for b, c in zip(boxes, classes) if c == 0]
    if not glove_boxes:
        return None

    # If no pitcher box, return largest glove box
    if pitcher_box is None:
        return list(map(int, glove_boxes[0]))

    px1, py1, px2, py2 = pitcher_box
    pitcher_center = ((px1 + px2) / 2, (py1 + py2) / 2)

    def dist(box):
        x1, y1, x2, y2 = box
        glove_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        return np.linalg.norm(np.array(glove_center) - np.array(pitcher_center))

    best_box = min(glove_boxes, key=dist)
    return [int(v) for v in best_box]


def process_video(video_path):

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    keypoint_sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame skip
        if frame_idx % FRAME_PRINT == 0:
            print(frame_idx)

        # YOLO pose inference
        results = pose_model(frame, verbose=False)[0]

        if results.keypoints is None:
            print(f"[WARN] No person detected at frame {frame_idx}")
            frame_idx += 1
            continue

        persons = results.keypoints.xy
        if len(persons) == 0:
            frame_idx += 1
            continue

        idx = pick_pitcher_person(persons, frame.shape)

        person_kpts = persons[idx]
        person_kpts = person_kpts.cpu().numpy()

        # Compute pitcher bounding box
        x_min, y_min = np.min(person_kpts, axis=0)
        x_max, y_max = np.max(person_kpts, axis=0)
        pitcher_box = [x_min, y_min, x_max, y_max]

        glove_box = detect_glove_yolo(frame, pitcher_box)
        if glove_box is None:
            glove_box = [-1, -1, -1, -1]
            print(f"[WARN] No glove detected at frame {frame_idx}")



        keypoint_sequence.append({
            "keypoints": person_kpts,
            "glove_box": glove_box
        })

        frame_idx += 1

    cap.release()

    keypoint_sequence = np.array(keypoint_sequence)
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{name}.npy")

    np.save(out_path, keypoint_sequence)

    print(f"Saved keypoints: {out_path} shape={keypoint_sequence.shape}")


def main():
    if not os.path.exists(VIDEO_DIR):
        raise FileNotFoundError(f"Data directory not found: {VIDEO_DIR}")

    for label in os.listdir(VIDEO_DIR):
        label_dir = os.path.join(VIDEO_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        print(f"\nProcessing label: {label}")

        for filename in os.listdir(label_dir):
            if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue

            video_path = os.path.join(label_dir, filename)
            print(f"  - {filename}")

            process_video(video_path)


if __name__ == "__main__":
    main()