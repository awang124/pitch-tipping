import argparse

import cv2
import os
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.getenv("EXTRACTION_DATA_DIR")
if DATA_DIR is None:
    raise ValueError("Missing EXTRACTION_DATA_DIR in .env")

# argument
parser = argparse.ArgumentParser(description="Extract keypoints from videos.")
parser.add_argument("--pitcher", required=True, help="Pitcher name")
args = parser.parse_args()
PITCHER_NAME = args.pitcher

VIDEO_DIR = os.path.join(DATA_DIR, PITCHER_NAME, "pitches")
OUTPUT_DIR = os.path.join(DATA_DIR, PITCHER_NAME, "frames")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_frames(video_path, out_dir, fps=10):
    os.makedirs(out_dir, exist_ok=True)
    vid = cv2.VideoCapture(video_path)
    original_fps = vid.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / fps)

    frame_count = 0
    saved = 0

    while True:
        ret, frame = vid.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            cv2.imwrite(f"{out_dir}/frame_{saved:05d}.jpg", frame)
            saved += 1
        frame_count += 1

    vid.release()
    print(f"Saved {saved} frames")

for label in os.listdir(VIDEO_DIR):
    for vid in os.listdir(os.path.join(VIDEO_DIR, label)):
        extract_frames(f"{VIDEO_DIR}/{label}/{vid}", f"{OUTPUT_DIR}/{label}")