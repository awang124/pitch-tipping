import cv2
import numpy as np
import os
from dotenv import load_dotenv
import argparse


#TO RUN: video-data-extraction/plot-points --pitcher <pitcher> --pitch <pitch type folder> --video <video file>

#load path to video
load_dotenv()
DATA_DIR = os.getenv("EXTRACTION_DATA_DIR")
if DATA_DIR is None:
    raise ValueError("EXTRACTION_DATA_DIR must be set in your .env file!")

# 17-keypoint skeleton structure (COCO format for YOLOv8-pose)
SKELETON_EDGES = [
    (5, 7), (7, 9),     # Left arm
    (6, 8), (8, 10),    # Right arm
    (11, 13), (13, 15), # Left leg
    (12, 14), (14, 16), # Right leg
    (5, 6),             # Shoulders
    (11, 12),           # Hips
    (5, 11), (6, 12)    # Torso diagonals
]

def draw_skeleton(frame, keypoints):
    """Draw COCO skeleton on a frame."""
    # draw keypoints
    for x, y in keypoints:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # draw connections
    for a, b in SKELETON_EDGES:
        x1, y1 = keypoints[a]
        x2, y2 = keypoints[b]
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                 (0, 255, 255), 2)

    return frame


def overlay_keypoints_on_video(video_path, npy_path, frame_skip, output_path):
    # Load data as objects
    data = np.load(npy_path, allow_pickle=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = fps / frame_skip

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    frame_idx = 0
    kpt_idx = 0

    print(f"Writing annotated video → {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0 and kpt_idx < len(data):
            frame_data = data[kpt_idx]

            keypoints = frame_data["keypoints"]
            glove_box = frame_data["glove_box"]

            # Draw keypoints
            frame = draw_skeleton(frame, keypoints)

            # Draw glove box if valid
            x1, y1, x2, y2 = glove_box
            if x1 != -1:  # your "invalid" flag
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)),
                    (0, 0, 255), 3
                )

            kpt_idx += 1

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("Done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Overlay keypoints onto a video")
    parser.add_argument("--pitcher", required=True, help="Pitcher name")
    parser.add_argument("--pitch", required=True, help="Pitch type folder (fastball/changeup/etc)")
    parser.add_argument("--video", required=True, help="Video filename only (not full path)")

    args = parser.parse_args()

    # Build paths to files
    video_file = os.path.join(
        DATA_DIR, args.pitcher, "pitches", args.pitch, args.video
    )
    video_basename = os.path.splitext(args.video)[0]
    npy_file = os.path.join(
        DATA_DIR, args.pitcher, "keypoints", f"{video_basename}.npy"
    )
    os.makedirs(os.path.join(DATA_DIR, args.pitcher, "overlays"), exist_ok=True)
    output_file = os.path.join(
        DATA_DIR, args.pitcher, "overlays",
        f"overlay_{args.video}"
    )

    # Validate 
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video not found: {video_file}")

    if not os.path.exists(npy_file):
        raise FileNotFoundError(f"Numpy keypoints not found: {npy_file}")


    # Test usage
    frame_skip = 1

    overlay_keypoints_on_video(video_file, npy_file, frame_skip, output_file)
