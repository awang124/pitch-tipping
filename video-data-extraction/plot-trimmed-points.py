
import cv2
import numpy as np
import os
from dotenv import load_dotenv
import argparse

# Same skeleton as before
SKELETON_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), 
    (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)
]

def draw_skeleton(frame, keypoints):
    for x, y in keypoints:
        if x > 0 and y > 0: # Only draw detected points
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    for a, b in SKELETON_EDGES:
        x1, y1 = keypoints[a]
        x2, y2 = keypoints[b]
        if x1 > 0 and x2 > 0:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    return frame

def find_video_start_offset(original_npy_path, trimmed_data):
    """
    Finds which frame of the original video matches the first frame of the trimmed data.
    """
    full_data = np.load(original_npy_path, allow_pickle=True)
    first_trimmed_kpts = trimmed_data[0]["keypoints"]
    
    for idx, frame_data in enumerate(full_data):
        # Compare keypoints (using a small epsilon for float math safety)
        if np.allclose(frame_data["keypoints"], first_trimmed_kpts, atol=1e-3):
            return idx
    return 0

def overlay_trimmed_video(video_path, trimmed_npy_path, original_npy_path, output_path):
    trimmed_data = np.load(trimmed_npy_path, allow_pickle=True)
    
    # 1. Find where the video should actually start
    start_frame = find_video_start_offset(original_npy_path, trimmed_data)
    print(f"Syncing: Trimmed start matches original video frame {start_frame}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 2. Set the video to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    kpt_idx = 0
    while kpt_idx < len(trimmed_data):
        ret, frame = cap.read()
        if not ret:
            break

        frame_data = trimmed_data[kpt_idx]
        keypoints = frame_data["keypoints"]
        glove_box = frame_data["glove_box"]

        # Draw
        frame = draw_skeleton(frame, keypoints)
        x1, y1, x2, y2 = glove_box
        if x1 != -1:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

        out.write(frame)
        kpt_idx += 1

    cap.release()
    out.release()
    print(f"Trimmed overlay saved to: {output_path}")

if __name__ == "__main__":
    load_dotenv()
    DATA_DIR = os.getenv("EXTRACTION_DATA_DIR")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pitcher", required=True)
    parser.add_argument("--pitch", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    # Paths
    video_basename = os.path.splitext(args.video)[0]
    video_file = os.path.join(DATA_DIR, args.pitcher, "pitches", args.pitch, args.video)
    
    # We need BOTH the original and the trimmed to find the sync point
    orig_npy = os.path.join(DATA_DIR, args.pitcher, "keypoints", f"{video_basename}.npy")
    trim_npy = os.path.join(DATA_DIR, args.pitcher, "trimmed_keypoints", f"trimmed_{video_basename}.npy")
    
    output_file = os.path.join(DATA_DIR, args.pitcher, "overlays", f"trimmed_overlay_{args.video}")

    overlay_trimmed_video(video_file, trim_npy, orig_npy, output_file)