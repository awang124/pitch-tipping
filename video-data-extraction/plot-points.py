import cv2
import numpy as np
import os

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
    # load keypoints (num_frames, num_kpts, 2)
    keypoints = np.load(npy_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_fps = fps / frame_skip

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    frame_idx = 0
    kpt_idx = 0

    print(f"Writing annotated video → {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only annotate frames that we extracted keypoints for
        if frame_idx % frame_skip == 0:
            if kpt_idx < len(keypoints):
                frame = draw_skeleton(frame, keypoints[kpt_idx])
                kpt_idx += 1

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print("Done!")


if __name__ == "__main__":
    # Test usage
    video_file = "SARG/pitch-tipping/data/test-pitcher/pitches/changeup/5182923-uhd_2160_3840_25fps.mp4"
    npy_file = "SARG/pitch-tipping/data/test-pitcher/keypoints/changeup_5182923-uhd_2160_3840_25fps.npy"
    frame_skip = 2
    output_file = "SARG/pitch-tipping/data/test-pitcher/pitches/changeup/overlay.mp4"

    overlay_keypoints_on_video(video_file, npy_file, frame_skip, output_file)
