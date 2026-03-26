import numpy as np
import cv2
import argparse

# TO RUN:
# python superposition.py --file1 a.npy --file2 b.npy --output out.mp4

# ---------------- ARGUMENTS ----------------
parser = argparse.ArgumentParser(description="Overlay two keypoint sequences (shared origin)")
parser.add_argument("--file1", required=True)
parser.add_argument("--file2", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--size", type=int, default=720)
args = parser.parse_args()

SIZE = args.size
CENTER = np.array([SIZE // 2, SIZE // 2])

# ---------------- SKELETON ----------------
SKELETON_EDGES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12)
]

# ---------------- HELPERS ----------------
def pick_back_foot_index(data):
    """
    Majority vote across all frames to lock back foot to one ankle index.
    Matches the same logic used in normalize_keypoints.py.
    """
    votes = {15: 0, 16: 0}
    for frame in data:
        kpts = frame["keypoints"]
        left = kpts[15]
        right = kpts[16]
        if np.isnan(left).any() and np.isnan(right).any():
            continue
        if np.isnan(left).any():
            votes[16] += 1
        elif np.isnan(right).any():
            votes[15] += 1

        #compares y values right now ------- CHANGE
        elif left[1] > right[1]:
            votes[15] += 1
        else:
            votes[16] += 1

    #back foot locked to right foot only for now ------- CHANGE
    return 16 
    return 15 if votes[15] >= votes[16] else 16

def compute_initial_offset(sequence):
    """
    Get translation so frame 0 back foot (locked index) goes to CENTER.
    Uses the same majority-voted foot index for consistency.
    """
    back_foot_idx = pick_back_foot_index(sequence)
    kpts0 = sequence[0]["keypoints"]
    origin = kpts0[back_foot_idx].copy()
    return CENTER - origin

def draw_skeleton(frame, kpts, color):
    for x, y in kpts:
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    for a, b in SKELETON_EDGES:
        x1, y1 = kpts[a]
        x2, y2 = kpts[b]
        if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return frame

def draw_glove_box(frame, raw_glove_box, offset, color):
    """
    Draw glove bounding box using raw pixel coords shifted by the
    same offset applied to the skeleton.
    """
    x1, y1, x2, y2 = raw_glove_box
    if x1 == -1:
        return frame
    x1 = int(x1 + offset[0])
    y1 = int(y1 + offset[1])
    x2 = int(x2 + offset[0])
    y2 = int(y2 + offset[1])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame

# ---------------- MAIN ----------------
def main():
    data1 = np.load(args.file1, allow_pickle=True)
    data2 = np.load(args.file2, allow_pickle=True)
 
    length = min(len(data1), len(data2))
 
    offset1 = compute_initial_offset(data1)
    offset2 = compute_initial_offset(data2)
 
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (SIZE, SIZE))
 
    print(f"Writing overlay → {args.output}")
 
    for i in range(length):
        frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
 
        kpts1 = data1[i]["keypoints"] + offset1
        kpts2 = data2[i]["keypoints"] + offset2
 
        frame = draw_skeleton(frame, kpts1, (0, 255, 0))   # green
        frame = draw_skeleton(frame, kpts2, (0, 0, 255))   # red
 
        # Draw glove boxes using raw pixel coords + same offset as skeleton
        frame = draw_glove_box(frame, data1[i]["glove_box"], offset1, (0, 255, 0))
        frame = draw_glove_box(frame, data2[i]["glove_box"], offset2, (0, 0, 255))
 
        out.write(frame)
        if i % 50 == 0:
            print(f"Frame {i}/{length}")
 
    out.release()
    print("Done!")
 
 
if __name__ == "__main__":
    main()
