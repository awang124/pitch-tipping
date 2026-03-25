import numpy as np
import cv2
import argparse

# TO RUN:
# python overlay_compare.py --file1 a.npy --file2 b.npy --output out.mp4

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
def get_back_foot(kpts):
    left = kpts[15]
    right = kpts[16]

    if np.isnan(left).any():
        return right
    if np.isnan(right).any():
        return left

    return left if left[1] > right[1] else right


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


def compute_initial_offset(sequence):
    """Get translation so frame 0 back foot goes to CENTER"""
    kpts0 = sequence[0]["keypoints"]
    origin = get_back_foot(kpts0)
    return CENTER - origin


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

        # Draw both on SAME frame
        frame = draw_skeleton(frame, kpts1, (0, 255, 0))   # green
        frame = draw_skeleton(frame, kpts2, (0, 0, 255))   # red

        out.write(frame)

        if i % 50 == 0:
            print(f"Frame {i}/{length}")

    out.release()
    print("Done!")


if __name__ == "__main__":
    main()
