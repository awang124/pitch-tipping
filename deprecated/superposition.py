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
parser.add_argument("--highlight", type=int, default=None,
                    help="Frame index to highlight and pause on for 0.5 seconds (e.g. 29)")
args = parser.parse_args()

SIZE   = args.size
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
        kpts  = frame["keypoints"]
        left  = kpts[15]
        right = kpts[16]
        if np.isnan(left).any() and np.isnan(right).any():
            continue
        if np.isnan(left).any():
            votes[16] += 1
        elif np.isnan(right).any():
            votes[15] += 1
        elif left[1] > right[1]:
            votes[15] += 1
        else:
            votes[16] += 1
    # back foot locked to right foot only for now ------- CHANGE
    return 16


def compute_initial_offset(sequence, scale):
    """
    Get translation so frame 0 back foot (locked index) goes to CENTER.
    Uses the same majority-voted foot index for consistency.
    """
    back_foot_idx = pick_back_foot_index(sequence)
    kpts0  = sequence[0]["keypoints"] * scale   # scale first
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


def draw_glove_box(frame, raw_glove_box, scale, offset, color):
    """
    Draw glove bounding box using raw pixel coords shifted by the
    same offset applied to the skeleton.
    """
    x1, y1, x2, y2 = raw_glove_box
    if x1 == -1:
        return frame
    x1 = int(x1 * scale + offset[0])
    y1 = int(y1 * scale + offset[1])
    x2 = int(x2 * scale + offset[0])
    y2 = int(y2 * scale + offset[1])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    return frame


def draw_highlight(frame, size, frame_idx):
    """
    Draw a yellow border and frame-number label to indicate the highlighted frame.
    """
    border = 6
    cv2.rectangle(frame, (border, border), (size - border, size - border), (0, 215, 255), border)
    label = f"frame {frame_idx}"
    cv2.putText(
        frame, label,
        (border * 2 + 4, border * 2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA,
    )
    return frame


def draw_legend(frame, name1, name2, size):
    """
    Draw a color-keyed filename legend at the bottom of the frame.
    Green dot = file1, Red dot = file2.
    """
    margin      = 12
    dot_radius  = 6
    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = 0.55
    thickness   = 1
    line_height = 24

    # file2 (red) on the very bottom row, file1 (green) one row above
    entries = [
        (name2, (0, 0, 255)),
        (name1, (0, 255, 0)),
    ]

    for row, (label, color) in enumerate(entries):
        y       = size - margin - row * line_height
        dot_x   = margin + dot_radius
        dot_y   = y - dot_radius // 2
        text_x  = margin + dot_radius * 2 + 8
        text_y  = y

        cv2.circle(frame, (dot_x, dot_y), dot_radius, color, -1)
        cv2.putText(
            frame, label,
            (text_x, text_y),
            font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA,
        )

    return frame

def compute_scale(sequence, target_torso_px=80):
    """
    Compute a scale factor so the torso height (hip mid → shoulder mid)
    matches target_torso_px across sequences, regardless of camera distance.
    Averages over all valid frames for stability.
    """
    torso_lengths = []
    for frame in sequence:
        kpts = frame["keypoints"]
        # Shoulders: 5, 6  |  Hips: 11, 12
        pts = [kpts[5], kpts[6], kpts[11], kpts[12]]
        if any(np.isnan(p).any() for p in pts):
            continue
        shoulder_mid = (kpts[5] + kpts[6]) / 2
        hip_mid      = (kpts[11] + kpts[12]) / 2
        torso_lengths.append(np.linalg.norm(shoulder_mid - hip_mid))

    if not torso_lengths:
        return 1.0  # fallback: no valid frames

    return target_torso_px / np.mean(torso_lengths)


# ---------------- MAIN ----------------

def main():
    data1 = np.load(args.file1, allow_pickle=True)
    data2 = np.load(args.file2, allow_pickle=True)
    length = min(len(data1), len(data2))

    scale1 = compute_scale(data1)
    scale2 = compute_scale(data2)

    offset1 = compute_initial_offset(data1, scale1)
    offset2 = compute_initial_offset(data2, scale2)

    # Strip directory path — show just the filename in the legend
    name1 = args.file1.split("/")[-1]
    name2 = args.file2.split("/")[-1]

    fps    = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(args.output, fourcc, fps, (SIZE, SIZE))

    pause_frames   = int(fps * 0.5)   # 0.5 seconds worth of duplicate frames
    highlight_idx  = args.highlight

    print(f"Writing overlay → {args.output}")
    print(f"  green : {name1}")
    print(f"  red   : {name2}")
    if highlight_idx is not None:
        print(f"  pause : frame {highlight_idx} (0.5 s highlight)")

    for i in range(length):
        frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

        kpts1 = data1[i]["keypoints"] * scale1 + offset1
        kpts2 = data2[i]["keypoints"] * scale2 + offset2

        frame = draw_skeleton(frame,  kpts1, (0, 255, 0))
        frame = draw_skeleton(frame,  kpts2, (0, 0, 255))

        frame = draw_glove_box(frame, data1[i]["glove_box"], scale1, offset1, (0, 255, 0))
        frame = draw_glove_box(frame, data2[i]["glove_box"], scale2, offset2, (0, 0, 255))

        frame = draw_legend(frame, name1, name2, SIZE)

        # Highlight + pause on the specified frame
        if highlight_idx is not None and i == highlight_idx:
            frame = draw_highlight(frame, SIZE, i)
            for _ in range(pause_frames):
                out.write(frame)            # repeat frame to simulate pause
        else:
            out.write(frame)

    out.release()
    print("Done!")


if __name__ == "__main__":
    main()

