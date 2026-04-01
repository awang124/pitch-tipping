import os
from dotenv import load_dotenv
import numpy as np
import argparse

# RUN THIS BEFORE TRIMMING TO NORMALIZE KEYPOINTS WITH BACK FOOT AS ORIGIN:
# python normalize_keypoints.py --pitcher <pitcher>

# ---------------- ARGUMENTS ----------------
parser = argparse.ArgumentParser(description="Normalize keypoints with back foot as origin.")
parser.add_argument("--pitcher", required=True, help="Pitcher name")
args = parser.parse_args()

PITCHER_NAME = args.pitcher

load_dotenv()
DATA_DIR = os.getenv("EXTRACTION_DATA_DIR")
if DATA_DIR is None:
    raise ValueError("Missing EXTRACTION_DATA_DIR in .env")

INPUT_DIR = os.path.join(DATA_DIR, PITCHER_NAME, "keypoints")
OUTPUT_DIR = os.path.join(DATA_DIR, PITCHER_NAME, "keypoints_normalized")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- KEYPOINT LOGIC ----------------

def pick_back_foot_index(kpts):
    """
    Determine which ankle index (15=left, 16=right) is the back foot,
    decided ONCE per sequence using a majority vote across all frames.
    This prevents the origin from flickering between feet mid-delivery.
    """
    votes = {15: 0, 16: 0}
 
    for frame in kpts:
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


def normalize_keypoints(kpts, back_foot_idx):
    """
    Translate all keypoints so the locked back foot is (0, 0).
    Returns normalized keypoints and the origin used.
    """
    origin = kpts[back_foot_idx].copy()
    return kpts - origin, origin
 
 
def normalize_glove_box(glove_box, origin):
    x1, y1, x2, y2 = glove_box
    if x1 == -1:
        return glove_box
    return [
        x1 - origin[0],
        y1 - origin[1],
        x2 - origin[0],
        y2 - origin[1]
    ]


# ---------------- PROCESS FILE ----------------
def process_file(file_path):
    data = np.load(file_path, allow_pickle=True)
 
    # Lock back foot index once for the whole sequence via majority vote
    back_foot_idx = pick_back_foot_index(data)
    print(f"  Back foot locked to: {'LEFT' if back_foot_idx == 15 else 'RIGHT'} ankle (index {back_foot_idx})")
 
    normalized_sequence = []
    for frame in data:
        kpts = frame["keypoints"]
        norm_kpts, origin = normalize_keypoints(kpts, back_foot_idx)
        norm_glove_box = normalize_glove_box(frame["glove_box"], origin)
 
        normalized_sequence.append({
            "keypoints": norm_kpts,
            "glove_box": norm_glove_box,        # normalized (for spatial ML analysis)
            "raw_glove_box": frame["glove_box"]  # original pixel coords (for visualizer)
        })
 
    normalized_sequence = np.array(normalized_sequence, dtype=object)
    filename = os.path.basename(file_path)
    name, _ = os.path.splitext(filename)
    out_path = os.path.join(OUTPUT_DIR, f"{name}_normalized.npy")
    np.save(out_path, normalized_sequence)
    print(f"  Saved: {out_path} shape={normalized_sequence.shape}")
 
 
# ---------------- MAIN ----------------
def main():
    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Keypoint directory not found: {INPUT_DIR}")
 
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".npy"):
            continue
        file_path = os.path.join(INPUT_DIR, filename)
        print(f"Processing: {filename}")
        process_file(file_path)
 
 
if __name__ == "__main__":
    main()
 
