import os
from dotenv import load_dotenv
import numpy as np
import argparse

# TO RUN:
# python normalize_keypoints.py --pitcher <pitcher>

# ---------------- ARGUMENTS ----------------
parser = argparse.ArgumentParser(description="Normalize keypoints with back foot as origin.")
parser.add_argument("--pitcher", required=True, help="Pitcher name")
parser.add_argument("--data_dir", required=True, help="Base data directory")
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

def get_back_foot(kpts):
    """
    kpts: (17, 2)
    returns: (2,) back foot coordinate
    """
    left_ankle = kpts[15]
    right_ankle = kpts[16]

    # Handle missing points (NaNs)
    if np.isnan(left_ankle).any():
        return right_ankle
    if np.isnan(right_ankle).any():
        return left_ankle

    # Larger y = lower in image
    return left_ankle if left_ankle[1] > right_ankle[1] else right_ankle


def normalize_keypoints(kpts):
    """
    Translate all keypoints so back foot is (0,0)
    """
    origin = get_back_foot(kpts)
    return kpts - origin

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

    normalized_sequence = []

    for frame in data:
        kpts = frame["keypoints"]

        norm_kpts = normalize_keypoints(kpts)

        normalized_sequence.append({
            "keypoints": norm_kpts,
            "glove_box": frame["glove_box"]  # unchanged
        })

    normalized_sequence = np.array(normalized_sequence, dtype=object)

    filename = os.path.basename(file_path)
    name, _ = os.path.splitext(filename)
    out_path = os.path.join(OUTPUT_DIR, f"{name}_normalized.npy")

    np.save(out_path, normalized_sequence)

    print(f"Saved: {out_path} shape={normalized_sequence.shape}")


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
