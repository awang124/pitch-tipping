import argparse

from dotenv import load_dotenv
import numpy as np
import os
import glob



load_dotenv()
DATA_DIR = os.getenv("EXTRACTION_DATA_DIR")
if DATA_DIR is None:
    raise ValueError("Missing EXTRACTION_DATA_DIR in .env")

parser = argparse.ArgumentParser(description="Extract keypoints from videos.")
parser.add_argument("--pitcher", required=True, help="Pitcher name")
args = parser.parse_args()
PITCHER_NAME = args.pitcher

NPY_DIR = os.path.join(DATA_DIR, PITCHER_NAME, "keypoints")
OUTPUT_DIR = os.path.join(DATA_DIR, PITCHER_NAME, "trimmed_keypoints")

def find_knee_apex(sequence):
    """
    Finds the frame index where the knee is highest (minimum Y).
    Expects a list/array of dictionaries with a 'keypoints' key.
    """
    highest_y = float('inf')
    apex_idx = 0

    for i, frame_data in enumerate(sequence):
        # Extract the keypoints array (17, 2) from the dictionary
        kpts = frame_data["keypoints"]
        
        # YOLO/COCO Indices: 13 (Left Knee), 14 (Right Knee)
        # kpts[idx, 1] is the Y coordinate
        lk_y = kpts[13, 1]
        rk_y = kpts[14, 1]

        # Filter out 0.0 (undetected)
        current_min = min([y for y in [lk_y, rk_y] if y > 0] or [float('inf')])

        if current_min < highest_y:
            highest_y = current_min
            apex_idx = i

    return apex_idx

def process_and_trim(window_size=100):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Find all .npy files
    files = glob.glob(os.path.join(NPY_DIR, "*.npy"))

    for file_path in files:
        # allow_pickle=True is required because you saved a list of dictionaries
        sequence = np.load(file_path, allow_pickle=True)
        
        if len(sequence) == 0:
            continue

        apex_idx = find_knee_apex(sequence)
        
        # --- WINDOW CALCULATION ---
        # Calculate start and end with 'clamping' to stay within array bounds
        start_frame = max(0, apex_idx - window_size)
        end_frame = min(len(sequence), apex_idx + window_size - (window_size // 2))
        
        # Slice the 200-frame (approx) chunk
        trimmed_sequence = sequence[start_frame:end_frame]
        
        # Save output
        file_name = os.path.basename(file_path)
        out_path = os.path.join(OUTPUT_DIR, f"trimmed_{file_name}")
        np.save(out_path, trimmed_sequence)
        
        print(f"File name: {file_name}  - Apex Frame: {apex_idx}")

def main():
    # Update these paths to match your extraction output

    print("Starting Knee-Apex Trimming...")
    process_and_trim()
    print("Done!")

if __name__ == "__main__":
    main()