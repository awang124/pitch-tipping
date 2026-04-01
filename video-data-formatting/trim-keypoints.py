import argparse
import glob
import os
 
import numpy as np
from dotenv import load_dotenv
 
load_dotenv()
DATA_DIR = os.getenv("EXTRACTION_DATA_DIR")
if DATA_DIR is None:
    raise ValueError("Missing EXTRACTION_DATA_DIR in .env")
 
parser = argparse.ArgumentParser(description="Trim keypoint sequences to a fixed window around the knee apex.")
parser.add_argument("--pitcher", required=True, help="Pitcher name")
parser.add_argument("--frames-before", type=int, default=100, metavar="N",
                    help="Frames to include before the knee apex (default: 100)")
parser.add_argument("--frames-after", type=int, default=50, metavar="N",
                    help="Frames to include after the knee apex (default: 50)")
args = parser.parse_args()
 
PITCHER_NAME  = args.pitcher
FRAMES_BEFORE = args.frames_before
FRAMES_AFTER  = args.frames_after
WINDOW_SIZE   = FRAMES_BEFORE + FRAMES_AFTER   # 150 total
NPY_DIR       = os.path.join(DATA_DIR, PITCHER_NAME, "keypoints_normalized")
OUTPUT_DIR    = os.path.join(DATA_DIR, PITCHER_NAME, "trimmed_keypoints")
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def find_knee_apex(sequence) -> int:
    """
    Return the frame index where the lift knee is highest (minimum Y value).
    Filters out undetected keypoints (y == 0.0).
    """
    highest_y = float("inf")
    apex_idx  = 0
 
    for i, frame_data in enumerate(sequence):
        kpts = frame_data["keypoints"]          # (17, 2)
        lk_y = kpts[13, 1]                      # Left Knee
        rk_y = kpts[14, 1]                      # Right Knee
        valid = [y for y in [lk_y, rk_y] if y > 0]
        if not valid:
            continue
        current_min = min(valid)
        if current_min < highest_y:
            highest_y = current_min
            apex_idx  = i
 
    return apex_idx
 
 
def make_padding_frame(reference_frame: dict) -> dict:
    """
    Return a blank frame shaped like a real frame (keypoints zeroed, box zeroed).
    Used to pad sequences that are shorter than WINDOW_SIZE.
    """
    kpts_shape = np.array(reference_frame["keypoints"]).shape
    return {
        "keypoints": np.zeros(kpts_shape, dtype=float),
        "glove_box": [0.0, 0.0, 0.0, 0.0],
    }
 
 
def enforce_window(sequence, apex_idx: int, frames_before: int, frames_after: int):
    """
    Build an output sequence of exactly (frames_before + frames_after) frames,
    always anchored so that apex_idx lands at position `frames_before`.
 
    Zero-padding is applied unconditionally on both sides:
      - If fewer than `frames_before` real frames exist before the apex,
        the deficit is filled with zero-frames at the front.
      - If fewer than `frames_after` real frames exist after the apex,
        the deficit is filled with zero-frames at the end.
 
    No window shifting — the apex is always at the same relative position
    in every output file, making frame-index features directly comparable
    across sequences.
 
    Returns (trimmed_sequence, warning_message_or_None).
    """
    n           = len(sequence)
    window_size = frames_before + frames_after
    ref_frame   = sequence[0]
    warnings    = []
 
    # --- Frames BEFORE apex ---
    real_before = min(apex_idx, frames_before)          # how many real frames we can take
    pad_before  = frames_before - real_before           # zero-frames needed at the front
    before_frames = (
        [make_padding_frame(ref_frame)] * pad_before
        + list(sequence[apex_idx - real_before : apex_idx])
    )
    if pad_before:
        warnings.append(f"padded {pad_before} zero-frames before apex")
 
    # --- Frames AFTER apex (inclusive of apex itself) ---
    frames_available_after = n - apex_idx               # includes the apex frame
    real_after  = min(frames_available_after, frames_after)
    pad_after   = frames_after - real_after
    after_frames = (
        list(sequence[apex_idx : apex_idx + real_after])
        + [make_padding_frame(ref_frame)] * pad_after
    )
    if pad_after:
        warnings.append(f"padded {pad_after} zero-frames after apex")
 
    trimmed = before_frames + after_frames
 
    # Safety truncate (should never trigger, but guarantees the contract)
    if len(trimmed) > window_size:
        trimmed = trimmed[:window_size]
        warnings.append(f"truncated to {window_size}")
 
    assert len(trimmed) == window_size, (
        f"BUG: expected {window_size} frames, got {len(trimmed)}"
    )
 
    warning = "; ".join(warnings) if warnings else None
    return trimmed, warning
 
 
# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------
 
def process_and_trim():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    files = glob.glob(os.path.join(NPY_DIR, "*.npy"))
    if not files:
        print(f"  No .npy files found in: {NPY_DIR}")
        return
 
    ok_count      = 0
    warning_count = 0
    skip_count    = 0
 
    print(f"  Window      : {FRAMES_BEFORE} before apex + {FRAMES_AFTER} after apex = {WINDOW_SIZE} frames total")
    print(f"  Source dir  : {NPY_DIR}")
    print(f"  Output dir  : {OUTPUT_DIR}")
    print(f"  Files found : {len(files)}\n")
    print(f"  {'File':<45} {'Apex':>5} {'Raw':>5} {'Out':>5}  Notes")
    print("  " + "-" * 80)
 
    for file_path in sorted(files):
        file_name = os.path.basename(file_path)
        sequence  = np.load(file_path, allow_pickle=True)
 
        if len(sequence) == 0:
            print(f"  {file_name:<45} {'—':>5} {'0':>5} {'—':>5}  SKIPPED (empty file)")
            skip_count += 1
            continue
 
        apex_idx = find_knee_apex(sequence)
        raw_len  = len(sequence)
 
        trimmed, warning = enforce_window(sequence, apex_idx, FRAMES_BEFORE, FRAMES_AFTER)
 
        # Verify before saving
        assert len(trimmed) == WINDOW_SIZE, (
            f"Frame count assertion failed for {file_name}: got {len(trimmed)}"
        )
 
        out_path = os.path.join(OUTPUT_DIR, f"trimmed_{file_name}")
        np.save(out_path, np.array(trimmed, dtype=object))
 
        notes = f"WARNING: {warning}" if warning else "ok"
        print(f"  {file_name:<45} {apex_idx:>5} {raw_len:>5} {len(trimmed):>5}  {notes}")
 
        if warning:
            warning_count += 1
        else:
            ok_count += 1
 
    print()
    print(f"  Done.  {ok_count} clean  |  {warning_count} with warnings  |  {skip_count} skipped")
 
 
def main():
    print(f"\nStarting knee-apex trimming for: {PITCHER_NAME}")
    process_and_trim()
 
 
if __name__ == "__main__":
    main()
