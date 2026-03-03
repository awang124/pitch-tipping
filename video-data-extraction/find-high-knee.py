import os
from dotenv import load_dotenv
import numpy as np
import cv2
from scipy.signal import find_peaks

# -------------------------
# CONFIG
# -------------------------
LEFT_KNEE = 13
RIGHT_KNEE = 14

SQUARE_SIZE = 150
COLOR = (0, 0, 255)   # Red square
THICKNESS = 6

# -------------------------
# LOAD ENV + NPY
# -------------------------
load_dotenv()
BASE_DIR = os.getenv("EXTRACTION_DATA_DIR")
if BASE_DIR is None:
    raise ValueError("Missing EXTRACTION_DATA_DIR in .env")

NPY_PATH = os.path.join(BASE_DIR, "kieran/keypoints/IMG_6116.npy")
VIDEO_PATH = os.path.join(BASE_DIR, "kieran/Four-Seam-Fastball/IMG_6116.mov")
OUTPUT_PATH = os.path.join(BASE_DIR, "kieran/knee_up_vid/IMG_6116.mov")

print("Loading:", NPY_PATH)
data = np.load(NPY_PATH)  # shape: (frames, 17, 2)
frame_count = data.shape[0]

# -------------------------
# FIND PEAK FRAMES
# -------------------------
def find_knee_peak(knee_y):
    smooth = np.convolve(knee_y, np.ones(5)/5, mode='same')
    inv = -smooth
    peaks, _ = find_peaks(inv, prominence=3, distance=5)
    return int(peaks[0]) if len(peaks) > 0 else None

# left knee
knee_yl = data[:, LEFT_KNEE, 1]
left_peak = find_knee_peak(knee_yl)
print("Left knee peak frame:", left_peak)

# right knee
knee_yr = data[:, RIGHT_KNEE, 1]
right_peak = find_knee_peak(knee_yr)
print("Right knee peak frame:", right_peak)

# -------------------------
# OPEN VIDEO FOR DRAWING
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Could not open video at " + VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

frame_index = 0

print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw square for LEFT knee
    if left_peak is not None and frame_index == left_peak:
        x, y = data[frame_index, LEFT_KNEE]
        x1 = int(x - SQUARE_SIZE // 2)
        y1 = int(y - SQUARE_SIZE // 2)
        x2 = int(x + SQUARE_SIZE // 2)
        y2 = int(y + SQUARE_SIZE // 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, THICKNESS)

    # Draw square for RIGHT knee
    if right_peak is not None and frame_index == right_peak:
        x, y = data[frame_index, RIGHT_KNEE]
        x1 = int(x - SQUARE_SIZE // 2)
        y1 = int(y - SQUARE_SIZE // 2)
        x2 = int(x + SQUARE_SIZE // 2)
        y2 = int(y + SQUARE_SIZE // 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, THICKNESS)

    out.write(frame)
    frame_index += 1

cap.release()
out.release()

print("Done!")
print("Saved output video →", OUTPUT_PATH)
