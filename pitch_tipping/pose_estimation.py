# This script processes raw baseball pitching videos to extract:
#   - 2D keypoints (17 body joints) for the pitcher using YOLOv8-pose.
#   - Glove (mitt) bounding boxes using a custom YOLOv8 model (best2.pt).
# It then automatically truncates each video to the most relevant part of the
# windup (from about 3 seconds before the pitch release until the release frame).
# The result is saved as a .npy file containing a list of dictionaries, each with
# "keypoints" (17x2 array) and "glove_box" (x1,y1,x2,y2).


get_ipython().system('pip install ultralytics')

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO


def countFrames(video):
    """Return total number of frames in a video file."""
    cap = cv2.VideoCapture(video)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames


def collectVideos(root):
    """
    Recursively find all .MOV files in a directory tree.
    These are the raw pitching clips.
    """
    videos = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.upper().endswith(".MOV"):
                videos.append(os.path.join(dirpath, f))
    return videos


def identifyPitcher(keypoints, frame_shape):
    """
    When multiple people are detected, pick the most likely pitcher.
    Heuristics:
      - Larger bounding box area (x_diff * y_diff) indicates the person closer to camera.
      - Horizontal position near the image center (expected_x = width/2) is typical for pitchers.
    Returns index of the chosen person in the keypoints list.
    """
    if len(keypoints) == 1:
        return 0
    h, w = frame_shape[:2]
    expected_x = 0.5 * w
    idx, score = 0, float("-inf")

    for i, pts in enumerate(keypoints):
        xs, ys = pts[:, 0], pts[:, 1]
        x_diff = torch.max(pts[:, 0]) - torch.min(pts[:, 0])
        y_diff = torch.max(pts[:, 1]) - torch.min(pts[:, 1])
        hori = 1 - torch.abs(torch.nanmean(xs) - expected_x) / w
        # Score = 10 * (area) + horizontal centering bonus
        if (s := 10 * x_diff * y_diff + hori) > score:
            idx, score = i, s
    return idx


def detect_glove_yolo(frame, glove_model, pitcher_box=None):
    """
    Detect the pitcher's glove (baseball mitt) using a custom YOLO model.
    If pitcher_box (bounding box of the pitcher's body) is provided, we choose the
    glove detection that is closest to the pitcher's center. Otherwise, return the
    first detected glove.
    Returns [x1, y1, x2, y2] or None if no glove found.
    """
    results = glove_model(frame, conf=0.05, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    glove_boxes = [b for b, c in zip(boxes, classes) if c == 0]
    if not glove_boxes:
        return None
    if pitcher_box is None:
        return list(map(int, glove_boxes[0]))
    px1, py1, px2, py2 = pitcher_box
    pitcher_center = ((px1 + px2) / 2, (py1 + py2) / 2)

    def dist(box):
        x1, y1, x2, y2 = box
        glove_center = ((x1 + x2) / 2, (y1 + y2) / 2)
        return np.linalg.norm(np.array(glove_center) - np.array(pitcher_center))

    best_box = min(glove_boxes, key=dist)
    return [int(v) for v in best_box]


def getThrowingArm(keypoints):
    """
    Determine the pitcher's throwing arm (right or left) by comparing the horizontal
    range of motion of the right wrist relative to right shoulder vs left wrist to left shoulder.
    The arm that moves more horizontally during the pitch is the throwing arm.
    keypoints: array of shape (T, 17, 2)
    """
    rel_x_right = keypoints[:, 10, 0] - keypoints[:, 6, 0]   # right wrist - right shoulder
    rel_x_left = keypoints[:, 9, 0] - keypoints[:, 5, 0]     # left wrist - left shoulder

    # Interpolate any NaN values (missing detections)
    for arr in [rel_x_right, rel_x_left]:
        if np.any(np.isnan(arr)):
            valid = ~np.isnan(arr)
            arr[:] = np.interp(np.arange(len(arr)), np.where(valid)[0], arr[valid])

    range_right = np.nanmax(rel_x_right) - np.nanmin(rel_x_right)
    range_left = np.nanmax(rel_x_left) - np.nanmin(rel_x_left)
    return "right" if range_right > range_left else "left"


def getForwardDirection(rel_x):
    """
    Determine whether the pitcher is throwing toward positive or negative x direction.
    rel_x is the horizontal displacement of the wrist relative to the shoulder.
    We compare the mean of the first 10% of the sequence to the last 10% – if the
    wrist moves forward (increasing x for a right‑handed pitcher), direction = 1.
    Otherwise direction = -1.
    """
    start = int(0.1 * len(rel_x))
    end = int(0.9 * len(rel_x))
    if end <= start:
        return 1
    start_mean = np.nanmean(rel_x[:start])
    end_mean = np.nanmean(rel_x[end:])
    return 1 if end_mean > start_mean else -1


def findPitchStart(keypoints):
    """
    Find the frame index where the pitcher releases the ball (the "start" of the pitch).
    Algorithm:
      1. Identify throwing arm.
      2. Compute relative horizontal position: wrist_x - shoulder_x.
      3. Multiply by forward direction so that positive means "moving forward".
      4. Smooth the signal with a moving average.
      5. After an initial search offset (30 frames), find the minimum of the smoothed signal.
         This minimum corresponds to the moment the arm is farthest back (cocking phase).
      6. Then move forward until the signal rises above the minimum by a threshold (5% of range)
         and the next few frames are consistently increasing. That rising edge marks the
         beginning of the forward acceleration → release point.
    Returns index of the release frame (pitch start).
    """
    arm = getThrowingArm(keypoints)
    shoulder_idx, wrist_idx = (6, 10) if arm == "right" else (5, 9)
    shoulder_x = keypoints[:, shoulder_idx, 0]
    wrist_x = keypoints[:, wrist_idx, 0]
    rel_x = wrist_x - shoulder_x

    if np.any(np.isnan(rel_x)):
        valid = ~np.isnan(rel_x)
        rel_x = np.interp(np.arange(len(rel_x)), np.where(valid)[0], rel_x[valid])

    direction = getForwardDirection(rel_x)
    signed_rel = direction * rel_x

    kernel = np.ones(window := 5) / window
    signed_rel_smooth = np.convolve(signed_rel, kernel, mode="same")
    if len(signed_rel_smooth) <= (search_start := 30):
        return 0
    min_idx = np.argmin(signed_rel_smooth[search_start:]) + search_start

    range_after = np.max(signed_rel_smooth[min_idx:]) - np.min(signed_rel_smooth[min_idx:])
    threshold = 0.05 * range_after
    for i in range(min_idx + 1, len(signed_rel_smooth)):
        if signed_rel_smooth[i] > signed_rel_smooth[min_idx] + threshold:
            # Require three consecutive increasing frames to avoid noise
            if i + 3 < len(signed_rel_smooth) and all(
                signed_rel_smooth[j] > signed_rel_smooth[j - 1] for j in range(i + 1, i + 4)
            ):
                return i
            return i
    return min_idx


def processVideo(pose_model, glove_model, video, savedir):
    """
    Main processing function for one video.
    Steps:
      - Read video frame by frame.
      - For each frame, run pose estimation and identify the pitcher.
      - Run glove detection and associate the glove with the pitcher.
      - Store keypoints and glove box in a list.
      - After reading all frames, automatically find the pitch release frame (end index)
        and set start index as (end - 180) to keep ~3 seconds before release (assuming 60 fps).
      - Truncate the sequence and save as a .npy file.
    """
    cap = cv2.VideoCapture(video)
    frame_idx, kp_seq = 0, []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = pose_model(frame, verbose=False)[0]
        if not (res.keypoints and len(persons := res.keypoints.xy)):
            frame_idx += 1
            continue
        idx = identifyPitcher(persons, frame.shape)
        person_kpts = persons.cpu().numpy()
        target_kpts = person_kpts[idx]
        try:
            x_min, y_min = np.min(target_kpts, axis=0)
            x_max, y_max = np.max(target_kpts, axis=0)
            pitcher_box = [x_min, y_min, x_max, y_max]
        except:
            pitcher_box = None

        glove_box = detect_glove_yolo(frame, glove_model, pitcher_box)
        if glove_box is None:
            glove_box = [-1, -1, -1, -1]

        kp_seq.append({"keypoints": target_kpts, "glove_box": glove_box})
        frame_idx += 1

    cap.release()
    if not kp_seq:
        print("[ERR] No keypoints extracted")
        return

    # --- Automatic truncation to the most relevant part of the windup ---
    import warnings
    pure_kpts = np.array([frame["keypoints"] for frame in kp_seq])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        end = findPitchStart(pure_kpts) + 1   # release frame index
        start = max(0, end - 180)             # keep ~3 seconds before release (60 fps)
    truncated_kp_seq = kp_seq[start:end]

    os.makedirs(savedir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video))[0]
    np.save(os.path.join(savedir, f"{base}.npy"), np.array(truncated_kp_seq))


# EXECUTION
# from google.colab import drive
# drive.mount("/content/drive")
# 
# pose_model = YOLO("yolov8x-pose.pt")
# # Ensure best2.pt is uploaded to Colab directly before running!
# glove_model = YOLO("best2.pt")
# if torch.cuda.is_available():
#     pose_model.to("cuda")
#     glove_model.to("cuda")
# 
# videos = collectVideos("/content/drive/My Drive/Pitch-Tipping/Pitches")
# print(f"Found {len(videos)} video(s) to process.")
# 
# get_ipython().run_cell_magic('time', '', 'for video in videos:\n    print(f"\\n--- Processing: {video} ---")\n    processVideo(pose_model, glove_model, video, "/content/drive/My Drive/Pitch-Tipping/Keypoints")\n')
