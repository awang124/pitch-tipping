"""
pitch_tipping_analyzer.py
=========================
Detects all 33 MediaPipe body landmarks and computes pitch-tipping
features from the batter's viewpoint — works on still images AND video.

ANALYSIS WINDOW
---------------
  setup (set position)  ->  max external rotation (arm fully loaded back)

Usage
-----
  python pitch_tipping_analyzer.py --input pitcher.jpg
  python pitch_tipping_analyzer.py --input pitcher.mp4
  python pitch_tipping_analyzer.py --input pitcher.mp4 --output out.mp4 --auto_window
  python pitch_tipping_analyzer.py --input pitcher.mp4 --json features.json

Setup
-----
  pip install mediapipe opencv-python numpy

  Download the model file before running:
  curl -L -o pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

  Pass the model path via --model (default: ./pose_landmarker.task)
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

BaseOptions          = mp.tasks.BaseOptions
PoseLandmarker       = mp_vision.PoseLandmarker
PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
VisionRunningMode    = mp_vision.RunningMode

# Pose skeleton connections — index pairs matching the 33-landmark topology
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(15,21),
    (17,19),(12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

# Landmark indices (replaces the removed PoseLandmark enum)
NOSE          = 0
LEFT_EAR      = 7
RIGHT_EAR     = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER= 12
LEFT_ELBOW    = 13
RIGHT_ELBOW   = 14
LEFT_WRIST    = 15
RIGHT_WRIST   = 16
LEFT_HIP      = 23
RIGHT_HIP     = 24
LEFT_KNEE     = 25
LEFT_ANKLE    = 27
RIGHT_ANKLE   = 28

LANDMARK_NAMES = {
    0:"NOSE",1:"LEFT_EYE_INNER",2:"LEFT_EYE",3:"LEFT_EYE_OUTER",
    4:"RIGHT_EYE_INNER",5:"RIGHT_EYE",6:"RIGHT_EYE_OUTER",
    7:"LEFT_EAR",8:"RIGHT_EAR",9:"MOUTH_LEFT",10:"MOUTH_RIGHT",
    11:"LEFT_SHOULDER",12:"RIGHT_SHOULDER",13:"LEFT_ELBOW",14:"RIGHT_ELBOW",
    15:"LEFT_WRIST",16:"RIGHT_WRIST",17:"LEFT_PINKY",18:"RIGHT_PINKY",
    19:"LEFT_INDEX",20:"RIGHT_INDEX",21:"LEFT_THUMB",22:"RIGHT_THUMB",
    23:"LEFT_HIP",24:"RIGHT_HIP",25:"LEFT_KNEE",26:"RIGHT_KNEE",
    27:"LEFT_ANKLE",28:"RIGHT_ANKLE",29:"LEFT_HEEL",30:"RIGHT_HEEL",
    31:"LEFT_FOOT_INDEX",32:"RIGHT_FOOT_INDEX",
}


# ─────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────

def lm_xyz(landmarks, idx) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z])


def angle_at_vertex(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    ba = a - vertex
    bc = c - vertex
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def vector_angle_from_horizontal(p1: np.ndarray, p2: np.ndarray) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(np.degrees(np.arctan2(-dy, dx)))


def lateral_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(abs(p1[0] - p2[0]))


def normalised_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


# ─────────────────────────────────────────────
# Feature groups  [setup -> max external rotation]
# ─────────────────────────────────────────────

def shoulder_features(lm) -> dict:
    l_sh = lm_xyz(lm, LEFT_SHOULDER)
    r_sh = lm_xyz(lm, RIGHT_SHOULDER)
    r_hp = lm_xyz(lm, RIGHT_HIP)
    return {
        "shoulder_tilt_deg":        vector_angle_from_horizontal(r_sh, l_sh),
        "shoulder_width_norm":      lateral_distance(l_sh, r_sh),
        "front_shoulder_open_norm": float(l_sh[0] - r_hp[0]),
    }


def glove_arm_features(lm) -> dict:
    l_sh  = lm_xyz(lm, LEFT_SHOULDER)
    l_el  = lm_xyz(lm, LEFT_ELBOW)
    l_wr  = lm_xyz(lm, LEFT_WRIST)
    r_sh  = lm_xyz(lm, RIGHT_SHOULDER)
    mid_x = float((l_sh[0] + r_sh[0]) / 2)
    l_hp   = lm_xyz(lm, LEFT_HIP)
    r_hp   = lm_xyz(lm, RIGHT_HIP)
    hp_mid = (l_hp + r_hp) / 2
    return {
        "glove_elbow_angle_deg":    angle_at_vertex(l_sh, l_el, l_wr),
        "glove_wrist_height_norm":  float(l_wr[1]),
        "glove_above_hip_norm": float(hp_mid[1] - l_wr[1]),
        "glove_wrist_lateral_norm": float(l_wr[0] - mid_x),
        "glove_elbow_flare_deg":    vector_angle_from_horizontal(l_sh, l_el),
    }


def throwing_arm_load_features(lm) -> dict:
    r_sh = lm_xyz(lm, RIGHT_SHOULDER)
    r_el = lm_xyz(lm, RIGHT_ELBOW)
    r_wr = lm_xyz(lm, RIGHT_WRIST)
    r_hp = lm_xyz(lm, RIGHT_HIP)
    return {
        "throw_elbow_angle_deg":     angle_at_vertex(r_sh, r_el, r_wr),
        "throw_elbow_height_norm":   float(r_el[1] - r_sh[1]),
        "throw_elbow_hip_dist_norm": normalised_distance(r_el, r_hp),
        "throw_upper_arm_angle_deg": vector_angle_from_horizontal(r_sh, r_el),
        "throw_forearm_angle_deg":   vector_angle_from_horizontal(r_el, r_wr),
    }


def trunk_features(lm) -> dict:
    l_sh   = lm_xyz(lm, LEFT_SHOULDER)
    r_sh   = lm_xyz(lm, RIGHT_SHOULDER)
    l_hp   = lm_xyz(lm, LEFT_HIP)
    r_hp   = lm_xyz(lm, RIGHT_HIP)
    sh_mid = (l_sh + r_sh) / 2
    hp_mid = (l_hp + r_hp) / 2
    spine  = sh_mid - hp_mid
    return {
        "hip_tilt_deg":           vector_angle_from_horizontal(r_hp, l_hp),
        "trunk_lean_deg":         float(np.degrees(np.arctan2(spine[0], -spine[1]))),
        "hip_shoulder_sep_norm":  float(sh_mid[0] - hp_mid[0]),
        "hip_forward_drive_norm": float(hp_mid[1]),
    }


def head_features(lm) -> dict:
    nose  = lm_xyz(lm, NOSE)
    l_ear = lm_xyz(lm, LEFT_EAR)
    r_ear = lm_xyz(lm, RIGHT_EAR)
    l_sh  = lm_xyz(lm, LEFT_SHOULDER)
    r_sh  = lm_xyz(lm, RIGHT_SHOULDER)
    sh_mid_x = float((l_sh[0] + r_sh[0]) / 2)
    sh_mid_y = float((l_sh[1] + r_sh[1]) / 2)
    return {
        "head_tilt_deg":            vector_angle_from_horizontal(r_ear, l_ear),
        "head_lateral_offset_norm": float(nose[0] - sh_mid_x),
        "head_forward_norm":        float(nose[1] - sh_mid_y),
        "head_rotation_proxy_norm": lateral_distance(l_ear, r_ear),
    }


def leg_and_stride_features(lm) -> dict:
    l_hp = lm_xyz(lm, LEFT_HIP)
    l_kn = lm_xyz(lm, LEFT_KNEE)
    l_an = lm_xyz(lm, LEFT_ANKLE)
    r_an = lm_xyz(lm, RIGHT_ANKLE)
    return {
        "stride_knee_angle_deg":    angle_at_vertex(l_hp, l_kn, l_an),
        "stride_knee_height_norm":  float(l_kn[1]),
        "stride_foot_lateral_norm": float(l_an[0]),
        "stride_length_norm":       normalised_distance(l_an, r_an),
    }


def hand_break_features(lm) -> dict:
    l_wr = lm_xyz(lm, LEFT_WRIST)
    r_wr = lm_xyz(lm, RIGHT_WRIST)
    r_sh = lm_xyz(lm, RIGHT_SHOULDER)
    return {
        "wrist_separation_norm":  normalised_distance(l_wr, r_wr),
        "wrist_vertical_offset":  float(r_wr[1] - l_wr[1]),
        "hand_break_height_norm": float((l_wr[1] + r_wr[1]) / 2),
        "throw_wrist_back_norm":  float(r_wr[0] - r_sh[0]),
    }


# ─────────────────────────────────────────────
# Master feature extractor
# ─────────────────────────────────────────────

@dataclass
class PitchFeatures:
    frame_idx:    int  = 0
    shoulder:     dict = field(default_factory=dict)
    glove_arm:    dict = field(default_factory=dict)
    throwing_arm: dict = field(default_factory=dict)
    trunk:        dict = field(default_factory=dict)
    head:         dict = field(default_factory=dict)
    legs:         dict = field(default_factory=dict)
    hand_break:   dict = field(default_factory=dict)

    def flat(self) -> dict:
        out = {"frame_idx": self.frame_idx}
        for group in [self.shoulder, self.glove_arm, self.throwing_arm,
                      self.trunk, self.head, self.legs, self.hand_break]:
            out.update(group)
        return out

def normalize_to_body(landmarks) -> list:
    l_sh = lm_xyz(landmarks, LEFT_SHOULDER)
    r_sh = lm_xyz(landmarks, RIGHT_SHOULDER)
    l_hp = lm_xyz(landmarks, LEFT_HIP)
    r_hp = lm_xyz(landmarks, RIGHT_HIP)

    origin = (l_hp + r_hp) / 2          # hip midpoint — becomes (0,0)
    sh_mid = (l_sh + r_sh) / 2
    scale  = np.linalg.norm(sh_mid - origin)  # spine length — becomes 1.0

    if scale < 1e-6:
        return landmarks  # fallback if pose is degenerate

    class NormalizedLandmark:
        def __init__(self, x, y, z, visibility):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = visibility

    normalized = []
    for lm in landmarks:
        pt = (np.array([lm.x, lm.y, lm.z]) - origin) / scale
        normalized.append(NormalizedLandmark(pt[0], pt[1], pt[2], lm.visibility))

    return normalized

def extract_features(landmarks, frame_idx: int = 0) -> PitchFeatures:
    lm = normalize_to_body(landmarks)
    # landmarks is result.pose_landmarks[0] — a list of 33 NormalizedLandmark objects
    return PitchFeatures(
        frame_idx    = frame_idx,
        shoulder     = shoulder_features(lm),
        glove_arm    = glove_arm_features(lm),
        throwing_arm = throwing_arm_load_features(lm),
        trunk        = trunk_features(lm),
        head         = head_features(lm),
        legs         = leg_and_stride_features(lm),
        hand_break   = hand_break_features(lm),
    )


# ─────────────────────────────────────────────
# Max ER detection
# ─────────────────────────────────────────────

def find_max_er_frame(all_features: list) -> int:
    if not all_features:
        return 0
    # wrist furthest back + wrist separation together are more robust than
    # elbow height alone (immune to drop-elbow pitchers)
    scores = [
        f.hand_break.get("throw_wrist_back_norm", 0) +
        f.hand_break.get("wrist_separation_norm", 0) +
        f.throwing_arm.get("throw_elbow_hip_dist_norm", 0)
        for f in all_features
    ]
    return int(np.argmax(scores))


# ─────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────

def draw_skeleton(frame: np.ndarray, landmarks: list, img_w: int, img_h: int):
    """Draw skeleton connections and numbered landmark dots onto frame in-place."""
    pts = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        if landmarks[a].visibility > 0.3 and landmarks[b].visibility > 0.3:
            cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2, cv2.LINE_AA)
    for idx, (px, py) in enumerate(pts):
        if landmarks[idx].visibility > 0.3:
            cv2.circle(frame, (px, py), 4, (255, 80, 80), -1, cv2.LINE_AA)
            cv2.putText(frame, str(idx), (px + 4, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (255, 255, 0), 1)

def draw_glove_height_indicator(frame: np.ndarray, landmarks: list, img_w: int, img_h: int):
    l_hp = landmarks[LEFT_HIP]
    r_hp = landmarks[RIGHT_HIP]
    l_wr = landmarks[LEFT_WRIST]

    hp_mid_px = (int((l_hp.x + r_hp.x) / 2 * img_w),
                 int((l_hp.y + r_hp.y) / 2 * img_h))
    wr_px     = (int(l_wr.x * img_w), int(l_wr.y * img_h))

    # vertical purple line from hip height down/up to glove wrist
    cv2.line(frame, (wr_px[0], hp_mid_px[1]), wr_px, (0, 255, 0), 2, cv2.LINE_AA)

    # dot at each endpoint
    cv2.circle(frame, (wr_px[0], hp_mid_px[1]), 5, (0, 255, 0), -1)
    cv2.circle(frame, wr_px, 5, (0, 255, 0), -1)

    # label
    glove_above_hip = hp_mid_px[1] - wr_px[1]
    cv2.putText(frame, f"glove_above_hip: {glove_above_hip}px", (hp_mid_px[0] + 0, hp_mid_px[1] + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)


def draw_feature_text(frame: np.ndarray, features: PitchFeatures,
                      is_in_window: bool = True,
                      anchor: str = "top_left") -> np.ndarray:
    flat   = features.flat()
    line_h = 18
    box_w  = 140
    box_h  = line_h * (len(flat) + 2) + 10

    img_h, img_w = frame.shape[:2]

    if anchor == "top_left":
        x, y   = 10, 20
        box_x1, box_y1 = 0, 0
        box_x2, box_y2 = box_w, box_h
    elif anchor == "top_right":
        x, y   = 500, 20
        box_x1, box_y1 = 490, 0
        box_x2, box_y2 = 490 + box_w, box_h
        # box_x1, box_y1 = img_w - box_w, 0
        # box_x2, box_y2 = img_w, box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    label = "IN WINDOW" if is_in_window else "OUT OF WINDOW"
    color = (0, 255, 100) if is_in_window else (60, 60, 200)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1, cv2.LINE_AA)
    y += line_h

    for key, val in flat.items():
        text = f"frame: {val}" if key == "frame_idx" else \
               f"{key}: {val:+.2f}" if isinstance(val, float) else f"{key}: {val}"
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.20, (0, 255, 180), 1, cv2.LINE_AA)
        y += line_h
    return frame

def sort_poses_left_to_right(pose_landmarks_list):
    def hip_mid_x(landmarks):
        l_hip = landmarks[LEFT_HIP]
        r_hip = landmarks[RIGHT_HIP]
        return (l_hip.x + r_hip.x) / 2
    return sorted(pose_landmarks_list, key=hip_mid_x)


# ─────────────────────────────────────────────
# Image processor
# ─────────────────────────────────────────────

def process_image(path: str, model_path: str,
                  output_path: Optional[str] = None) -> Optional[PitchFeatures]:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    h, w = img.shape[:2]
    rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=2,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_img)

    if not result.pose_landmarks:
        print("No pose detected in image.")
        return None

    annotated = img.copy()
    all_features = []
    sorted_poses = sort_poses_left_to_right(result.pose_landmarks)
    for person_idx, landmarks in enumerate(sorted_poses):
        draw_skeleton(annotated, landmarks, w, h)
        draw_glove_height_indicator(annotated, landmarks, w, h)
        features = extract_features(landmarks, person_idx)
        anchor = "top_right" if person_idx == 1 else "top_left"
        annotated = draw_feature_text(annotated, features, is_in_window=True, anchor=anchor)
        all_features.append([features])

    out = output_path or path.replace(".", "_annotated.")
    cv2.imwrite(out, annotated)
    print(f"Annotated image saved -> {out}")
    print(all_features)
    return all_features


# ─────────────────────────────────────────────
# Video processor
# ─────────────────────────────────────────────

def process_video(path: str, model_path: str,
                  output_path: Optional[str] = None,
                  auto_window: bool = False) -> list:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    all_frames       = []
    all_landmarks    = []   # None or list-of-33 per frame
    all_features_raw = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            result = landmarker.detect_for_video(mp_img, ts_ms)

            all_frames.append(frame)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]
                all_landmarks.append(lms)
                all_features_raw.append(extract_features(lms, frame_idx))
            else:
                all_landmarks.append(None)
                all_features_raw.append(None)

            frame_idx += 1

    cap.release()

    valid_features = [f for f in all_features_raw if f is not None]
    if auto_window and valid_features:
        max_er_frame = find_max_er_frame(valid_features)
        print(f"Auto window: frame 0 -> max ER frame {max_er_frame}")
    else:
        max_er_frame = len(all_frames) - 1

    out_path = output_path or path.replace(".", "_annotated.")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    collected = []
    for i, (frame, lms, features) in enumerate(
            zip(all_frames, all_landmarks, all_features_raw)):
        annotated = frame.copy()
        in_window = i <= max_er_frame

        if lms is not None:
            draw_skeleton(annotated, lms, w, h)
        if features is not None:
            annotated = draw_feature_text(annotated, features, is_in_window=in_window)
            if in_window:
                collected.append(features)

        writer.write(annotated)

        if i % 30 == 0:
            print(f"  Writing... {i}/{total} ({i/max(total,1)*100:.0f}%)")

    writer.release()
    print(f"Annotated video saved -> {out_path}")
    print(f"Frames in tipping window: {len(collected)} / {total}")
    return collected


# ─────────────────────────────────────────────
# JSON export
# ─────────────────────────────────────────────

def save_features_json(features, output_path: str):
    data = [asdict(f) for f in features] if isinstance(features, list) else asdict(features)
    with open(output_path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"Feature data saved -> {output_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pitch tipping analyzer — setup to max external rotation"
    )
    parser.add_argument("--input",       required=True)
    parser.add_argument("--output",      required=False)
    parser.add_argument("--json",        required=False)
    parser.add_argument("--model",       default="pose_landmarker.task",
                        help="Path to pose_landmarker.task model file")
    parser.add_argument("--auto_window", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Download with:")
        print("  curl -L -o pose_landmarker.task https://storage.googleapis.com/"
              "mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
        return

    ext        = os.path.splitext(args.input)[1].lower()
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    if ext in image_exts:
        features = process_image(args.input, args.model, args.output)
        # for 2 people, includes 2 lists of features
        # print(features)
        if features:
            print("\n-- Feature summary --")
            for person_idx, person in enumerate(features):
                print(f"\n-- Person {person_idx} --")
                for f in person:
                    for k, v in f.flat().items():
                        print(f"  {k:<42} {v:+.4f}" if isinstance(v, float) else
                            f"  {k:<42} {v}")
            if args.json:
                save_features_json(features, args.json)

    elif ext in video_exts:
        all_features = process_video(args.input, args.model, args.output,
                                     auto_window=args.auto_window)
        print(f"\n-- {len(all_features)} frames in tipping window --")
        if args.json and all_features:
            save_features_json(all_features, args.json)
    else:
        print(f"Unsupported file type: {ext}")


if __name__ == "__main__":
    main()