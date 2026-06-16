"""MediaPipe Pose Landmarker wrapper for video extraction."""
import os
from contextlib import contextmanager

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

DEFAULT_MODEL_NAME = "pose_landmarker.task"
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)

BaseOptions = mp_tasks.BaseOptions
PoseLandmarker = mp_vision.PoseLandmarker
PoseLandmarkerOptions = mp_vision.PoseLandmarkerOptions
VisionRunningMode = mp_vision.RunningMode


def resolve_model_path(base_dir=None):
    """Resolve pose_landmarker.task under project root or cwd."""
    if base_dir:
        candidate = os.path.join(os.path.abspath(base_dir), DEFAULT_MODEL_NAME)
        if os.path.isfile(candidate):
            return candidate
    cwd_candidate = os.path.join(os.getcwd(), DEFAULT_MODEL_NAME)
    if os.path.isfile(cwd_candidate):
        return cwd_candidate
    return os.path.join(os.getcwd(), DEFAULT_MODEL_NAME)


def ensure_model_path(model_path):
    """Raise a clear error if the .task model file is missing."""
    if os.path.isfile(model_path):
        return model_path
    raise FileNotFoundError(
        f"MediaPipe model not found: {model_path}\n"
        "Download with:\n"
        f"  curl -L -o {DEFAULT_MODEL_NAME} {MODEL_DOWNLOAD_URL}"
    )


def _make_options(model_path, running_mode, num_poses=2):
    return PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=running_mode,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


@contextmanager
def video_landmarker(model_path, num_poses=2):
    """Context manager for VIDEO-mode pose landmarker."""
    model_path = ensure_model_path(model_path)
    options = _make_options(model_path, VisionRunningMode.VIDEO, num_poses=num_poses)
    landmarker = PoseLandmarker.create_from_options(options)
    try:
        yield landmarker
    finally:
        landmarker.close()


@contextmanager
def image_landmarker(model_path, num_poses=2):
    """Context manager for single-frame IMAGE-mode pose landmarker."""
    model_path = ensure_model_path(model_path)
    options = _make_options(model_path, VisionRunningMode.IMAGE, num_poses=num_poses)
    landmarker = PoseLandmarker.create_from_options(options)
    try:
        yield landmarker
    finally:
        landmarker.close()


def detect_video_frame(landmarker, bgr_frame, timestamp_ms):
    """Run VIDEO-mode detection on one BGR frame. Returns list of landmark lists."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_img, int(timestamp_ms))
    return result.pose_landmarks or []


def detect_image_frame(landmarker, bgr_frame):
    """Run IMAGE-mode detection on one BGR frame. Returns list of landmark lists."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    return result.pose_landmarks or []
