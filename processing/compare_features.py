"""Pitch Compare feature highlights aligned with Tab 3 biomechanical features."""
import numpy as np

from processing.kpt_utils import isValidKpt, trimToWindupOnly
from processing.tab3_helper import _compute_angles

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
RIGHT_ELBOW = 8
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13

COMPARE_FEATURE_OPTIONS = [
    ("None", ""),
    ("Glove Speed", "glove_speed"),
    ("Hip-Shoulder Separation", "hip_shoulder_sep"),
    ("Left Knee Height", "left_knee_max_height"),
    ("Right Elbow Angle", "right_elbow"),
    ("Right Wrist Speed", "right_wrist_speed"),
    ("Trunk Inclination", "trunk_inclination"),
    ("Pelvis Rotation", "pelvis_rotation"),
]

_FEATURE_SPECS = {
    "glove_speed": {
        "label": "Glove Speed",
        "joints": [],
        "edges": [],
        "glove": True,
    },
    "hip_shoulder_sep": {
        "label": "Hip-Shoulder Separation",
        "joints": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
        "edges": [(LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP)],
        "glove": False,
    },
    "left_knee_max_height": {
        "label": "Left Knee Height",
        "joints": [LEFT_KNEE],
        "edges": [(LEFT_HIP, LEFT_KNEE)],
        "glove": False,
    },
    "right_elbow": {
        "label": "Right Elbow Angle",
        "joints": [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST],
        "edges": [(RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST)],
        "glove": False,
    },
    "right_wrist_speed": {
        "label": "Right Wrist Speed",
        "joints": [RIGHT_WRIST],
        "edges": [(RIGHT_ELBOW, RIGHT_WRIST)],
        "glove": False,
    },
    "trunk_inclination": {
        "label": "Trunk Inclination",
        "joints": [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
        "edges": [(LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)],
        "glove": False,
    },
    "pelvis_rotation": {
        "label": "Pelvis Rotation",
        "joints": [LEFT_HIP, RIGHT_HIP],
        "edges": [(LEFT_HIP, RIGHT_HIP)],
        "glove": False,
    },
}


def _gloveCenter(glove):
    if glove is None or len(glove) < 4 or glove[0] == -1:
        return None
    return np.array([(glove[0] + glove[2]) / 2, (glove[1] + glove[3]) / 2], dtype=np.float32)


def _hipMid(kpts):
    if not isValidKpt(kpts[LEFT_HIP]) or not isValidKpt(kpts[RIGHT_HIP]):
        return None
    return (kpts[LEFT_HIP] + kpts[RIGHT_HIP]) / 2


def _featureScalar(kpts, glove, feature_key):
    """Single-frame scalar for a Tab 3 compare feature."""
    if feature_key == "glove_speed":
        center = _gloveCenter(glove)
        return float(np.linalg.norm(center)) if center is not None else 0.0
    angles = _compute_angles(kpts)
    if feature_key == "hip_shoulder_sep":
        return float(angles.get("hip_shoulder_sep", 0.0))
    if feature_key == "right_elbow":
        return float(angles.get("right_elbow", 0.0))
    if feature_key == "trunk_inclination":
        return float(angles.get("trunk_inclination", 0.0))
    if feature_key == "pelvis_rotation":
        return float(angles.get("pelvis_rotation", 0.0))
    if feature_key == "left_knee_max_height":
        hip = _hipMid(kpts)
        if hip is None or not isValidKpt(kpts[LEFT_KNEE]):
            return 0.0
        return float(hip[1] - kpts[LEFT_KNEE, 1])
    if feature_key == "right_wrist_speed":
        if not isValidKpt(kpts[RIGHT_WRIST]):
            return 0.0
        return float(np.linalg.norm(kpts[RIGHT_WRIST]))
    return 0.0


def resolveCompareFeature(feature_key):
    """
    Return highlight spec for Pitch Compare.

    Keys: is_glove, joints, edges, label, feature_key
    """
    if not feature_key:
        return {
            "feature_key": "",
            "label": "None",
            "joints": [],
            "edges": [],
            "is_glove": False,
        }
    spec = _FEATURE_SPECS.get(feature_key, {})
    return {
        "feature_key": feature_key,
        "label": spec.get("label", feature_key.replace("_", " ").title()),
        "joints": list(spec.get("joints", [])),
        "edges": list(spec.get("edges", [])),
        "is_glove": bool(spec.get("glove", False)),
    }


def featureSeparation(kpts1, kpts2, glove1, glove2, feature_key):
    """Absolute difference in feature scalar between two poses at one synced frame."""
    if not feature_key:
        return 0.0
    v1 = _featureScalar(kpts1, glove1, feature_key)
    v2 = _featureScalar(kpts2, glove2, feature_key)
    if feature_key == "glove_speed":
        c1, c2 = _gloveCenter(glove1), _gloveCenter(glove2)
        if c1 is None or c2 is None:
            return 0.0
        return float(np.linalg.norm(c1 - c2))
    return abs(v1 - v2)


def scoreCompareFeature(kpts_path, feature_key):
    """Motion score for one pitch — max frame-to-frame change in the feature proxy."""
    if not feature_key:
        return 0.0
    try:
        raw = np.load(kpts_path, allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.dtype == object:
            raw = raw.tolist()
        if not raw:
            return 0.0
        if isinstance(raw[0], dict):
            raw, _ = trimToWindupOnly(raw)
            if len(raw) < 2:
                return 0.0
            values = []
            for frame in raw:
                kpts = np.array(frame["keypoints"], dtype=np.float32)
                glove = np.array(frame.get("glove_box", [-1] * 4), dtype=np.float32)
                values.append(_featureScalar(kpts, glove, feature_key))
            if feature_key == "glove_speed":
                centers = [_gloveCenter(np.array(f.get("glove_box", [-1] * 4))) for f in raw]
                centers = [c for c in centers if c is not None]
                if len(centers) < 2:
                    return 0.0
                return float(np.max(np.linalg.norm(np.diff(np.array(centers), axis=0), axis=1)))
            return float(np.max(np.abs(np.diff(values)))) if len(values) > 1 else 0.0
    except Exception:
        return 0.0
    return 0.0
