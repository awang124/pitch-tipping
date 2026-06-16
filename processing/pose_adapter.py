"""Convert MediaPipe 33-landmark poses to COCO-17 keypoints for downstream tabs."""
import numpy as np

# MediaPipe index -> COCO-17 index
MP_TO_COCO = (
    (0, 0),
    (2, 1),
    (5, 2),
    (7, 3),
    (8, 4),
    (11, 5),
    (12, 6),
    (13, 7),
    (14, 8),
    (15, 9),
    (16, 10),
    (23, 11),
    (24, 12),
    (25, 13),
    (26, 14),
    (27, 15),
    (28, 16),
)

GLOVE_LANDMARKS = (15, 17, 19, 21)
MIN_VISIBILITY = 0.3


def mp_landmarks_to_coco17(landmarks, img_w, img_h, min_vis=MIN_VISIBILITY):
    """Map MediaPipe pose landmarks to COCO-17 pixel coordinates."""
    kpts = np.zeros((17, 2), dtype=np.float32)
    for mp_idx, coco_idx in MP_TO_COCO:
        lm = landmarks[mp_idx]
        if lm.visibility >= min_vis:
            kpts[coco_idx] = (lm.x * img_w, lm.y * img_h)
    return kpts


def landmarks_mean_confidence(landmarks, min_vis=MIN_VISIBILITY):
    """Mean visibility over mapped body joints."""
    vis = [landmarks[i].visibility for i, _ in MP_TO_COCO if landmarks[i].visibility >= min_vis]
    return float(np.mean(vis)) if vis else 0.0


def landmarks_per_joint_confidence(landmarks, min_vis=MIN_VISIBILITY):
    """Per COCO-17 joint visibility (0 if below threshold)."""
    conf = np.zeros(17, dtype=np.float32)
    for mp_idx, coco_idx in MP_TO_COCO:
        if landmarks[mp_idx].visibility >= min_vis:
            conf[coco_idx] = landmarks[mp_idx].visibility
    return conf


def glove_box_from_landmarks(landmarks, img_w, img_h, pad_frac=0.15, min_vis=MIN_VISIBILITY):
    """Bounding box around left-hand landmarks (glove side)."""
    pts = []
    for idx in GLOVE_LANDMARKS:
        lm = landmarks[idx]
        if lm.visibility >= min_vis:
            pts.append((lm.x * img_w, lm.y * img_h))
    if len(pts) < 2:
        return [-1, -1, -1, -1]
    xs, ys = zip(*pts)
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    x1 = max(0, x1 - pad_frac * bw)
    y1 = max(0, y1 - pad_frac * bh)
    x2 = min(img_w, x2 + pad_frac * bw)
    y2 = min(img_h, y2 + pad_frac * bh)
    return [float(x1), float(y1), float(x2), float(y2)]


def poses_to_coco_batch(pose_landmarks_list, img_w, img_h):
    """
    Convert a list of MediaPipe landmark lists to COCO-17 arrays.

    Returns (persons_np, confs_np) with shapes (N, 17, 2) and (N, 17).
    """
    if not pose_landmarks_list:
        return np.zeros((0, 17, 2), dtype=np.float32), np.zeros((0, 17), dtype=np.float32)
    persons, confs = [], []
    for landmarks in pose_landmarks_list:
        persons.append(mp_landmarks_to_coco17(landmarks, img_w, img_h))
        confs.append(landmarks_per_joint_confidence(landmarks))
    return np.array(persons, dtype=np.float32), np.array(confs, dtype=np.float32)
