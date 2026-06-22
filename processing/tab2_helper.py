import os
import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from processing.kpt_utils import (
    DELIVERY_POST_FRAMES,
    DELIVERY_PRE_FRAMES,
    MIN_DELIVERY_FRAMES,
    findDeliverySyncFrame,
    isValidKpt,
    loadKeypointMeta,
    trimToWindupOnly,
    videoPathFromKeypoints,
)

SKELETON_EDGES = [
    (5, 7), (7, 9),     # LEFT ARM
    (6, 8), (8, 10),    # RIGHT ARM
    (5, 6),             # SHOULDERS
    (5, 11), (6, 12),   # TORSO
    (11, 13), (13, 15), # LEFT LEG
    (12, 14), (14, 16), # RIGHT LEG
    (11, 12)            # HIPS
]

KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

def scorePitch(file, feature_grp):
    """
    Compute a score for a pitch file based on a kinematic feature.
    For joint features, returns maximum frame‑to‑frame displacement.
    For glove, returns maximum speed of glove center.

    Parameters
    ----------
    file: Path to .npy keypoint file.
    feature_grp: String name of the feature (e.g., 'Right Elbow').

    Returns
    -------
    Score value (higher = more motion).
    """
    try:
        if not len(data := np.load(file, allow_pickle=True)):
            return 0
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.tolist()
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data, _ = trimToWindupOnly(data)
        if "glove" in (feat := feature_grp.split(" (")[0].lower().replace(" ", "_")):
            if not (valid_gloves := [
                d["glove_box"] for d in data if "glove_box" in d.keys()
            ]):
                return 0
            if not len(g_centers := np.array([
                ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in valid_gloves
            ])):
                return 0
            return np.max(np.linalg.norm(np.diff(g_centers, axis=0), axis=1))
        seq = np.array([d["keypoints"] for d in data])
        seq, _ = trimToWindupOnly(seq)
        if len(seq) < 2:
            return 0
        idx = {
            "left_shoulder": 5, "right_shoulder": 6, "right_elbow": 8,
            "left_hip": 11, "right_hip": 12, "left_knee": 13, "right_knee": 14,
        }.get(feat, 13)
        return np.max(np.linalg.norm(np.diff(seq[:, idx, :], axis=0), axis=1))
    except:
        return 0

def findBestPitchPair(kpts_root, feature_grp):
    """
    Find fastball / offspeed pitch pair with largest score difference.

    Parameters
    ----------
    kpts_root: Directory containing .npy keypoint files.
    feature_grp: Kinematic feature name.

    Returns
    -------
    Tuple (relative path of best fastball, relative path of best offspeed).
    """
    numpy_files = [
        os.path.join(root_dir, f) for root_dir, _, files
        in os.walk(kpts_root) for f in files if f.endswith(".npy")
    ]
    fastballs = [
        f for f in numpy_files if any(t in os.path.basename(f).lower()
        for t in ["fastball", "cutter", "four_seam", "two_seam"])
    ]
    offspeeds = [
        f for f in numpy_files if any(t in os.path.basename(f).lower()
        for t in ["offspeed", "changeup", "curveball", "slider", "splitter"])
    ]
    if not min(len(fastballs), len(offspeeds)):
        return [], []
    fb_scores = [(f, scorePitch(f, feature_grp)) for f in fastballs]
    os_scores = [(f, scorePitch(f, feature_grp)) for f in offspeeds]
    best_fb = os.path.relpath(max(fb_scores, key=lambda x: x[1])[0], kpts_root)
    best_os = os.path.relpath(max(os_scores, key=lambda x: x[1])[0], kpts_root)
    return best_fb, best_os


def _loadPitchSequenceArrays(kpts_path):
    """Load keypoints, glove boxes, and per-frame video indices from a .npy file."""
    kpts_path = os.path.abspath(kpts_path)
    raw = np.load(kpts_path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.tolist()
    if not raw:
        return None, None, None

    if isinstance(raw[0], dict):
        glove = np.array(
            [frame.get("glove_box", [-1] * 4) for frame in raw], dtype=np.float32
        )
        kpts = np.array([frame["keypoints"] for frame in raw], dtype=np.float32)
        video_frames = np.array(
            [int(frame.get("video_frame", -1)) for frame in raw], dtype=np.int32
        )
    else:
        glove = np.full((len(raw), 4), -1.0, dtype=np.float32)
        kpts = np.array(raw, dtype=np.float32)
        video_frames = np.full(len(raw), -1, dtype=np.int32)

    valid_idx = [i for i, frame in enumerate(kpts) if np.max(frame) > 0]
    if valid_idx:
        kpts = kpts[valid_idx]
        glove = glove[valid_idx]
        video_frames = video_frames[valid_idx]
    return kpts, glove, video_frames


def _trimWindupWithVideoFrames(kpts, glove, video_frames):
    """Slice to [0, windup_end) while keeping video_frame tags aligned."""
    if len(kpts) == 0:
        return kpts, glove, video_frames, {"windup_trim_ok": False}
    trimmed, info = trimToWindupOnly(kpts)
    end = info.get("windup_end")
    if end is None or end >= len(kpts):
        return kpts, glove, video_frames, info
    return trimmed, glove[:end], video_frames[:end], info


def loadAndAlignPitchPair(kpts1_path, kpts2_path, trim_delivery=True):
    """
    Load two pitch sequences and compute a DTW alignment path (Tab 2 logic).

    Parameters
    ----------
    kpts1_path, kpts2_path: Paths to .npy keypoint files.
    trim_delivery: If True, trim to windup-only window before DTW (Tab 2 default).
                   If False, use the full extracted sequence.

    Returns
    -------
    dict with keys kpts1, glove1, video_frames1, kpts2, glove2, video_frames2, path,
    windup_end1, windup_end2, windup_trim_ok1, windup_trim_ok2.
    """
    kpts1, glove1, vf1 = _loadPitchSequenceArrays(kpts1_path)
    kpts2, glove2, vf2 = _loadPitchSequenceArrays(kpts2_path)
    if kpts1 is None or kpts2 is None or len(kpts1) == 0 or len(kpts2) == 0:
        raise ValueError("One or both keypoint files are empty.")

    windup_meta = {"windup_end1": None, "windup_end2": None,
                   "windup_trim_ok1": False, "windup_trim_ok2": False}
    if trim_delivery:
        kpts1, glove1, vf1, info1 = _trimWindupWithVideoFrames(kpts1, glove1, vf1)
        kpts2, glove2, vf2, info2 = _trimWindupWithVideoFrames(kpts2, glove2, vf2)
        windup_meta = {
            "windup_end1": info1.get("windup_end"),
            "windup_end2": info2.get("windup_end"),
            "windup_trim_ok1": info1.get("windup_trim_ok", False),
            "windup_trim_ok2": info2.get("windup_trim_ok", False),
        }

    feat1 = extractDTWFeatures(kpts1)
    feat2 = extractDTWFeatures(kpts2)
    _, path = fastdtw(feat1, feat2, dist=euclidean)

    return {
        "kpts1": kpts1,
        "glove1": glove1,
        "video_frames1": vf1,
        "kpts2": kpts2,
        "glove2": glove2,
        "video_frames2": vf2,
        "path": path,
        **windup_meta,
    }


def filterValidPoints(kpts, indices, frame_width=None, frame_height=None):
    """
    Extract valid keypoints for given indices.

    Parameters
    ----------
    kpts: Array of keypoints (17, 2).
    indices: List of joint indices.

    Returns
    -------
    Tuple (array of valid points, list of corresponding indices).
    """
    pts = []
    valid_indices = []
    for i in indices:
        x, y = kpts[i]
        if isValidKpt((x, y), frame_width, frame_height):
            pts.append([x, y])
            valid_indices.append(i)
    return np.array(pts, dtype=np.float32).reshape(-1, 2), valid_indices

def drawSkeleton(frame, kpts, color, thickness=2, valid_only=False):
    """
    Draw pose skeleton on a frame.

    Parameters
    ----------
    frame: Image to draw on.
    kpts: Array of keypoints (17, 2).
    color: BGR color tuple.
    thickness: Line thickness.
    valid_only: Not used in this version.
    """
    for i, (x, y) in enumerate(kpts):
        if isValidKpt((x, y)):
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    for (a, b) in SKELETON_EDGES:
        if a < len(kpts) and b < len(kpts):
            x1, y1 = kpts[a]
            x2, y2 = kpts[b]
            if isValidKpt((x1, y1)) and isValidKpt((x2, y2)):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def extractDTWFeatures(kpts_sequence):
    """
    Extract multi‑feature signal for DTW alignment
    using knee Y, hip Y, and mean velocity.

    Parameters
    ----------
    kpts_sequence: Array (T, 17, 2) of keypoints.

    Returns
    -------
    Array of shape (T, 3) with features.
    """
    features = []
    prev_frame = None
    for frame in kpts_sequence:
        knee = (frame[13, 1] + frame[14, 1]) /\
            2 if isValidKpt(frame[13]) and isValidKpt(frame[14]) else 0
        hip = (frame[11, 1] + frame[12, 1]) / 2\
            if isValidKpt(frame[11]) and isValidKpt(frame[12]) else 0
        if prev_frame is not None:
            vel = np.mean(np.linalg.norm(frame - prev_frame, axis=1))
        else:
            vel = 0
        prev_frame = frame
        features.append([knee, hip, vel])
    return np.array(features)

def getInterpolatedAnchors(seq):
    """
    Compute interpolated anchor points for every frame in a keypoint sequence.
    Anchors are used for spatial alignment. The function first tries to use
    the hip midpoint, then the shoulder midpoint, then the nose. Frames
    where no anchor is available are filled by linear interpolation.

    Parameters
    ----------
    seq: Array of shape (T, 17, 2) containing skeleton keypoints over T frames.

    Returns
    -------
    Array of shape (T, 2) with an (x, y) anchor for each frame.
    Returns an array of zeros if no valid anchor is ever found.
    """
    anchors = []
    for frame in seq:
        if isValidKpt(frame[11]) and isValidKpt(frame[12]):
            anchors.append((frame[11] + frame[12]) / 2)
        elif isValidKpt(frame[5]) and isValidKpt(frame[6]):
            anchors.append((frame[5] + frame[6]) / 2)
        elif isValidKpt(frame[0]):
            anchors.append(frame[0])
        else:
            anchors.append(None)

    anchors = np.array(anchors)
    valid_indices = np.where([a is not None for a in anchors])[0]
    if len(valid_indices) == 0:
        return np.zeros((len(seq), 2))
    
    all_indices = np.arange(len(seq))
    valid_anchors = np.stack(anchors[valid_indices])
    interp_x = np.interp(all_indices, valid_indices, valid_anchors[:, 0])
    interp_y = np.interp(all_indices, valid_indices, valid_anchors[:, 1])
    return np.stack([interp_x, interp_y], axis=1)

def getTorsoLength(kpts):
    """
    Estimate the torso length from a keypoint sequence.
    For each frame where both shoulders and both hips are valid, the
    torso length is computed as the Euclidean distance between the
    shoulder midpoint and the hip midpoint. The function returns the
    median over all valid frames to obtain a robust, scale‑invariant
    reference length.

    Parameters
    ----------
    kpts: Array of shape (T, 17, 2) containing skeleton keypoints over T frames.

    Returns
    -------
    Median torso length in pixels. 1.0 if no valid frames are found
    """
    lengths = []
    for frame in kpts:
        if isValidKpt(frame[5]) and isValidKpt(frame[6])\
            and isValidKpt(frame[11]) and isValidKpt(frame[12]):
            s_mid = (frame[5] + frame[6]) / 2
            h_mid = (frame[11] + frame[12]) / 2
            lengths.append(np.linalg.norm(s_mid - h_mid))
    return np.median(lengths) if lengths else 1.0

MIN_TORSO_PX = 50
CROP_HALF_W = 2.2
CROP_HALF_H = 2.8
CROP_FALLBACK_HALF_W = 3.0
CROP_FALLBACK_HALF_H = 3.5
MIN_CROP_W = 200
MIN_CROP_H = 300
CROP_COVERAGE_THRESHOLD = 0.80
PERCENTILE_LO = 5
PERCENTILE_HI = 95
TORSO_OUTLIER_RATIO = 2.5

def trimToWindupWindow(kpts, glove):
    """Trim sequence to [0, windup_end) for tipping DTW alignment."""
    if len(kpts) == 0:
        return kpts, glove
    trimmed, info = trimToWindupOnly(kpts)
    end = info.get("windup_end")
    if end is None or end >= len(kpts):
        return kpts, glove
    return trimmed, glove[:end]

def trimToDeliveryWindow(kpts, glove, pre=DELIVERY_PRE_FRAMES, post=DELIVERY_POST_FRAMES):
    """
    Trim a sequence to a fixed window ending near release.
    Both sequences trimmed this way place the delivery sync frame at the same
    relative index, which stabilizes DTW when clips start at different phases.
    """
    if len(kpts) == 0:
        return kpts, glove
    end = findDeliverySyncFrame(kpts)
    start = max(0, end - pre)
    stop = min(len(kpts), end + post)
    target_len = pre + post
    if stop - start < MIN_DELIVERY_FRAMES:
        return kpts, glove
    return kpts[start:stop], glove[start:stop]

def getPairTorsoLengths(kpts1, kpts2):
    """
    Torso lengths for spatial alignment and crop sizing.
    When one estimate is an outlier vs the other, use the larger value for
    both so scale/crop are not driven by a bad detection on one clip.
    """
    len1 = getTorsoLength(kpts1)
    len2 = getTorsoLength(kpts2)
    if len1 > 0 and len2 > 0:
        ratio = max(len1, len2) / min(len1, len2)
        if ratio > TORSO_OUTLIER_RATIO:
            robust = max(len1, len2)
            return robust, robust
    return len1, len2

def _videoPathFromKeypoints(kpts_path, base_dir):
    """Map Keypoints/.../foo.npy to Pitches/.../foo.mov or .mp4."""
    return videoPathFromKeypoints(kpts_path, base_dir)

def _inferFrameSizeFromKeypoints(kpts1, kpts2):
    """Infer landscape vs portrait canvas from keypoint coordinate spans."""
    pts = []
    for kpts in (kpts1, kpts2):
        for frame in kpts:
            for k in range(len(frame)):
                if isValidKpt(frame[k]):
                    pts.append(frame[k])
    if not pts:
        return 1080, 1920
    pts = np.array(pts, dtype=np.float32)
    x_hi = float(np.percentile(pts[:, 0], 98))
    y_hi = float(np.percentile(pts[:, 1], 98))
    margin = 1.12
    if x_hi > y_hi * 1.25:
        width = max(1920, int(x_hi * margin))
        height = max(1080, int(y_hi * margin))
    else:
        width = max(1080, int(x_hi * margin))
        height = max(1920, int(y_hi * margin))
    return width, height

def resolveFrameSize(kpts_path, base_dir=None, kpts1=None, kpts2=None):
    """
    Resolve source video frame dimensions for a keypoint file.
    Prefers sidecar metadata, then paired Pitches/ video, then keypoint bounds.
    """
    meta = loadKeypointMeta(kpts_path)
    if meta.get("frame_width") and meta.get("frame_height"):
        return int(meta["frame_width"]), int(meta["frame_height"])

    video_path = _videoPathFromKeypoints(kpts_path, base_dir)
    if video_path:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if width > 0 and height > 0:
            return width, height

    if kpts1 is not None and kpts2 is not None and len(kpts1) and len(kpts2):
        return _inferFrameSizeFromKeypoints(kpts1, kpts2)
    return 1080, 1920

def _hasValidAnchorCenter(anchors1):
    """True if at least one non-zero anchor exists."""
    return np.any(np.linalg.norm(anchors1, axis=1) > 1e-3)

def _fallbackCenter(kpts1):
    """Body center from valid keypoints when anchors are unavailable."""
    pts = []
    for frame in kpts1:
        pts.extend(_collectFramePoints(frame))
    if pts:
        return np.median(np.array(pts, dtype=np.float32), axis=0)
    return np.array([540.0, 960.0], dtype=np.float32)

def _expandBoxSymmetric(center, x1, y1, x2, y2, pts):
    """Expand crop around fixed center to include all given points."""
    cx, cy = float(center[0]), float(center[1])
    half_w = max(cx - x1, x2 - cx)
    half_h = max(cy - y1, y2 - cy)
    for pt in pts:
        x, y = float(pt[0]), float(pt[1])
        half_w = max(half_w, abs(x - cx))
        half_h = max(half_h, abs(y - cy))
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h

def _coverageInCrop(kpts1, x1, y1, x2, y2):
    """Fraction of valid sequence-1 keypoints that fall inside the crop."""
    total, inside = 0, 0
    for frame in kpts1:
        for k in range(17):
            if isValidKpt(frame[k]):
                total += 1
                x, y = frame[k]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    inside += 1
    return inside / total if total > 0 else 0.0

def _collectAlignedMotionPoints(
    kpts1, kpts2, glove1, glove2, anchors1, anchors2, global_scale, path, is_normalized,
):
    """Collect aligned skeleton/glove points across the DTW path."""
    all_pts = []
    for i, j in path:
        all_pts.extend(_collectFramePoints(kpts1[i], glove1[i]))
        kpts2_aligned = _alignKptsToBase(
            kpts2[j], anchors1[i], anchors2[j], global_scale, is_normalized
        )
        g2_aligned = _alignGloveToBase(
            glove2[j], anchors1[i], anchors2[j], global_scale, is_normalized
        )
        all_pts.extend(_collectFramePoints(kpts2_aligned, g2_aligned))
    return all_pts

def _torsoCropBox(center, torso, half_w_mult, half_h_mult):
    """Build anchor-centered crop from torso multiples."""
    half_w = half_w_mult * torso
    half_h = half_h_mult * torso
    return (
        center[0] - half_w, center[1] - half_h,
        center[0] + half_w, center[1] + half_h,
    )

def _clampCropRect(x1, y1, x2, y2, frame_width, frame_height, torso):
    """Clamp crop to frame bounds and enforce minimum size."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)
    if x2 - x1 < MIN_CROP_W:
        cx = (x1 + x2) / 2
        x1 = max(0, int(cx - MIN_CROP_W / 2))
        x2 = min(frame_width, x1 + MIN_CROP_W)
    if y2 - y1 < MIN_CROP_H:
        cy = (y1 + y2) / 2
        y1 = max(0, int(cy - MIN_CROP_H / 2))
        y2 = min(frame_height, y1 + MIN_CROP_H)
    if x2 <= x1:
        x2 = min(frame_width, x1 + max(int(torso * CROP_HALF_W * 2), MIN_CROP_W))
    if y2 <= y1:
        y2 = min(frame_height, y1 + max(int(torso * CROP_HALF_H * 2), MIN_CROP_H))
    return x1, y1, x2, y2

def _collectFramePoints(kpts, glove=None):
    """Collect valid skeleton and glove coordinates from a single frame."""
    pts = [kpts[k] for k in range(len(kpts)) if isValidKpt(kpts[k])]
    if glove is not None and glove[0] != -1:
        x1, y1, x2, y2 = glove
        pts.extend([[x1, y1], [x2, y2]])
    return pts

def _alignKptsToBase(kpts2, anchor1, anchor2, global_scale, is_normalized):
    """Spatially align sequence-2 keypoints to sequence-1 anchor for one frame."""
    kpts2_aligned = np.copy(kpts2)
    if not is_normalized:
        for k in range(len(kpts2_aligned)):
            if isValidKpt(kpts2_aligned[k]):
                kpts2_aligned[k] = (kpts2_aligned[k] - anchor2) * global_scale + anchor1
    return kpts2_aligned

def _alignGloveToBase(glove, anchor1, anchor2, global_scale, is_normalized):
    """Spatially align a glove box to sequence-1 coordinates."""
    if glove[0] == -1 or is_normalized:
        return None
    shift_vec = anchor1 - anchor2 * global_scale
    x1, y1, x2, y2 = glove
    nx1, ny1 = np.array([x1, y1]) * global_scale + shift_vec
    nx2, ny2 = np.array([x2, y2]) * global_scale + shift_vec
    return [nx1, ny1, nx2, ny2]

def computePitcherCropRect(
    kpts1, kpts2, glove1, glove2, anchors1, anchors2,
    global_scale, path, frame_width, frame_height, is_normalized,
):
    """
    Compute a fixed anchor-centered crop rectangle for the pitcher region.
    Uses torso-scaled dimensions with percentile motion-envelope expansion
    and validation with a wider fallback box when needed.
    """
    center = (
        np.median(anchors1, axis=0)
        if _hasValidAnchorCenter(anchors1)
        else _fallbackCenter(kpts1)
    )
    torso = max(getTorsoLength(kpts1), getTorsoLength(kpts2), MIN_TORSO_PX)

    def build_crop(half_w_mult, half_h_mult):
        x1, y1, x2, y2 = _torsoCropBox(center, torso, half_w_mult, half_h_mult)
        motion_pts = _collectAlignedMotionPoints(
            kpts1, kpts2, glove1, glove2, anchors1, anchors2,
            global_scale, path, is_normalized,
        )
        if motion_pts:
            pts = np.array(motion_pts, dtype=np.float32)
            lo = np.percentile(pts, PERCENTILE_LO, axis=0)
            hi = np.percentile(pts, PERCENTILE_HI, axis=0)
            envelope = np.array([
                lo, hi, [lo[0], hi[1]], [hi[0], lo[1]],
            ], dtype=np.float32)
            x1, y1, x2, y2 = _expandBoxSymmetric(center, x1, y1, x2, y2, envelope)
        return _clampCropRect(x1, y1, x2, y2, frame_width, frame_height, torso)

    x1, y1, x2, y2 = build_crop(CROP_HALF_W, CROP_HALF_H)
    coverage = _coverageInCrop(kpts1, x1, y1, x2, y2)
    crop_w, crop_h = x2 - x1, y2 - y1
    used_fallback = (
        coverage < CROP_COVERAGE_THRESHOLD
        or crop_w < MIN_CROP_W
        or crop_h < MIN_CROP_H
        or not _hasValidAnchorCenter(anchors1)
    )
    if used_fallback:
        x1, y1, x2, y2 = build_crop(CROP_FALLBACK_HALF_W, CROP_FALLBACK_HALF_H)

    return x1, y1, x2, y2, used_fallback

def visualizeDifferences(
    kpts1_path, kpts2_path, output_path, video_bg=None, threshold=15.0,
    highlight="", base_dir=None,
):
    """
    Create a RANSAC‑aligned superimposition video of two pitch deliveries.
    - Load and parse keypoint sequences.
    - Optionally normalize to canvas center.
    - Compute DTW alignment using multi‑feature signals.
    - Anchor‑based scaling and translation to align sequence 2 to sequence 1.
    - Render skeletons, glove boxes, and optional kinematic feature highlighting.

    Parameters
    ----------
    kpts1_path: Path to first .npy keypoint file.
    kpts2_path: Path to second .npy keypoint file.
    output_path: Path to save the output .mp4.
    video_bg: Optional path to background video.
    threshold: Not used in current implementation.
    highlight: String name of feature to highlight (e.g., 'glove', 'right_elbow').
    base_dir: Root directory for Pitches/ and Keypoints/ (used to resolve frame size).

    Returns
    -------
    dict with key ``crop_fallback`` (bool) if a wider fallback crop was used.
    """
    kpts1 = np.load(kpts1_path, allow_pickle=True)
    kpts2 = np.load(kpts2_path, allow_pickle=True)
    
    if isinstance(kpts1, np.ndarray) and kpts1.dtype == object:
        kpts1 = kpts1.tolist()
    if isinstance(kpts2, np.ndarray) and kpts2.dtype == object:
        kpts2 = kpts2.tolist()

    ### SEPARATE POSE AND GLOVE SEQUENCES
    if len(kpts1) and isinstance(kpts1[0], dict):
        glove1 = np.array(
            [frame.get("glove_box", [-1] * 4) for frame in kpts1], dtype=np.float32
        )
        kpts1 = np.array([frame["keypoints"] for frame in kpts1], dtype=np.float32)
    else:
        glove1 = np.full((len(kpts1), 4), -1.0, dtype=np.float32)

    if len(kpts2) and isinstance(kpts2[0], dict):
        glove2 = np.array(
            [frame.get("glove_box", [-1] * 4) for frame in kpts2], dtype=np.float32
        )
        kpts2 = np.array([frame["keypoints"] for frame in kpts2], dtype=np.float32)
    else:
        glove2 = np.full((len(kpts2), 4), -1.0, dtype=np.float32)

    ### STRIP PADDING FRAMES
    valid_idx1 = [i for i, frame in enumerate(kpts1) if np.max(frame) > 0]
    valid_idx2 = [i for i, frame in enumerate(kpts2) if np.max(frame) > 0]
    
    if len(valid_idx1) > 0:
        kpts1 = np.array(kpts1)[valid_idx1]
        glove1 = glove1[valid_idx1]
    if len(valid_idx2) > 0:
        kpts2 = np.array(kpts2)[valid_idx2]
        glove2 = glove2[valid_idx2]

    ### TRIM TO WINDUP-ONLY WINDOW (phase-normalize before DTW)
    kpts1, glove1 = trimToWindupWindow(kpts1, glove1)
    kpts2, glove2 = trimToWindupWindow(kpts2, glove2)
    
    ### DETERMINE NORMALIZATION
    width, height, fps, frames_bg = 1080, 1920, 30, []
    is_normalized = False
    if len(kpts1) > 0:
        valid_pts = kpts1[kpts1 != 0]
        if len(valid_pts) > 0 and np.mean(np.abs(valid_pts)) < 50:
            is_normalized = True
            
        if is_normalized:
            offset_x = width // 2
            offset_y = (height // 2) + 200
            offset = np.array([offset_x, offset_y], dtype=np.float32)
            
            ### APPLY OFFSET TO ALL KEYPOINTS
            for i in range(len(kpts1)):
                kpts1[i] = np.where(kpts1[i] != -1, kpts1[i] + offset, kpts1[i])
            for i in range(len(kpts2)):
                kpts2[i] = np.where(kpts2[i] != -1, kpts2[i] + offset, kpts2[i])
                
            if glove1[0][0] != -1:
                glove1 += np.array([offset_x, offset_y, offset_x, offset_y], dtype=np.float32)
            if glove2[0][0] != -1:
                glove2 += np.array([offset_x, offset_y, offset_x, offset_y], dtype=np.float32)
    
    ### TEMPORAL ALIGNMENT
    feat1 = extractDTWFeatures(kpts1)
    feat2 = extractDTWFeatures(kpts2)
    distance, path = fastdtw(feat1, feat2, dist=euclidean)
    
    ### DETERMINE HIGHLIGHT MODE
    is_glove_highlight = "glove" in highlight.lower()
    highlight_idx = -1
    if not is_glove_highlight and highlight:
        clean_name = highlight.lower().replace("_", " ")
        for i, name in enumerate(KEYPOINT_NAMES):
            if name.lower() in clean_name:
                highlight_idx = i
                break
    
    ### PREPARE VIDEO WRITER
    width, height, fps, frames_bg = 1080, 1920, 30, []
    if video_bg:
        cap = cv2.VideoCapture(video_bg)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_bg.append(frame)
        cap.release()
    elif is_normalized:
        pass
    else:
        width, height = resolveFrameSize(
            kpts1_path, base_dir=base_dir, kpts1=kpts1, kpts2=kpts2
        )
        w2, h2 = resolveFrameSize(kpts2_path, base_dir=base_dir, kpts1=kpts1, kpts2=kpts2)
        width = max(width, w2)
        height = max(height, h2)
    
    ### ALIGNMENT REFERENCES
    anchors1 = getInterpolatedAnchors(kpts1)
    anchors2 = getInterpolatedAnchors(kpts2)
    
    ### GLOBAL SCALE FACTOR
    len1, len2 = getPairTorsoLengths(kpts1, kpts2)
    global_scale = len1 / len2 if len2 > 0 else 1.0

    ### FIXED PITCHER CROP (anchor-centered, computed once)
    crop_x1, crop_y1, crop_x2, crop_y2, crop_fallback = computePitcherCropRect(
        kpts1, kpts2, glove1, glove2, anchors1, anchors2,
        global_scale, path, width, height, is_normalized,
    )
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (crop_w, crop_h)
    )
    
    ### RENDER ALIGNED FRAMES
    max_overall_dist = 0
    max_overall_joint = ""
    max_overall_frame = 0

    for idx, (i, j) in enumerate(path):
        if video_bg and i >= len(frames_bg):
            break
        
        anchor1 = anchors1[i]
        anchor2 = anchors2[j]
        kpts2_aligned = _alignKptsToBase(
            kpts2[j], anchor1, anchor2, global_scale, is_normalized
        )
    
        ### RECORD DEVIATIONS
        max_dist_in_frame = 0
        max_joint_in_frame = ""
        differences = []
        for k in range(17):
            p1 = kpts1[i][k]
            p2 = kpts2_aligned[k]
            if isValidKpt(p1) and isValidKpt(p2):
                dist = euclidean(p1, p2)
                differences.append((k, p1, p2, dist))
                if dist > max_dist_in_frame:
                    max_dist_in_frame = dist
                    max_joint_in_frame = KEYPOINT_NAMES[k]
        if max_dist_in_frame > max_overall_dist:
            max_overall_dist = max_dist_in_frame
            max_overall_joint = max_joint_in_frame
            max_overall_frame = idx
    
        ### PREPARE CANVAS
        if video_bg:
            canvas = frames_bg[i].copy()
            canvas = cv2.addWeighted(canvas, 0.3, np.zeros_like(canvas), 0.7, 0)
        else:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
        ### DRAW ALIGNED SKELETONS
        drawSkeleton(canvas, kpts1[i], (255, 255, 255), thickness=2)
        drawSkeleton(canvas, kpts2_aligned, (255, 100, 50), thickness=2)
    
        ### DRAW ALIGNED GLOVE BOXES
        g2 = glove2[j]
        g2_aligned = _alignGloveToBase(
            g2, anchor1, anchor2, global_scale, is_normalized
        )
        g1 = glove1[i]
        if g1[0] != -1:
            cv2.rectangle(
                canvas, (int(g1[0]), int(g1[1])),
                (int(g1[2]), int(g1[3])), (255, 255, 255), 2
            )
        if g2_aligned is not None:
            cv2.rectangle(
                canvas, (int(g2_aligned[0]), int(g2_aligned[1])),
                (int(g2_aligned[2]), int(g2_aligned[3])), (255, 100, 50), 2
            )
    
        ### HIGHLIGHT SELECTED FEATURE
        if is_glove_highlight:
            if g1[0] != -1:
                cx, cy = int((g1[0] + g1[2]) / 2), int((g1[1] + g1[3]) / 2)
                cv2.circle(canvas, (cx, cy), 40, (0, 0, 255), 4)
            if g2_aligned is not None:
                cx = int((g2_aligned[0] + g2_aligned[2]) / 2)
                cy = int((g2_aligned[1] + g2_aligned[3]) / 2)
                cv2.circle(canvas, (cx, cy), 40, (0, 255, 255), 4)
        elif highlight_idx != -1:
            p1 = kpts1[i][highlight_idx]
            p2 = kpts2_aligned[highlight_idx]
            if isValidKpt(p1):
                cv2.circle(canvas, (int(p1[0]), int(p1[1])), 25, (0, 0, 255), 4)
            if isValidKpt(p2):
                cv2.circle(canvas, (int(p2[0]), int(p2[1])), 25, (0, 255, 255), 4)
            if isValidKpt(p1) and isValidKpt(p2):
                cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 3)
    
        ### ADD TEXT OVERLAYS AND CROP TO PITCHER REGION
        canvas = canvas[crop_y1:crop_y2, crop_x1:crop_x2]
        cv2.putText(
            canvas, f"Pitch 1 (Base): White", (20, crop_h - 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            canvas, f"Pitch 2 (Aligned): Blue", (20, crop_h - 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 50), 2
        )
        if highlight:
            cv2.putText(
                canvas, f"FLAG TARGET: {highlight.upper()}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
            )
        out.write(canvas)
    out.release()
    return {"crop_fallback": crop_fallback}

