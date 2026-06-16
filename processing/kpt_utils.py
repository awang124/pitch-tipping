import json
import os
import numpy as np

DELIVERY_PRE_FRAMES = 90
DELIVERY_POST_FRAMES = 15
DELIVERY_POST_KNEE_FRAMES = 20
PEAK_KNEE_SKIP_FRAC = 0.10
MIN_DELIVERY_FRAMES = 20
META_SUFFIX = "_meta.json"
ARM_JOINT_INDICES = (5, 6, 7, 8, 9, 10)
WARMUP_FRAMES = 25
HYSTERESIS_MARGIN = 0.6
TRACK_SWITCH_MAX_PX = 120
JITTER_RETRY_THRESHOLD = 80.0


def isStretchLandscapePath(path, frame_width=None, frame_height=None):
    """True for stretch-folder clips in landscape orientation."""
    if "stretch" not in (path or "").lower().replace("\\", "/"):
        return False
    if frame_width and frame_height:
        return frame_width >= frame_height
    return True


def computeArmMotionDelta(kpts_prev, kpts_curr):
    """Sum of wrist/elbow displacements between two pose frames."""
    if kpts_prev is None:
        return 0.0
    total = 0.0
    for idx in ARM_JOINT_INDICES:
        p = kpts_prev[idx]
        c = kpts_curr[idx]
        if (p[0] == 0 and p[1] == 0) or (c[0] == 0 and c[1] == 0):
            continue
        total += float(np.linalg.norm(c - p))
    return total


def computeArmMotionScore(kpts_sequence):
    """Total arm joint displacement across a keypoint sequence."""
    if len(kpts_sequence) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(kpts_sequence)):
        total += computeArmMotionDelta(kpts_sequence[i - 1], kpts_sequence[i])
    return total


def _personAnchorFromKpts(kpts):
    if (kpts[11, 0] != 0 or kpts[11, 1] != 0) and (kpts[12, 0] != 0 or kpts[12, 1] != 0):
        return (kpts[11] + kpts[12]) / 2
    if (kpts[5, 0] != 0 or kpts[5, 1] != 0) and (kpts[6, 0] != 0 or kpts[6, 1] != 0):
        return (kpts[5] + kpts[6]) / 2
    valid = kpts[(kpts[:, 0] != 0) | (kpts[:, 1] != 0)]
    return np.mean(valid, axis=0) if len(valid) else np.array([0.0, 0.0])


def metaPathForNpy(npy_path):
    """Sidecar metadata path for a keypoint .npy file."""
    return npy_path.replace(".npy", META_SUFFIX)


def saveKeypointMeta(npy_path, meta):
    """Persist extraction metadata alongside a .npy keypoint file."""
    with open(metaPathForNpy(npy_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def loadKeypointMeta(npy_path):
    """Load sidecar metadata if present."""
    path = metaPathForNpy(npy_path)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def isValidKpt(kpt, frame_width=None, frame_height=None):
    """
    Check if a keypoint coordinate is valid.
    Rejects NaN, zero vector, and coords clearly outside frame bounds.
    """
    x, y = float(kpt[0]), float(kpt[1])
    if np.isnan(x) or np.isnan(y) or (x == 0 and y == 0):
        return False
    if frame_width and frame_height:
        margin = max(frame_width, frame_height) * 0.15
        if x < -margin or y < -margin:
            return False
        if x > frame_width + margin or y > frame_height + margin:
            return False
    return True


def loadKeypointSequence(npy_path):
    """
    Load keypoint sequence and glove boxes from a .npy file.

    Returns
    -------
    kpts: (T, 17, 2) float32 array
    gloves: (T, 4) float32 array
    meta: dict from sidecar if available
    """
    raw = np.load(npy_path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.tolist()
    meta = loadKeypointMeta(npy_path)
    if not raw:
        return np.zeros((0, 17, 2), dtype=np.float32), np.zeros((0, 4), dtype=np.float32), meta
    if isinstance(raw[0], dict):
        gloves = np.array(
            [frame.get("glove_box", [-1, -1, -1, -1]) for frame in raw], dtype=np.float32
        )
        kpts = np.array([frame["keypoints"] for frame in raw], dtype=np.float32)
    else:
        gloves = np.full((len(raw), 4), -1.0, dtype=np.float32)
        kpts = np.array(raw, dtype=np.float32)
    return kpts, gloves, meta


def stripPaddingFrames(kpts, gloves=None):
    """Remove leading zero-padded frames from Tab 1 output."""
    valid_idx = [i for i, frame in enumerate(kpts) if np.max(frame) > 0]
    if not valid_idx:
        return kpts, gloves
    start = valid_idx[0]
    kpts = kpts[start:]
    if gloves is not None:
        gloves = gloves[start:]
    return kpts, gloves


def getThrowingArm(kpts):
    """True if right-handed based on wrist horizontal range vs shoulder."""
    if kpts.ndim != 3 or len(kpts) == 0:
        return True
    relxr = kpts[:, 10, 0] - kpts[:, 6, 0]
    relxl = kpts[:, 9, 0] - kpts[:, 5, 0]
    for arr in (relxr, relxl):
        if np.any(np.isnan(arr)) and np.any(valid := ~np.isnan(arr)):
            arr[:] = np.interp(np.arange(len(arr)), np.where(valid)[0], arr[valid])
    ranger = np.nanmax(relxr) - np.nanmin(relxr) if not np.all(np.isnan(relxr)) else 0
    rangel = np.nanmax(relxl) - np.nanmin(relxl) if not np.all(np.isnan(relxl)) else 0
    return ranger > rangel


def getForwardDirection(relx):
    """1 if moving left-to-right, -1 if right-to-left."""
    if (end := int(0.9 * len(relx))) <= (start := int(0.1 * len(relx))):
        return 1
    smean = np.nanmean(relx[:start]) if not np.all(np.isnan(relx[:start])) else 0
    emean = np.nanmean(relx[end:]) if not np.all(np.isnan(relx[end:])) else 0
    return 2 * (emean > smean) - 1


def findWindupEnd(kpts):
    """Frame index where windup ends using throwing-side wrist-shoulder displacement."""
    if kpts.ndim != 3 or len(kpts) == 0:
        return 0
    arm = getThrowingArm(kpts)
    shoulder, wrist = (6, 10) if arm else (5, 9)
    shoulder_x, wrist_x = kpts[:, shoulder, 0], kpts[:, wrist, 0]
    relx = wrist_x - shoulder_x
    if np.any(np.isnan(relx)) and np.any(valid := ~np.isnan(relx)):
        relx = np.interp(np.arange(len(relx)), np.where(valid)[0], relx[valid])
    srel = np.convolve(relx * getForwardDirection(relx), np.ones(5) / 5, mode="same")
    if len(srel) <= (search_start := min(30, len(srel) - 1)):
        return 0
    minidx = np.argmin(srel[search_start:]) + search_start
    range_after = np.max(srel[minidx:]) - np.min(srel[minidx:])
    for i in range(minidx + 1, len(srel)):
        if srel[i] > srel[minidx] + 0.05 * range_after:
            return i
    return minidx


def findPeakKneeFrame(kpts):
    """Frame of maximum knee lift (min knee Y), ignoring early noisy frames."""
    if kpts.ndim != 3 or len(kpts) == 0:
        return 0
    knee_y = (kpts[:, 13, 1] + kpts[:, 14, 1]) / 2
    valid = (kpts[:, 13, 1] != 0) | (kpts[:, 14, 1] != 0)
    start = int(len(kpts) * PEAK_KNEE_SKIP_FRAC)
    search = knee_y.astype(np.float64)
    search[:start] = np.nan
    search[~valid] = np.nan
    if not np.any(~np.isnan(search)):
        return len(kpts) // 2
    return int(np.nanargmin(search))


def findDeliverySyncFrame(kpts):
    """
    Hybrid delivery sync: windup end with peak-knee fallback for stretch/short clips.
    """
    if kpts.ndim != 3 or len(kpts) == 0:
        return 0
    windup = findWindupEnd(kpts)
    peak = findPeakKneeFrame(kpts)
    if windup <= peak + 5:
        return min(len(kpts) - 1, peak + DELIVERY_POST_KNEE_FRAMES)
    if len(kpts) > 200 and windup < len(kpts) * 0.12:
        return min(len(kpts) - 1, peak + DELIVERY_POST_KNEE_FRAMES + 30)
    return windup


def trimToDeliveryWindow(kpts, pre=DELIVERY_PRE_FRAMES, post=DELIVERY_POST_FRAMES):
    """Trim sequence to fixed window ending near delivery sync frame."""
    if len(kpts) == 0:
        return kpts, 0
    end = findDeliverySyncFrame(kpts)
    start = max(0, end - pre)
    stop = min(len(kpts), end + post)
    if stop - start < MIN_DELIVERY_FRAMES:
        return kpts, end
    return kpts[start:stop], end


def computeQualityMetrics(kpts, gloves, frame_width=None, frame_height=None):
    """
    Compute QA metrics for a keypoint sequence.

    Returns dict with valid_frame_pct, glove_pct, hip_jitter, sync_frame, etc.
    """
    fw = frame_width or 0
    fh = frame_height or 0
    valid_frames = 0
    joint_counts = []
    anchors = []
    for frame in kpts:
        if np.max(frame) <= 0:
            continue
        valid_frames += 1
        n_valid = sum(
            1 for k in range(17) if isValidKpt(frame[k], fw or None, fh or None)
        )
        joint_counts.append(n_valid)
        if isValidKpt(frame[11], fw or None, fh or None) and isValidKpt(
            frame[12], fw or None, fh or None
        ):
            anchors.append((frame[11] + frame[12]) / 2)

    glove_pct = (
        sum(1 for g in gloves if g[0] != -1) / len(gloves) * 100 if len(gloves) else 0
    )
    hip_jitter = float(np.std(anchors, axis=0).mean()) if len(anchors) > 5 else 999.0
    sync_frame = findDeliverySyncFrame(kpts) if len(kpts) else 0
    peak_knee = findPeakKneeFrame(kpts) if len(kpts) else 0
    delivery_hint = "stretch" if sync_frame <= peak_knee + 5 else "windup"

    return {
        "valid_frame_pct": round(valid_frames / max(len(kpts), 1) * 100, 1),
        "avg_joint_pct": round(np.mean(joint_counts) / 17 * 100, 1) if joint_counts else 0,
        "glove_pct": round(glove_pct, 1),
        "hip_jitter_px": round(hip_jitter, 1),
        "sync_frame": int(sync_frame),
        "peak_knee_frame": int(peak_knee),
        "delivery_hint": delivery_hint,
        "num_frames": len(kpts),
    }


def auditStatus(metrics):
    """PASS / WARN / FAIL from QA metrics."""
    if metrics["hip_jitter_px"] > 100 or metrics["avg_joint_pct"] < 50:
        return "FAIL"
    if metrics["hip_jitter_px"] > 40 or metrics["glove_pct"] < 20:
        return "WARN"
    return "PASS"


def parseKeypointSequence(raw):
    """
    Normalize loaded .npy keypoint data into arrays for overlay / QA.

    Returns (kpts_seq, glove_seq, video_frames, first_valid_idx).
    """
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw = raw.tolist()
    if not raw:
        return None, None, None, 0

    if isinstance(raw[0], dict):
        glove_seq = np.array(
            [frame.get("glove_box", [-1, -1, -1, -1]) for frame in raw],
            dtype=np.float32,
        )
        kpts_seq = np.array([frame["keypoints"] for frame in raw], dtype=np.float32)
        video_frames = np.array(
            [int(frame.get("video_frame", -1)) for frame in raw],
            dtype=np.int32,
        )
    else:
        glove_seq = np.full((len(raw), 4), -1.0, dtype=np.float32)
        kpts_seq = np.array(raw, dtype=np.float32)
        video_frames = np.full(len(raw), -1, dtype=np.int32)

    valid_mask = [np.max(frame) > 0 for frame in kpts_seq]
    first_valid = next((i for i, v in enumerate(valid_mask) if v), 0)
    return (
        kpts_seq[first_valid:],
        glove_seq[first_valid:],
        video_frames[first_valid:],
        first_valid,
    )


def fullKeypointsPath(kpts_path):
    """Deprecated companion path from an earlier two-file save; prefer main .npy."""
    base, ext = os.path.splitext(kpts_path)
    if base.endswith("_full"):
        return kpts_path
    return f"{base}_full{ext}"


def resolveOverlayKeypointsPath(kpts_path):
    """
    Pick the best on-disk keypoint file for full-video overlay.

    Prefers a legacy *_full.npy companion when the main .npy is still a short
    delivery-window extract from an older pipeline version.
    """
    full_path = fullKeypointsPath(kpts_path)
    if not os.path.isfile(full_path):
        return kpts_path
    try:
        main_len = len(np.load(kpts_path, allow_pickle=True))
        full_len = len(np.load(full_path, allow_pickle=True))
        if full_len > main_len:
            return full_path
    except Exception:
        pass
    return kpts_path


def buildVideoFrameLookup(kpts_seq, glove_seq, video_frames, video_start_frame):
    """Map source video frame index to (keypoints, glove_box)."""
    lookup = {}
    has_video_frames = video_frames is not None and np.any(video_frames >= 0)
    for i in range(len(kpts_seq)):
        if np.max(kpts_seq[i]) <= 0:
            continue
        if has_video_frames and video_frames[i] >= 0:
            vf = int(video_frames[i])
        else:
            vf = int(video_start_frame + i)
        lookup[vf] = (kpts_seq[i], glove_seq[i])
    return lookup


def resolveVideoAlignStart(kpts_path, n_kpt_frames, total_video_frames):
    """
    Map keypoint index 0 (after stripping zero-padding) to source video frame.
    """
    meta = loadKeypointMeta(kpts_path)
    if meta.get("video_align_start") is not None:
        return int(meta["video_align_start"])
    try:
        raw = np.load(kpts_path, allow_pickle=True)
        _, _, video_frames, _ = parseKeypointSequence(raw)
        if video_frames is not None:
            mapped = video_frames[video_frames >= 0]
            if len(mapped):
                return int(mapped[0])
    except Exception:
        pass
    return max(0, total_video_frames - n_kpt_frames)


def videoPathFromKeypoints(kpts_path, base_dir=None):
    """Map Keypoints/.../foo.npy to Pitches/.../foo.mov or .mp4."""
    kpts_path = os.path.abspath(kpts_path)
    rel = None
    if base_dir:
        kp_root = os.path.join(os.path.abspath(base_dir), "Keypoints")
        if kpts_path.startswith(kp_root):
            rel = os.path.relpath(kpts_path, kp_root)
    if rel is None:
        parts = kpts_path.replace("\\", "/").split("/")
        if "Keypoints" in parts:
            idx = parts.index("Keypoints")
            rel = os.path.join(*parts[idx + 1:])
    if rel is None:
        return None
    pitch_root = (
        os.path.join(os.path.abspath(base_dir), "Pitches")
        if base_dir
        else os.path.dirname(kpts_path).replace("Keypoints", "Pitches")
    )
    stem = os.path.splitext(rel)[0]
    for ext in (".mov", ".mp4", ".MOV", ".MP4"):
        candidate = os.path.join(pitch_root, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None
