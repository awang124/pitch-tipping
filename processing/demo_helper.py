import os

import cv2
import numpy as np
from scipy.spatial.distance import euclidean

from processing.kpt_utils import (
    isValidKpt,
    loadKeypointMeta,
    resolveOverlayKeypointsPath,
    resolveVideoAlignStart,
    videoPathFromKeypoints,
)
from processing.tab2_helper import loadAndAlignPitchPair, scorePitch
from processing.tab5_helper import (
    KEYPOINT_NAMES,
    _composeOverlayFrame,
    _resolveHighlight,
)

FB_TOKENS = ("fastball", "cutter", "four_seam", "two_seam")
OS_TOKENS = ("offspeed", "changeup", "curveball", "slider", "splitter", "off_")


def isFastballFilename(name):
    lower = name.lower()
    return any(t in lower for t in FB_TOKENS)


def isOffspeedFilename(name):
    lower = name.lower()
    if "unlabel" in lower:
        return False
    return any(t in lower for t in OS_TOKENS)


def listPitcherFolders(keypoint_root):
    if not os.path.isdir(keypoint_root):
        return []
    folders = []
    for name in sorted(os.listdir(keypoint_root)):
        path = os.path.join(keypoint_root, name)
        if os.path.isdir(path) and any(f.endswith(".npy") for f in os.listdir(path)):
            folders.append(name)
    return folders


def listPitchFiles(keypoint_root, pitcher, pitch_filter):
    folder = os.path.join(keypoint_root, pitcher)
    if not os.path.isdir(folder):
        return []
    files = []
    for f in sorted(os.listdir(folder)):
        if not f.endswith(".npy"):
            continue
        if pitch_filter(f):
            files.append(f)
    return files


def _openVideoContext(kpts_path, base_dir, n_kpt_frames):
    video_path = videoPathFromKeypoints(kpts_path, base_dir)
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"No source video for {kpts_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta = loadKeypointMeta(kpts_path)
    fw = meta.get("frame_width") or vid_w
    fh = meta.get("frame_height") or vid_h
    video_start = resolveVideoAlignStart(kpts_path, n_kpt_frames, total_frames)
    return {
        "cap": cap,
        "video_path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "vid_w": vid_w,
        "vid_h": vid_h,
        "fw": fw,
        "fh": fh,
        "video_start": video_start,
    }


def _videoFrameForKptIndex(video_frames, kpt_idx, video_start):
    if video_frames is not None and kpt_idx < len(video_frames) and video_frames[kpt_idx] >= 0:
        return int(video_frames[kpt_idx])
    return int(video_start + kpt_idx)


def _buildSyncedTimeline(aligned, ctx1, ctx2):
    """
    Build a full-pitch timeline: parallel pre-roll, DTW-synced delivery, parallel post-roll.

    Each entry is (vf1, vf2, kpt_idx1, kpt_idx2) where kpt indices are None outside
    the DTW-synced delivery phase.
    """
    path = aligned["path"]
    vf1_arr = aligned["video_frames1"]
    vf2_arr = aligned["video_frames2"]
    vs1, vs2 = ctx1["video_start"], ctx2["video_start"]
    n1, n2 = len(aligned["kpts1"]), len(aligned["kpts2"])
    te1, te2 = ctx1["total_frames"], ctx2["total_frames"]

    dtw_pairs = []
    for i, j in path:
        dtw_pairs.append((
            _videoFrameForKptIndex(vf1_arr, i, vs1),
            _videoFrameForKptIndex(vf2_arr, j, vs2),
            i,
            j,
        ))

    if not dtw_pairs:
        return []

    first_vf1, first_vf2, _, _ = dtw_pairs[0]
    last_vf1, last_vf2, _, _ = dtw_pairs[-1]

    timeline = []

    # Pre-roll: both videos from frame 0 up to the longer wind-up lead-in.
    for t in range(max(first_vf1, first_vf2)):
        timeline.append((min(t, te1 - 1), min(t, te2 - 1), None, None))

    # Delivery: DTW-synced pairs (Tab 2 alignment).
    prev = None
    for vf1, vf2, i, j in dtw_pairs:
        key = (vf1, vf2)
        if key == prev:
            continue
        prev = key
        timeline.append((vf1, vf2, i, j))

    # Post-roll: parallel follow-through after keypoint coverage ends.
    tail1 = te1 - last_vf1 - 1
    tail2 = te2 - last_vf2 - 1
    for k in range(1, max(tail1, tail2) + 1):
        timeline.append((
            min(last_vf1 + k, te1 - 1),
            min(last_vf2 + k, te2 - 1),
            None,
            None,
        ))

    return timeline


def _readVideoFrame(cap, frame_idx, fallback_w, fallback_h):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ret, frame = cap.read()
    if not ret or frame is None:
        return np.zeros((fallback_h, fallback_w, 3), dtype=np.uint8)
    return frame


def _scaleToHeight(frame, target_h):
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    scale = target_h / h
    new_w = max(1, int(w * scale))
    return cv2.resize(frame, (new_w, target_h))


def _scaleToWidth(frame, target_w):
    h, w = frame.shape[:2]
    if w == target_w:
        return frame
    scale = target_w / w
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (target_w, new_h))


def _hipAnchor(kpts):
    if isValidKpt(kpts[11]) and isValidKpt(kpts[12]):
        return (kpts[11] + kpts[12]) / 2
    if isValidKpt(kpts[5]) and isValidKpt(kpts[6]):
        return (kpts[5] + kpts[6]) / 2
    return np.array([0.0, 0.0])


def _jointSeparation(kpts1, kpts2, highlight_idx):
    if highlight_idx < 0:
        return 0.0
    p1 = kpts1[highlight_idx]
    p2 = kpts2[highlight_idx]
    if not isValidKpt(p1) or not isValidKpt(p2):
        return 0.0
    a1 = _hipAnchor(kpts1)
    a2 = _hipAnchor(kpts2)
    return float(np.linalg.norm((p1 - a1) - (p2 - a2)))


def _pitchLabel(path):
    return os.path.splitext(os.path.basename(path))[0]


def _suggestedAction(highlight, fb_score, os_score, fb_path, os_path):
    joint = highlight.replace("_", " ") if highlight else "delivery motion"
    fb_name = _pitchLabel(fb_path)
    os_name = _pitchLabel(os_path)
    if not highlight:
        return "Compare overall delivery timing and arm slot between fastball and offspeed."
    if fb_score > os_score * 1.1:
        return (
            f"Fastball shows more {joint} movement ({fb_score:.0f} vs {os_score:.0f} px/frame). "
            f"Watch for reduced {joint} activity on {os_name} as a live read."
        )
    if os_score > fb_score * 1.1:
        return (
            f"Offspeed shows more {joint} movement ({os_score:.0f} vs {fb_score:.0f} px/frame). "
            f"Use {joint} differences at release to spot {os_name} early."
        )
    return (
        f"{joint.title()} motion is similar between pitches — try another joint "
        "or use Find Best Pair to surface a clearer cue."
    )


def renderPitchComparison(
    kpts1_path,
    kpts2_path,
    output_path,
    *,
    layout="side_by_side",
    highlight="",
    skeleton_opacity=0.85,
    base_dir=None,
    label1="FASTBALL",
    label2="OFFSPEED",
):
    """
    Render DTW-synced side-by-side or stacked comparison on full pitch footage.

    Uses Tab 2 DTW alignment during the delivery phase, with parallel pre/post
    roll so the entire source video is shown at natural pacing.

    Returns metadata dict for the insights panel.
    """
    kpts1_path = resolveOverlayKeypointsPath(kpts1_path)
    kpts2_path = resolveOverlayKeypointsPath(kpts2_path)

    aligned = loadAndAlignPitchPair(kpts1_path, kpts2_path, trim_delivery=False)
    ctx1 = _openVideoContext(kpts1_path, base_dir, len(aligned["kpts1"]))
    ctx2 = _openVideoContext(kpts2_path, base_dir, len(aligned["kpts2"]))

    timeline = _buildSyncedTimeline(aligned, ctx1, ctx2)
    if not timeline:
        raise RuntimeError("Could not build comparison timeline.")

    is_glove_highlight, highlight_idx = _resolveHighlight(highlight)
    if is_glove_highlight:
        highlight_idx = -1

    fps = min(ctx1["fps"], ctx2["fps"])
    panel_h = max(ctx1["vid_h"], ctx2["vid_h"])
    panel_w = max(ctx1["vid_w"], ctx2["vid_w"])

    peak_sep = 0.0
    peak_frame = 0
    dtw_frame_count = 0

    frames_out = []
    for step, (vf1, vf2, ki1, ki2) in enumerate(timeline):
        frame1 = _readVideoFrame(ctx1["cap"], vf1, ctx1["vid_w"], ctx1["vid_h"])
        frame2 = _readVideoFrame(ctx2["cap"], vf2, ctx2["vid_w"], ctx2["vid_h"])

        kpts1 = aligned["kpts1"][ki1] if ki1 is not None else None
        kpts2 = aligned["kpts2"][ki2] if ki2 is not None else None
        glove1 = aligned["glove1"][ki1] if ki1 is not None else np.array([-1] * 4)
        glove2 = aligned["glove2"][ki2] if ki2 is not None else np.array([-1] * 4)

        if ki1 is not None and ki2 is not None:
            dtw_frame_count += 1
            sep = _jointSeparation(kpts1, kpts2, highlight_idx)
            if sep > peak_sep:
                peak_sep = sep
                peak_frame = step

        if kpts1 is not None:
            panel1 = _composeOverlayFrame(
                frame1, kpts1, glove1,
                fw=ctx1["fw"], fh=ctx1["fh"], vid_h=ctx1["vid_h"],
                highlight=highlight,
                highlight_idx=highlight_idx,
                is_glove_highlight=False,
                skeleton_opacity=skeleton_opacity,
                frame_label=label1,
                draw_glove=False,
            )
        else:
            panel1 = frame1.copy()

        if kpts2 is not None:
            panel2 = _composeOverlayFrame(
                frame2, kpts2, glove2,
                fw=ctx2["fw"], fh=ctx2["fh"], vid_h=ctx2["vid_h"],
                highlight=highlight,
                highlight_idx=highlight_idx,
                is_glove_highlight=False,
                skeleton_opacity=skeleton_opacity,
                frame_label=label2,
                draw_glove=False,
            )
        else:
            panel2 = frame2.copy()

        cv2.putText(
            panel1, label1, (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (56, 189, 248), 2,
        )
        cv2.putText(
            panel2, label2, (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 100, 255), 2,
        )

        if layout == "stacked":
            p1 = _scaleToWidth(panel1, panel_w)
            p2 = _scaleToWidth(panel2, panel_w)
            composed = np.vstack([p1, p2])
        else:
            p1 = _scaleToHeight(panel1, panel_h)
            p2 = _scaleToHeight(panel2, panel_h)
            composed = np.hstack([p1, p2])

        frames_out.append(composed)

    ctx1["cap"].release()
    ctx2["cap"].release()

    out_h, out_w = frames_out[0].shape[:2]
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )
    for frame in frames_out:
        out.write(frame)
    out.release()

    highlight_label = highlight.replace("_", " ").title() if highlight else "None"
    if highlight_idx >= 0:
        highlight_label = KEYPOINT_NAMES[highlight_idx]

    fb_score = scorePitch(kpts1_path, highlight or "left_knee")
    os_score = scorePitch(kpts2_path, highlight or "left_knee")

    duration_sec = round(len(frames_out) / fps, 1)

    return {
        "peak_separation_px": round(peak_sep, 1),
        "peak_frame_idx": peak_frame,
        "highlight_joint": highlight_label,
        "fb_score": round(float(fb_score), 1),
        "os_score": round(float(os_score), 1),
        "dtw_path_length": dtw_frame_count,
        "output_frames": len(frames_out),
        "duration_sec": duration_sec,
        "suggested_action": _suggestedAction(
            highlight,
            fb_score,
            os_score,
            kpts1_path,
            kpts2_path,
        ),
    }
