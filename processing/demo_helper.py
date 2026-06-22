import os

import cv2
import numpy as np

from processing.compare_features import (
    featureSeparation,
    resolveCompareFeature,
    scoreCompareFeature,
)
from processing.kpt_utils import (
    loadKeypointMeta,
    resolveOverlayKeypointsPath,
    resolvePeakKneeFrame,
    resolveVideoAlignStart,
    videoPathFromKeypoints,
)
from processing.tab2_helper import _loadPitchSequenceArrays, loadAndAlignPitchPair
from processing.tab5_helper import _composeOverlayFrame

FB_TOKENS = ("fastball", "cutter", "four_seam", "two_seam")
OS_TOKENS = ("offspeed", "changeup", "curveball", "slider", "splitter", "off_")

# BGR — high-contrast pair for 50% overlay blend
FB_SKELETON_COLOR = (255, 255, 0)    # cyan
FB_HIGHLIGHT_COLOR = (255, 255, 120)
OS_SKELETON_COLOR = (255, 0, 255)    # magenta
OS_HIGHLIGHT_COLOR = (255, 80, 255)


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


def _videoContext(kpts_path, base_dir, n_kpt_frames):
    video_path = videoPathFromKeypoints(kpts_path, base_dir)
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError(f"No source video for {kpts_path}")
    meta = loadKeypointMeta(kpts_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    fw = meta.get("frame_width") or vid_w
    fh = meta.get("frame_height") or vid_h
    video_start = resolveVideoAlignStart(kpts_path, n_kpt_frames, total_frames)
    return {
        "video_path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "vid_w": vid_w,
        "vid_h": vid_h,
        "fw": fw,
        "fh": fh,
        "video_start": video_start,
    }


def _preloadVideo(video_path):
    """Read entire video once — avoids per-frame seek during render."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, vid_w, vid_h


def _videoFrameForKptIndex(video_frames, kpt_idx, video_start):
    if video_frames is not None and kpt_idx < len(video_frames) and video_frames[kpt_idx] >= 0:
        return int(video_frames[kpt_idx])
    return int(video_start + kpt_idx)


def _peakKneeAnchor(kpts, video_frames, video_start):
    """Video frame at peak leg kick — sync point for side-by-side playback."""
    resolved = resolvePeakKneeFrame(kpts)
    anchor_kpt = resolved if resolved is not None else max(0, len(kpts) // 2)
    anchor_kpt = min(anchor_kpt, len(kpts) - 1)
    anchor_vf = _videoFrameForKptIndex(video_frames, anchor_kpt, video_start)
    info = {
        "peak_knee_frame": anchor_kpt,
        "peak_knee_ok": resolved is not None,
    }
    return anchor_vf, anchor_kpt, info


def _buildLinearAnchorTimeline(total1, total2, anchor_vf1, anchor_vf2):
    """
    Align peak-knee anchors on one output frame; both videos advance every step.

    When a video has not started or has ended, its index falls outside [0, total)
    and the caller shows a black panel — no frame-hold pauses.
    """
    sync_out = max(anchor_vf1, anchor_vf2)
    tail = max(total1 - anchor_vf1, total2 - anchor_vf2)
    timeline = []
    for t in range(sync_out + tail):
        vf1 = t - sync_out + anchor_vf1
        vf2 = t - sync_out + anchor_vf2
        timeline.append((vf1, vf2))
    return timeline, sync_out


def _kptIndexForVideoFrame(video_frames, vf, video_start, n_kpts):
    if vf < 0:
        return None
    if video_frames is not None and len(video_frames):
        hits = np.where(video_frames == vf)[0]
        if len(hits):
            return int(hits[0])
        valid = [(i, int(video_frames[i])) for i in range(len(video_frames)) if video_frames[i] >= 0]
        candidates = [i for i, v in valid if v <= vf]
        if candidates:
            return candidates[-1]
    ki = vf - video_start
    if 0 <= ki < n_kpts:
        return int(ki)
    return None


def _frameAtIndex(frames, idx, fallback_w, fallback_h):
    if idx < 0 or idx >= len(frames):
        return np.zeros((fallback_h, fallback_w, 3), dtype=np.uint8)
    return frames[idx]


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


def _scaleToSize(frame, target_w, target_h):
    h, w = frame.shape[:2]
    if w == target_w and h == target_h:
        return frame
    return cv2.resize(frame, (target_w, target_h))


def _blendOverlay(base, top, top_alpha=0.5):
    """Blend top panel over base at the given opacity."""
    return cv2.addWeighted(base, 1.0 - top_alpha, top, top_alpha, 0)


def _dtwPathLength(aligned):
    """Count of unique DTW-synced windup frame pairs (metrics only)."""
    path = aligned["path"]
    count = 0
    prev = None
    for i, j in path:
        key = (i, j)
        if key == prev:
            continue
        prev = key
        count += 1
    return count


def _pitchLabel(path):
    return os.path.splitext(os.path.basename(path))[0]


def _suggestedAction(feature_key, feature_label, fb_score, os_score, fb_path, os_path):
    feat = feature_label or "delivery motion"
    fb_name = _pitchLabel(fb_path)
    os_name = _pitchLabel(os_path)
    if not feature_key:
        return "Compare overall delivery timing and arm slot between fastball and offspeed."
    if fb_score > os_score * 1.1:
        return (
            f"Fastball shows more {feat} ({fb_score:.0f} vs {os_score:.0f}). "
            f"Watch for reduced {feat} on {os_name} as a live read."
        )
    if os_score > fb_score * 1.1:
        return (
            f"Offspeed shows more {feat} ({os_score:.0f} vs {fb_score:.0f}). "
            f"Use {feat} differences to spot {os_name} early."
        )
    return f"{feat} is similar between pitches — try another feature for a clearer cue."


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
    Render side-by-side or 50% opacity overlay comparison with peak leg-kick anchor sync.

    Playback: both videos advance every output frame at native rate; the shorter
    clip gets black padding before/after. DTW on windup-only motion drives metrics
    only (peak separation), not the video timeline.
    """
    kpts1_path = resolveOverlayKeypointsPath(kpts1_path)
    kpts2_path = resolveOverlayKeypointsPath(kpts2_path)

    kpts1, glove1, vf1_arr = _loadPitchSequenceArrays(kpts1_path)
    kpts2, glove2, vf2_arr = _loadPitchSequenceArrays(kpts2_path)
    ctx1 = _videoContext(kpts1_path, base_dir, len(kpts1))
    ctx2 = _videoContext(kpts2_path, base_dir, len(kpts2))

    frames1, fps1, _, _ = _preloadVideo(ctx1["video_path"])
    frames2, fps2, _, _ = _preloadVideo(ctx2["video_path"])
    te1, te2 = len(frames1), len(frames2)

    anchor_vf1, anchor_kpt1, pk_info1 = _peakKneeAnchor(kpts1, vf1_arr, ctx1["video_start"])
    anchor_vf2, anchor_kpt2, pk_info2 = _peakKneeAnchor(kpts2, vf2_arr, ctx2["video_start"])
    timeline, sync_out_frame = _buildLinearAnchorTimeline(te1, te2, anchor_vf1, anchor_vf2)

    aligned = loadAndAlignPitchPair(kpts1_path, kpts2_path, trim_delivery=True)

    feat = resolveCompareFeature(highlight)
    feature_key = feat["feature_key"]
    feature_label = feat["label"]

    dtw_frame_count = _dtwPathLength(aligned)

    fps = min(fps1, fps2, ctx1["fps"], ctx2["fps"])
    panel_h = max(ctx1["vid_h"], ctx2["vid_h"], frames1[0].shape[0] if frames1 else 0)
    panel_w = max(ctx1["vid_w"], ctx2["vid_w"], frames1[0].shape[1] if frames1 else 0)

    peak_sep = 0.0
    peak_step = 0
    frames_out = []
    for step, (vf1, vf2) in enumerate(timeline):
        frame1 = _frameAtIndex(frames1, vf1, ctx1["vid_w"], ctx1["vid_h"])
        frame2 = _frameAtIndex(frames2, vf2, ctx2["vid_w"], ctx2["vid_h"])

        ki1 = _kptIndexForVideoFrame(vf1_arr, vf1, ctx1["video_start"], len(kpts1))
        ki2 = _kptIndexForVideoFrame(vf2_arr, vf2, ctx2["video_start"], len(kpts2))

        if ki1 is not None and ki2 is not None:
            sep = featureSeparation(
                kpts1[ki1], kpts2[ki2], glove1[ki1], glove2[ki2], feature_key,
            )
            if sep > peak_sep:
                peak_sep = sep
                peak_step = step

        if ki1 is not None:
            panel1 = _composeOverlayFrame(
                frame1, kpts1[ki1], glove1[ki1],
                fw=ctx1["fw"], fh=ctx1["fh"], vid_h=ctx1["vid_h"],
                highlight=feature_key,
                highlight_idx=-1,
                is_glove_highlight=feat["is_glove"],
                skeleton_opacity=skeleton_opacity,
                frame_label=label1,
                draw_glove=feat["is_glove"],
                skeleton_color=FB_SKELETON_COLOR,
                highlight_color=FB_HIGHLIGHT_COLOR,
                highlight_joints=feat["joints"],
                highlight_edges=feat["edges"],
                highlight_label=feature_label,
            )
        else:
            panel1 = frame1.copy()

        if ki2 is not None:
            panel2 = _composeOverlayFrame(
                frame2, kpts2[ki2], glove2[ki2],
                fw=ctx2["fw"], fh=ctx2["fh"], vid_h=ctx2["vid_h"],
                highlight=feature_key,
                highlight_idx=-1,
                is_glove_highlight=feat["is_glove"],
                skeleton_opacity=skeleton_opacity,
                frame_label=label2,
                draw_glove=feat["is_glove"],
                skeleton_color=OS_SKELETON_COLOR,
                highlight_color=OS_HIGHLIGHT_COLOR,
                highlight_joints=feat["joints"],
                highlight_edges=feat["edges"],
                highlight_label=feature_label,
            )
        else:
            panel2 = frame2.copy()

        if layout == "stacked":
            p1 = _scaleToSize(panel1, panel_w, panel_h)
            p2 = _scaleToSize(panel2, panel_w, panel_h)
            composed = _blendOverlay(p1, p2, top_alpha=0.5)
            cv2.putText(
                composed, label1, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, FB_SKELETON_COLOR, 2,
            )
            cv2.putText(
                composed, label2, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, OS_SKELETON_COLOR, 2,
            )
        else:
            cv2.putText(
                panel1, label1, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, FB_SKELETON_COLOR, 2,
            )
            cv2.putText(
                panel2, label2, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, OS_SKELETON_COLOR, 2,
            )
            p1 = _scaleToHeight(panel1, panel_h)
            p2 = _scaleToHeight(panel2, panel_h)
            composed = np.hstack([p1, p2])

        frames_out.append(composed)

    if not frames_out:
        raise RuntimeError("Could not build comparison timeline.")

    out_h, out_w = frames_out[0].shape[:2]
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h)
    )
    for frame in frames_out:
        out.write(frame)
    out.release()

    fb_score = scoreCompareFeature(kpts1_path, feature_key)
    os_score = scoreCompareFeature(kpts2_path, feature_key)

    duration_sec = round(len(frames_out) / fps, 1)
    peak_time_sec = round(peak_step / fps, 1)

    return {
        "peak_separation_px": round(peak_sep, 1),
        "peak_time_sec": peak_time_sec,
        "peak_frame_idx": peak_step,
        "highlight_feature": feature_label,
        "highlight_joint": feature_label,
        "fb_score": round(float(fb_score), 1),
        "os_score": round(float(os_score), 1),
        "dtw_path_length": dtw_frame_count,
        "output_frames": len(frames_out),
        "duration_sec": duration_sec,
        "sync_anchor_frame": sync_out_frame,
        "sync_landmark": "peak_knee",
        "anchor_kpt1": anchor_kpt1,
        "anchor_kpt2": anchor_kpt2,
        "anchor_vf1": anchor_vf1,
        "anchor_vf2": anchor_vf2,
        "peak_knee_ok1": pk_info1.get("peak_knee_ok", False),
        "peak_knee_ok2": pk_info2.get("peak_knee_ok", False),
        "windup_end1": aligned.get("windup_end1"),
        "windup_end2": aligned.get("windup_end2"),
        "windup_trim_ok1": aligned.get("windup_trim_ok1", False),
        "windup_trim_ok2": aligned.get("windup_trim_ok2", False),
        "suggested_action": _suggestedAction(
            feature_key,
            feature_label,
            fb_score,
            os_score,
            kpts1_path,
            kpts2_path,
        ),
    }
