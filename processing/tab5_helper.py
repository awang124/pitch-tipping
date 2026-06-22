import cv2
import numpy as np

from processing.kpt_utils import (
    buildVideoFrameLookup,
    isValidKpt,
    loadKeypointMeta,
    parseKeypointSequence,
    resolveOverlayKeypointsPath,
    resolveVideoAlignStart,
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

HIGHLIGHT_MAP = {
    "glove":           None,
    "left_knee":       13,
    "right_knee":      14,
    "right_elbow":     8,
    "left_shoulder":   5,
    "right_shoulder":  6,
    "left_hip":        11,
    "right_hip":       12,
}

def drawSkeleton(
    frame, kpts, color, thickness=2, highlight_idx=-1,
    highlight_color=(0, 255, 255), frame_width=None, frame_height=None,
    highlight_joints=None, highlight_edges=None,
):
    """
    Draw pose skeleton on a frame, with optional joint highlighting.

    Parameters
    ----------
    frame: Image to draw on.
    kpts: Array of keypoints (17, 2).
    color: BGR color tuple for skeleton.
    thickness: Line thickness.
    highlight_idx: Joint index to highlight, or -1 for none.
    highlight_color: BGR color for highlighted joint.
    highlight_joints: Optional set/list of joint indices to emphasize.
    highlight_edges: Optional list of (a, b) skeleton edges to emphasize.
    """
    joint_set = set(highlight_joints or [])
    if highlight_idx >= 0:
        joint_set.add(highlight_idx)
    edge_set = {tuple(e) for e in (highlight_edges or [])}
    edge_set |= {(b, a) for a, b in edge_set}

    for i, (x, y) in enumerate(kpts):
        if isValidKpt((x, y), frame_width, frame_height):
            if i in joint_set:
                cv2.circle(frame, (int(x), int(y)), 10, highlight_color, -1)
                cv2.circle(frame, (int(x), int(y)), 14, highlight_color, 2)
            else:
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    for (a, b) in SKELETON_EDGES:
        if a < len(kpts) and b < len(kpts):
            x1, y1 = kpts[a]
            x2, y2 = kpts[b]
            if isValidKpt((x1, y1), frame_width, frame_height) and isValidKpt(
                (x2, y2), frame_width, frame_height
            ):
                if (a, b) in edge_set or a in joint_set or b in joint_set:
                    edge_color = highlight_color
                    edge_thickness = max(thickness + 1, 3)
                else:
                    edge_color = color
                    edge_thickness = thickness
                cv2.line(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)),
                    edge_color, edge_thickness,
                )


def _resolveHighlight(highlight):
    is_glove_highlight = "glove" in highlight.lower()
    highlight_idx = -1
    if not is_glove_highlight and highlight:
        clean = highlight.lower().replace("_", " ")
        for i, name in enumerate(KEYPOINT_NAMES):
            if name.lower() in clean:
                highlight_idx = i
                break
    return is_glove_highlight, highlight_idx


def _composeOverlayFrame(
    frame,
    kpts,
    glove,
    *,
    fw,
    fh,
    vid_h,
    highlight,
    highlight_idx,
    is_glove_highlight,
    skeleton_opacity,
    frame_label,
    draw_glove=True,
    skeleton_color=(0, 220, 255),
    highlight_color=(0, 80, 255),
    highlight_joints=None,
    highlight_edges=None,
    highlight_label="",
):
    overlay = frame.copy()
    cv2.addWeighted(frame, 0.6, np.zeros_like(frame), 0.4, 0, overlay)

    skeleton_layer = overlay.copy()
    drawSkeleton(
        skeleton_layer, kpts,
        color=skeleton_color,
        thickness=2,
        highlight_idx=highlight_idx,
        highlight_color=highlight_color,
        frame_width=fw,
        frame_height=fh,
        highlight_joints=highlight_joints,
        highlight_edges=highlight_edges,
    )
    cv2.addWeighted(skeleton_layer, skeleton_opacity, overlay, 1 - skeleton_opacity, 0, overlay)

    if draw_glove and glove[0] != -1:
        gx1, gy1, gx2, gy2 = int(glove[0]), int(glove[1]), int(glove[2]), int(glove[3])
        box_color = highlight_color if is_glove_highlight else skeleton_color
        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), box_color, 2)
        if is_glove_highlight:
            cx, cy = (gx1 + gx2) // 2, (gy1 + gy2) // 2
            cv2.circle(overlay, (cx, cy), 35, highlight_color, 3)

    cv2.putText(
        overlay,
        frame_label,
        (20, vid_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1,
    )

    if highlight or highlight_label:
        label = (highlight_label or highlight).replace("_", " ").upper()
        cv2.putText(
            overlay, f"HIGHLIGHT: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, highlight_color, 2,
        )

    return overlay


def overlaySkeletonOnVideo(
    kpts_path,
    video_path,
    output_path,
    highlight="",
    skeleton_opacity=0.85,
):
    """
    Overlay extracted keypoint skeleton onto the full source video.

    Renders every source frame. Skeleton and glove box are drawn on frames
    that have keypoints, aligned via per-frame video_frame tags.

    Parameters
    ----------
    kpts_path: Path to .npy keypoint file.
    video_path: Path to source .mp4 / .mov video file.
    output_path: Path to save rendered output .mp4.
    highlight: Feature name string (e.g. 'right_elbow', 'glove', or '').
    skeleton_opacity: Opacity of skeleton overlay layer (0.0 - 1.0).

    Returns
    -------
    Tuple (success bool, message str).
    """
    kpts_path = resolveOverlayKeypointsPath(kpts_path)
    try:
        raw = np.load(kpts_path, allow_pickle=True)
    except Exception as e:
        return False, f"Failed to load keypoint file: {e}"

    kpts_seq, glove_seq, video_frames, _ = parseKeypointSequence(raw)
    if kpts_seq is None or len(kpts_seq) == 0:
        return False, "Keypoint file is empty."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Could not open source video file."

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta = loadKeypointMeta(kpts_path)
    fw = meta.get("frame_width") or vid_w
    fh = meta.get("frame_height") or vid_h

    video_start_frame = resolveVideoAlignStart(
        kpts_path, len(kpts_seq), total_video_frames
    )
    is_glove_highlight, highlight_idx = _resolveHighlight(highlight)
    lookup = buildVideoFrameLookup(
        kpts_seq, glove_seq, video_frames, video_start_frame
    )

    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h)
    )
    if not out.isOpened():
        cap.release()
        return False, "Could not open video writer. Check output path."

    rendered = 0
    overlay_frames = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for video_idx in range(total_video_frames):
        ret, frame = cap.read()
        if not ret:
            break

        rendered += 1
        hit = lookup.get(video_idx)
        if hit is None:
            out.write(frame)
            continue

        kpts, glove = hit
        overlay_frames += 1
        out.write(
            _composeOverlayFrame(
                frame, kpts, glove,
                fw=fw, fh=fh, vid_h=vid_h,
                highlight=highlight,
                highlight_idx=highlight_idx,
                is_glove_highlight=is_glove_highlight,
                skeleton_opacity=skeleton_opacity,
                frame_label=f"Frame {video_idx + 1} / {total_video_frames}",
            )
        )

    cap.release()
    out.release()

    if rendered == 0:
        return False, "No frames were rendered."

    legacy_note = ""
    if len(kpts_seq) < total_video_frames * 0.5:
        legacy_note = " — re-extract in Tab 1 for keypoints across the full pitch"
    return True, (
        f"Rendered {rendered} frames ({overlay_frames} with skeleton){legacy_note}"
    )
