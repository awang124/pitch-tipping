import cv2
import numpy as np

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
    "right_elbow":     8,
    "left_shoulder":   5,
    "right_shoulder":  6,
    "left_hip":        11,
    "right_hip":       12,
}

def isValidKpt(kpt):
    """
    Check if a keypoint coordinate is valid (not NaN, not zero vector).

    Parameters
    ----------
    kpt: (x, y) tuple.

    Returns
    -------
    True if valid.
    """
    x, y = kpt
    return not (np.isnan(x) or np.isnan(y) or (x == 0 and y == 0))

def drawSkeleton(frame, kpts, color, thickness=2, highlight_idx=-1, highlight_color=(0, 255, 255)):
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
    """
    for i, (x, y) in enumerate(kpts):
        if isValidKpt((x, y)):
            if i == highlight_idx:
                cv2.circle(frame, (int(x), int(y)), 10, highlight_color, -1)
                cv2.circle(frame, (int(x), int(y)), 14, highlight_color, 2)
            else:
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    for (a, b) in SKELETON_EDGES:
        if a < len(kpts) and b < len(kpts):
            x1, y1 = kpts[a]
            x2, y2 = kpts[b]
            if isValidKpt((x1, y1)) and isValidKpt((x2, y2)):
                edge_color = highlight_color if (
                    a == highlight_idx or b == highlight_idx
                ) else color
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), edge_color, thickness)

def overlaySkeletonOnVideo(
    kpts_path, video_path, output_path, highlight="", skeleton_opacity=0.85
):
    """
    Overlay extracted keypoint skeleton onto its source video.

    Synchronizes the 180-frame keypoint window to the last 180 frames
    of the source video (matching the windup truncation logic in Tab 1).
    Draws skeleton and glove box on each frame, darkens background slightly
    for contrast, and optionally highlights a kinematic feature.

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
    ### LOAD KEYPOINTS
    try:
        raw = np.load(kpts_path, allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.dtype == object:
            raw = raw.tolist()
    except Exception as e:
        return False, f"Failed to load keypoint file: {e}"

    if not raw:
        return False, "Keypoint file is empty."

    ### PARSE KEYPOINTS AND GLOVE BOXES
    if isinstance(raw[0], dict):
        glove_seq = np.array(
            [frame.get("glove_box", [-1, -1, -1, -1]) for frame in raw], dtype=np.float32
        )
        kpts_seq = np.array([frame["keypoints"] for frame in raw], dtype=np.float32)
    else:
        glove_seq = np.full((len(raw), 4), -1.0, dtype=np.float32)
        kpts_seq = np.array(raw, dtype=np.float32)

    ### STRIP ZERO-PADDING FROM FRONT
    valid_mask = [np.max(frame) > 0 for frame in kpts_seq]
    first_valid = next((i for i, v in enumerate(valid_mask) if v), 0)
    kpts_seq = kpts_seq[first_valid:]
    glove_seq = glove_seq[first_valid:]
    n_kpt_frames = len(kpts_seq)

    ### OPEN SOURCE VIDEO
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Could not open source video file."

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ### ALIGN: KEYPOINTS CORRESPOND TO LAST n_kpt_frames OF VIDEO
    # Tab 1 truncates to the 180 frames ending at windup peak,
    # so we align keyframe index 0 to video frame (total - n_kpt_frames).
    video_start_frame = max(0, total_video_frames - n_kpt_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_start_frame)

    ### RESOLVE HIGHLIGHT
    is_glove_highlight = "glove" in highlight.lower()
    highlight_idx = -1
    if not is_glove_highlight and highlight:
        clean = highlight.lower().replace("_", " ")
        for i, name in enumerate(KEYPOINT_NAMES):
            if name.lower() in clean:
                highlight_idx = i
                break

    ### PREPARE VIDEO WRITER
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vid_w, vid_h)
    )
    if not out.isOpened():
        cap.release()
        return False, "Could not open video writer. Check output path."

    ### RENDER FRAMES
    kpt_idx = 0
    while kpt_idx < n_kpt_frames:
        ret, frame = cap.read()
        if not ret:
            break

        kpts = kpts_seq[kpt_idx]
        glove = glove_seq[kpt_idx]

        ### DARKEN BACKGROUND SLIGHTLY FOR CONTRAST
        overlay = frame.copy()
        cv2.addWeighted(frame, 0.6, np.zeros_like(frame), 0.4, 0, overlay)

        ### DRAW SKELETON
        skeleton_layer = overlay.copy()
        drawSkeleton(
            skeleton_layer, kpts,
            color=(0, 220, 255),
            thickness=2,
            highlight_idx=highlight_idx,
            highlight_color=(0, 80, 255),
        )

        ### BLEND SKELETON LAYER ONTO FRAME
        cv2.addWeighted(skeleton_layer, skeleton_opacity, overlay, 1 - skeleton_opacity, 0, overlay)

        ### DRAW GLOVE BOX
        if glove[0] != -1:
            gx1, gy1, gx2, gy2 = int(glove[0]), int(glove[1]), int(glove[2]), int(glove[3])
            box_color = (0, 80, 255) if is_glove_highlight else (0, 220, 255)
            cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), box_color, 2)
            if is_glove_highlight:
                ### PULSING HIGHLIGHT RING AROUND GLOVE
                cx, cy = (gx1 + gx2) // 2, (gy1 + gy2) // 2
                cv2.circle(overlay, (cx, cy), 35, (0, 80, 255), 3)

        ### FRAME COUNTER
        cv2.putText(
            overlay,
            f"Frame {kpt_idx + 1} / {n_kpt_frames}",
            (20, vid_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1,
        )

        ### HIGHLIGHT LABEL
        if highlight:
            label = highlight.replace("_", " ").upper()
            cv2.putText(
                overlay, f"HIGHLIGHT: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 2,
            )

        out.write(overlay)
        kpt_idx += 1

    cap.release()
    out.release()

    if kpt_idx == 0:
        return False, "No frames were rendered. Video may be shorter than keypoint window."

    return True, f"Rendered {kpt_idx} frames."