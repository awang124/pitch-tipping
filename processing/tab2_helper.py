import os
import cv2
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

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

# ──────────────────────────────────────────────────────────────────────────────────────────────────
# FEATURE DIFFERENCE SELECTION
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def scorePitch(file, feature_grp):
    try:
        if not len(data := np.load(file, allow_pickle=True)):
            return 0
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
        idx = {
            "left_shoulder": 5, "right_shoulder": 6, "right_elbow": 8,
            "left_hip": 11, "right_hip": 12, "left_knee": 13
        }.get(feat, 13)
        return np.max(np.linalg.norm(np.diff(seq[:, idx, :], axis=0), axis=1))
    except:
        return 0

def findBestPitchPair(kpts_root, feature_grp):
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

# ──────────────────────────────────────────────────────────────────────────────────────────────────
# SEQUENCE DIFFERENCE VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def isValidKpt(kpt):
    x, y = kpt
    return not (np.isnan(x) or np.isnan(y) or (x == 0 and y == 0))

def filterValidPoints(kpts, indices):
    pts = []
    valid_indices = []
    for i in indices:
        x, y = kpts[i]
        if isValidKpt((x, y)):
            pts.append([x, y])
            valid_indices.append(i)
    return np.array(pts, dtype=np.float32).reshape(-1, 2), valid_indices

def drawSkeleton(frame, kpts, color, thickness=2, valid_only=False):
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
    features = []
    for frame in kpts_sequence:
        if isValidKpt(frame[13]) and isValidKpt(frame[14]):
            val = (frame[13, 1] + frame[14, 1]) / 2
        elif isValidKpt(frame[13]):
            val = frame[13, 1]
        elif isValidKpt(frame[14]):
            val = frame[14, 1]
        else:
            val = 0
        features.append([val])
    return np.array(features)

def visualizeDifferences(
    kpts1_path, kpts2_path, output_path, video_bg=None, threshold=15.0, highlight=""
):
    kpts1 = np.load(kpts1_path, allow_pickle=True)
    kpts2 = np.load(kpts2_path, allow_pickle=True)
    
    if isinstance(kpts1, np.ndarray) and kpts1.dtype == object:
        kpts1 = kpts1.tolist()
    if isinstance(kpts2, np.ndarray) and kpts2.dtype == object:
        kpts2 = kpts2.tolist()
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

    # Strip empty padding frames (-1 arrays) so DTW perfectly aligns the actual active sequence
    valid_idx1 = [i for i, frame in enumerate(kpts1) if np.max(frame) > 0]
    valid_idx2 = [i for i, frame in enumerate(kpts2) if np.max(frame) > 0]
    
    if len(valid_idx1) > 0:
        kpts1 = np.array(kpts1)[valid_idx1]
        glove1 = glove1[valid_idx1]
    if len(valid_idx2) > 0:
        kpts2 = np.array(kpts2)[valid_idx2]
        glove2 = glove2[valid_idx2]
        
    width, height, fps, frames_bg = 1080, 1920, 30, []
    is_normalized = False
    # If the keypoints are mean-normalized, they will be centered around (0,0) and have negative bounds.
    # OpenCV draws (0,0) at the top-left corner, meaning they render entirely off-screen.
    # We must globally translate them to the absolute center of the canvas.
    if len(kpts1) > 0 and np.nanmin(kpts1) < 0:
        is_normalized = True
        offset_x = width // 2
        offset_y = (height // 2) + 200  # Shift slightly lower for aesthetic centering
        offset = np.array([offset_x, offset_y], dtype=np.float32)
        
        # Shift all valid keypoints
        for i in range(len(kpts1)):
            kpts1[i] = np.where(kpts1[i] != -1, kpts1[i] + offset, kpts1[i])
        for i in range(len(kpts2)):
            kpts2[i] = np.where(kpts2[i] != -1, kpts2[i] + offset, kpts2[i])
            
        # Shift glove boxes if valid
        if glove1[0][0] != -1:
            glove1 += np.array([offset_x, offset_y, offset_x, offset_y], dtype=np.float32)
        if glove2[0][0] != -1:
            glove2 += np.array([offset_x, offset_y, offset_x, offset_y], dtype=np.float32)
            
    feat1 = extractDTWFeatures(kpts1)
    feat2 = extractDTWFeatures(kpts2)
    is_glove_highlight = "glove" in highlight.lower()
    highlight_idx = -1
    if not is_glove_highlight and highlight:
        clean_name = highlight.lower().replace("_", " ")
        for i, name in enumerate(KEYPOINT_NAMES):
            if name.lower() in clean_name:
                highlight_idx = i
                break
    distance, path = fastdtw(feat1, feat2, dist=euclidean)
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
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    ALL_KEYPOINTS = list(range(17))
    max_overall_dist = 0
    max_overall_joint = ""
    max_overall_frame = 0
    for idx, (i, j) in enumerate(path):
        if video_bg and i >= len(frames_bg):
            break
        pts1, valid1 = filterValidPoints(kpts1[i], ALL_KEYPOINTS)
        pts2, valid2 = filterValidPoints(kpts2[j], ALL_KEYPOINTS)
        common_indices = set(valid1).intersection(valid2)
        kpts2_aligned = np.copy(kpts2[j])
        T = None
        if not is_normalized and len(common_indices) >= 3:
            pts1_common = np.array([kpts1[i][k] for k in common_indices], dtype=np.float32)
            pts2_common = np.array([kpts2[j][k] for k in common_indices], dtype=np.float32)
            T, _ = cv2.estimateAffinePartial2D(pts2_common, pts1_common, method=cv2.RANSAC)
            if T is not None:
                kpts2_aligned = cv2.transform(kpts2[j].reshape(-1, 1, 2), T).reshape(-1, 2)
            else:
                center1 = (kpts1[i][11] + kpts1[i][12]) / 2\
                    if isValidKpt(kpts1[i][11]) and isValidKpt(kpts1[i][12]) else None
                center2 = (kpts2[j][11] + kpts2[j][12]) / 2\
                    if isValidKpt(kpts2[j][11]) and isValidKpt(kpts2[j][12]) else None
                if center1 is not None and center2 is not None:
                    shift = center1 - center2
                    kpts2_aligned = kpts2[j] + shift
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
        if video_bg:
            canvas = frames_bg[i].copy()
            canvas = cv2.addWeighted(canvas, 0.3, np.zeros_like(canvas), 0.7, 0)
        else:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
        drawSkeleton(canvas, kpts1[i], (255, 255, 255), thickness=2)
        drawSkeleton(canvas, kpts2_aligned, (255, 100, 50), thickness=2)
        g2 = glove2[j]
        g2_aligned = None
        if T is not None and g2[0] != -1:
            x1, y1, x2, y2 = g2
            corners = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
            ).reshape(-1, 1, 2)
            corners_aligned = cv2.transform(corners, T).reshape(-1, 2)
            nx1, ny1 = np.min(corners_aligned, axis=0)
            nx2, ny2 = np.max(corners_aligned, axis=0)
            g2_aligned = [nx1, ny1, nx2, ny2]
        elif T is None and g2[0] != -1 and "shift" in locals():
            x1, y1, x2, y2 = g2
            g2_aligned = [x1 + shift[0], y1 + shift[1], x2 + shift[0], y2 + shift[1]]
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
        cv2.putText(
            canvas, f"Pitch 1 (Base): White", (40, height - 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )
        cv2.putText(
            canvas, f"Pitch 2 (Aligned): Blue", (40, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 50), 2
        )
        if highlight:
            cv2.putText(
                canvas, f"FLAG TARGET: {highlight.upper()}", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
            )
        out.write(canvas)
    out.release()

