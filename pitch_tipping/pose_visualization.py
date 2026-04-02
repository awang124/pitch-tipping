# This script creates side‑by‑side or overlaid visualizations of two pitching
# sequences. It uses Dynamic Time Warping (DTW) to align the two motions in time,
# then renders a video where:
#   - One pitch is shown in white skeleton + glove box.
#   - The other pitch is aligned (using affine transformation) and shown in blue.
# Differences in body position are highlighted. The output helps spot mechanical
# differences between two pitch types (e.g., fastball vs. curveball).


get_ipython().system('pip install fastdtw')

from google.colab import drive
drive.mount('/content/drive')

import cv2
import numpy as np
import os
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Skeleton edges (COCO format, indices 0‑16)
SKELETON_EDGES = [
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    (5, 6),           # shoulders
    (5, 11), (6, 12), # torso sides
    (11, 13), (13, 15), # left leg
    (12, 14), (14, 16), # right leg
    (11, 12)          # hips
]

KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

def is_valid_kpt(kpt):
    """Check if a keypoint (x,y) is valid (not NaN and not (0,0))."""
    x, y = kpt
    return not (np.isnan(x) or np.isnan(y) or (x == 0 and y == 0))

def filter_valid_points(kpts, indices):
    """
    Extract only those keypoints whose indices are in `indices` and are valid.
    Returns (array of valid points, list of original indices that were valid).
    """
    pts = []
    valid_indices = []
    for i in indices:
        x, y = kpts[i]
        if is_valid_kpt((x, y)):
            pts.append([x, y])
            valid_indices.append(i)
    return np.array(pts, dtype=np.float32).reshape(-1, 2), valid_indices

def draw_skeleton(frame, kpts, color, thickness=2):
    """
    Draw keypoints (as circles) and skeleton edges on a frame.
    kpts: array of shape (17,2) with keypoint coordinates.
    color: (B,G,R) tuple.
    """
    for i, (x, y) in enumerate(kpts):
        if is_valid_kpt((x, y)):
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    for (a, b) in SKELETON_EDGES:
        if a < len(kpts) and b < len(kpts):
            x1, y1 = kpts[a]
            x2, y2 = kpts[b]
            if is_valid_kpt((x1, y1)) and is_valid_kpt((x2, y2)):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def extract_dtw_features(kpts_sequence):
    """
    Extract a simple 1D feature for DTW alignment: the average y‑coordinate of the knees.
    This provides a rough representation of the overall vertical motion of the lower body.
    Returns a (T,1) array.
    """
    features = []
    for frame in kpts_sequence:
        if is_valid_kpt(frame[13]) and is_valid_kpt(frame[14]):
            val = (frame[13, 1] + frame[14, 1]) / 2
        elif is_valid_kpt(frame[13]):
            val = frame[13, 1]
        elif is_valid_kpt(frame[14]):
            val = frame[14, 1]
        else:
            val = 0
        features.append([val])
    return np.array(features)


def visualize_differences(kpts1_path, kpts2_path, output_path, video_bg=None, threshold=15.0):
    """
    Main function: load two pitch sequences, align them with DTW, and generate a video
    that shows the aligned motions superimposed.
    
    Parameters:
      kpts1_path: path to .npy file of the first (base) pitch
      kpts2_path: path to .npy file of the second (compared) pitch
      output_path: where to save the resulting .mp4 video
      video_bg: optional original video file to use as background (otherwise black)
      threshold: (unused, kept for compatibility)
    """
    # Load sequences (handle both raw keypoints and the dict format with glove_box)
    kpts1 = np.load(kpts1_path, allow_pickle=True)
    kpts2 = np.load(kpts2_path, allow_pickle=True)

    if len(kpts1) > 0 and isinstance(kpts1[0], dict):
        glove1 = np.array([frame.get("glove_box", [-1, -1, -1, -1]) for frame in kpts1], dtype=np.float32)
        kpts1 = np.array([frame["keypoints"] for frame in kpts1], dtype=np.float32)
    else:
        glove1 = np.full((len(kpts1), 4), -1.0, dtype=np.float32)

    if len(kpts2) > 0 and isinstance(kpts2[0], dict):
        glove2 = np.array([frame.get("glove_box", [-1, -1, -1, -1]) for frame in kpts2], dtype=np.float32)
        kpts2 = np.array([frame["keypoints"] for frame in kpts2], dtype=np.float32)
    else:
        glove2 = np.full((len(kpts2), 4), -1.0, dtype=np.float32)

    # Compute DTW alignment based on simple knee‑height feature
    feat1 = extract_dtw_features(kpts1)
    feat2 = extract_dtw_features(kpts2)

    print("Calculating Dynamic Time Warping...")
    distance, path = fastdtw(feat1, feat2, dist=euclidean)
    print(f"DTW Distance: {distance:.2f}, Alignment Path Length: {len(path)}")

    # Prepare video writer
    width, height = 1080, 1920
    fps = 30
    frames_bg = []
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

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    ALL_KEYPOINTS = list(range(17))
    max_overall_dist, max_overall_joint, max_overall_frame = 0, "", 0

    print("Synthesizing differences...")
    for idx, (i, j) in enumerate(path):
        if video_bg and i >= len(frames_bg):
            break

        # Extract points for current aligned frames
        pts1, valid1 = filter_valid_points(kpts1[i], ALL_KEYPOINTS)
        pts2, valid2 = filter_valid_points(kpts2[j], ALL_KEYPOINTS)

        common_indices = set(valid1).intersection(valid2)
        kpts2_aligned = np.copy(kpts2[j])
        T, shift = None, None

        # Estimate an affine transformation (rotation + translation) to align the second pose
        # onto the first, using the common keypoints.
        if len(common_indices) >= 3:
            pts1_common = np.array([kpts1[i][k] for k in common_indices], dtype=np.float32)
            pts2_common = np.array([kpts2[j][k] for k in common_indices], dtype=np.float32)
            T, _ = cv2.estimateAffinePartial2D(pts2_common, pts1_common, method=cv2.RANSAC)
            if T is not None:
                kpts2_aligned = cv2.transform(kpts2[j].reshape(-1, 1, 2), T).reshape(-1, 2)
            else:
                # Fallback: simple translation using hip centers
                center1 = (kpts1[i][11] + kpts1[i][12]) / 2 if is_valid_kpt(kpts1[i][11]) and is_valid_kpt(kpts1[i][12]) else None
                center2 = (kpts2[j][11] + kpts2[j][12]) / 2 if is_valid_kpt(kpts2[j][11]) and is_valid_kpt(kpts2[j][12]) else None
                if center1 is not None and center2 is not None:
                    shift = center1 - center2
                    kpts2_aligned = kpts2[j] + shift

        # Compute maximum joint displacement in this frame (for logging)
        max_dist_in_frame, max_joint_in_frame = 0, ""
        for k in range(17):
            p1 = kpts1[i][k]
            p2 = kpts2_aligned[k]
            if is_valid_kpt(p1) and is_valid_kpt(p2):
                dist = euclidean(p1, p2)
                if dist > max_dist_in_frame:
                    max_dist_in_frame = dist
                    max_joint_in_frame = KEYPOINT_NAMES[k]

        if max_dist_in_frame > max_overall_dist:
            max_overall_dist, max_overall_joint, max_overall_frame = max_dist_in_frame, max_joint_in_frame, idx

        # Prepare canvas (either from background video or black)
        canvas = frames_bg[i].copy() if video_bg else np.zeros((height, width, 3), dtype=np.uint8)
        if video_bg:
            canvas = cv2.addWeighted(canvas, 0.3, np.zeros_like(canvas), 0.7, 0)

        # Draw both skeletons
        draw_skeleton(canvas, kpts1[i], (255, 255, 255), thickness=2)   # white = base pitch
        draw_skeleton(canvas, kpts2_aligned, (255, 100, 50), thickness=2) # blue = compared pitch

        # Draw glove boxes (if present), applying the same alignment to the second glove
        g1 = glove1[i]
        if g1[0] != -1:
            cv2.rectangle(canvas, (int(g1[0]), int(g1[1])), (int(g1[2]), int(g1[3])), (255, 255, 255), 2)

        g2 = glove2[j]
        g2_aligned = None
        if T is not None and g2[0] != -1:
            corners = np.array([[g2[0], g2[1]], [g2[2], g2[1]], [g2[2], g2[3]], [g2[0], g2[3]]], dtype=np.float32).reshape(-1, 1, 2)
            corners_aligned = cv2.transform(corners, T).reshape(-1, 2)
            nx1, ny1 = np.min(corners_aligned, axis=0)
            nx2, ny2 = np.max(corners_aligned, axis=0)
            g2_aligned = [nx1, ny1, nx2, ny2]
        elif shift is not None and g2[0] != -1:
            g2_aligned = [g2[0] + shift[0], g2[1] + shift[1], g2[2] + shift[0], g2[3] + shift[1]]

        if g2_aligned is not None:
            cv2.rectangle(canvas, (int(g2_aligned[0]), int(g2_aligned[1])), (int(g2_aligned[2]), int(g2_aligned[3])), (255, 100, 50), 2)

        # Add legend
        cv2.putText(canvas, f"Pitch 1 (Base): White", (40, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, f"Pitch 2 (Aligned): Blue", (40, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 50), 2)

        out.write(canvas)
        if idx % 50 == 0:
            print(f"Rendered {idx} frames...")

    out.release()
    print(f"Saved {output_path}!")
    print(f"\n--- OVERALL PITCH DEVIATION ---")
    print(f"Most Divergent Joint: {max_overall_joint}")
    print(f"Max Displacement: {max_overall_dist:.1f}px (at frame {max_overall_frame})")


# # Option A: Compare a single pair of pitches
# BASE_PITCH = "/content/drive/My Drive/Pitch-Tipping/Keypoints/fastball_0.npy"
# COMPARE_PITCH = "/content/drive/My Drive/Pitch-Tipping/Keypoints/curveball_0.npy"
# OUTPUT_VIDEO = "/content/drive/My Drive/Pitch-Tipping/Visualizations/fastball_vs_curveball.mp4"
# os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
# visualize_differences(BASE_PITCH, COMPARE_PITCH, OUTPUT_VIDEO)

# # Option B: Batch run against all .npy files in the Keypoints directory
# import glob
# BASE_PITCH = "/content/drive/My Drive/Pitch-Tipping/Keypoints/fastball_0.npy"
# OUTPUT_DIR = "/content/drive/My Drive/Pitch-Tipping/Visualizations/"
# npy_files = glob.glob("/content/drive/My Drive/Pitch-Tipping/Keypoints/*.npy")
# for pitch_path in npy_files:
#     if os.path.abspath(pitch_path) == os.path.abspath(BASE_PITCH):
#         continue
#     base_name = os.path.splitext(os.path.basename(BASE_PITCH))[0]
#     pitch_name = os.path.splitext(os.path.basename(pitch_path))[0]
#     out_video = os.path.join(OUTPUT_DIR, f"{base_name}_vs_{pitch_name}.mp4")
#     print(f"\n--- Generating: {base_name} vs {pitch_name} ---")
#     visualize_differences(BASE_PITCH, pitch_path, out_video)

