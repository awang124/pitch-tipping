import cv2
import numpy as np
import argparse
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# COCO keypoint connections for drawing skeleton
SKELETON_EDGES = [
    (5, 7), (7, 9),   # left arm
    (6, 8), (8, 10),  # right arm
    (5, 6),           # shoulders
    (5, 11), (6, 12), # torso
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
    x, y = kpt
    return not (np.isnan(x) or np.isnan(y) or (x == 0 and y == 0))

def filter_valid_points(kpts, indices):
    pts = []
    valid_indices = []
    for i in indices:
        x, y = kpts[i]
        if is_valid_kpt((x, y)):
            pts.append([x, y])
            valid_indices.append(i)
    return np.array(pts, dtype=np.float32).reshape(-1, 2), valid_indices

def draw_skeleton(frame, kpts, color, thickness=2, valid_only=False):
    """Draw keypoints and skeleton on frame."""
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
    """Extract a 1D signal to align the pitches (e.g., knee lifting)."""
    features = []
    for frame in kpts_sequence:
        if is_valid_kpt(frame[13]) and is_valid_kpt(frame[14]):
            val = (frame[13, 1] + frame[14, 1]) / 2  # Mean knee Y
        elif is_valid_kpt(frame[13]):
            val = frame[13, 1]
        elif is_valid_kpt(frame[14]):
            val = frame[14, 1]
        else:
            val = 0 # Fallback
        features.append([val])
    return np.array(features)

def visualize_differences(kpts1_path, kpts2_path, output_path, video_bg=None, threshold=15.0, highlight=""):
    kpts1 = np.load(kpts1_path, allow_pickle=True)
    kpts2 = np.load(kpts2_path, allow_pickle=True)

    # Handle dictwrapper legacy npy or pure arrays
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

    feat1 = extract_dtw_features(kpts1)
    feat2 = extract_dtw_features(kpts2)

    is_glove_highlight = "glove" in highlight.lower()
    highlight_idx = -1
    if not is_glove_highlight and highlight:
        clean_name = highlight.lower().replace("_", " ")
        for i, name in enumerate(KEYPOINT_NAMES):
            if name.lower() in clean_name:
                highlight_idx = i
                break

    print("Calculating Dynamic Time Warping...")
    distance, path = fastdtw(feat1, feat2, dist=euclidean)
    print(f"DTW Distance: {distance:.2f}, Alignment Path Length: {len(path)}")

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
            if not ret: break
            frames_bg.append(frame)
        cap.release()
        
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    
    ALL_KEYPOINTS = list(range(17))

    max_overall_dist = 0
    max_overall_joint = ""
    max_overall_frame = 0

    print("Synthesizing differences...")
    for idx, (i, j) in enumerate(path):
        if video_bg and i >= len(frames_bg):
            break
            
        pts1, valid1 = filter_valid_points(kpts1[i], ALL_KEYPOINTS)
        pts2, valid2 = filter_valid_points(kpts2[j], ALL_KEYPOINTS)

        # Spatial alignment
        common_indices = set(valid1).intersection(valid2)
        kpts2_aligned = np.copy(kpts2[j])
        
        if len(common_indices) >= 3:
            pts1_common = np.array([kpts1[i][k] for k in common_indices], dtype=np.float32)
            pts2_common = np.array([kpts2[j][k] for k in common_indices], dtype=np.float32)
            
            # Robust mapping
            T, _ = cv2.estimateAffinePartial2D(pts2_common, pts1_common, method=cv2.RANSAC)
            if T is not None:
                kpts2_aligned = cv2.transform(kpts2[j].reshape(-1, 1, 2), T).reshape(-1, 2)
            else:
                # Fallback to simple translation based on hips
                center1 = (kpts1[i][11] + kpts1[i][12]) / 2 if is_valid_kpt(kpts1[i][11]) and is_valid_kpt(kpts1[i][12]) else None
                center2 = (kpts2[j][11] + kpts2[j][12]) / 2 if is_valid_kpt(kpts2[j][11]) and is_valid_kpt(kpts2[j][12]) else None
                if center1 is not None and center2 is not None:
                    shift = center1 - center2
                    kpts2_aligned = kpts2[j] + shift
                
        # Calculate differences mapping to Pitch 1's skeleton space
        max_dist_in_frame = 0
        max_joint_in_frame = ""
        
        differences = []
        for k in range(17):
            p1 = kpts1[i][k]
            p2 = kpts2_aligned[k]
            if is_valid_kpt(p1) and is_valid_kpt(p2):
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
            # Dim the background to make skeletons pop
            canvas = cv2.addWeighted(canvas, 0.3, np.zeros_like(canvas), 0.7, 0)
        else:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw Pitch 1 Skeleton (White) - Baseball "base"
        draw_skeleton(canvas, kpts1[i], (255, 255, 255), thickness=2)
        # Draw Pitch 2 Skeleton (Blueish)
        draw_skeleton(canvas, kpts2_aligned, (255, 100, 50), thickness=2)

        # Calculate transformed glove_box for Pitch 2
        g2 = glove2[j]
        g2_aligned = None
        if T is not None and g2[0] != -1:
            x1, y1, x2, y2 = g2
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32).reshape(-1, 1, 2)
            corners_aligned = cv2.transform(corners, T).reshape(-1, 2)
            nx1, ny1 = np.min(corners_aligned, axis=0)
            nx2, ny2 = np.max(corners_aligned, axis=0)
            g2_aligned = [nx1, ny1, nx2, ny2]
        elif T is None and g2[0] != -1 and 'shift' in locals():
            x1, y1, x2, y2 = g2
            g2_aligned = [x1 + shift[0], y1 + shift[1], x2 + shift[0], y2 + shift[1]]

        # Draw Pitch 1 Glove Box
        g1 = glove1[i]
        if g1[0] != -1:
            cv2.rectangle(canvas, (int(g1[0]), int(g1[1])), (int(g1[2]), int(g1[3])), (255, 255, 255), 2)
            
        # Draw Pitch 2 Glove Box
        if g2_aligned is not None:
            cv2.rectangle(canvas, (int(g2_aligned[0]), int(g2_aligned[1])), (int(g2_aligned[2]), int(g2_aligned[3])), (255, 100, 50), 2)

        # --- AI HIGHLIGHTER ---
        if is_glove_highlight:
            if g1[0] != -1:
                cx, cy = int((g1[0]+g1[2])/2), int((g1[1]+g1[3])/2)
                cv2.circle(canvas, (cx, cy), 40, (0, 0, 255), 4) # Glowing red circle on White Pitch
            if g2_aligned is not None:
                cx, cy = int((g2_aligned[0]+g2_aligned[2])/2), int((g2_aligned[1]+g2_aligned[3])/2)
                cv2.circle(canvas, (cx, cy), 40, (0, 255, 255), 4) # Yellow circle on Aligned Pitch
                
        elif highlight_idx != -1:
            p1 = kpts1[i][highlight_idx]
            p2 = kpts2_aligned[highlight_idx]
            if is_valid_kpt(p1):
                cv2.circle(canvas, (int(p1[0]), int(p1[1])), 25, (0, 0, 255), 4)
            if is_valid_kpt(p2):
                cv2.circle(canvas, (int(p2[0]), int(p2[1])), 25, (0, 255, 255), 4)
            # Draw a bright line connecting the two divergent joints
            if is_valid_kpt(p1) and is_valid_kpt(p2):
                cv2.line(canvas, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 3)

        # (Difference highlight lines removed as requested)

        # --- HUD Analytics Overlay ---
        # Legend at Bottom
        cv2.putText(canvas, f"Pitch 1 (Base): White", (40, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, f"Pitch 2 (Aligned): Blue", (40, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 50), 2)

        if highlight:
            cv2.putText(canvas, f"AI FLAG TARGET: {highlight.upper()}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(canvas)
        
        if idx % 50 == 0:
            print(f"Rendered {idx} frames...")

    out.release()
    print(f"Saved {output_path}!")
    print(f"\n--- OVERALL PITCH DEVIATION ---")
    print(f"Most Divergent Joint: {max_overall_joint}")
    print(f"Max Displacement: {max_overall_dist:.1f}px (at frame {max_overall_frame})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and compute discrepancies between two pitches.")
    parser.add_argument("--kpts1", required=True, help="Base tracking .npy file (Pitch 1)")
    parser.add_argument("--kpts2", required=True, help="Comparison tracking .npy file (Pitch 2)")
    parser.add_argument("--video1", default=None, help="(Optional) Base video for background")
    parser.add_argument("--out", default="pitch_differences_hud.mp4", help="Output video path")
    parser.add_argument("--threshold", type=float, default=25.0, help="Distance threshold to color RED")
    parser.add_argument("--highlight", type=str, default="", help="Feature name to highlight (e.g. 'glove', 'left_knee')")
    
    args = parser.parse_args()
    visualize_differences(args.kpts1, args.kpts2, args.out, video_bg=args.video1, threshold=args.threshold, highlight=args.highlight)
