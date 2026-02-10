import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

# YOLOv8 pose keypoint connections (skeleton)
POSE_CONNECTIONS = [
    (0, 1), (0, 2),  # nose to eyes
    (1, 3), (2, 4),  # eyes to ears
    (0, 5), (0, 6),  # nose to shoulders
    (5, 6),  # shoulders
    (5, 7), (7, 9),  # left arm
    (6, 8), (8, 10),  # right arm
    (5, 11), (6, 12),  # shoulders to hips
    (11, 12),  # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16),  # right leg
]

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

def visualize_sequence(keypoint_file, num_frames=5):
    data = np.load(keypoint_file)
    print(f"Loaded: {keypoint_file}")
    print(f"Shape: {data.shape}")
    print(f"Total frames: {data.shape[0]}")
    
    total_frames = data.shape[0]
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    fig, axes = plt.subplots(1, num_frames, figsize=(4*num_frames, 5))
    if num_frames == 1:
        axes = [axes]
    
    for i, frame_idx in enumerate(frame_indices):
        ax = axes[i]
        keypoints = data[frame_idx]
        
        ax.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=50, zorder=3)
        
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                x_coords = [keypoints[start_idx, 0], keypoints[end_idx, 0]]
                y_coords = [keypoints[start_idx, 1], keypoints[end_idx, 1]]
                ax.plot(x_coords, y_coords, 'b-', linewidth=2, zorder=2)
        
        ax.invert_yaxis()
        ax.set_title(f'Frame {frame_idx}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    filename = os.path.basename(keypoint_file)
    fig.suptitle(f'Pitch Sequence: {filename}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visualize_keypoints.py <keypoint_file.npy>")
        sys.exit(1)
    
    visualize_sequence(sys.argv[1])
