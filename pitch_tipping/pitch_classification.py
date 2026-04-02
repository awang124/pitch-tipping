# This script processes the keypoint sequences produced by pose_estimation.py.
# It extracts over 100 kinematic features (angles, velocities, timings, symmetry,
# coordination, glove motion, etc.) and then:
#   1. Performs statistical tests (Mann‑Whitney U) to find features that differ
#      significantly between pitch types.
#   2. Uses DTW (Dynamic Time Warping) to compute distances between sequences
#      and adds mean‑distance‑to‑each‑class as features.
#   3. Builds binary classification models (Random Forest & L1‑regularized Logistic
#      Regression) to distinguish fastballs from off‑speed pitches, identifying the
#      most important kinematic "tells".


get_ipython().system('pip install fastdtw')

from google.colab import drive
drive.mount("/content/drive")

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fastdtw import fastdtw
from scipy.special import comb
from scipy import stats, signal
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean

from itertools import combinations
from IPython.display import display

# Keypoint indices (COCO format, 17 points)
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def loadSequences(root):
    """
    Load all .npy keypoint files from a directory tree.
    Each file is expected to contain a list of dictionaries with "keypoints" and "glove_box".
    The pitch type (label) is inferred from the filename (e.g., "curveball_0.npy" -> "curveball").
    Returns a list of (sequence_array, label) tuples.
    """
    seqs = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if not (f := f.lower()).endswith(".npy"):
                continue
            if f.startswith("unlabeled"):
                continue
            # Map filename patterns to pitch types
            match f:
                case name if "curveball" in name:
                    label = "curveball"
                case name if "cutter" in name:
                    label = "cutter"
                case name if "four_seam_fastball" in name:
                    label = "four_seam_fastball"
                case name if "slider" in name:
                    label = "slider"
                case name if "splitter" in name:
                    label = "splitter"
                case name if "two_seam_fastball" in name:
                    label = "two_seam_fastball"
                case name if "changeup" in name:
                    label = "changeup"
                case name if "fastball" in name:
                    label = "fastball"
                case _:
                    continue
            seqs.append((np.load(os.path.join(dirpath, f), allow_pickle=True), label))
    return seqs


# FEATURE EXTRACTION

# Helper functions
def _time_to_peak(series):
    """Normalized time (0‑1) of the maximum value in the series."""
    if len(series) == 0:
        return np.nan
    return np.argmax(series) / len(series)

def _time_to_min(series):
    """Normalized time (0‑1) of the minimum value in the series."""
    if len(series) == 0:
        return np.nan
    return np.argmin(series) / len(series)

def _velocity(series, dt):
    """First derivative using central differences."""
    return np.gradient(series, dt)

def _acceleration(series, dt):
    """Second derivative."""
    return np.gradient(np.gradient(series, dt), dt)

def _angular_velocity(angle_series, dt):
    """
    Angular velocity in degrees per second.
    Unwraps angles to avoid jumps around 360°→0°.
    """
    unwrapped = np.unwrap(np.deg2rad(angle_series)) * 180 / np.pi
    return np.gradient(unwrapped, dt)

def _path_length(trajectory):
    """Total Euclidean distance traveled along a trajectory (sum of step lengths)."""
    return np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))

def _angle_between_points(p1, p2, p3):
    """Angle at p2 formed by points p1-p2-p3, returned in degrees."""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def _forearm_angle(shoulder, elbow, wrist, reference='vertical'):
    """
    Angle of the forearm (elbow→wrist) relative to vertical or horizontal.
    For vertical: 0° = straight down, 180° = straight up.
    For horizontal: 0° = pointing right (positive x), 180° = left.
    """
    forearm = np.array(wrist) - np.array(elbow)
    if reference == 'vertical':
        ref_vec = np.array([0, -1])   # downward (y increases downward)
    elif reference == 'horizontal':
        ref_vec = np.array([1, 0])    # right
    else:
        raise ValueError("reference must be 'vertical' or 'horizontal'")
    dot = np.dot(forearm, ref_vec)
    norm_prod = np.linalg.norm(forearm) * np.linalg.norm(ref_vec) + 1e-8
    angle = np.degrees(np.arccos(np.clip(dot / norm_prod, -1.0, 1.0)))
    return angle

def _compute_angles(frame):
    """
    Compute a set of joint angles and body orientations for a single frame (17x2).
    Returns a dictionary of angle names → angle value (degrees).
    """
    angles = {}

    # Arm angles (shoulder and elbow)
    angles['left_shoulder'] = _angle_between_points(
        frame[LEFT_HIP], frame[LEFT_SHOULDER], frame[LEFT_ELBOW])
    angles['left_elbow'] = _angle_between_points(
        frame[LEFT_SHOULDER], frame[LEFT_ELBOW], frame[LEFT_WRIST])
    angles['right_shoulder'] = _angle_between_points(
        frame[RIGHT_HIP], frame[RIGHT_SHOULDER], frame[RIGHT_ELBOW])
    angles['right_elbow'] = _angle_between_points(
        frame[RIGHT_SHOULDER], frame[RIGHT_ELBOW], frame[RIGHT_WRIST])

    # Leg angles (hip and knee)
    angles['left_hip'] = _angle_between_points(
        frame[LEFT_SHOULDER], frame[LEFT_HIP], frame[LEFT_KNEE])
    angles['left_knee'] = _angle_between_points(
        frame[LEFT_HIP], frame[LEFT_KNEE], frame[LEFT_ANKLE])
    angles['right_hip'] = _angle_between_points(
        frame[RIGHT_SHOULDER], frame[RIGHT_HIP], frame[RIGHT_KNEE])
    angles['right_knee'] = _angle_between_points(
        frame[RIGHT_HIP], frame[RIGHT_KNEE], frame[RIGHT_ANKLE])

    # Trunk inclination (spine angle relative to vertical)
    shoulder_mid = (frame[LEFT_SHOULDER] + frame[RIGHT_SHOULDER]) / 2
    hip_mid = (frame[LEFT_HIP] + frame[RIGHT_HIP]) / 2
    trunk_vec = shoulder_mid - hip_mid
    vertical = np.array([0, -1])
    angles['trunk_inclination'] = np.degrees(np.arccos(
        np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-8)))

    # Pelvis rotation (angle of line connecting hips, relative to horizontal)
    hip_line = frame[RIGHT_HIP] - frame[LEFT_HIP]
    angles['pelvis_rotation'] = np.degrees(np.arctan2(hip_line[1], hip_line[0]))

    # Shoulder rotation (angle of line connecting shoulders)
    shoulder_line = frame[RIGHT_SHOULDER] - frame[LEFT_SHOULDER]
    angles['shoulder_rotation'] = np.degrees(np.arctan2(shoulder_line[1], shoulder_line[0]))

    # Hip‑shoulder separation (X‑factor) – difference in rotation angles
    angles['hip_shoulder_sep'] = angles['shoulder_rotation'] - angles['pelvis_rotation']

    # Head tilt (angle of line connecting ears)
    ear_line = frame[RIGHT_EAR] - frame[LEFT_EAR] if RIGHT_EAR in frame and LEFT_EAR in frame else np.array([0, 0])
    angles['head_tilt'] = np.degrees(np.arctan2(ear_line[1], ear_line[0]))

    # Forearm angle relative to vertical – proxy for arm slot (release height)
    angles['right_forearm_vertical'] = _forearm_angle(
        frame[RIGHT_SHOULDER], frame[RIGHT_ELBOW], frame[RIGHT_WRIST], 'vertical')

    return angles


def extractFeatures(seq, class_label, fps=60):
    """
    Extract a comprehensive set of kinematic features from one pitch sequence.
    Input:
      seq: either a list of dicts (with "keypoints" and "glove_box") or a numpy array (T,17,2)
      class_label: string (pitch type)
      fps: frames per second (default 60)
    Returns a dictionary of feature names → values, including the class label.
    """
    # Handle input format: if seq contains dicts, separate keypoints and glove boxes
    if len(seq) > 0 and isinstance(seq[0], dict):
        glove_seq = np.array([frame.get("glove_box", [-1, -1, -1, -1]) for frame in seq], dtype=np.float32)
        seq = np.array([frame["keypoints"] for frame in seq], dtype=np.float32)
    else:
        glove_seq = np.full((len(seq), 4), -1.0, dtype=np.float32)

    T = seq.shape[0]
    dt = 1.0 / fps
    duration = T * dt

    # Compute angles for each frame
    angles_list = [_compute_angles(seq[t]) for t in range(T)]
    angle_names = angles_list[0].keys()
    angles = {name: np.array([a[name] for a in angles_list], dtype=float)
              for name in angle_names}

    # Position data for key points (for linear velocities, distances, etc.)
    pos = {
        'left_knee': seq[:, LEFT_KNEE, :],
        'right_knee': seq[:, RIGHT_KNEE, :],
        'left_ankle': seq[:, LEFT_ANKLE, :],
        'right_ankle': seq[:, RIGHT_ANKLE, :],
        'left_shoulder': seq[:, LEFT_SHOULDER, :],
        'right_shoulder': seq[:, RIGHT_SHOULDER, :],
        'left_hip': seq[:, LEFT_HIP, :],
        'right_hip': seq[:, RIGHT_HIP, :],
        'left_wrist': seq[:, LEFT_WRIST, :],
        'right_wrist': seq[:, RIGHT_WRIST, :],
        'left_elbow': seq[:, LEFT_ELBOW, :],
        'right_elbow': seq[:, RIGHT_ELBOW, :],
    }

    # Torso length (used to normalize distances, making features scale‑invariant)
    shoulder_mid = (seq[0, LEFT_SHOULDER] + seq[0, RIGHT_SHOULDER]) / 2
    hip_mid = (seq[0, LEFT_HIP] + seq[0, RIGHT_HIP]) / 2
    torso_length = euclidean(shoulder_mid, hip_mid)

    features = {'class': class_label, 'duration': duration}

    # 1. Basic angle statistics (mean, std, range)
    for name, series in angles.items():
        series_clean = series[~np.isnan(series)]
        if len(series_clean) == 0:
            continue
        features[f'{name}_mean'] = np.mean(series_clean)
        features[f'{name}_std'] = np.std(series_clean)
        features[f'{name}_range'] = np.nanmax(series) - np.nanmin(series)

    # 2. Temporal features (when do key events happen?)
    left_knee_y = pos['left_knee'][:, 1]
    features['time_to_peak_left_knee'] = _time_to_peak(left_knee_y)

    if 'right_elbow' in angles:
        features['time_to_min_right_elbow'] = _time_to_min(angles['right_elbow'])   # max flexion
        features['time_to_max_right_elbow'] = _time_to_peak(angles['right_elbow'])
    if 'left_elbow' in angles:
        features['time_to_min_left_elbow'] = _time_to_min(angles['left_elbow'])
        features['time_to_max_left_elbow'] = _time_to_peak(angles['left_elbow'])
    if 'hip_shoulder_sep' in angles:
        features['time_to_max_hip_shoulder_sep'] = _time_to_peak(angles['hip_shoulder_sep'])
        features['time_to_min_hip_shoulder_sep'] = _time_to_min(angles['hip_shoulder_sep'])
    if 'right_forearm_vertical' in angles:
        features['time_to_min_forearm_angle'] = _time_to_min(angles['right_forearm_vertical'])
        features['time_to_max_forearm_angle'] = _time_to_peak(angles['right_forearm_vertical'])

    # 3. Angular velocities (peak, mean, timing)
    for joint in ['right_elbow', 'left_elbow', 'right_shoulder', 'left_shoulder',
                  'right_knee', 'left_knee', 'hip_shoulder_sep', 'pelvis_rotation',
                  'shoulder_rotation']:
        if joint in angles:
            ang_vel = _angular_velocity(angles[joint], dt)
            features[f'{joint}_ang_vel_max'] = np.nanmax(np.abs(ang_vel))
            features[f'{joint}_ang_vel_mean'] = np.nanmean(np.abs(ang_vel))
            features[f'{joint}_time_to_max_ang_vel'] = _time_to_peak(np.abs(ang_vel))

    # Linear speed of right wrist (throwing hand)
    rw_x, rw_y = pos['right_wrist'][:, 0], pos['right_wrist'][:, 1]
    vel_x = _velocity(rw_x, dt)
    vel_y = _velocity(rw_y, dt)
    speed_rw = np.sqrt(vel_x**2 + vel_y**2)
    features['right_wrist_speed_max'] = np.max(speed_rw)
    features['right_wrist_speed_mean'] = np.mean(speed_rw)
    features['right_wrist_time_to_max_speed'] = _time_to_peak(speed_rw)

    # Linear speed of left knee (drive leg)
    lk_x, lk_y = pos['left_knee'][:, 0], pos['left_knee'][:, 1]
    vel_x_lk = _velocity(lk_x, dt)
    vel_y_lk = _velocity(lk_y, dt)
    speed_lk = np.sqrt(vel_x_lk**2 + vel_y_lk**2)
    features['left_knee_speed_max'] = np.max(speed_lk)
    features['left_knee_speed_mean'] = np.mean(speed_lk)
    features['left_knee_time_to_max_speed'] = _time_to_peak(speed_lk)

    # 4. Key positions and distances (normalized by torso length)
    features['left_knee_max_height'] = np.max(left_knee_y)
    stride = np.linalg.norm(pos['left_ankle'][-1] - pos['right_ankle'][-1])
    features['stride_width_norm'] = stride / torso_length
    dist_rw_rs = np.linalg.norm(pos['right_wrist'] - pos['right_shoulder'], axis=1)
    features['min_dist_wrist_shoulder_norm'] = np.min(dist_rw_rs) / torso_length
    left_knee_start_x = pos['left_knee'][0, 0]
    left_knee_forward = pos['left_knee'][:, 0] - left_knee_start_x
    features['left_knee_max_forward_norm'] = np.max(left_knee_forward) / torso_length

    # Center of mass (centroid of all keypoints) – range and final position
    centroid = np.nanmean(seq, axis=1)  # (T,2)
    features['com_vertical_range_norm'] = (np.max(centroid[:,1]) - np.min(centroid[:,1])) / torso_length
    features['com_horizontal_range_norm'] = (np.max(centroid[:,0]) - np.min(centroid[:,0])) / torso_length
    features['com_final_vertical_norm'] = centroid[-1, 1] / torso_length
    features['com_final_horizontal_norm'] = centroid[-1, 0] / torso_length

    # 5. Trajectory shape (path length)
    features['left_knee_path_length_norm'] = _path_length(pos['left_knee']) / torso_length
    features['right_wrist_path_length_norm'] = _path_length(pos['right_wrist']) / torso_length

    # 6. Symmetry / asymmetry at the moment of peak left knee height
    peak_idx = np.argmax(left_knee_y)
    for side in ['shoulder', 'elbow', 'hip', 'knee']:
        left_name = f'left_{side}'
        right_name = f'right_{side}'
        if left_name in angles and right_name in angles:
            features[f'{side}_angle_diff_at_peak'] = angles[left_name][peak_idx] - angles[right_name][peak_idx]

    # Range ratios (left/right) to capture asymmetry
    for side in ['shoulder', 'elbow', 'hip', 'knee']:
        left_name = f'left_{side}'
        right_name = f'right_{side}'
        if left_name in angles and right_name in angles:
            left_range = np.nanmax(angles[left_name]) - np.nanmin(angles[left_name])
            right_range = np.nanmax(angles[right_name]) - np.nanmin(angles[right_name])
            if right_range != 0:
                features[f'{side}_range_ratio'] = left_range / right_range
            else:
                features[f'{side}_range_ratio'] = np.nan

    # 7. Coordination (correlations between different body parts)
    if 'right_elbow' in angles:
        # Pearson correlation (ignoring phase)
        pearson = np.corrcoef(left_knee_y, angles['right_elbow'])[0, 1]
        features['corr_left_knee_right_elbow'] = pearson
        # Cross‑correlation max to capture lagged relationships
        corr = signal.correlate(left_knee_y - np.mean(left_knee_y),
                                angles['right_elbow'] - np.mean(angles['right_elbow']),
                                mode='full')
        features['max_cross_corr_left_knee_right_elbow'] = np.max(np.abs(corr))
    if 'hip_shoulder_sep' in angles:
        pearson = np.corrcoef(left_knee_y, angles['hip_shoulder_sep'])[0, 1]
        features['corr_left_knee_hip_shoulder_sep'] = pearson

    # 8. Timing differences between events (e.g., knee peak vs. elbow flexion)
    if 'right_elbow' in angles:
        t_knee = _time_to_peak(left_knee_y)
        t_elbow_min = _time_to_min(angles['right_elbow'])
        features['time_diff_knee_peak_elbow_min'] = t_knee - t_elbow_min
    if 'hip_shoulder_sep' in angles:
        t_hip_sep_max = _time_to_peak(angles['hip_shoulder_sep'])
        features['time_diff_knee_peak_hip_sep_max'] = t_knee - t_hip_sep_max

    # 9. Glove (mitt) features – if available
    valid_glove = glove_seq[:, 0] != -1
    if np.sum(valid_glove) > 2:
        # Interpolate missing glove detections
        glove_seq_interp = np.copy(glove_seq)
        for i in range(4):
            arr = glove_seq_interp[:, i]
            arr[~valid_glove] = np.interp(np.arange(len(arr))[~valid_glove],
                                          np.arange(len(arr))[valid_glove],
                                          arr[valid_glove])

        glove_centers = np.stack([(glove_seq_interp[:, 0] + glove_seq_interp[:, 2]) / 2,
                                  (glove_seq_interp[:, 1] + glove_seq_interp[:, 3]) / 2], axis=1)

        features['glove_max_height_norm'] = np.max(glove_centers[:, 1]) / torso_length
        features['glove_min_height_norm'] = np.min(glove_centers[:, 1]) / torso_length
        features['glove_height_range_norm'] = (np.max(glove_centers[:, 1]) - np.min(glove_centers[:, 1])) / torso_length
        g_vel_x = _velocity(glove_centers[:, 0], dt)
        g_vel_y = _velocity(glove_centers[:, 1], dt)
        g_speed = np.sqrt(g_vel_x**2 + g_vel_y**2)
        features['glove_speed_max'] = np.max(g_speed)
        features['glove_speed_mean'] = np.mean(g_speed)
        features['glove_time_to_max_speed'] = _time_to_peak(g_speed)
        shoulder_mids = (seq[:, LEFT_SHOULDER] + seq[:, RIGHT_SHOULDER]) / 2
        dist_to_shoulder = np.linalg.norm(glove_centers - shoulder_mids, axis=1)
        features['glove_dist_to_shoulder_min_norm'] = np.min(dist_to_shoulder) / torso_length
        features['glove_dist_to_shoulder_mean_norm'] = np.mean(dist_to_shoulder) / torso_length

    return features

# STATISTICAL TESTING (Mann‑Whitney U with Bonferroni correction)

def pairwiseTests(df, feature_cols, class_col="class", alpha=0.05):
    """
    Perform Mann‑Whitney U tests for every feature across every pair of pitch classes.
    Returns a DataFrame with test statistics, p‑values, effect sizes (U/(n1*n2)),
    and a "significant" flag after Bonferroni correction for multiple comparisons.
    """
    classes = df[class_col].unique()
    pairs = list(combinations(classes, 2))
    res = []

    for feature in feature_cols:
        for c1, c2 in pairs:
            data1 = df[df[class_col] == c1][feature].dropna()
            data2 = df[df[class_col] == c2][feature].dropna()
            if len(data1) < 2 or len(data2) < 2:
                continue
            U, p = stats.mannwhitneyu(data1, data2, alternative="two-sided")
            n1, n2 = len(data1), len(data2)
            effect_size = U / (n1 * n2)
            res.append({
                "feature": feature,
                "class1": c1,
                "class2": c2,
                "U": U,
                "p": p,
                "effect_size": effect_size,
                "n1": n1,
                "n2": n2
            })
    result_df = pd.DataFrame(res)
    # Bonferroni correction: number of comparisons = comb(len(classes),2)
    result_df["significant"] = result_df["p"] < alpha / comb(len(classes), 2, exact=True)
    return result_df


# EXECUTION

seqs = loadSequences("/content/drive/My Drive/Pitch-Tipping/Keypoints")
features = pd.DataFrame([extractFeatures(seq, label) for seq, label in seqs])
featureCols = [col for col in features.columns if col != "class"]
display(pairwiseTests(features, featureCols).sort_values(
    ["effect_size", "p"],
    key=lambda x: (x - 0.5).abs() if x.name == "effect_size" else x,
    ascending=[False, True]
)[:10])

# Re‑label to binary groups (fastball vs. off‑speed)
relabel = dict.fromkeys(["cutter", "four_seam_fastball", "two_seam_fastball"], "fastball")
relabel.update(dict.fromkeys(["curveball", "slider", "splitter"], "offspeed"))
seqs = loadSequences("/content/drive/My Drive/Pitch-Tipping/Keypoints")
seqs = list(map(lambda pair: (pair[0], relabel[pair[1]]), seqs))
features = pd.DataFrame([extractFeatures(seq, label) for seq, label in seqs])
features, dtw_mat = add_dtw_features(seqs, [label for _, label in seqs], features)
featureCols = [col for col in features.columns if col != "class"]
display(pairwiseTests(features, featureCols).sort_values(
    ["effect_size", "p"],
    key=lambda x: (x - 0.5).abs() if x.name == "effect_size" else x,
    ascending=[False, True]
)[:10])


# MACHINE LEARNING: Binary classification (Fastball vs. Offspeed)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

df_ml = features.copy()

# Map pitch types to binary labels (0 = fastball, 1 = offspeed)
fastballs = ["four_seam_fastball", "two_seam_fastball", "cutter"]
offspeeds = ["curveball", "slider", "splitter", "changeup"]

safe_fastballs = ["fastball", "cutter", "four_seam_fastball", "two_seam_fastball"]
safe_offspeeds = ["offspeed", "changeup", "curveball", "slider", "splitter"]

def label_bin(val):
    s = str(val).lower().strip()
    if any(fb in s for fb in safe_fastballs):
        return 0
    if any(osp in s for osp in safe_offspeeds):
        return 1
    return -1

df_ml['binary_class'] = df_ml['class'].apply(label_bin)
print(f"\n🔍 Raw Labels Found: {list(df_ml['class'].unique())}")
print(f"🔢 Mapped Binary Classes: {dict(df_ml['binary_class'].value_counts())}")

df_ml = df_ml[df_ml['binary_class'] != -1]   # remove any unrecognized

X_raw = df_ml.drop(columns=['class', 'binary_class'])
# Drop features with >20% missing values, then median‑impute the rest
X_clean = X_raw.dropna(axis=1, thresh=int(len(X_raw)*0.8))
X_clean = X_clean.fillna(X_clean.median())
y = df_ml['binary_class']

print(f"Dataset ready. Fastballs: {sum(y==0)}, Offspeed: {sum(y==1)}, Total Features: {X_clean.shape[1]}")
feature_names = X_clean.columns

# --- Random Forest (interpretable feature importances) ---
print("\n--- Model 1: Random Forest Classifier ---")
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, class_weight="balanced")
rf.fit(X_clean, y)

importances = rf.feature_importances_
top_indices = np.argsort(importances)[::-1][:5]

print("🚨 TOP 5 KINEMATIC TELLS (Random Forest):")
for i in top_indices:
    print(f"{feature_names[i]}: {importances[i]*100:.1f}% AI Importance")

# --- Logistic Regression with L1 penalty (automatic feature selection) ---
print("\n--- Model 2: Logistic Regression (L1 Lasso) ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

lr = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=42, class_weight="balanced")
lr.fit(X_scaled, y)

coefs = lr.coef_[0]
nonzero_indices = np.where(coefs != 0)[0]
sorted_nonzero = sorted(nonzero_indices, key=lambda i: abs(coefs[i]), reverse=True)

print(f"L1 Lasso mathematically deleted {len(feature_names) - len(nonzero_indices)} random noisy features!")
print("🚨 TOP COMMANDING KINEMATIC TELLS (L1 Logistic Regression):")
for i in sorted_nonzero[:5]:
    direction = "HIGHER" if coefs[i] > 0 else "LOWER"
    print(f"{feature_names[i]}: Coefficient Magnitude {abs(coefs[i]):.3f} (Mechanic is numerically {direction} on Offspeed pitches)")

print("\n🎯 >> Type the highest ranking Feature Name directly into Dashboard Tab 2 -> Level 4 AI Highlighter! <<")
