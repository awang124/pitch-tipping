import os
import numpy as np
import scipy as sc
import pandas as pd
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW     = 7
RIGHT_ELBOW    = 8
LEFT_WRIST     = 9
RIGHT_WRIST    = 10
LEFT_HIP       = 11
RIGHT_HIP      = 12
LEFT_KNEE      = 13
RIGHT_KNEE     = 14
LEFT_ANKLE     = 15
RIGHT_ANKLE    = 16

# ──────────────────────────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def _time_to_peak(series):
    return np.argmax(series) / len(series) if len(series) else np.nan

def _time_to_min(series):
    return np.argmin(series) / len(series) if len(series) else np.nan

def _velocity(series, dt):
    return np.gradient(series, dt)

def _angular_velocity(angle_series, dt):
    return np.gradient(np.unwrap(np.deg2rad(angle_series)) * 180 / np.pi, dt)

def _path_length(trajectory):
    return np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))

def _angle_between_points(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def _forearm_angle(shoulder, elbow, wrist, reference="vertical"):
    ref_vec = np.array([0, -1]) if reference == "vertical" else np.array([1, 0])
    dot = np.dot(forearm := np.array(wrist) - np.array(elbow), ref_vec)
    norm_prod = np.linalg.norm(forearm) * np.linalg.norm(ref_vec) + 1e-8
    return np.degrees(np.arccos(np.clip(dot / norm_prod, -1.0, 1.0)))

def _compute_angles(frame):
    angles = {}
    angles["left_shoulder"] = _angle_between_points(
        frame[LEFT_HIP], frame[LEFT_SHOULDER], frame[LEFT_ELBOW]
    )
    angles["left_elbow"] = _angle_between_points(
        frame[LEFT_SHOULDER], frame[LEFT_ELBOW], frame[LEFT_WRIST]
    )
    angles["right_shoulder"] = _angle_between_points(
        frame[RIGHT_HIP], frame[RIGHT_SHOULDER], frame[RIGHT_ELBOW]
    )
    angles["right_elbow"] = _angle_between_points(
        frame[RIGHT_SHOULDER], frame[RIGHT_ELBOW], frame[RIGHT_WRIST]
    )
    angles["left_hip"] = _angle_between_points(
        frame[LEFT_SHOULDER], frame[LEFT_HIP], frame[LEFT_KNEE]
    )
    angles["left_knee"] = _angle_between_points(
        frame[LEFT_HIP], frame[LEFT_KNEE], frame[LEFT_ANKLE]
    )
    angles["right_hip"] = _angle_between_points(
        frame[RIGHT_SHOULDER], frame[RIGHT_HIP], frame[RIGHT_KNEE]
    )
    angles["right_knee"] = _angle_between_points(
        frame[RIGHT_HIP], frame[RIGHT_KNEE], frame[RIGHT_ANKLE]
    )
    shoulder_mid = (frame[LEFT_SHOULDER] + frame[RIGHT_SHOULDER]) / 2
    hip_mid = (frame[LEFT_HIP] + frame[RIGHT_HIP]) / 2
    trunk_vec = shoulder_mid - hip_mid
    angles["trunk_inclination"] = np.degrees(np.arccos(
        np.dot(trunk_vec, np.array([0, -1])) / (np.linalg.norm(trunk_vec) + 1e-8)
    ))
    hip_line = frame[RIGHT_HIP] - frame[LEFT_HIP]
    angles["pelvis_rotation"] = np.degrees(np.arctan2(hip_line[1], hip_line[0]))
    shoulder_line = frame[RIGHT_SHOULDER] - frame[LEFT_SHOULDER]
    angles["shoulder_rotation"] = np.degrees(np.arctan2(shoulder_line[1], shoulder_line[0]))
    angles["hip_shoulder_sep"] = angles["shoulder_rotation"] - angles["pelvis_rotation"]
    angles["right_forearm_vertical"] = _forearm_angle(
        frame[RIGHT_SHOULDER], frame[RIGHT_ELBOW], frame[RIGHT_WRIST], "vertical"
    )
    return angles

def extractFeatures(seq, label, fps=60):
    if len(seq) and isinstance(seq[0], dict):
        glove_seq = np.array(
            [frame.get("glove_box", [-1] * 4) for frame in seq], dtype=np.float32
        )
        seq = np.array([frame["keypoints"] for frame in seq], dtype=np.float32)
    else:
        glove_seq = np.full((len(seq), 4), -1.0, dtype=np.float32)
    T = seq.shape[0]
    dt = 1.0 / fps
    duration = T * dt
    angles_list = [_compute_angles(seq[t]) for t in range(T)]
    angles = {
        name: np.array([a[name] for a in angles_list], dtype=np.float32)
        for name in angles_list[0].keys()
    }
    pos = {
        "left_knee": seq[:, LEFT_KNEE, :], "right_knee": seq[:, RIGHT_KNEE, :],
        "left_ankle": seq[:, LEFT_ANKLE, :], "right_ankle": seq[:, RIGHT_ANKLE, :],
        "left_shoulder": seq[:, LEFT_SHOULDER, :], "right_shoulder": seq[:, RIGHT_SHOULDER, :],
        "left_hip": seq[:, LEFT_HIP, :], "right_hip": seq[:, RIGHT_HIP, :],
        "left_wrist": seq[:, LEFT_WRIST, :], "right_wrist": seq[:, RIGHT_WRIST, :],
        "left_elbow": seq[:, LEFT_ELBOW, :], "right_elbow": seq[:, RIGHT_ELBOW, :],
    }
    shoulder_mid = (seq[0, LEFT_SHOULDER] + seq[0, RIGHT_SHOULDER]) / 2
    hip_mid = (seq[0, LEFT_HIP] + seq[0, RIGHT_HIP]) / 2
    torso_length = sc.spatial.distance.euclidean(shoulder_mid, hip_mid)
    features = {"class": label, "duration": duration}
    for name, series in angles.items():
        series_clean = series[~np.isnan(series)]
        if len(series_clean) == 0: continue
        features[f"{name}_mean"] = np.mean(series_clean)
        features[f"{name}_std"] = np.std(series_clean)
        features[f"{name}_range"] = np.nanmax(series) - np.nanmin(series)
    left_knee_y = pos["left_knee"][:, 1]
    features["time_to_peak_left_knee"] = _time_to_peak(left_knee_y)
    if "right_elbow" in angles:
        features["time_to_min_right_elbow"] = _time_to_min(angles["right_elbow"])
        features["time_to_max_right_elbow"] = _time_to_peak(angles["right_elbow"])
    if "hip_shoulder_sep" in angles:
        features["time_to_max_hip_shoulder_sep"] = _time_to_peak(angles["hip_shoulder_sep"])
    if "right_forearm_vertical" in angles:
        features["time_to_min_forearm_angle"] = _time_to_min(angles["right_forearm_vertical"])
    for joint in [
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder", "left_knee", "right_knee"
    ]:
        if joint in angles:
            ang_vel = _angular_velocity(angles[joint], dt)
            features[f"{joint}_ang_vel_max"] = np.nanmax(np.abs(ang_vel))
            features[f"{joint}_ang_vel_mean"] = np.nanmean(np.abs(ang_vel))
    rw_x, rw_y = pos["right_wrist"][:, 0], pos["right_wrist"][:, 1]
    speed_rw = np.sqrt(_velocity(rw_x, dt) ** 2 + _velocity(rw_y, dt) ** 2)
    features["right_wrist_speed_max"] = np.max(speed_rw)
    lk_x, lk_y = pos["left_knee"][:, 0], pos["left_knee"][:, 1]
    speed_lk = np.sqrt(_velocity(lk_x, dt) ** 2 + _velocity(lk_y, dt) ** 2)
    features["left_knee_speed_max"] = np.max(speed_lk)
    features["left_knee_max_height"] = np.max(left_knee_y)
    valid_glove = glove_seq[:, 0] != -1
    if np.sum(valid_glove) > 2:
        glove_seq_interp = np.copy(glove_seq)
        for i in range(4):
            arr = glove_seq_interp[:, i]
            arr[~valid_glove] = np.interp(
                np.arange(len(arr))[~valid_glove],
                np.arange(len(arr))[valid_glove],
                arr[valid_glove]
            )
        glove_centers = np.stack([
            (glove_seq_interp[:, 0] + glove_seq_interp[:, 2]) / 2,
            (glove_seq_interp[:, 1] + glove_seq_interp[:, 3]) / 2
        ], axis=1)
        g_vel_x = _velocity(glove_centers[:, 0], dt)
        g_vel_y = _velocity(glove_centers[:, 1], dt)
        g_speed = np.sqrt(g_vel_x ** 2 + g_vel_y ** 2)
        features["glove_speed_max"] = np.max(g_speed)
        features["glove_speed_mean"] = np.mean(g_speed)
    return features

# ──────────────────────────────────────────────────────────────────────────────────────────────────
# PITCH LABELING
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def loadSequences(root):
    seqs = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if not f.lower().endswith(".npy"):
                continue
            elif "unlabeled" in (name := f.lower()):
                continue
            elif "changeup" in name:
                label = "changeup"
            elif "curveball" in name:
                label = "curveball"
            elif "cutter" in name:
                label = "cutter"
            elif "four_seam_fastball" in name:
                label = "four_seam_fastball"
            elif "slider" in name:
                label = "slider"
            elif "splitter" in name:
                label = "splitter"
            elif "two_seam_fastball" in name:
                label = "two_seam_fastball"
            elif "fast" in name:
                label = "fastball"
            elif "off" in name:
                label = "offspeed"
            else:
                label = "other"
            seqs.append((np.load(os.path.join(dirpath, f), allow_pickle=True), label))
    return seqs

# ──────────────────────────────────────────────────────────────────────────────────────────────────
# FEATURE SIGNIFICANCE (MANN-WHITNEY TEST)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def mannWhitney(df, alpha=0.05, n_show=10):
    classes = df["class"].unique()
    pairs = list(combinations(classes, 2))
    res = []
    for feature in df.columns:
        if feature == "class":
            continue
        for c1, c2 in pairs:
            data1 = df[df["class"] == c1][feature].dropna()
            data2 = df[df["class"] == c2][feature].dropna()
            if len(data1.unique()) < 2 or len(data2.unique()) < 2:
                continue
            U, p = sc.stats.mannwhitneyu(data1, data2, alternative="two-sided")
            n1, n2 = len(data1), len(data2)
            effect_size = U / (n1 * n2)
            res.append({
                "Feature": feature,
                "U Statistic": U,
                "p-value": p,
                "Effect Size": effect_size
            })
    resdf = pd.DataFrame(res)
    resdf["Significant"] = resdf["p-value"] < alpha /\
        sc.special.comb(len(classes), 2, exact=True)
    return resdf.sort_values(
        ["p-value", "Effect Size"],
        key=lambda x: x if x.name == "p-value" else (x - 0.5).abs(),
        ascending=[True, False]
    ).head(n_show)

# ──────────────────────────────────────────────────────────────────────────────────────────────────
# PITCH CLASSIFICATION
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def logisticRegression(df, n_show=5):
    X = df.drop(columns=["class"])
    X = X.fillna(X.median())
    y = df["class"]
    X = StandardScaler().fit_transform(X)
    coefs = LogisticRegression(
        l1_ratio=1, solver="liblinear", C=0.5, class_weight="balanced", random_state=0
    ).fit(X, y).coef_[0]
    idxs = sorted(
        np.where(coefs != 0)[0], key=lambda i: abs(coefs[i]), reverse=True
    )[:n_show]
    return [{"Feature": df.columns[i], "Coefficient": coefs[i]} for i in idxs]

def decisionTree(df, n_show=5):
    X = df.drop(columns=["class"])
    X = X.fillna(X.median())
    y = df["class"]
    X = StandardScaler().fit_transform(X)
    coefs = DecisionTreeClassifier(
        max_depth=10, class_weight="balanced", random_state=0
    ).fit(X, y).feature_importances_
    idxs = np.argsort(coefs)[::-1]
    idxs = [i for i in idxs if coefs[i] > 0][:n_show]
    return [{"Feature": df.columns[i], "Importance": coefs[i]} for i in idxs]

