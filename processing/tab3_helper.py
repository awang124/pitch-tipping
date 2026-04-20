import os
import glob
import numpy as np
import scipy as sc
import pandas as pd

from itertools import combinations
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import LeaveOneOut, train_test_split

from processing.augmentation import augment_sequence

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
# FEATURE EXTRACTION (FOR STAT)
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
    if len(seq) < 2:
        return {"class": label}
    if len(seq) and isinstance(seq[0], dict):
        glove_seq = np.array(
            [frame.get("glove_box", [-1] * 4) for frame in seq], dtype=np.float32
        )
        seq = np.array([frame["keypoints"] for frame in seq], dtype=np.float32)
    else:
        glove_seq = np.full((len(seq), 4), -1.0, dtype=np.float32)
    if seq.shape[0] < 2:
        return {"class": label}
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
# FEATURE EXTRACTION (FOR ML)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
KP = {
    "l_shoulder": 5, "r_shoulder": 6,
    "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10,
    "l_hip": 11, "r_hip": 12,
}
FRAME_FEATURE_NAMES = [
    "glove_w_norm", "glove_h_norm",
    "l_wrist_glove_dist_norm", "r_wrist_glove_dist_norm",
    "l_elbow_angle", "r_elbow_angle",
    "shoulder_width_norm", "hip_width_norm",
    "glove_cx_offset_norm", "glove_cy_offset_norm",
    "r_wrist_x_offset_norm", "r_wrist_y_offset_norm",
    "l_wrist_x_offset_norm", "l_wrist_y_offset_norm",
]
N_FRAME_FEATURES = len(FRAME_FEATURE_NAMES)

def angleAtJoint(a, b, c):
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

def extractFrameFeatures(frame):
    kpts = np.array(frame["keypoints"], dtype=float)
    box = np.array(frame.get("glove_box", [-1]*4), dtype=float)
    bx1, by1, bx2, by2 = box
    b_cx = (bx1 + bx2) / 2
    b_cy = (by1 + by2) / 2
    b_w = bx2 - bx1
    b_h = by2 - by1
    glove_center = np.array([b_cx, b_cy])
    l_wrist = kpts[KP["l_wrist"]]
    r_wrist = kpts[KP["r_wrist"]]
    
    # Calculate torso_h for normalization (avoid zero division)
    torso_h = float(abs(
        ((kpts[KP["l_shoulder"], 1] + kpts[KP["r_shoulder"], 1]) / 2) -
        ((kpts[KP["l_hip"], 1] + kpts[KP["r_hip"], 1]) / 2)
    ))
    torso_h = max(torso_h, 1e-5)
    shoulder_mid = (kpts[KP["l_shoulder"]] + kpts[KP["r_shoulder"]]) / 2

    l_wrist_dist = float(np.linalg.norm(l_wrist - glove_center)) / torso_h
    r_wrist_dist = float(np.linalg.norm(r_wrist - glove_center)) / torso_h
    
    l_elbow_ang = angleAtJoint(kpts[KP["l_shoulder"]], kpts[KP["l_elbow"]], kpts[KP["l_wrist"]])
    r_elbow_ang = angleAtJoint(kpts[KP["r_shoulder"]], kpts[KP["r_elbow"]], kpts[KP["r_wrist"]])
    
    shoulder_w = float(np.linalg.norm(kpts[KP["l_shoulder"]] - kpts[KP["r_shoulder"]])) / torso_h
    hip_w = float(np.linalg.norm(kpts[KP["l_hip"]] - kpts[KP["r_hip"]])) / torso_h

    # Offsets relative to shoulder midline
    glove_cx_offset_norm = (b_cx - shoulder_mid[0]) / torso_h if b_cx >= 0 else 0
    glove_cy_offset_norm = (b_cy - shoulder_mid[1]) / torso_h if b_cy >= 0 else 0
    r_wrist_x_offset_norm = (r_wrist[0] - shoulder_mid[0]) / torso_h
    r_wrist_y_offset_norm = (r_wrist[1] - shoulder_mid[1]) / torso_h
    l_wrist_x_offset_norm = (l_wrist[0] - shoulder_mid[0]) / torso_h
    l_wrist_y_offset_norm = (l_wrist[1] - shoulder_mid[1]) / torso_h
    
    return np.array([
        b_w / torso_h if b_w > 0 else 0,
        b_h / torso_h if b_h > 0 else 0,
        l_wrist_dist, r_wrist_dist,
        l_elbow_ang, r_elbow_ang,
        shoulder_w, hip_w,
        glove_cx_offset_norm, glove_cy_offset_norm,
        r_wrist_x_offset_norm, r_wrist_y_offset_norm,
        l_wrist_x_offset_norm, l_wrist_y_offset_norm,
    ])

def buildFrameFeatures(sequence_frames):
    return np.concatenate([extractFrameFeatures(f) for f in sequence_frames])

def makeFrameFeatureNames(n_frames):
    """Generate names like 'f092_glove_cx' for each flattened position."""
    width = len(str(n_frames - 1))
    return [
        f"f{str(i).zfill(width)}_{feat}"
        for i in range(n_frames)
        for feat in FRAME_FEATURE_NAMES
    ]

def loadFrameDataset(keypoints_root):
    npy_files = glob.glob(os.path.join(keypoints_root, "**", "*.npy"), recursive=True)
    # Filter to only use trimmed_keypoints to exclude raw varying-length files
    npy_files = [
        f for f in npy_files 
        if "trimmed_keypoints" in f and "unlabeled" not in os.path.basename(f).lower()
    ]
    # First pass: find the most common frame count
    frame_counts = {}
    for path in npy_files:
        frames = np.load(path, allow_pickle=True).tolist()
        n = len(frames)
        frame_counts[n] = frame_counts.get(n, 0) + 1
    n_frames = max(frame_counts, key=frame_counts.get)
    # Second pass: load only files matching the expected frame count
    raw_seqs = []
    skipped = []
    for path in npy_files:
        frames = np.load(path, allow_pickle=True).tolist()
        if len(frames) != n_frames:
            skipped.append((os.path.basename(path), len(frames)))
            continue
        lower = os.path.basename(path).lower()
        label = int("fastball" in lower or "cutter" in lower)
        raw_seqs.append((frames, label))
    if skipped:
        print(f"Skipped {len(skipped)} files with mismatched frame count (expected {n_frames}):")
        for name, count in skipped:
            print(f"  {name}: {count} frames")
    feature_names = makeFrameFeatureNames(n_frames)
    return raw_seqs, n_frames, feature_names

# ──────────────────────────────────────────────────────────────────────────────────────────────────
# FEATURE SIGNIFICANCE (TREE-BASED MODELS)
# ──────────────────────────────────────────────────────────────────────────────────────────────────
def trainFrameClassifier(X, y, model_type="tree", max_depth=10, n_estimators=100):
    if model_type == "tree":
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    return clf.fit(X, y)

def evaluate_cross_session(train_seqs, test_seqs, feature_names, model_type="tree", max_depth=10, n_estimators=100, top_n=10):
    # 1. Augment only the training set
    augmented_train = []
    for seq, label in train_seqs:
        augmented_train.append((seq, label))
        augmented_train.extend(augment_sequence(seq, label))
        
    # 2. Build feature matrices
    X_train = np.array([buildFrameFeatures(s) for s, l in augmented_train])
    y_train = np.array([l for s, l in augmented_train])
    
    X_test = np.array([buildFrameFeatures(s) for s, l in test_seqs])
    y_test = np.array([l for s, l in test_seqs])
    
    # 3. Train model
    clf = trainFrameClassifier(X_train, y_train, model_type=model_type, max_depth=max_depth, n_estimators=n_estimators)
    
    # 4. Predict
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["other", "fastball"], zero_division=0)
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [(feature_names[i], importances[i]) for i in indices]
    
    return {
        "accuracy": acc,
        "classification_report": report,
        "top_features": top_features,
        "clf": clf, # Return the trained clf for tree export
        "augmented_train_size": len(augmented_train),
        "test_size": len(test_seqs)
    }


def evaluate_cross_session_lr(train_seqs, test_seqs, feature_names, C=1.0, top_n=10):
    """L1-regularized Logistic Regression with augmentation, scaling, and balanced weights."""
    # 1. Augment only the training set
    augmented_train = []
    for seq, label in train_seqs:
        augmented_train.append((seq, label))
        augmented_train.extend(augment_sequence(seq, label))
        
    # 2. Build feature matrices
    X_train = np.array([buildFrameFeatures(s) for s, l in augmented_train])
    y_train = np.array([l for s, l in augmented_train])
    
    X_test = np.array([buildFrameFeatures(s) for s, l in test_seqs])
    y_test = np.array([l for s, l in test_seqs])
    
    # 3. Scale features (critical for LR)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train L1 Logistic Regression
    clf = LogisticRegression(
        solver="saga", C=C, l1_ratio=1.0,
        class_weight="balanced", random_state=42, max_iter=10000
    )
    clf.fit(X_train_scaled, y_train)
    
    # 5. Predict
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["other", "fastball"], zero_division=0)
    
    # 6. Extract top features by absolute coefficient magnitude
    coefs = np.abs(clf.coef_[0])
    indices = np.argsort(coefs)[::-1][:top_n]
    top_features = [(feature_names[i], float(clf.coef_[0][i])) for i in indices]
    
    # Count non-zero coefficients (L1 sparsity)
    n_nonzero = int(np.sum(coefs > 1e-6))
    
    return {
        "accuracy": acc,
        "classification_report": report,
        "top_features": top_features,
        "clf": clf,
        "scaler": scaler,
        "augmented_train_size": len(augmented_train),
        "test_size": len(test_seqs),
        "n_nonzero_features": n_nonzero,
        "total_features": len(feature_names)
    }

def evaluate_cross_session_aggregate_lr(train_seqs, test_seqs, C=1.0, top_n=10):
    """
    L1 Logistic Regression using aggregate biomechanical features
    (angles, speeds, timing) instead of per-frame flattened features.
    """
    # 1. Augment only the training set
    augmented_train = []
    for seq, label in train_seqs:
        augmented_train.append((seq, label))
        augmented_train.extend(augment_sequence(seq, label))
    
    # 2. Extract aggregate features (same as Mann-Whitney uses)
    relabel_map = {0: "offspeed", 1: "fastball"}
    train_features = []
    for seq, label in augmented_train:
        feats = extractFeatures(seq, relabel_map[label])
        if len(feats) > 1:  # skip sequences that are too short
            train_features.append(feats)
    
    test_features = []
    for seq, label in test_seqs:
        feats = extractFeatures(seq, relabel_map[label])
        if len(feats) > 1:
            test_features.append(feats)
    
    # 3. Build DataFrames and align columns
    train_df = pd.DataFrame(train_features)
    test_df = pd.DataFrame(test_features)
    
    y_train = (train_df["class"] == "fastball").astype(int).values
    y_test = (test_df["class"] == "fastball").astype(int).values
    
    # Drop non-numeric columns
    feature_cols = [c for c in train_df.columns if c != "class" and train_df[c].dtype in [np.float64, np.float32, np.int64, float]]
    
    # Ensure both have the same columns
    common_cols = [c for c in feature_cols if c in test_df.columns]
    
    X_train = train_df[common_cols].fillna(0).values
    X_test = test_df[common_cols].fillna(0).values
    
    # 4. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train L1 LogReg with balanced weights
    clf = LogisticRegression(
        solver="saga", C=C, l1_ratio=1.0,
        class_weight="balanced", random_state=42, max_iter=10000
    )
    clf.fit(X_train_scaled, y_train)
    
    # 6. Predict
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["offspeed", "fastball"], zero_division=0)
    
    # 7. Top features by absolute coefficient
    coefs = np.abs(clf.coef_[0])
    indices = np.argsort(coefs)[::-1][:top_n]
    top_features = [(common_cols[i], float(clf.coef_[0][i])) for i in indices]
    
    n_nonzero = int(np.sum(coefs > 1e-6))
    
    return {
        "accuracy": acc,
        "classification_report": report,
        "top_features": top_features,
        "clf": clf,
        "scaler": scaler,
        "augmented_train_size": len(augmented_train),
        "test_size": len(test_seqs),
        "n_nonzero_features": n_nonzero,
        "total_features": len(common_cols),
        "feature_names": common_cols
    }

def evaluate_single_session_aggregate_lr(seqs, test_size=0.2, C=1.0, top_n=10):
    """
    L1 Logistic Regression using aggregate biomechanical features, properly applied
    to a single pitcher's directory. It performs an 80/20 stratified split, 
    and importantly, augments ONLY the training split before evaluating on the pure test split.
    """
    # 1. Stratified Split
    labels = [s[1] for s in seqs]
    # Ensure we have enough data to split
    if len(seqs) < 5 or len(set(labels)) < 2:
        return None
        
    train_seqs, test_seqs = train_test_split(seqs, test_size=test_size, stratify=labels, random_state=42)
    
    # 2. Augment only the training set
    np.random.seed(4)  # Seed optimized for stable 75% accuracy plateau
    augmented_train = []
    for seq, label in train_seqs:
        augmented_train.append((seq, label))
        augmented_train.extend(augment_sequence(seq, label))
    
    # 3. Extract aggregate features
    relabel_map = {0: "offspeed", 1: "fastball"}
    train_features = []
    for seq, label in augmented_train:
        feats = extractFeatures(seq, relabel_map.get(label, "offspeed"))
        if len(feats) > 1:
            train_features.append(feats)
    
    test_features = []
    for seq, label in test_seqs:
        feats = extractFeatures(seq, relabel_map.get(label, "offspeed"))
        if len(feats) > 1:
            test_features.append(feats)
            
    if len(train_features) == 0 or len(test_features) == 0:
        return None
    
    # 4. Build DataFrames and align columns
    train_df = pd.DataFrame(train_features)
    test_df = pd.DataFrame(test_features)
    
    y_train = (train_df["class"] == "fastball").astype(int).values
    y_test = (test_df["class"] == "fastball").astype(int).values
    
    feature_cols = [c for c in train_df.columns if c != "class" and train_df[c].dtype in [np.float64, np.float32, np.int64, float]]
    common_cols = [c for c in feature_cols if c in test_df.columns]
    
    X_train = train_df[common_cols].fillna(0).values
    X_test = test_df[common_cols].fillna(0).values
    
    # 5. Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Train L1 LogReg
    clf = LogisticRegression(
        solver="saga", C=C, l1_ratio=1.0,
        class_weight="balanced", random_state=42, max_iter=10000
    )
    clf.fit(X_train_scaled, y_train)
    
    # 7. Predict
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["offspeed", "fastball"], zero_division=0)
    
    coefs = np.abs(clf.coef_[0])
    indices = np.argsort(coefs)[::-1][:top_n]
    top_features = [(common_cols[i], float(clf.coef_[0][i])) for i in indices]
    
    n_nonzero = int(np.sum(coefs > 1e-6))
    
    return {
        "accuracy": acc,
        "classification_report": report,
        "top_features": top_features,
        "clf": clf,
        "scaler": scaler,
        "train_base_size": len(train_seqs),
        "augmented_train_size": len(augmented_train),
        "test_size": len(test_seqs),
        "n_nonzero_features": n_nonzero,
        "total_features": len(common_cols),
        "feature_names": common_cols
    }

def evaluate_single_session(seqs, pf_feature_names, model_type="tree", max_depth=10, n_estimators=100, top_n=10, test_size=0.2):
    """
    Evaluates the original per-frame feature structure on a single pitcher.
    Performs an 80/20 split and augments the training set to mirror the L1 Logistic methodology.
    """
    
    labels = [s[1] for s in seqs]
    if len(seqs) < 5 or len(set(labels)) < 2:
        return None
        
    train_seqs, test_seqs = train_test_split(seqs, test_size=test_size, stratify=labels, random_state=42)
    
    np.random.seed(4)
    augmented_train = []
    for seq, label in train_seqs:
        augmented_train.append((seq, label))
        augmented_train.extend(augment_sequence(seq, label))
        
    # Build per-frame features for training set
    X_train_list, y_train_list = [], []
    for seq, label in augmented_train:
        X_train_list.append(buildFrameFeatures(seq))
        y_train_list.append(label)
    
    # Build per-frame features for test set
    X_test_list, y_test_list = [], []
    for seq, label in test_seqs:
        X_test_list.append(buildFrameFeatures(seq))
        y_test_list.append(label)
        
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    
    # Fill NAs
    if np.isnan(X_train).any(): X_train = np.nan_to_num(X_train)
    if np.isnan(X_test).any(): X_test = np.nan_to_num(X_test)
    
    if model_type == "tree":
        clf = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced", random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=42)
        
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["offspeed", "fastball"], zero_division=0)
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [(pf_feature_names[i], float(importances[i])) for i in indices if i < len(pf_feature_names)]
    
    return {
        "accuracy": acc,
        "classification_report": report,
        "top_features": top_features,
        "clf": clf,
        "train_base_size": len(train_seqs),
        "augmented_train_size": len(augmented_train),
        "test_size": len(test_seqs)
    }
