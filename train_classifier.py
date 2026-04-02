import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy import stats, signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Keypoint indices
LEFT_SHOULDER = 5 ; RIGHT_SHOULDER = 6
LEFT_ELBOW = 7 ; RIGHT_ELBOW = 8
LEFT_WRIST = 9 ; RIGHT_WRIST = 10
LEFT_HIP = 11 ; RIGHT_HIP = 12
LEFT_KNEE = 13 ; RIGHT_KNEE = 14
LEFT_ANKLE = 15 ; RIGHT_ANKLE = 16

def _time_to_peak(series):
    if len(series) == 0: return np.nan
    return np.argmax(series) / len(series)

def _time_to_min(series):
    if len(series) == 0: return np.nan
    return np.argmin(series) / len(series)

def _velocity(series, dt):
    return np.gradient(series, dt)

def _angular_velocity(angle_series, dt):
    unwrapped = np.unwrap(np.deg2rad(angle_series)) * 180 / np.pi
    return np.gradient(unwrapped, dt)

def _path_length(trajectory):
    return np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))

def _angle_between_points(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def _forearm_angle(shoulder, elbow, wrist, reference='vertical'):
    forearm = np.array(wrist) - np.array(elbow)
    ref_vec = np.array([0, -1]) if reference == 'vertical' else np.array([1, 0])
    dot = np.dot(forearm, ref_vec)
    norm_prod = np.linalg.norm(forearm) * np.linalg.norm(ref_vec) + 1e-8
    return np.degrees(np.arccos(np.clip(dot / norm_prod, -1.0, 1.0)))

def _compute_angles(frame):
    angles = {}
    angles['left_shoulder'] = _angle_between_points(frame[LEFT_HIP], frame[LEFT_SHOULDER], frame[LEFT_ELBOW])
    angles['left_elbow'] = _angle_between_points(frame[LEFT_SHOULDER], frame[LEFT_ELBOW], frame[LEFT_WRIST])
    angles['right_shoulder'] = _angle_between_points(frame[RIGHT_HIP], frame[RIGHT_SHOULDER], frame[RIGHT_ELBOW])
    angles['right_elbow'] = _angle_between_points(frame[RIGHT_SHOULDER], frame[RIGHT_ELBOW], frame[RIGHT_WRIST])
    angles['left_hip'] = _angle_between_points(frame[LEFT_SHOULDER], frame[LEFT_HIP], frame[LEFT_KNEE])
    angles['left_knee'] = _angle_between_points(frame[LEFT_HIP], frame[LEFT_KNEE], frame[LEFT_ANKLE])
    angles['right_hip'] = _angle_between_points(frame[RIGHT_SHOULDER], frame[RIGHT_HIP], frame[RIGHT_KNEE])
    angles['right_knee'] = _angle_between_points(frame[RIGHT_HIP], frame[RIGHT_KNEE], frame[RIGHT_ANKLE])

    shoulder_mid = (frame[LEFT_SHOULDER] + frame[RIGHT_SHOULDER]) / 2
    hip_mid = (frame[LEFT_HIP] + frame[RIGHT_HIP]) / 2
    trunk_vec = shoulder_mid - hip_mid
    angles['trunk_inclination'] = np.degrees(np.arccos(np.dot(trunk_vec, np.array([0, -1])) / (np.linalg.norm(trunk_vec) + 1e-8)))

    hip_line = frame[RIGHT_HIP] - frame[LEFT_HIP]
    angles['pelvis_rotation'] = np.degrees(np.arctan2(hip_line[1], hip_line[0]))

    shoulder_line = frame[RIGHT_SHOULDER] - frame[LEFT_SHOULDER]
    angles['shoulder_rotation'] = np.degrees(np.arctan2(shoulder_line[1], shoulder_line[0]))
    angles['hip_shoulder_sep'] = angles['shoulder_rotation'] - angles['pelvis_rotation']
    angles['right_forearm_vertical'] = _forearm_angle(frame[RIGHT_SHOULDER], frame[RIGHT_ELBOW], frame[RIGHT_WRIST], 'vertical')
    return angles

def extractFeaturesDeepseek(seq, class_label, fps=60):
    if len(seq) > 0 and isinstance(seq[0], dict):
        glove_seq = np.array([frame.get("glove_box", [-1, -1, -1, -1]) for frame in seq], dtype=np.float32)
        seq = np.array([frame["keypoints"] for frame in seq], dtype=np.float32)
    else:
        glove_seq = np.full((len(seq), 4), -1.0, dtype=np.float32)
        
    T = seq.shape[0]
    dt = 1.0 / fps
    duration = T * dt
    angles_list = [_compute_angles(seq[t]) for t in range(T)]
    angle_names = angles_list[0].keys()
    angles = {name: np.array([a[name] for a in angles_list], dtype=float) for name in angle_names}

    pos = {
        'left_knee': seq[:, LEFT_KNEE, :], 'right_knee': seq[:, RIGHT_KNEE, :],
        'left_ankle': seq[:, LEFT_ANKLE, :], 'right_ankle': seq[:, RIGHT_ANKLE, :],
        'left_shoulder': seq[:, LEFT_SHOULDER, :], 'right_shoulder': seq[:, RIGHT_SHOULDER, :],
        'left_hip': seq[:, LEFT_HIP, :], 'right_hip': seq[:, RIGHT_HIP, :],
        'left_wrist': seq[:, LEFT_WRIST, :], 'right_wrist': seq[:, RIGHT_WRIST, :],
        'left_elbow': seq[:, LEFT_ELBOW, :], 'right_elbow': seq[:, RIGHT_ELBOW, :],
    }

    shoulder_mid = (seq[0, LEFT_SHOULDER] + seq[0, RIGHT_SHOULDER]) / 2
    hip_mid = (seq[0, LEFT_HIP] + seq[0, RIGHT_HIP]) / 2
    torso_length = euclidean(shoulder_mid, hip_mid)

    features = {'class': class_label, 'duration': duration}

    for name, series in angles.items():
        series_clean = series[~np.isnan(series)]
        if len(series_clean) == 0: continue
        features[f'{name}_mean'] = np.mean(series_clean)
        features[f'{name}_std'] = np.std(series_clean)
        features[f'{name}_range'] = np.nanmax(series) - np.nanmin(series)

    left_knee_y = pos['left_knee'][:, 1]
    features['time_to_peak_left_knee'] = _time_to_peak(left_knee_y)

    if 'right_elbow' in angles:
        features['time_to_min_right_elbow'] = _time_to_min(angles['right_elbow'])
        features['time_to_max_right_elbow'] = _time_to_peak(angles['right_elbow'])

    if 'hip_shoulder_sep' in angles:
        features['time_to_max_hip_shoulder_sep'] = _time_to_peak(angles['hip_shoulder_sep'])

    if 'right_forearm_vertical' in angles:
        features['time_to_min_forearm_angle'] = _time_to_min(angles['right_forearm_vertical'])

    for joint in ['right_elbow', 'left_elbow', 'right_shoulder', 'left_shoulder', 'right_knee', 'left_knee']:
        if joint in angles:
            ang_vel = _angular_velocity(angles[joint], dt)
            features[f'{joint}_ang_vel_max'] = np.nanmax(np.abs(ang_vel))
            features[f'{joint}_ang_vel_mean'] = np.nanmean(np.abs(ang_vel))

    rw_x, rw_y = pos['right_wrist'][:, 0], pos['right_wrist'][:, 1]
    speed_rw = np.sqrt(_velocity(rw_x, dt)**2 + _velocity(rw_y, dt)**2)
    features['right_wrist_speed_max'] = np.max(speed_rw)

    lk_x, lk_y = pos['left_knee'][:, 0], pos['left_knee'][:, 1]
    speed_lk = np.sqrt(_velocity(lk_x, dt)**2 + _velocity(lk_y, dt)**2)
    features['left_knee_speed_max'] = np.max(speed_lk)
    features['left_knee_max_height'] = np.max(left_knee_y)
    
    valid_glove = glove_seq[:, 0] != -1
    if np.sum(valid_glove) > 2:
        glove_seq_interp = np.copy(glove_seq)
        for i in range(4):
            arr = glove_seq_interp[:, i]
            arr[~valid_glove] = np.interp(np.arange(len(arr))[~valid_glove], np.arange(len(arr))[valid_glove], arr[valid_glove])
        glove_centers = np.stack([(glove_seq_interp[:, 0] + glove_seq_interp[:, 2]) / 2,
                                  (glove_seq_interp[:, 1] + glove_seq_interp[:, 3]) / 2], axis=1)

        g_vel_x = _velocity(glove_centers[:, 0], dt)
        g_vel_y = _velocity(glove_centers[:, 1], dt)
        g_speed = np.sqrt(g_vel_x**2 + g_vel_y**2)
        features['glove_speed_max'] = np.max(g_speed)
        features['glove_speed_mean'] = np.mean(g_speed)

    return features

def loadSequences(root):
    seqs = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if not f.lower().endswith(".npy"): continue
            if "unlabeled" in f.lower(): continue
            name = f.lower()
            label = "other"
            if "curveball" in name: label = "curveball"
            elif "cutter" in name: label = "cutter"
            elif "four_seam_fastball" in name: label = "four_seam_fastball"
            elif "slider" in name: label = "slider"
            elif "splitter" in name: label = "splitter"
            elif "two_seam_fastball" in name: label = "two_seam_fastball"
            elif "changeup" in name: label = "changeup"
            elif "fastball" in name: label = "fastball"
            elif "offspeed" in name: label = "offspeed"
            seqs.append((np.load(os.path.join(dirpath, f), allow_pickle=True), label))
    return seqs

def train_models(keypoints_dir):
    seqs = loadSequences(keypoints_dir)
    if not seqs:
        return {"error": "No .npy pitch sequences found in Keypoints directory."}
        
    features = pd.DataFrame([extractFeaturesDeepseek(seq, label) for seq, label in seqs])
    
    # Binary Relabeling
    safe_fastballs = ["fastball", "cutter", "four_seam", "two_seam"]
    safe_offspeeds = ["offspeed", "changeup", "curveball", "slider", "splitter"]
    
    def label_bin(val):
        s = str(val).lower()
        if any(fb in s for fb in safe_fastballs): return 0
        if any(osp in s for osp in safe_offspeeds): return 1
        return -1
        
    features['binary_class'] = features['class'].apply(label_bin)
    df_ml = features[features['binary_class'] != -1]
    
    if len(df_ml) < 4:
        return {"error": "Not enough labeled Fastball/Offspeed pitches to train ML models (need at least 4)."}
        
    X_raw = df_ml.drop(columns=['class', 'binary_class'])
    X_clean = X_raw.dropna(axis=1, thresh=int(len(X_raw)*0.8)).fillna(X_raw.median())
    y = df_ml['binary_class']
    feature_names = list(X_clean.columns)

    # Statistical Testing (Mann-Whitney U)
    from scipy.special import comb
    from itertools import combinations
    df_ml_stats = df_ml.copy()
    df_ml_stats['Pitch Type'] = df_ml_stats['binary_class'].map({0: 'Fastball', 1: 'Offspeed'})
    
    def pairwiseTests(df, feature_cols, class_col="Pitch Type", alpha=0.05):
        classes = df[class_col].unique()
        pairs = list(combinations(classes, 2))
        res = []
        for feature in feature_cols:
            for c1, c2 in pairs:
                data1 = df[df[class_col] == c1][feature].dropna()
                data2 = df[df[class_col] == c2][feature].dropna()
                if len(data1) < 2 or len(data2) < 2: continue
                # Skip constant features to avoid ValueError in Mann-Whitney
                if len(data1.unique()) == 1 and len(data2.unique()) == 1 and data1.iloc[0] == data2.iloc[0]: continue
                try:
                    U, p = stats.mannwhitneyu(data1, data2, alternative="two-sided")
                    n1, n2 = len(data1), len(data2)
                    effect_size = U / (n1 * n2)
                    res.append({
                        "Feature": feature,
                        "p-value": float(p),
                        "Effect Size": float(effect_size)
                    })
                except Exception:
                    pass
        df_res = pd.DataFrame(res)
        if not df_res.empty:
            df_res["Significant"] = df_res["p-value"] < alpha / comb(len(classes), 2, exact=True)
            df_res = df_res.sort_values(
                ["Effect Size", "p-value"],
                key=lambda x: (x - 0.5).abs() if x.name == "Effect Size" else x,
                ascending=[False, True]
            )
        return df_res

    stats_df = pairwiseTests(df_ml_stats, feature_names)
    stats_records = stats_df.head(15).to_dict('records') if not stats_df.empty else []

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42, class_weight="balanced")
    rf.fit(X_clean, y)
    importances = rf.feature_importances_
    top_rf_indices = np.argsort(importances)[::-1][:5]
    top_rf = [{"feature": feature_names[i], "importance": float(importances[i])} for i in top_rf_indices]

    # Logistic Regression L1
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    lr = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, random_state=42, class_weight="balanced")
    lr.fit(X_scaled, y)
    coefs = lr.coef_[0]
    nonzero_indices = np.where(coefs != 0)[0]
    sorted_nonzero = sorted(nonzero_indices, key=lambda i: abs(coefs[i]), reverse=True)[:5]
    top_lr = [{"feature": feature_names[i], "coefficient": float(coefs[i])} for i in sorted_nonzero]
    
    stats = {
        "fastballs": int(sum(y==0)),
        "offspeeds": int(sum(y==1)),
        "total_features": int(X_clean.shape[1]),
        "deleted_features": len(feature_names) - len(nonzero_indices)
    }

    return {"top_rf": top_rf, "top_lr": top_lr, "stats": stats, "pairwise_stats": stats_records}

if __name__ == "__main__":
    import pprint
    pprint.pprint(train_models("research/video-data-extraction/"))
