import os
import copy
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

def _time_to_peak(series):
    """Normalized frame index of first maximum in a 1D series."""
    return np.argmax(series) / len(series) if len(series) else np.nan

def _time_to_min(series):
    """Normalized frame index of first minimum in a 1D series."""
    return np.argmin(series) / len(series) if len(series) else np.nan

def _velocity(series, dt):
    """Numeric first derivative."""
    return np.gradient(series, dt)

def _angular_velocity(angle_series, dt):
    """Angular velocity (degrees per second)."""
    return np.gradient(np.unwrap(np.deg2rad(angle_series)) * 180 / np.pi, dt)

def _path_length(trajectory):
    """Total path length along a trajectory."""
    return np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))

def _angle_between_points(p1, p2, p3):
    """Angle at p2 formed by p1-p2-p3 (degrees)."""
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def _forearm_angle(shoulder, elbow, wrist, reference="vertical"):
    """
    Compute angle between forearm and a reference direction.

    Parameters
    ----------
    shoulder, elbow, wrist: (x, y) coordinates.
    reference: 'vertical' or 'horizontal'.

    Returns
    -------
    Angle in degrees.
    """
    ref_vec = np.array([0, -1]) if reference == "vertical" else np.array([1, 0])
    dot = np.dot(forearm := np.array(wrist) - np.array(elbow), ref_vec)
    norm_prod = np.linalg.norm(forearm) * np.linalg.norm(ref_vec) + 1e-8
    return np.degrees(np.arccos(np.clip(dot / norm_prod, -1.0, 1.0)))

def _compute_angles(frame):
    """
    Compute all joint angles and biomechanical angles for a single frame.

    Parameters
    ----------
    frame: Array (17, 2) of keypoints.

    Returns
    -------
    Dictionary mapping angle names to values.
    """
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
    """
    Extract aggregate (pitch‑level) biomechanical features from a
    keypoint sequence, including joint angle statistics, timing landmarks,
    angular velocities, wrist / glove speeds, and normalized positions.

    Parameters
    ----------
    seq: List of per‑frame dicts with 'keypoints' and 'glove_box'.
    label: Pitch class label (string).
    fps: Sampling rate (default 60).

    Returns
    -------
    Dictionary of features (includes 'class' and 'duration').
    """
    if len(seq) < 2:
        return {"class": label}
    if len(seq) and isinstance(seq[0], dict):
        glove_seq = np.array(
            [frame.get("glove_box", [-1] * 4) for frame in seq], dtype=np.float32
        )

        ### HANDLE UNEXPECTED KEYPOINTS
        kpts_list = []
        for frame in seq:
            kp = np.array(frame["keypoints"], dtype=np.float32)
            if kp.shape != (17, 2):
                if kp.shape == (17, 3): kp = kp[:, :2]
                else: kp = np.zeros((17, 2), dtype=np.float32)
            kpts_list.append(kp)
        seq = np.array(kpts_list, dtype=np.float32)
    else:
        glove_seq = np.full((len(seq), 4), -1.0, dtype=np.float32)

    ### GET CLASS, FRAMERATE, DURATION
    if seq.shape[0] < 2:
        return {"class": label}
    T = seq.shape[0]
    dt = 1.0 / fps
    duration = T * dt

    ### COMPUTE PER-FRAME JOINT ANGLES
    angles_list = [_compute_angles(seq[t]) for t in range(T)]
    angles = {
        name: np.array([a[name] for a in angles_list], dtype=np.float32)
        for name in angles_list[0].keys()
    }

    ### GET RAW JOINT POSITIONS
    pos = {
        "left_knee": seq[:, LEFT_KNEE, :], "right_knee": seq[:, RIGHT_KNEE, :],
        "left_ankle": seq[:, LEFT_ANKLE, :], "right_ankle": seq[:, RIGHT_ANKLE, :],
        "left_shoulder": seq[:, LEFT_SHOULDER, :], "right_shoulder": seq[:, RIGHT_SHOULDER, :],
        "left_hip": seq[:, LEFT_HIP, :], "right_hip": seq[:, RIGHT_HIP, :],
        "left_wrist": seq[:, LEFT_WRIST, :], "right_wrist": seq[:, RIGHT_WRIST, :],
        "left_elbow": seq[:, LEFT_ELBOW, :], "right_elbow": seq[:, RIGHT_ELBOW, :],
    }

    ### GET TORSO LENGTH (NORMALIZER)
    shoulder_mid = (seq[0, LEFT_SHOULDER] + seq[0, RIGHT_SHOULDER]) / 2
    hip_mid = (seq[0, LEFT_HIP] + seq[0, RIGHT_HIP]) / 2
    torso_length = max(sc.spatial.distance.euclidean(shoulder_mid, hip_mid), 1e-8)

    features = {"class": label, "duration": duration}

    ### JOINT ANGLE STATISTICS
    for name, series in angles.items():
        series_clean = series[~np.isnan(series)]
        if len(series_clean) == 0: continue
        features[f"{name}_mean"] = np.mean(series_clean)
        features[f"{name}_std"] = np.std(series_clean)
        features[f"{name}_range"] = np.nanmax(series) - np.nanmin(series)

    ### TIMING LANDMARKS
    left_knee_y = pos["left_knee"][:, 1]
    features["time_to_peak_left_knee"] = _time_to_peak(left_knee_y)
    if "right_elbow" in angles:
        features["time_to_min_right_elbow"] = _time_to_min(angles["right_elbow"])
        features["time_to_max_right_elbow"] = _time_to_peak(angles["right_elbow"])
    if "hip_shoulder_sep" in angles:
        features["time_to_max_hip_shoulder_sep"] = _time_to_peak(angles["hip_shoulder_sep"])
    if "right_forearm_vertical" in angles:
        features["time_to_min_forearm_angle"] = _time_to_min(angles["right_forearm_vertical"])

    ### ANGULAR VELOCITIES
    for joint in [
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder", "left_knee", "right_knee"
    ]:
        if joint in angles:
            ang_vel = _angular_velocity(angles[joint], dt)
            features[f"{joint}_ang_vel_max"] = np.nanmax(np.abs(ang_vel))
            features[f"{joint}_ang_vel_mean"] = np.nanmean(np.abs(ang_vel))

    ### NORMALIZED WRIST / KNEE SPEEDS
    rw_x, rw_y = pos["right_wrist"][:, 0], pos["right_wrist"][:, 1]
    speed_rw = np.sqrt(_velocity(rw_x, dt) ** 2 + _velocity(rw_y, dt) ** 2) / torso_length
    features["right_wrist_speed_max"] = np.max(speed_rw)
    lk_x, lk_y = pos["left_knee"][:, 0], pos["left_knee"][:, 1]
    speed_lk = np.sqrt(_velocity(lk_x, dt) ** 2 + _velocity(lk_y, dt) ** 2) / torso_length
    features["left_knee_speed_max"] = np.max(speed_lk)
    features["left_knee_max_height"] = np.max(left_knee_y) / torso_length

    ### GLOVE SPEED
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
        g_speed = np.sqrt(g_vel_x ** 2 + g_vel_y ** 2) / torso_length
        features["glove_speed_max"] = np.max(g_speed)
        features["glove_speed_mean"] = np.mean(g_speed)
    return features

def extractFrameFeatures(frame):
    """
    Extract feature vector for a single frame, including glove
    size, wrist‑glove distances, elbow angles, widths, offsets.

    Parameters
    ----------
    frame: Dict with 'keypoints' and 'glove_box'.

    Returns
    -------
    1D numpy array of features.
    """
    kpts = np.array(frame["keypoints"], dtype=float)
    box = np.array(frame.get("glove_box", [-1]*4), dtype=float)

    ### GLOVE POSITION / SIZE
    bx1, by1, bx2, by2 = box
    b_cx = (bx1 + bx2) / 2
    b_cy = (by1 + by2) / 2
    b_w = bx2 - bx1
    b_h = by2 - by1
    glove_center = np.array([b_cx, b_cy])

    l_wrist = kpts[KP["l_wrist"]]
    r_wrist = kpts[KP["r_wrist"]]
    
    ### GET TORSO HEIGHT (NORMALIZER)
    torso_h = float(abs(
        ((kpts[KP["l_shoulder"], 1] + kpts[KP["r_shoulder"], 1]) / 2) -
        ((kpts[KP["l_hip"], 1] + kpts[KP["r_hip"], 1]) / 2)
    ))
    torso_h = max(torso_h, 1e-5)
    shoulder_mid = (kpts[KP["l_shoulder"]] + kpts[KP["r_shoulder"]]) / 2

    ### WRIST-GLOVE DISTANCES
    l_wrist_dist = float(np.linalg.norm(l_wrist - glove_center)) / torso_h
    r_wrist_dist = float(np.linalg.norm(r_wrist - glove_center)) / torso_h

    ### ELBOW ANGLES
    l_elbow_ang = _angle_between_points(
        kpts[KP["l_shoulder"]], kpts[KP["l_elbow"]], kpts[KP["l_wrist"]]
    )
    r_elbow_ang = _angle_between_points(
        kpts[KP["r_shoulder"]], kpts[KP["r_elbow"]], kpts[KP["r_wrist"]]
    )
    
    ### SHOULDER / HIP WIDTHS
    shoulder_w = float(np.linalg.norm(
        kpts[KP["l_shoulder"]] - kpts[KP["r_shoulder"]]
    )) / torso_h
    hip_w = float(np.linalg.norm(kpts[KP["l_hip"]] - kpts[KP["r_hip"]])) / torso_h

    ### OFFSETS (RELATIVE TO SHOULDER MIDPOINT)
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
    """Concatenate per-frame features for an entire sequence."""
    return np.concatenate([extractFrameFeatures(f) for f in sequence_frames])

def makeFrameFeatureNames(n_frames):
    """
    Generate numerically indexed names for flattened per-frame features.

    Parameters
    ----------
    n_frames: Number of frames in a sequence.

    Returns
    -------
    List of strings (e.g., 'f000_glove_w_norm', 'f001_...').
    """
    width = len(str(n_frames - 1))
    return [
        f"f{str(i).zfill(width)}_{feat}"
        for i in range(n_frames)
        for feat in FRAME_FEATURE_NAMES
    ]

def loadFrameDataset(keypoints_root):
    """
    Load keypoint files suitable for per‑frame analysis: filters out
    unlabeled files, keeps only sequences with the majority frame count.

    Parameters
    ----------
    keypoints_root: Directory containing .npy files.

    Returns
    -------
    Tuple (list of (frames, label) tuples, n_frames, feature_names).
    """
    npy_files = [
        f for f in glob.glob(os.path.join(keypoints_root, "**", "*.npy"), recursive=True)
        if not os.path.basename(f).endswith("_full.npy")
    ]
    npy_files.sort()
    npy_files = [
        f for f in npy_files 
        if "unlabel" not in os.path.basename(f).lower()
    ]

    ### COUNT FRAMES
    frame_counts = {}
    for path in npy_files:
        frames = np.load(path, allow_pickle=True).tolist()
        n = len(frames)
        frame_counts[n] = frame_counts.get(n, 0) + 1
        
    if not frame_counts:
        return [], 0, []
        
    ### LOAD KEYPOINT SEQUENCES (normalize length to modal frame count)
    n_frames = max(frame_counts, key=frame_counts.get)
    raw_seqs = []
    skipped = []
    for path in npy_files:
        frames = np.load(path, allow_pickle=True).tolist()
        n = len(frames)
        if n > n_frames:
            frames = frames[-n_frames:]
        elif n < n_frames:
            pad = n_frames - n
            zero_f = {
                k: (np.zeros_like(v) if hasattr(v, "__len__") else 0)
                for k, v in frames[0].items()
            }
            frames = [zero_f] * pad + frames
        lower = os.path.basename(path).lower()
        label = int("fastball" in lower or "cutter" in lower)
        raw_seqs.append((frames, label))
    feature_names = makeFrameFeatureNames(n_frames)
    return raw_seqs, n_frames, feature_names

def loadSequences(root):
    """
    Load all .npy keypoint files from a directory tree and assign
    labels based on filename keywords: unlabeled files are skipped.

    Parameters
    ----------
    root: Root directory to search.

    Returns
    -------
    List of (sequence, label) tuples.
    """
    seqs = []
    file_paths = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            file_paths.append(os.path.join(dirpath, f))
    file_paths.sort()
    
    for path in file_paths:
        f = os.path.basename(path)
        if not (name := f.lower()).endswith(".npy"):
            continue
        if name.endswith("_full.npy"):
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
            continue
        seqs.append((np.load(path, allow_pickle=True), label))
    return seqs

def mannWhitney(df, alpha=0.05, n_show=10):
    """
    Perform Mann‑Whitney U-Tests all features between pitch classes.

    Parameters
    ----------
    df: DataFrame with a 'class' column.
    alpha: Significance level (unused in this snippet, kept for compatibility).
    n_show: Number of top results to return.

    Returns
    -------
    DataFrame sorted by p‑value and effect size.
    """
    classes = df["class"].unique()
    pairs = list(combinations(classes, 2))
    res = []
    
    ### TEST EACH FEATURE
    for feature in df.columns:
        if feature == "class":
            continue
        for c1, c2 in pairs:
    
            ### ISOLATE CLASSES 1 / 2
            data1 = df[df["class"] == c1][feature].dropna()
            data2 = df[df["class"] == c2][feature].dropna()
            if len(data1.unique()) < 2 or len(data2.unique()) < 2:
                continue
    
            ### COMPUTE MANN-WHITNEY
            U, p = sc.stats.mannwhitneyu(data1, data2, alternative="two-sided")
            n1, n2 = len(data1), len(data2)
            effect_size = U / (n1 * n2)
            res.append({
                "Feature": feature,
                "U Statistic": U,
                "p-value": p,
                "Effect Size": effect_size
            })
    
    ### SORT DATAFRAME BY P-VALUE / EFFECT SIZE
    resdf = pd.DataFrame(res)
    resdf["Significant"] = resdf["p-value"] < alpha /\
        sc.special.comb(len(classes), 2, exact=True)
    return resdf.sort_values(
        ["p-value", "Effect Size"],
        key=lambda x: x if x.name == "p-value" else (x - 0.5).abs(),
        ascending=[True, False]
    ).head(n_show)

def addNoise(frames, sigma=0.005):
    """
    Augment a sequence by adding Gaussian noise.

    Parameters
    ----------
    frames: List of frame dicts.
    sigma: Standard deviation of noise.

    Returns
    -------
    List of augmented frame dicts.
    """
    aug_frames = []
    for frame in frames:
        new_frame = copy.deepcopy(frame)
        noise = np.random.normal(0, sigma, new_frame["keypoints"].shape).astype(np.float32)
        new_frame["keypoints"] += noise
        aug_frames.append(new_frame)
    return aug_frames

def timeWarp(frames, factor):
    """
    Augment a sequence by temporally resampling using linear
    interpolation, padding / cropping to preserve length.

    Parameters
    ----------
    frames: List of frame dicts.
    factor: Resampling factor (<1 for faster, >1 for slower).

    Returns
    -------
    List of augmented frame dicts (same length as input).
    """
    orig_len = len(frames)
    new_len = int(orig_len * factor)
    orig_t = np.linspace(0, 1, orig_len)
    new_t = np.linspace(0, 1, new_len)
    
    ### ISOLATE POSE / GLOVE
    seq_kpt = np.array([f["keypoints"] for f in frames])
    has_glove = "glove_box" in frames[0]
    if has_glove:
        seq_glove = np.array([f["glove_box"] for f in frames])
        
    ### INTERPOLATE POSE / GLOVE
    f_kpt = sc.interpolate.interp1d(orig_t, seq_kpt, axis=0, kind='linear')
    resampled_kpt = f_kpt(new_t)
    if has_glove:
        f_glove = sc.interpolate.interp1d(orig_t, seq_glove, axis=0, kind='linear')
        resampled_glove = f_glove(new_t)
        
    ### COMPOSE NEW KEYPOINT SEQUENCE
    resampled_frames = []
    for i in range(new_len):
        frame = {"keypoints": resampled_kpt[i].astype(np.float32)}
        if has_glove:
            frame["glove_box"] = resampled_glove[i].tolist()
        resampled_frames.append(frame)
        
    ### PAD IF NECESSARY
    if new_len < orig_len:
        pad_len = orig_len - new_len
        pad_frames = [copy.deepcopy(resampled_frames[0]) for _ in range(pad_len)]
        final_frames = pad_frames + resampled_frames

    ### CROP IF NECESSARY
    elif new_len > orig_len:
        crop_len = new_len - orig_len
        final_frames = resampled_frames[crop_len:]
    else:
        final_frames = resampled_frames
        
    return final_frames

def augmentSequence(frames, label):
    """
    Generate synthetic variants of a single pitch sequence.
    Includes noise additions, time warping, and both.

    Parameters
    ----------
    frames: List of frame dicts.
    label: Class label (int or string).

    Returns
    -------
    List of (augmented_frames, label) tuples.
    """
    variants = []
    
    variants.append((addNoise(frames, sigma=0.005), label))
    variants.append((addNoise(frames, sigma=0.01), label))
    
    variants.append((timeWarp(frames, factor=0.9), label))
    variants.append((timeWarp(frames, factor=0.95), label))
    variants.append((timeWarp(frames, factor=1.05), label))
    variants.append((timeWarp(frames, factor=1.1), label))
    
    variants.append((addNoise(timeWarp(frames, factor=0.95), sigma=0.005), label))
    variants.append((addNoise(timeWarp(frames, factor=1.05), sigma=0.005), label))
    return variants

def trainFrameClassifier(X, y, model_type="tree", max_depth=10, n_estimators=100):
    """
    Train a decision tree on per‑frame feature data.

    Parameters
    ----------
    X: Feature matrix.
    y: Labels.
    model_type: 'tree' or 'rf' (we only use 'tree').
    max_depth: Tree depth limit.
    n_estimators: Number of trees for random forest.

    Returns
    -------
    Trained classifier.
    """
    if model_type == "tree":
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    return clf.fit(X, y)

def evaluate_cross_session(
    train_seqs, test_seqs, feature_names, model_type="tree",
    max_depth=10, n_estimators=100, top_n=10
):
    """
    Train a per‑frame decision tree on one pitcher and evaluate on another.

    Parameters
    ----------
    train_seqs: training sequences.
    test_seqs: test sequences.
    feature_names: flattened per-frame feature names.
    model_type: 'tree' or 'rf' (we only use 'tree').
    max_depth: maximum tree depth.
    n_estimators: number of random forest trees.
    top_n: Number of top features to return.

    Returns
    -------
    Dictionary with accuracy, classification_report, top_features, clf, sizes.
    """
    ### AUGMENT TRAINING SET
    augmented_train = []
    for seq, label in train_seqs:
        augmented_train.append((seq, label))
        augmented_train.extend(augmentSequence(seq, label))
        
    ### CONSTRUCT MATRICES
    X_train = np.array([buildFrameFeatures(s) for s, l in augmented_train])
    y_train = np.array([l for s, l in augmented_train])
    X_test = np.array([buildFrameFeatures(s) for s, l in test_seqs])
    y_test = np.array([l for s, l in test_seqs])
    
    ### TRAIN
    clf = trainFrameClassifier(
        X_train, y_train, model_type=model_type,
        max_depth=max_depth, n_estimators=n_estimators
    )
    
    ### TEST
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["other", "fastball"], zero_division=0
    )
    
    ### GET DISCRIMINATIVE FEATURES
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [(feature_names[i], importances[i]) for i in indices]
    
    return {
        "accuracy": acc,
        "classification_report": report,
        "top_features": top_features,
        "clf": clf,
        "augmented_train_size": len(augmented_train),
        "test_size": len(test_seqs)
    }

def evaluate_cross_session_lr(train_seqs, test_seqs, feature_names, C=1.0, top_n=10):
    """
    Train a per‑frame logistic regression on one pitcher and evaluate on another.

    Parameters
    ----------
    train_seqs: training sequences.
    test_seqs: test sequences.
    C: inverse regularization strength.
    top_n: Number of top features to return.

    Returns
    -------
    Dictionary with accuracy, classification_report, top_features, clf, sizes.
    """
    ### AUGMENT TRAINING SET
    augmented_train = []
    for seq, label in train_seqs:
        augmented_train.append((seq, label))
        augmented_train.extend(augmentSequence(seq, label))
    
    ### CREATE MATRICES
    X_train = np.array([buildFrameFeatures(s) for s, l in augmented_train])
    X_test = np.array([buildFrameFeatures(s) for s, l in test_seqs])
    y_train = np.array([l for s, l in augmented_train])
    y_test = np.array([l for s, l in test_seqs])
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ### TRAIN
    clf = LogisticRegression(
        l1_ratio=1, solver='liblinear', C=C,
        class_weight="balanced", random_state=42, max_iter=10000
    )
    clf.fit(X_train_scaled, y_train)
    
    ### TEST
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["other", "fastball"], zero_division=0
    )

    ### GET DISCRIMINATIVE FEATURES
    coefs = np.abs(clf.coef_[0])
    indices = np.argsort(coefs)[::-1][:top_n]
    top_features = [(feature_names[i], float(clf.coef_[0][i])) for i in indices]
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
    Train an aggregate logistic regression on one pitcher and evaluate on another.

    Parameters
    ----------
    train_seqs: training sequences.
    test_seqs: test sequences.
    C: inverse regularization strength.
    top_n: Number of top features to return.

    Returns
    -------
    Dictionary with accuracy, classification_report, top_features, clf, sizes.
    """
    ### AUGMENT TRAINING SET
    augmented_train = []
    for seq, label in train_seqs:
        augmented_train.append((seq, label))
        augmented_train.extend(augmentSequence(seq, label))
    
    ### GET TRAINING FEATURES
    relabel_map = {0: "offspeed", 1: "fastball"}
    train_features = []
    for seq, label in augmented_train:
        feats = extractFeatures(seq, relabel_map[label])
        if len(feats) > 1:
            train_features.append(feats)
    
    ### GET TEST FEATURES
    test_features = []
    for seq, label in test_seqs:
        feats = extractFeatures(seq, relabel_map[label])
        if len(feats) > 1:
            test_features.append(feats)
    
    ### CREATE MATRICES
    train_df = pd.DataFrame(train_features)
    test_df = pd.DataFrame(test_features)
    y_train = (train_df["class"] == "fastball").astype(int).values
    y_test = (test_df["class"] == "fastball").astype(int).values
    
    feature_cols = [
        c for c in train_df.columns if c != "class"
        and train_df[c].dtype in [np.float64, np.float32, np.int64, float]
    ]
    common_cols = [c for c in feature_cols if c in test_df.columns]
    X_train = train_df[common_cols].fillna(0).values
    X_test = test_df[common_cols].fillna(0).values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ### TRAIN
    clf = LogisticRegression(
        l1_ratio=1, solver='liblinear', C=C,
        class_weight="balanced", random_state=42, max_iter=10000
    )
    clf.fit(X_train_scaled, y_train)
    
    ### TEST
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["offspeed", "fastball"], zero_division=0
    )

    ### GET DISCRIMINATIVE FEATURES
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
    Train an aggregate logistic regression on a single pitcher.

    Parameters
    ----------
    seqs: training and test sequences.
    test_size: proportion of sequences to form test set.
    C: inverse regularization strength.
    top_n: Number of top features to return.

    Returns
    -------
    Dictionary with accuracy, classification_report, top_features, clf, sizes.
    """
    labels = [s[1] for s in seqs]
    if len(seqs) < 5 or len(set(labels)) < 2:
        return None
        
    train_seqs, test_seqs = train_test_split(
        seqs, test_size=test_size, stratify=labels, random_state=1
    )
    np.random.seed(1)

    ### AUGMENT TRAINING SET
    augmented_train = []
    for seq, label in train_seqs:
        augmented_train.append((seq, label))
        augmented_train.extend(augmentSequence(seq, label))
    
    ### GET TRAINING FEATURES
    relabel_map = {0: "offspeed", 1: "fastball"}
    train_features = []
    for seq, label in augmented_train:
        feats = extractFeatures(seq, relabel_map.get(label, "offspeed"))
        if len(feats) > 1:
            train_features.append(feats)
    
    ### GET TEST FEATURES
    test_features = []
    for seq, label in test_seqs:
        feats = extractFeatures(seq, relabel_map.get(label, "offspeed"))
        if len(feats) > 1:
            test_features.append(feats)
            
    if len(train_features) == 0 or len(test_features) == 0:
        return None
    
    ### CREATE MATRICES
    train_df = pd.DataFrame(train_features)
    test_df = pd.DataFrame(test_features)
    y_train = (train_df["class"] == "fastball").astype(int).values
    y_test = (test_df["class"] == "fastball").astype(int).values
    
    feature_cols = [
        c for c in train_df.columns if c != "class"
        and train_df[c].dtype in [np.float64, np.float32, np.int64, float]
    ]
    common_cols = [c for c in feature_cols if c in test_df.columns]
    X_train = train_df[common_cols].fillna(0).values
    X_test = test_df[common_cols].fillna(0).values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ### TRAIN
    clf = LogisticRegression(
        l1_ratio=1, solver='liblinear', C=C,
        random_state=1, max_iter=10000
    )
    clf.fit(X_train_scaled, y_train)
    
    ### TEST
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["offspeed", "fastball"], zero_division=0
    )

    ### GET DISCRIMINATIVE FEATURES
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

def evaluate_single_session(
    seqs, pf_feature_names, model_type="tree",
    max_depth=10, n_estimators=100, top_n=10, test_size=0.2
):
    """
    Train an per-frame decision tree on a single pitcher.

    Parameters
    ----------
    seqs: training and test sequences.
    feature_names: flattened per-frame feature names.
    model_type: 'tree' or 'rf' (we only use 'tree').
    max_depth: Tree depth limit.
    n_estimators: Number of trees for random forest.
    top_n: Number of top features to return.
    test_size: proportion of sequences to form test set.

    Returns
    -------
    Dictionary with accuracy, classification_report, top_features, clf, sizes.
    """
    labels = [s[1] for s in seqs]
    if len(seqs) < 5 or len(set(labels)) < 2:
        return None
        
    train_seqs, test_seqs = train_test_split(
        seqs, test_size=test_size, stratify=labels, random_state=42
    )
    np.random.seed(4)

    ### AUGMENT TRAINING SET
    augmented_train = []
    for seq, label in train_seqs:
        augmented_train.append((seq, label))
        augmented_train.extend(augmentSequence(seq, label))
        
    ### GET TRAINING FEATURES
    X_train_list, y_train_list = [], []
    for seq, label in augmented_train:
        X_train_list.append(buildFrameFeatures(seq))
        y_train_list.append(label)
    
    ### GET TEST FEATURES
    X_test_list, y_test_list = [], []
    for seq, label in test_seqs:
        X_test_list.append(buildFrameFeatures(seq))
        y_test_list.append(label)
        
    ### CREATE MATRICES
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    
    if np.isnan(X_train).any(): X_train = np.nan_to_num(X_train)
    if np.isnan(X_test).any(): X_test = np.nan_to_num(X_test)
    
    ### TRAIN
    if model_type == "tree":
        clf = DecisionTreeClassifier(
            max_depth=max_depth, class_weight="balanced", random_state=42
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced", random_state=42
        )
        
    clf.fit(X_train, y_train)

    ### TEST
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=["offspeed", "fastball"], zero_division=0
    )

    ### GET DISCRIMINATIVE FEATURES
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [
        (pf_feature_names[i], float(importances[i]))
        for i in indices if i < len(pf_feature_names)
    ]
    
    return {
        "accuracy": acc,
        "classification_report": report,
        "top_features": top_features,
        "clf": clf,
        "train_base_size": len(train_seqs),
        "augmented_train_size": len(augmented_train),
        "test_size": len(test_seqs)
    }

