"""
pitch_classifier.py
====================
Decision-tree-based classifier that predicts whether a pitch is a fastball
from numpy files containing keypoint sequences and glove-box data.

Each .npy file stores a list of dicts:
    [
        {
            "keypoints": np.ndarray,  # shape (17, 2) — COCO 17-point (x, y)
            "glove_box": [x1, y1, x2, y2]
        },
        ...   # ~200 frames
    ]

Files are labeled by filename:
    fastball_01.npy  -> label = 1
    curveball_03.npy -> label = 0
"""

import os
import glob
import argparse
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# COCO 17-keypoint index reference
# ---------------------------------------------------------------------------
# 0  Nose          1  Left Eye       2  Right Eye
# 3  Left Ear      4  Right Ear      5  Left Shoulder
# 6  Right Shoulder 7 Left Elbow    8  Right Elbow
# 9  Left Wrist   10  Right Wrist  11  Left Hip
# 12 Right Hip    13  Left Knee    14  Right Knee
# 15 Left Ankle   16  Right Ankle

KP = {
    "nose": 0, "l_eye": 1, "r_eye": 2,
    "l_ear": 3, "r_ear": 4,
    "l_shoulder": 5, "r_shoulder": 6,
    "l_elbow": 7, "r_elbow": 8,
    "l_wrist": 9, "r_wrist": 10,
    "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14,
    "l_ankle": 15, "r_ankle": 16,
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_frame_features(frame: dict) -> np.ndarray:
    """
    Extract a feature vector from a single frame dict.

    Features (per frame):
        - All 17 keypoints flattened: (x, y) -> 34 values
        - Glove box: x1, y1, x2, y2, center_x, center_y, width, height -> 8 values
        - Wrist-to-glove distance (L2) for both wrists -> 2 values
        - Elbow angle (left and right) -> 2 values
        - Shoulder width -> 1 value
        - Hip width -> 1 value
        - Torso height (avg shoulder y to avg hip y) -> 1 value
        Total: 49 values per frame
    """
    kpts = np.array(frame["keypoints"], dtype=float)  # (17, 2)
    box  = np.array(frame["glove_box"], dtype=float)  # (4,)

    # ---- Glove box derived values ----
    bx1, by1, bx2, by2 = box
    b_cx  = (bx1 + bx2) / 2
    b_cy  = (by1 + by2) / 2
    b_w   = bx2 - bx1
    b_h   = by2 - by1
    box_feats = np.array([bx1, by1, bx2, by2, b_cx, b_cy, b_w, b_h])

    # ---- Wrist-to-glove-center distances ----
    l_wrist = kpts[KP["l_wrist"]]
    r_wrist = kpts[KP["r_wrist"]]
    glove_center = np.array([b_cx, b_cy])
    l_wrist_glove_dist = float(np.linalg.norm(l_wrist - glove_center))
    r_wrist_glove_dist = float(np.linalg.norm(r_wrist - glove_center))

    # ---- Elbow angles ----
    def angle_at_joint(a, b, c):
        """Angle at point b formed by a-b-c (degrees)."""
        ba = a - b
        bc = c - b
        cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos_val, -1, 1))))

    l_elbow_angle = angle_at_joint(
        kpts[KP["l_shoulder"]], kpts[KP["l_elbow"]], kpts[KP["l_wrist"]]
    )
    r_elbow_angle = angle_at_joint(
        kpts[KP["r_shoulder"]], kpts[KP["r_elbow"]], kpts[KP["r_wrist"]]
    )

    # ---- Body proportions ----
    shoulder_width = float(np.linalg.norm(kpts[KP["l_shoulder"]] - kpts[KP["r_shoulder"]]))
    hip_width      = float(np.linalg.norm(kpts[KP["l_hip"]]      - kpts[KP["r_hip"]]))
    avg_shoulder_y = float((kpts[KP["l_shoulder"], 1] + kpts[KP["r_shoulder"], 1]) / 2)
    avg_hip_y      = float((kpts[KP["l_hip"], 1]      + kpts[KP["r_hip"], 1])      / 2)
    torso_height   = float(abs(avg_hip_y - avg_shoulder_y))

    body_feats = np.array([shoulder_width, hip_width, torso_height])

    frame_vec = np.concatenate([
        kpts.flatten(),                                      # 34
        box_feats,                                           # 8
        [l_wrist_glove_dist, r_wrist_glove_dist],           # 2
        [l_elbow_angle, r_elbow_angle],                     # 2
        body_feats,                                          # 3
    ])
    return frame_vec  # length 49


def aggregate_sequence(frames: list) -> np.ndarray:
    """
    Aggregate per-frame features across the full sequence into a single
    fixed-length vector using mean, std, min, and max.

    Output length: 49 features * 4 stats = 196 values
    """
    frame_matrix = np.array([extract_frame_features(f) for f in frames])  # (T, 49)

    mean_vec = frame_matrix.mean(axis=0)
    std_vec  = frame_matrix.std(axis=0)
    min_vec  = frame_matrix.min(axis=0)
    max_vec  = frame_matrix.max(axis=0)

    return np.concatenate([mean_vec, std_vec, min_vec, max_vec])  # (196,)


def make_feature_names() -> list[str]:
    """Return human-readable feature names matching aggregate_sequence output."""
    base_names = []

    for i in range(17):
        name = [k for k, v in KP.items() if v == i][0]
        base_names += [f"{name}_x", f"{name}_y"]

    base_names += [
        "glove_x1", "glove_y1", "glove_x2", "glove_y2",
        "glove_cx", "glove_cy", "glove_w", "glove_h",
        "l_wrist_glove_dist", "r_wrist_glove_dist",
        "l_elbow_angle", "r_elbow_angle",
        "shoulder_width", "hip_width", "torso_height",
    ]

    full_names = []
    for stat in ["mean", "std", "min", "max"]:
        for name in base_names:
            full_names.append(f"{stat}_{name}")

    return full_names


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def is_fastball(filename: str) -> int:
    """Return 1 if filename contains 'fastball', else 0."""
    return 1 if "fastball" in os.path.basename(filename).lower() else 0


def load_dataset(data_dir: str):
    """
    Load all .npy files in data_dir and return (X, y, filenames).

    X: list of feature vectors (one per file)
    y: list of labels (1 = fastball, 0 = not fastball)
    """
    pattern = os.path.join(data_dir, "**", "*.npy")
    files = glob.glob(pattern, recursive=True)

    if not files:
        raise FileNotFoundError(f"No .npy files found in '{data_dir}'")

    X, y, names = [], [], []
    for f in sorted(files):
        frames = np.load(f, allow_pickle=True).tolist()
        X.append(aggregate_sequence(frames))
        y.append(is_fastball(f))
        names.append(os.path.basename(f))
        print(f"  Loaded: {os.path.basename(f):40s}  label={'fastball' if y[-1] else 'other'}")

    return np.array(X), np.array(y), names


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

class PitchClassifier:
    """
    Decision-tree classifier with optional export of the tree structure.
    Use this first — it's interpretable and shows exactly which features matter.
    """

    def __init__(self, max_depth: int = 10):
        self.clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self.feature_names = make_feature_names()

    def train(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)

    def classify(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def export_tree(self, path: str = "tree.txt"):
        tree_text = export_text(self.clf, feature_names=self.feature_names, max_depth=50)
        with open(path, "w") as f:
            f.write(tree_text)
        print(f"Decision tree saved to: {path}")

    def feature_importance_report(self, top_n: int = 20) -> str:
        importances = self.clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        lines = [f"\n{'Rank':<6} {'Feature':<40} {'Importance':>10}",
                 "-" * 58]
        for rank, idx in enumerate(indices[:top_n], 1):
            lines.append(f"{rank:<6} {self.feature_names[idx]:<40} {importances[idx]:>10.4f}")
        return "\n".join(lines)


class PitchClassifierRF:
    """
    Random-forest classifier — higher accuracy, less interpretable.
    Switch to this once you're happy with your features.
    """

    def __init__(self, n_estimators: int = 100):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            random_state=42
        )
        self.feature_names = make_feature_names()

    def train(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)

    def classify(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def feature_importance_report(self, top_n: int = 20) -> str:
        importances = self.clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        lines = [f"\n{'Rank':<6} {'Feature':<40} {'Importance':>10}",
                 "-" * 58]
        for rank, idx in enumerate(indices[:top_n], 1):
            lines.append(f"{rank:<6} {self.feature_names[idx]:<40} {importances[idx]:>10.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parseargs():
    parser = argparse.ArgumentParser(description="Fastball classifier from keypoint .npy files")
    parser.add_argument("--data",       required=True,  help="Directory containing .npy files")
    parser.add_argument("--model",      default="tree", choices=["tree", "rf"],
                        help="Model type: 'tree' (DecisionTree) or 'rf' (RandomForest)")
    parser.add_argument("--max_depth",  type=int, default=10,
                        help="Max depth for DecisionTree (ignored for RF)")
    parser.add_argument("--test_size",  type=float, default=0.2,
                        help="Fraction of data to use for testing (default: 0.2)")
    parser.add_argument("--tree_out",   default="tree.txt",
                        help="Path to write the decision tree text (tree model only)")
    parser.add_argument("--top_n",      type=int, default=20,
                        help="Number of top features to show in importance report")
    return parser.parse_args()


def main():
    args = parseargs()

    # ---- Load data ----
    print(f"\nLoading data from: {args.data}")
    X, y, filenames = load_dataset(args.data)
    print(f"\nLoaded {len(X)} sequences — {y.sum()} fastball, {(1-y).sum()} other")

    # ---- Train / test split ----
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, filenames, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ---- Build & train ----
    if args.model == "tree":
        classifier = PitchClassifier(max_depth=args.max_depth)
    else:
        classifier = PitchClassifierRF()

    print(f"\nTraining {args.model} classifier...")
    classifier.train(X_train, y_train)

    # ---- Evaluate ----
    outputs = classifier.classify(X_test)
    correct = int((outputs == y_test).sum())
    print(f"\nAccuracy: {correct} / {len(y_test)}  ({correct / len(y_test):.4f})")
    print("\nClassification Report:")
    print(classification_report(y_test, outputs, target_names=["other", "fastball"]))

    # ---- Error analysis ----
    print("Per-file predictions (test set):")
    print(f"  {'File':<40} {'True':>8} {'Pred':>8}")
    print("  " + "-" * 58)
    label_str = {0: "other", 1: "fastball"}
    for name, true, pred in zip(names_test, y_test, outputs):
        marker = "  <-- WRONG" if true != pred else ""
        print(f"  {name:<40} {label_str[true]:>8} {label_str[pred]:>8}{marker}")

    # ---- Feature importance ----
    print(f"\nTop {args.top_n} most important features:")
    print(classifier.feature_importance_report(top_n=args.top_n))

    # ---- Export tree ----
    if args.model == "tree":
        classifier.export_tree(path=args.tree_out)


if __name__ == "__main__":
    main()