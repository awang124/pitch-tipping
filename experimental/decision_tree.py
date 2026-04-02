"""
Decision-tree-based classifier that predicts whether a pitch is a fastball
from numpy keypoint-sequence files.
 
Each .npy file stores a list of dicts (one per frame):
    [
        {
            "keypoints": np.ndarray,  # shape (17, 2) — COCO (x, y)
            "glove_box": [x1, y1, x2, y2]
        },
        ...  # fixed-length sequence
    ]
 
Files are labeled by filename:
    fastball_01.npy  -> label = 1 (fastball)
    curveball_03.npy -> label = 0 (other)
 
Feature strategy:
    Rather than collapsing the sequence into aggregate stats (mean/std/etc.),
    each frame is treated as its own slot. The model sees features like
    "glove_cx at frame 90" or "r_wrist_glove_dist at frame 45", allowing the
    decision tree to identify discriminative moments in the pitch timeline.
 
    Per-frame features (15 values each):
        glove_cx, glove_cy, glove_w, glove_h,
        l_wrist_glove_dist, r_wrist_glove_dist,
        l_elbow_angle, r_elbow_angle,
        shoulder_width, hip_width, torso_height,
        r_wrist_x, r_wrist_y,
        l_wrist_x, l_wrist_y
 
    Total features: 15 * N_FRAMES (e.g. 15 * 200 = 3000)
"""
 
import os
import glob
import argparse
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from diagnostics import sanity_check_feature, cross_validate
 
# ---------------------------------------------------------------------------
# COCO 17-keypoint index reference
# ---------------------------------------------------------------------------
# 0  Nose        1  L Eye       2  R Eye
# 3  L Ear       4  R Ear       5  L Shoulder
# 6  R Shoulder  7  L Elbow     8  R Elbow
# 9  L Wrist    10  R Wrist    11  L Hip
# 12 R Hip      13  L Knee     14  R Knee
# 15 L Ankle    16  R Ankle
 
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
 
# Names of the features extracted per frame (order must match extract_frame_features)
FRAME_FEATURE_NAMES = [
    "glove_cx", "glove_cy", "glove_w", "glove_h",
    "l_wrist_glove_dist", "r_wrist_glove_dist",
    "l_elbow_angle", "r_elbow_angle",
    "shoulder_width", "hip_width", "torso_height",
    "r_wrist_x", "r_wrist_y",
    "l_wrist_x", "l_wrist_y",
]
N_FRAME_FEATURES = len(FRAME_FEATURE_NAMES)  # 15
 
 
# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
 
def angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at joint b, formed by the a->b->c chain."""
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))
 
 
def extract_frame_features(frame: dict) -> np.ndarray:
    """
    Extract a 15-element feature vector from a single frame.
    Order matches FRAME_FEATURE_NAMES exactly.
    """
    kpts = np.array(frame["keypoints"], dtype=float)  # (17, 2)
    box  = np.array(frame["glove_box"],  dtype=float)  # (4,)
 
    bx1, by1, bx2, by2 = box
    b_cx = (bx1 + bx2) / 2
    b_cy = (by1 + by2) / 2
    b_w  = bx2 - bx1
    b_h  = by2 - by1
 
    glove_center = np.array([b_cx, b_cy])
    l_wrist = kpts[KP["l_wrist"]]
    r_wrist = kpts[KP["r_wrist"]]
 
    l_wrist_dist = float(np.linalg.norm(l_wrist - glove_center))
    r_wrist_dist = float(np.linalg.norm(r_wrist - glove_center))
 
    l_elbow_ang = angle_at_joint(
        kpts[KP["l_shoulder"]], kpts[KP["l_elbow"]], kpts[KP["l_wrist"]]
    )
    r_elbow_ang = angle_at_joint(
        kpts[KP["r_shoulder"]], kpts[KP["r_elbow"]], kpts[KP["r_wrist"]]
    )
 
    shoulder_w = float(np.linalg.norm(kpts[KP["l_shoulder"]] - kpts[KP["r_shoulder"]]))
    hip_w      = float(np.linalg.norm(kpts[KP["l_hip"]]      - kpts[KP["r_hip"]]))
    torso_h    = float(abs(
        ((kpts[KP["l_shoulder"], 1] + kpts[KP["r_shoulder"], 1]) / 2) -
        ((kpts[KP["l_hip"],      1] + kpts[KP["r_hip"],      1]) / 2)
    ))
 
    return np.array([
        b_cx, b_cy, b_w, b_h,                   # glove box geometry
        l_wrist_dist, r_wrist_dist,              # wrist-to-glove distances
        l_elbow_ang, r_elbow_ang,                # elbow angles
        shoulder_w, hip_w, torso_h,              # body proportions
        float(r_wrist[0]), float(r_wrist[1]),    # throwing-hand wrist position
        float(l_wrist[0]), float(l_wrist[1]),    # glove-hand wrist position
    ])
 
 
def build_feature_vector(frames: list) -> np.ndarray:
    """
    Flatten all per-frame features into one vector, preserving frame order.
 
    Each frame contributes N_FRAME_FEATURES values at fixed positions, so the
    tree can learn rules like "glove_cx at frame 092 < 0.45".
 
    Output shape: (N_FRAMES * N_FRAME_FEATURES,)
    """
    return np.concatenate([extract_frame_features(f) for f in frames])
 
 
def make_feature_names(n_frames: int) -> list[str]:
    """
    Return names like 'f090_glove_cx', 'f091_r_wrist_glove_dist', etc.
    Zero-padded indices keep names in temporal order when sorted alphabetically.
    """
    width = len(str(n_frames - 1))
    return [
        f"f{str(i).zfill(width)}_{feat}"
        for i in range(n_frames)
        for feat in FRAME_FEATURE_NAMES
    ]
 
 
# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
 
def label_from_filename(path: str) -> int:
    """1 if filename contains 'fastball' or 'cutter' (case-insensitive), else 0."""
    return 1 if "fastball" in os.path.basename(path).lower() else 0
 
 
def load_dataset(data_dir: str):
    """
    Recursively load all labeled .npy files from data_dir.
 
    Returns:
        X         — np.ndarray of shape (n_files, n_features)
        y         — np.ndarray of shape (n_files,), values 0 or 1
        filenames — list of basenames, parallel to X and y
        n_frames  — number of frames per sequence (validated to be consistent)
    """
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True))
    files = [f for f in files if "unlabeled" not in os.path.basename(f).lower()]
 
    if not files:
        raise FileNotFoundError(f"No labeled .npy files found in: {data_dir}")
 
    X, y, names = [], [], []
    n_frames = None
 
    for path in files:
        frames = np.load(path, allow_pickle=True).tolist()
 
        if n_frames is None:
            n_frames = len(frames)
        elif len(frames) != n_frames:
            raise ValueError(
                f"Frame count mismatch in {os.path.basename(path)}: "
                f"expected {n_frames}, got {len(frames)}"
            )
 
        X.append(build_feature_vector(frames))
        y.append(label_from_filename(path))
        names.append(os.path.basename(path))
        tag = "fastball" if y[-1] else "other   "
        print(f"  [{tag}]  {os.path.basename(path)}")
 
    return np.array(X), np.array(y), names, n_frames
 
 
# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
 
def _importance_report(importances: np.ndarray, names: list[str], top_n: int) -> str:
    indices = np.argsort(importances)[::-1]
    lines = [
        f"\n{'Rank':<6} {'Feature':<44} {'Importance':>10}",
        "-" * 62,
    ]
    for rank, idx in enumerate(indices[:top_n], 1):
        lines.append(f"{rank:<6} {names[idx]:<44} {importances[idx]:>10.4f}")
    return "\n".join(lines)
 
 
class PitchClassifier:
    """
    DecisionTreeClassifier wrapper.
    Start here — the exported tree directly shows which frame/feature splits matter,
    e.g. '|--- f092_glove_cx <= 0.42' means frame 92's glove x-position is decisive.
    """
 
    def __init__(self, n_frames: int, max_depth: int = 10):
        self.feature_names = make_feature_names(n_frames)
        self.clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
 
    def train(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
 
    def classify(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)
 
    def export_tree(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        text = export_text(self.clf, feature_names=self.feature_names, max_depth=50)
        with open(path, "w") as f:
            f.write(text)
        print(f"  Decision tree written to: {path}")
 
    def feature_importance_report(self, top_n: int) -> str:
        return _importance_report(self.clf.feature_importances_, self.feature_names, top_n)
 
 
class PitchClassifierRF:
    """
    RandomForestClassifier wrapper.
    Higher accuracy, less interpretable. Graduate to this after validating features.
    """
 
    def __init__(self, n_frames: int, n_estimators: int = 100):
        self.feature_names = make_feature_names(n_frames)
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=None, random_state=42
        )
 
    def train(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
 
    def classify(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)
 
    def feature_importance_report(self, top_n: int) -> str:
        return _importance_report(self.clf.feature_importances_, self.feature_names, top_n)
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify pitches as fastball or other using per-frame keypoint sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pitcher", required=True, metavar="NAME",
        help="Pitcher subdirectory under EXTRACTION_DATA_DIR (e.g. shohei_ohtani)",
    )
    parser.add_argument(
        "--model", choices=["tree", "rf"], default="tree",
        help="'tree' = DecisionTree (interpretable), 'rf' = RandomForest (accurate)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=10, metavar="N",
        help="Max depth of the decision tree (tree model only)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, metavar="FRAC",
        help="Fraction of data held out for evaluation",
    )
    parser.add_argument(
        "--top-features", type=int, default=30, metavar="N",
        help="Number of top features to print in the importance report",
    )
    parser.add_argument(
        "--tree-out", default="tree.txt", metavar="PATH",
        help="Output path for the decision tree text file (tree model only)",
    )
    return parser.parse_args()
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main():
    load_dotenv()
    args = parse_args()
 
    data_root = os.getenv("EXTRACTION_DATA_DIR")
    if not data_root:
        raise EnvironmentError("EXTRACTION_DATA_DIR is not set in your .env file")
 
    keypoint_dir = os.path.join(data_root, args.pitcher, "trimmed_keypoints")
 
    # ---- Load ----
    print(f"\nLoading sequences from:\n  {keypoint_dir}\n")
    X, y, filenames, n_frames = load_dataset(keypoint_dir)
    n_fastball = int(y.sum())
    n_other    = int((1 - y).sum())
    print(f"\n{len(X)} sequences — {n_fastball} fastball, {n_other} other")
    print(f"Frames per sequence  : {n_frames}")
    print(f"Features per sequence: {X.shape[1]}  ({n_frames} frames × {N_FRAME_FEATURES} features/frame)")
 
    # ---- Split ----
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, filenames,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")
 
    # ---- Train ----
    print(f"\nTraining {args.model} classifier...")
    if args.model == "tree":
        classifier = PitchClassifier(n_frames=n_frames, max_depth=args.max_depth)
    else:
        classifier = PitchClassifierRF(n_frames=n_frames)
    classifier.train(X_train, y_train)
 
    # ---- Evaluate ----
    outputs = classifier.classify(X_test)
    correct = int((outputs == y_test).sum())
    print(f"\nAccuracy: {correct} / {len(y_test)}  ({correct / len(y_test):.4f})\n")
    print(classification_report(y_test, outputs, target_names=["other", "fastball"]))
 
    # ---- Per-file breakdown ----
    label_str = {0: "other", 1: "fastball"}
    print(f"  {'File':<44} {'True':>8} {'Pred':>8}")
    print("  " + "-" * 62)
    for name, true, pred in zip(names_test, y_test, outputs):
        wrong = "  <- WRONG" if true != pred else ""
        print(f"  {name:<44} {label_str[true]:>8} {label_str[pred]:>8}{wrong}")
 
    # ---- Feature importance ----
    print(f"\nTop {args.top_features} most important features:")
    print(classifier.feature_importance_report(top_n=args.top_features))
 
    sanity_check_feature(X, y, feature_names=classifier.feature_names, feature="f029_r_wrist_y")
    print(cross_validate(classifier, X, y))

    # ---- Export tree ----
    if args.model == "tree":
        print()
        classifier.export_tree(path=args.tree_out)
 
 
if __name__ == "__main__":
    main()

