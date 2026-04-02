"""
diagnostics.py
==============
Plug-in diagnostic functions for pitch_classifier.py.

Usage — add to the bottom of main() in pitch_classifier.py:

    from diagnostics import sanity_check_feature, cross_validate

    sanity_check_feature(X, y, feature_names=classifier.feature_names, feature="f029_r_wrist_y")
    cross_validate(classifier, X, y)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut


# ---------------------------------------------------------------------------
# 1. Sanity-check a single feature
# ---------------------------------------------------------------------------

def sanity_check_feature(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature: str = "f029_r_wrist_y",
):
    """
    Plot the distribution of one feature for fastball vs. other, and print
    basic stats so you can judge whether the split the tree found is real.

    Args:
        X             : full feature matrix (n_samples, n_features)
        y             : labels (0 = other, 1 = fastball)
        feature_names : list returned by make_feature_names() / classifier.feature_names
        feature       : exact feature name string to inspect (e.g. 'f029_r_wrist_y')
    """
    if feature not in feature_names:
        close = [n for n in feature_names if feature.split("_", 1)[-1] in n]
        raise ValueError(
            f"Feature '{feature}' not found.\n"
            f"Did you mean one of: {close[:8]}"
        )

    idx        = feature_names.index(feature)
    values     = X[:, idx]
    fastball   = values[y == 1]
    other      = values[y == 0]

    # ---- Console stats ----
    print(f"\n{'='*55}")
    print(f"  Sanity check: {feature}")
    print(f"{'='*55}")
    print(f"  {'':20s} {'fastball':>12} {'other':>12}")
    print(f"  {'-'*44}")
    print(f"  {'n':20s} {len(fastball):>12} {len(other):>12}")
    print(f"  {'mean':20s} {fastball.mean():>12.3f} {other.mean():>12.3f}")
    print(f"  {'std':20s} {fastball.std():>12.3f} {other.std():>12.3f}")
    print(f"  {'min':20s} {fastball.min():>12.3f} {other.min():>12.3f}")
    print(f"  {'max':20s} {fastball.max():>12.3f} {other.max():>12.3f}")

    overlap = (
        max(fastball.min(), other.min()),
        min(fastball.max(), other.max()),
    )
    if overlap[0] < overlap[1]:
        print(f"\n  Overlap region : [{overlap[0]:.3f}, {overlap[1]:.3f}]  ← ambiguous zone")
    else:
        print(f"\n  No overlap — groups are fully separated on this feature.")

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"Feature distribution: {feature}", fontsize=13, fontweight="bold")

    # Strip chart (all individual points)
    ax = axes[0]
    ax.scatter(fastball, np.ones(len(fastball)),  alpha=0.6, label="fastball", color="royalblue", s=60)
    ax.scatter(other,    np.zeros(len(other)),     alpha=0.6, label="other",    color="tomato",    s=60)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["other", "fastball"])
    ax.set_xlabel("Feature value")
    ax.set_title("Individual values (strip chart)")
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Histogram overlay
    ax = axes[1]
    bins = min(15, max(5, len(values) // 3))
    ax.hist(fastball, bins=bins, alpha=0.55, label="fastball", color="royalblue", density=True)
    ax.hist(other,    bins=bins, alpha=0.55, label="other",    color="tomato",    density=True)
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Density")
    ax.set_title("Histogram overlay")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("sanity_check.png", dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to: sanity_check.png", flush=True)
    plt.close()   # avoid blocking — open sanity_check.png directly instead


# ---------------------------------------------------------------------------
# 2. Leave-one-out cross-validation
# ---------------------------------------------------------------------------

def cross_validate(classifier, X: np.ndarray, y: np.ndarray):
    """
    Run Leave-One-Out cross-validation and print a per-sample result table.

    LOO is the right choice here because with only ~29 samples, every data
    point matters — you don't want any of them locked away in a held-out fold.

    Args:
        classifier : PitchClassifier or PitchClassifierRF instance (already constructed)
        X          : full feature matrix
        y          : labels
    """
    loo       = LeaveOneOut()
    total     = len(y)
    label_str = {0: "other", 1: "fastball"}

    print(f"\n{'='*55}")
    print(f"  Leave-One-Out Cross-Validation  ({total} folds)")
    print(f"{'='*55}")
    print(f"  {'Fold':>5}  {'True':>10}  {'Pred':>10}  {'Result'}")
    print(f"  {'-'*45}")

    # Single LOO loop — no redundant cross_val_score call
    preds = np.empty(total, dtype=int)
    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        classifier.clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = classifier.clf.predict(X[test_idx])
        # Flush each fold immediately so output appears in real time
        true, pred = y[test_idx[0]], preds[test_idx[0]]
        result = "✓" if true == pred else "✗  WRONG"
        print(f"  {fold+1:>5}  {label_str[true]:>10}  {label_str[pred]:>10}  {result}", flush=True)

    correct  = int((preds == y).sum())
    accuracy = correct / total

    print(f"\n  Accuracy : {correct} / {total}  ({accuracy:.3f})")
    for label, name in [(1, "fastball"), (0, "other")]:
        mask      = y == label
        class_acc = (preds[mask] == y[mask]).mean()
        print(f"  {name:>10} accuracy : {class_acc:.3f}  ({mask.sum()} samples)")

    if accuracy < 0.75:
        print("\n  ⚠  Accuracy below 0.75 — model may not be learning a reliable signal.")
        print("     Consider collecting more data before trusting any single feature split.")
    elif accuracy < 0.90:
        print("\n  ~ Moderate accuracy. More data or additional features may help.")
    else:
        print("\n  ✓ Strong LOO accuracy — the split looks real, not just noise.")