#!/usr/bin/env python3
"""
Audit keypoint extraction quality across Keypoints/ folders.

Usage:
    conda activate PitchTip
    python scripts/audit_keypoints.py [--root Keypoints] [--out audit_report.csv]
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.kpt_utils import (
    auditStatus,
    computeQualityMetrics,
    findWindupEnd,
    loadKeypointMeta,
    loadKeypointSequence,
    stripPaddingFrames,
)


def auditFile(path):
    kpts, gloves, meta = loadKeypointSequence(path)
    kpts, gloves = stripPaddingFrames(kpts, gloves)
    if not meta:
        meta = loadKeypointMeta(path)
    fw = meta.get("frame_width")
    fh = meta.get("frame_height")
    metrics = computeQualityMetrics(kpts, gloves, fw, fh)
    windup = findWindupEnd(kpts) if len(kpts) else 0
    status = auditStatus(metrics)
    x_min, x_max = float(kpts[:, :, 0].min()), float(kpts[:, :, 0].max())
    y_min, y_max = float(kpts[:, :, 1].min()), float(kpts[:, :, 1].max())
    x_span = round(x_max - x_min, 1) if len(kpts) else 0
    dominant_x = meta.get("dominant_hip_x_frac")
    if dominant_x is None and fw and len(kpts):
        anchors = []
        for frame in kpts:
            if frame[11, 0] != 0 and frame[12, 0] != 0:
                anchors.append((frame[11, 0] + frame[12, 0]) / 2)
        if anchors:
            dominant_x = round(float(np.median(anchors)) / fw, 3)
    return {
        "file": path,
        "status": status,
        "frames": metrics["num_frames"],
        "valid_frame_pct": metrics["valid_frame_pct"],
        "avg_joint_pct": metrics["avg_joint_pct"],
        "glove_pct": metrics["glove_pct"],
        "hip_jitter_px": metrics["hip_jitter_px"],
        "windup_end": windup,
        "sync_frame": metrics["sync_frame"],
        "delivery_hint": metrics["delivery_hint"],
        "x_min": round(x_min, 1),
        "x_max": round(x_max, 1),
        "x_span": x_span,
        "dominant_hip_x_frac": dominant_x,
        "y_min": round(y_min, 1),
        "y_max": round(y_max, 1),
        "retried_tracking": meta.get("retried_tracking", False),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit keypoint .npy files")
    parser.add_argument("--root", default="Keypoints", help="Keypoints root directory")
    parser.add_argument("--out", default="audit_report.csv", help="Output CSV path")
    args = parser.parse_args()

    rows = []
    folder_stats = {}
    for dirpath, _, files in os.walk(args.root):
        for f in sorted(files):
            if not f.endswith(".npy"):
                continue
            path = os.path.join(dirpath, f)
            row = auditFile(path)
            rows.append(row)
            folder = os.path.relpath(dirpath, args.root)
            folder_stats.setdefault(folder, []).append(row)

    if not rows:
        print(f"No .npy files found under {args.root}")
        return 1

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}\n")
    print(f"{'Folder':<25} {'n':>4} {'PASS':>5} {'WARN':>5} {'FAIL':>5} {'jitter':>8}")
    for folder, frows in sorted(folder_stats.items()):
        counts = {s: sum(1 for r in frows if r["status"] == s) for s in ("PASS", "WARN", "FAIL")}
        jitter = round(np.mean([r["hip_jitter_px"] for r in frows]), 1)
        print(
            f"{folder:<25} {len(frows):>4} {counts['PASS']:>5} "
            f"{counts['WARN']:>5} {counts['FAIL']:>5} {jitter:>8}"
        )

    stretch_rows = [r for r in rows if "stretch" in r["file"].lower()]
    if stretch_rows:
        print(f"\nStretch folders summary ({len(stretch_rows)} files):")
        print(f"  avg jitter: {np.mean([r['hip_jitter_px'] for r in stretch_rows]):.1f}")
        print(f"  avg x_span: {np.mean([r['x_span'] for r in stretch_rows]):.1f}")
        wrong_side = [
            r for r in stretch_rows
            if r.get("dominant_hip_x_frac") is not None and r["dominant_hip_x_frac"] < 0.45
        ]
        print(f"  likely umpire (hip_x < 0.45): {len(wrong_side)}")
        print(
            f"  PASS/WARN/FAIL: "
            f"{sum(1 for r in stretch_rows if r['status']=='PASS')}/"
            f"{sum(1 for r in stretch_rows if r['status']=='WARN')}/"
            f"{sum(1 for r in stretch_rows if r['status']=='FAIL')}"
        )

    fails = [r for r in rows if r["status"] == "FAIL"]
    if fails:
        print(f"\nTop FAIL files ({min(10, len(fails))}):")
        for r in sorted(fails, key=lambda x: -x["hip_jitter_px"])[:10]:
            print(f"  {r['file']} jitter={r['hip_jitter_px']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
