"""Paths and cache helpers for Visualizations/ (mirrors Pitches/ layout)."""
import json
import os
import re


def sanitizeStem(filename):
    """Filesystem-safe stem from a pitch filename."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    stem = stem.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def pitcherFromKeypointRel(rel_path):
    """Top-level folder name from a Keypoints-relative path."""
    return rel_path.replace("\\", "/").split("/")[0]


def vizDirForPitcher(base_dir, pitcher):
    """Visualizations/<pitcher>/ — parallel to Pitches/<pitcher>/."""
    path = os.path.join(base_dir, "Visualizations", pitcher)
    os.makedirs(path, exist_ok=True)
    return path


def compareArtifactBasename(fb_file, os_file, layout, highlight, opacity):
    """Unique name encoding pair, layout, highlight, and opacity."""
    fb = sanitizeStem(fb_file)
    os_f = sanitizeStem(os_file)
    hl = sanitizeStem(highlight) if highlight else "none"
    op = int(round(float(opacity) * 100))
    return f"{fb}__vs__{os_f}__{layout}__{hl}__op{op:02d}__peakknee"


def compareArtifactPaths(base_dir, pitcher, fb_file, os_file, layout, highlight, opacity):
    """Return dict of raw/web/metrics paths for one Pitch Compare render."""
    out_dir = vizDirForPitcher(base_dir, pitcher)
    base = compareArtifactBasename(fb_file, os_file, layout, highlight, opacity)
    return {
        "dir": out_dir,
        "base": base,
        "raw": os.path.join(out_dir, f"{base}.mp4"),
        "web": os.path.join(out_dir, f"web_{base}.mp4"),
        "metrics": os.path.join(out_dir, f"{base}_metrics.json"),
    }


def findPlayableVideo(artifact_paths):
    """Prefer browser-friendly web transcode, else raw mp4."""
    web = artifact_paths.get("web")
    raw = artifact_paths.get("raw")
    if web and os.path.isfile(web):
        return web
    if raw and os.path.isfile(raw):
        return raw
    return None


def comparisonIsCached(artifact_paths):
    return findPlayableVideo(artifact_paths) is not None


def loadCompareMetrics(metrics_path):
    if not metrics_path or not os.path.isfile(metrics_path):
        return None
    with open(metrics_path, encoding="utf-8") as fh:
        return json.load(fh)


def saveCompareMetrics(metrics_path, metrics):
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
