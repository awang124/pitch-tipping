#!/usr/bin/env python3
"""
Batch-extract keypoints for all videos under Pitches/.

Usage:
    conda activate PitchTip
    python scripts/batch_extract.py [--base-dir .] [--folder Randy_stretch] [--force]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.mediapipe_pose import resolve_model_path
from processing.tab1_helper import processVideo


class _Progress:
    def progress(self, _):
        pass


class _Status:
    def text(self, _):
        pass


def findVideos(pitch_root, folder=None):
    videos = []
    root = os.path.join(pitch_root, folder) if folder else pitch_root
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".mov", ".mp4")):
                videos.append(os.path.join(dirpath, f))
    return sorted(videos)


def kptPathForVideo(video_path, pitch_root, keypoint_root):
    rel = os.path.relpath(video_path, pitch_root)
    stem = os.path.splitext(rel)[0]
    return os.path.join(keypoint_root, stem + ".npy")


def main():
    parser = argparse.ArgumentParser(description="Batch keypoint extraction")
    parser.add_argument("--base-dir", default=".", help="Project root (Pitches/ and Keypoints/)")
    parser.add_argument("--folder", default=None, help="Process only this pitcher subfolder")
    parser.add_argument("--force", action="store_true", help="Re-extract even if .npy exists")
    parser.add_argument(
        "--num-poses", type=int, default=2, choices=(1, 2),
        help="MediaPipe poses per frame (use 1 for cropped/single-person clips)",
    )
    args = parser.parse_args()

    base = os.path.abspath(args.base_dir)
    pitch_root = os.path.join(base, "Pitches")
    keypoint_root = os.path.join(base, "Keypoints")

    if not os.path.isdir(pitch_root):
        print(f"Pitches directory not found: {pitch_root}")
        return 1

    model_path = resolve_model_path(base)
    print(f"MediaPipe model: {model_path}")
    print(f"num_poses: {args.num_poses}")
    progress, status = _Progress(), _Status()

    videos = findVideos(pitch_root, args.folder)
    if not videos:
        print("No videos found.")
        return 1

    done, skipped, failed = 0, 0, 0
    for video_path in videos:
        kpt_path = kptPathForVideo(video_path, pitch_root, keypoint_root)
        if os.path.isfile(kpt_path) and not args.force:
            skipped += 1
            continue
        os.makedirs(os.path.dirname(kpt_path), exist_ok=True)
        print(f"Processing {video_path} -> {kpt_path}")
        t0 = time.perf_counter()
        try:
            ok, meta = processVideo(
                video_path, kpt_path, model_path, progress, status,
                num_poses=args.num_poses,
            )
            elapsed = time.perf_counter() - t0
            if ok:
                done += 1
                print(
                    f"  OK frames={meta['num_extracted_frames']} "
                    f"jitter={meta.get('pitcher_track_jitter')} "
                    f"delivery={meta.get('delivery_hint')} "
                    f"hip_x={meta.get('dominant_hip_x_frac')} "
                    f"time={elapsed:.1f}s"
                )
            else:
                failed += 1
                print(f"  FAILED ({elapsed:.1f}s)")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")

    print(f"\nDone: {done} extracted, {skipped} skipped, {failed} failed")
    if done > 0:
        print("Run: python scripts/audit_keypoints.py")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
