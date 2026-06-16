#!/usr/bin/env python3
"""
Smoke tests for keypoint pipeline (Tabs 1-3, 5).

Usage:
    conda activate PitchTip
    python scripts/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    import processing.tab1_helper  # noqa: F401
    import processing.tab2_helper  # noqa: F401
    import processing.kpt_utils  # noqa: F401
    import processing.tab3_helper  # noqa: F401
    import processing.tab5_helper  # noqa: F401
    print("PASS imports")


def test_audit():
    from scripts.audit_keypoints import auditFile

    path = os.path.join(BASE, "Keypoints/Gburg16/fastball_01.npy")
    if not os.path.isfile(path):
        print("SKIP audit (no sample npy)")
        return
    row = auditFile(path)
    assert row["frames"] > 0
    print(f"PASS audit status={row['status']} jitter={row['hip_jitter_px']}")


def test_tab3_load():
    from processing.tab3_helper import loadFrameDataset

    root = os.path.join(BASE, "Keypoints/Gburg16")
    if not os.path.isdir(root):
        print("SKIP tab3 (no Gburg16)")
        return
    seqs, n_frames, names = loadFrameDataset(root)
    assert len(seqs) > 0
    assert n_frames > 0
    print(f"PASS tab3 loadFrameDataset n={len(seqs)} frames={n_frames}")


def test_tab2_viz():
    import processing.tab2_helper as t2

    k1 = os.path.join(BASE, "Keypoints/Gburg16/fastball_01.npy")
    k2 = os.path.join(BASE, "Keypoints/Gburg16/off_01.npy")
    if not (os.path.isfile(k1) and os.path.isfile(k2)):
        print("SKIP tab2 viz")
        return
    out = os.path.join(BASE, "Visualizations/smoke_test_pair.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    result = t2.visualizeDifferences(k1, k2, out, base_dir=BASE)
    assert os.path.isfile(out)
    assert "crop_fallback" in result
    print(f"PASS tab2 visualizeDifferences crop_fallback={result['crop_fallback']}")


def test_tab5_overlay():
    import processing.tab5_helper as t5
    from processing.kpt_utils import videoPathFromKeypoints

    kpt = os.path.join(BASE, "Keypoints/Gburg16/fastball_01.npy")
    vid = videoPathFromKeypoints(kpt, BASE)
    if not (os.path.isfile(kpt) and vid and os.path.isfile(vid)):
        print("SKIP tab5 overlay")
        return
    out = os.path.join(BASE, "Visualizations/smoke_test_overlay.mp4")
    ok, msg = t5.overlaySkeletonOnVideo(kpt, vid, out)
    assert ok, msg
    print(f"PASS tab5 overlay {msg}")


def test_tab1_preview():
    import processing.tab1_helper as t1
    from processing.kpt_utils import videoPathFromKeypoints

    kpt = os.path.join(BASE, "Keypoints/Gburg16/fastball_01.npy")
    vid = videoPathFromKeypoints(kpt, BASE)
    if not (os.path.isfile(kpt) and vid and os.path.isfile(vid)):
        print("SKIP tab1 preview")
        return
    img = t1.renderPreviewFrame(vid, kpt, 30)
    assert img is not None
    print(f"PASS tab1 preview shape={img.shape}")


def main():
    os.chdir(BASE)
    test_imports()
    test_audit()
    test_tab3_load()
    test_tab1_preview()
    test_tab5_overlay()
    test_tab2_viz()
    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
