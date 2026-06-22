import os
import shutil
import subprocess

import streamlit as st

import processing.demo_helper as demo
from processing.compare_features import COMPARE_FEATURE_OPTIONS
from processing.kpt_utils import videoPathFromKeypoints
from processing.viz_paths import (
    compareArtifactPaths,
    comparisonIsCached,
    findPlayableVideo,
    loadCompareMetrics,
    saveCompareMetrics,
)
from ui.components import (
    info_panel,
    insight_panel,
    section_label,
    success_panel,
    warn_panel,
)

FEATURES = [label for label, _ in COMPARE_FEATURE_OPTIONS]

LAYOUT_OPTIONS = {
    "Side by Side": "side_by_side",
    "Overlay": "stacked",
}


def _find_ffmpeg():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    for path in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if os.path.isfile(path):
            return path
    return None


def _highlight_key(name):
    if name == "None":
        return ""
    for label, key in COMPARE_FEATURE_OPTIONS:
        if label == name:
            return key
    return name.lower().replace(" ", "_")


def _transcode_for_web(ffmpeg, raw_path, web_path):
    if not ffmpeg or not os.path.isfile(raw_path):
        return False
    subprocess.run(
        [ffmpeg, "-y", "-i", raw_path, "-vcodec", "libx264", web_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return os.path.isfile(web_path)


def _render_comparison(
    fb_kpts, os_kpts, artifacts, layout, highlight, skeleton_opacity, base_dir,
):
    metrics = demo.renderPitchComparison(
        fb_kpts,
        os_kpts,
        artifacts["raw"],
        layout=layout,
        highlight=highlight,
        skeleton_opacity=skeleton_opacity,
        base_dir=base_dir,
        label1="FASTBALL",
        label2="OFFSPEED",
    )
    saveCompareMetrics(artifacts["metrics"], metrics)
    _transcode_for_web(_find_ffmpeg(), artifacts["raw"], artifacts["web"])
    return metrics


def render_compare_tab(base_dir):
    """Render Pitch Compare demo tab for presentation-ready FB vs OS comparison."""

    st.markdown(
        info_panel(
            "Pick a fastball and offspeed pitch, choose side-by-side or overlay layout, "
            "then generate a DTW-synced comparison on real footage. "
            "Cached comparisons load instantly from "
            "<code>Visualizations/&lt;pitcher&gt;/</code>."
        ),
        unsafe_allow_html=True,
    )

    keypoint_root = os.path.join(base_dir, "Keypoints")
    if not os.path.isdir(keypoint_root):
        st.markdown(
            warn_panel(
                f"No keypoint data in <code>{keypoint_root}</code>. "
                "Upload videos in Tab 1 first."
            ),
            unsafe_allow_html=True,
        )
        return

    pitchers = demo.listPitcherFolders(keypoint_root)
    if not pitchers:
        st.markdown(warn_panel("No pitcher folders found in Keypoints/."), unsafe_allow_html=True)
        return

    default_pitcher = "Gburg16_stretch_cropped" if "Gburg16_stretch_cropped" in pitchers else (
        "Gburg16" if "Gburg16" in pitchers else pitchers[0]
    )
    pitcher_idx = pitchers.index(default_pitcher) if default_pitcher in pitchers else 0

    ctrl1, ctrl2, ctrl3 = st.columns(3, gap="large")
    with ctrl1:
        pitcher = st.selectbox("Pitcher", pitchers, index=pitcher_idx, key="compare_pitcher")

    fb_files = demo.listPitchFiles(keypoint_root, pitcher, demo.isFastballFilename)
    os_files = demo.listPitchFiles(keypoint_root, pitcher, demo.isOffspeedFilename)

    if not fb_files:
        st.markdown(warn_panel(f"No fastball keypoints found for <code>{pitcher}</code>."), unsafe_allow_html=True)
        return
    if not os_files:
        st.markdown(warn_panel(f"No offspeed keypoints found for <code>{pitcher}</code>."), unsafe_allow_html=True)
        return

    if "compare_fb" not in st.session_state:
        st.session_state.compare_fb = fb_files[0]
    if "compare_os" not in st.session_state:
        st.session_state.compare_os = os_files[0]

    fb_idx = fb_files.index(st.session_state.compare_fb) if st.session_state.compare_fb in fb_files else 0
    os_idx = os_files.index(st.session_state.compare_os) if st.session_state.compare_os in os_files else 0

    with ctrl2:
        fb_pitch = st.selectbox("Fastball", fb_files, index=fb_idx, key="compare_fb_select")
    with ctrl3:
        os_pitch = st.selectbox("Offspeed", os_files, index=os_idx, key="compare_os_select")

    st.session_state.compare_fb = fb_pitch
    st.session_state.compare_os = os_pitch

    opt1, opt2, opt3 = st.columns(3, gap="large")
    with opt1:
        layout_label = st.radio("Layout", list(LAYOUT_OPTIONS.keys()), horizontal=True)
        layout = LAYOUT_OPTIONS[layout_label]
    with opt2:
        highlight_target = st.selectbox("Feature Highlight", FEATURES, key="compare_highlight")
    with opt3:
        skeleton_opacity = st.slider(
            "Skeleton Opacity", 0.3, 1.0, 0.85, 0.05, key="compare_opacity"
        )

    fb_kpts = os.path.join(keypoint_root, pitcher, fb_pitch)
    os_kpts = os.path.join(keypoint_root, pitcher, os_pitch)
    fb_video = videoPathFromKeypoints(fb_kpts, base_dir)
    os_video = videoPathFromKeypoints(os_kpts, base_dir)

    if not fb_video or not os_video:
        missing = []
        if not fb_video:
            missing.append(fb_pitch)
        if not os_video:
            missing.append(os_pitch)
        st.markdown(
            warn_panel(
                f"Missing source video for: {', '.join(missing)}. "
                "Ensure matching files exist in Pitches/."
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class='compare-badges'>
            <span class='pitch-badge-fastball'>Fastball — {fb_pitch}</span>
            <span class='pitch-badge-offspeed'>Offspeed — {os_pitch}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    highlight = _highlight_key(highlight_target)
    artifacts = compareArtifactPaths(
        base_dir, pitcher, fb_pitch, os_pitch, layout, highlight, skeleton_opacity,
    )
    play_path = findPlayableVideo(artifacts)
    cached = comparisonIsCached(artifacts)
    metrics = loadCompareMetrics(artifacts["metrics"]) if cached else None

    if cached:
        st.markdown(
            info_panel(
                f"Using cached comparison "
                f"<code>Visualizations/{pitcher}/{artifacts['base']}.mp4</code>"
            ),
            unsafe_allow_html=True,
        )

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        btn_label = "Regenerate Comparison" if cached else "Generate Comparison"
        generate = st.button(btn_label, type="primary", width="stretch")

    if generate:
        with st.spinner("Aligning pitches and rendering comparison…"):
            try:
                metrics = _render_comparison(
                    fb_kpts, os_kpts, artifacts, layout, highlight,
                    skeleton_opacity, base_dir,
                )
                play_path = findPlayableVideo(artifacts)
                if not play_path:
                    st.markdown(
                        warn_panel(
                            "Render finished but no playable video was found. "
                            "Install FFmpeg with <code>brew install ffmpeg</code>."
                        ),
                        unsafe_allow_html=True,
                    )
                    return
                st.markdown(
                    success_panel(
                        f"Comparison saved to <code>Visualizations/{pitcher}/</code>"
                    ),
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.markdown(warn_panel(f"Comparison failed: {e}"), unsafe_allow_html=True)
                return

    play_path = play_path or findPlayableVideo(artifacts)
    if play_path and os.path.isfile(play_path):
        if metrics is None:
            metrics = loadCompareMetrics(artifacts["metrics"])

        st.markdown(section_label("Comparison"), unsafe_allow_html=True)
        st.markdown("<div class='compare-panel'>", unsafe_allow_html=True)
        st.video(play_path)
        st.markdown("</div>", unsafe_allow_html=True)

        if metrics:
            peak_t = metrics.get("peak_time_sec")
            if peak_t is None:
                out_frames = metrics.get("output_frames", 0)
                dur = metrics.get("duration_sec", 0)
                if out_frames and dur:
                    peak_t = round(metrics.get("peak_frame_idx", 0) / out_frames * dur, 1)
                else:
                    peak_t = 0
            watch = (
                f"Peak <strong>{metrics.get('highlight_feature', metrics.get('highlight_joint', 'feature'))}</strong> "
                f"difference: <strong>{metrics.get('peak_separation_px', 0)}</strong> "
                f"at <strong>{peak_t}s</strong> into the video."
            )
            st.markdown(insight_panel("What to Watch", watch), unsafe_allow_html=True)
    elif not generate:
        st.markdown(
            info_panel("No cached comparison for this pair yet — click Generate Comparison."),
            unsafe_allow_html=True,
        )
