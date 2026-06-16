import os
import shutil
import subprocess

import streamlit as st

import processing.demo_helper as demo
import processing.tab2_helper as t2
from processing.kpt_utils import videoPathFromKeypoints
from ui.components import (
    info_panel,
    insight_panel,
    metric_card,
    section_label,
    success_panel,
    warn_panel,
)

FEATURES = [
    "None",
    "Left Knee",
    "Right Elbow",
    "Left Shoulder",
    "Right Shoulder",
    "Left Hip",
    "Right Hip",
]

LAYOUT_OPTIONS = {
    "Side by Side": "side_by_side",
    "Stacked": "stacked",
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
    return name.lower().replace(" ", "_")


def _cache_key(pitcher, fb, os_pitch, layout, highlight, opacity):
    return f"{pitcher}|{fb}|{os_pitch}|{layout}|{highlight}|{opacity:.2f}"


def _output_paths(base_dir, pitcher, fb, os_pitch, layout):
    out_dir = os.path.join(base_dir, "Visualizations")
    os.makedirs(out_dir, exist_ok=True)
    fb_stem = os.path.splitext(fb)[0]
    os_stem = os.path.splitext(os_pitch)[0]
    base = f"compare_{pitcher}_{fb_stem}_vs_{os_stem}_{layout}"
    raw = os.path.join(out_dir, f"{base}.mp4")
    web = os.path.join(out_dir, f"web_{base}.mp4")
    return raw, web


def render_compare_tab(base_dir):
    """Render Pitch Compare demo tab for presentation-ready FB vs OS comparison."""

    st.markdown(
        info_panel(
            "Pick a fastball and offspeed pitch, choose a layout and joint to highlight, "
            "then generate a DTW-synced comparison on real footage. "
            "Use <strong>Find Best Pair</strong> to auto-select the clearest tipping cue."
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

    default_pitcher = "Gburg16" if "Gburg16" in pitchers else pitchers[0]
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

    opt1, opt2, opt3, opt4 = st.columns(4, gap="large")
    with opt1:
        layout_label = st.radio("Layout", list(LAYOUT_OPTIONS.keys()), horizontal=True)
        layout = LAYOUT_OPTIONS[layout_label]
    with opt2:
        highlight_target = st.selectbox("Joint Highlight", FEATURES, key="compare_highlight")
    with opt3:
        skeleton_opacity = st.slider(
            "Skeleton Opacity", 0.3, 1.0, 0.85, 0.05, key="compare_opacity"
        )
    with opt4:
        st.markdown("<div style='padding-top: 1.65rem;'></div>", unsafe_allow_html=True)
        find_best = st.button("Find Best Pair", width="stretch")

    if find_best and highlight_target != "None":
        pitcher_root = os.path.join(keypoint_root, pitcher)
        with st.spinner(f"Scanning {pitcher} for best {highlight_target} contrast…"):
            best_fb, best_os = t2.findBestPitchPair(pitcher_root, highlight_target)
            if best_fb and best_os:
                st.session_state.compare_fb = os.path.basename(best_fb)
                st.session_state.compare_os = os.path.basename(best_os)
                st.rerun()
            else:
                st.markdown(
                    warn_panel("Could not find a fastball/offspeed pair for auto-selection."),
                    unsafe_allow_html=True,
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
    cache_key = _cache_key(pitcher, fb_pitch, os_pitch, layout, highlight, skeleton_opacity)
    raw_path, web_path = _output_paths(base_dir, pitcher, fb_pitch, os_pitch, layout)

    btn_col, _ = st.columns([1, 3])
    with btn_col:
        generate = st.button("Generate Comparison", type="primary", width="stretch")

    if generate or (
        st.session_state.get("compare_cache_key") == cache_key
        and os.path.exists(web_path if os.path.exists(web_path) else raw_path)
    ):
        if generate or st.session_state.get("compare_cache_key") != cache_key:
            with st.spinner("Aligning pitches and rendering comparison…"):
                try:
                    metrics = demo.renderPitchComparison(
                        fb_kpts,
                        os_kpts,
                        raw_path,
                        layout=layout,
                        highlight=highlight,
                        skeleton_opacity=skeleton_opacity,
                        base_dir=base_dir,
                        label1="FASTBALL",
                        label2="OFFSPEED",
                    )
                    st.session_state.compare_metrics = metrics
                    st.session_state.compare_cache_key = cache_key

                    ffmpeg = _find_ffmpeg()
                    if ffmpeg:
                        subprocess.run(
                            [ffmpeg, "-y", "-i", raw_path, "-vcodec", "libx264", web_path],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    else:
                        st.markdown(
                            warn_panel(
                                "FFmpeg not found — video may not play in browser. "
                                "Install with <code>brew install ffmpeg</code>."
                            ),
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    st.markdown(warn_panel(f"Comparison failed: {e}"), unsafe_allow_html=True)
                    return

            if generate:
                st.markdown(
                    success_panel(f"Comparison saved to <code>{raw_path}</code>"),
                    unsafe_allow_html=True,
                )

        play_path = web_path if os.path.exists(web_path) else raw_path
        if os.path.exists(play_path):
            st.markdown(section_label("Comparison"), unsafe_allow_html=True)
            st.markdown("<div class='compare-panel'>", unsafe_allow_html=True)
            st.video(play_path)
            st.markdown("</div>", unsafe_allow_html=True)

            metrics = st.session_state.get("compare_metrics", {})
            if metrics:
                st.markdown(section_label("Actionable Insights"), unsafe_allow_html=True)
                i1, i2, i3 = st.columns(3, gap="large")
                with i1:
                    watch = (
                        f"Peak <strong>{metrics.get('highlight_joint', 'joint')}</strong> "
                        f"separation: <strong>{metrics.get('peak_separation_px', 0)} px</strong> "
                        f"at synced frame {metrics.get('peak_frame_idx', 0)}."
                    )
                    st.markdown(insight_panel("What to Watch", watch), unsafe_allow_html=True)
                with i2:
                    why = (
                        f"Fastball motion score: <strong>{metrics.get('fb_score', 0)}</strong> · "
                        f"Offspeed: <strong>{metrics.get('os_score', 0)}</strong> "
                        f"({metrics.get('dtw_path_length', 0)} synced frames)."
                    )
                    st.markdown(insight_panel("Why It Matters", why), unsafe_allow_html=True)
                with i3:
                    action = metrics.get("suggested_action", "")
                    st.markdown(insight_panel("Suggested Action", action), unsafe_allow_html=True)

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown(
                        metric_card("Fastball", fb_pitch),
                        unsafe_allow_html=True,
                    )
                with r2:
                    st.markdown(
                        metric_card("Offspeed", os_pitch),
                        unsafe_allow_html=True,
                    )
                with r3:
                    st.markdown(
                        metric_card("Highlight", highlight_target),
                        unsafe_allow_html=True,
                    )
