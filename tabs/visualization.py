import os
import subprocess
import streamlit as st
import processing.tab2_helper as t2

from ui.components import info_panel, warn_panel, success_panel, section_label, metric_card

def render_visualization_tab(base_dir):
    """
    Render Tab 2 (Comparative Visualization).

    Allows selection of two keypoint sequences, optional auto‑selection
    of maximally different pair, and generation of a RANSAC‑aligned
    superimposition video with kinematic feature highlighting.

    Parameters
    ----------
    base_dir: str
        Root directory for Pitches/ and Keypoints/ folders.
    """

    ### TOP INSTRUCTION PANEL
    st.markdown(
        info_panel(
            "Select two keypoint sequences (filenames) to generate their superimposition. <br>"
            "Optionally highlight a kinematic feature to manually inspect tipping cues. <br>"
            "Optionally auto-select a sequence pair w/ maximal difference in highlighted feature."
        ),
        unsafe_allow_html=True,
    )

    ### KEYPOINTS DIRECTORY (JUST USE DEFAULT)
    tab2_data_root = st.text_input("Keypoint Directory", "Keypoints", key="tab2_root")
    keypoint_root = os.path.join(base_dir, tab2_data_root)

    ### ENSURE KEYPOINTS COMPUTED (TAB 1 BEFORE TAB 2)
    if not os.path.exists(keypoint_root):
        st.markdown(
            warn_panel(
                f"No keypoint sequences found in <code>{keypoint_root}</code>."
                " Upload your videos in Tab 01 to compute them."
            ),
            unsafe_allow_html=True,
        )
        return

    ### COLLECT ALL KEYPOINT FILES
    npy_files = []
    for r, _, f_list in os.walk(keypoint_root):
        for f in f_list:
            if f.endswith(".npy"):
                npy_files.append(os.path.relpath(os.path.join(r, f), keypoint_root))

    if len(npy_files) < 2:
        st.markdown(
            warn_panel(
                f"Fewer than two keypoint sequences found in <code>{keypoint_root}</code>."
                " Upload your videos in Tab 01 to compute more."
            ),
            unsafe_allow_html=True,
        )
        return

    ### DEFAULT SEQUENCE PAIR
    if "best_pitch1" not in st.session_state:
        st.session_state.best_pitch1 = npy_files[0]
    if "best_pitch2" not in st.session_state:
        st.session_state.best_pitch2 = npy_files[1] if len(npy_files) > 1 else npy_files[0]

    sel_col1, sel_col2 = st.columns(2, gap="large")
    base_idx = npy_files.index(st.session_state.best_pitch1)\
        if st.session_state.best_pitch1 in npy_files else 0
    comp_idx = npy_files.index(st.session_state.best_pitch2)\
        if st.session_state.best_pitch2 in npy_files else 1

    ### BOXES TO SELECT SEQUENCE PAIR
    with sel_col1:
        base_pitch = st.selectbox("Sequence 1 (White)", npy_files, index=base_idx)
    with sel_col2:
        compare_pitch = st.selectbox("Sequence 2 (Blue)", npy_files, index=comp_idx)

    st.session_state.best_pitch1 = base_pitch
    st.session_state.best_pitch2 = compare_pitch

    ### BOX TO SELECT FEATURE TO HIGHLIGHT
    FEATURES = [
        "None",
        "Glove",
        "Left Knee",
        "Right Elbow",
        "Left Shoulder",
        "Right Shoulder",
        "Left Hip",
        "Right Hip",
    ]
    feat_col, btn_col = st.columns([2, 2], gap="large")
    with feat_col:
        highlight_target = st.selectbox("Kinematic Feature", FEATURES)

    ### BUTTON TO SELECT EXTREMAL MATCHUP
    with btn_col:
        if highlight_target != "None":
            st.markdown("<div style='padding-top: 1.65rem;'></div>", unsafe_allow_html=True)
            if st.button(
                f"Auto-Select Extremal Matchup  →  {highlight_target.split(' (')[0]}",
                width="stretch",
            ):
                with st.spinner(
                    f"Scanning {len(npy_files)} sequences for maximum"
                    " {highlight_target.split(' (')[0]} difference:"
                ):
                    ### SELECT EXTREMAL MATCHUP
                    best_fb, best_os = t2.findBestPitchPair(keypoint_root, highlight_target)
                    if best_fb and best_os:
                        st.session_state.best_pitch1 = best_fb
                        st.session_state.best_pitch2 = best_os
                        st.rerun()
                    else:
                        st.markdown(
                            warn_panel(
                                "Not enough labeled fastball / offspeed pairs in"
                                " keypoints directory to perform auto-selection."
                            ),
                            unsafe_allow_html=True,
                        )

    ### BUTTON TO GENERATE SUPERIMPOSITION
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    gen_col, _ = st.columns([1, 3])
    with gen_col:
        generate = st.button("Generate Superimposition", type="primary", width="stretch")

    if generate:
        ### GET SELECTED KEYPOINT SEQUENCES
        out_dir = os.path.join(base_dir, "Visualizations")
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(base_pitch)[0].replace("/", "_").replace("\\", "_")
        comp_name = os.path.splitext(compare_pitch)[0].replace("/", "_").replace("\\", "_")
        out_video = os.path.join(out_dir, f"{base_name}_vs_{comp_name}.mp4")

        with st.spinner("Running DTW and RANSAC:"):
            highlight = (
                highlight_target.split(" (")[0].lower().replace(" ", "_")
                if highlight_target != "None"
                else ""
            )
            ### GENERATE SUPERIMPOSITION
            t2.visualizeDifferences(
                os.path.join(keypoint_root, base_pitch),
                os.path.join(keypoint_root, compare_pitch),
                out_video,
                highlight=highlight,
            )
            ### TRANSCODE OUTPUT FOR WEB BROWSER VIEWING
            web_video = os.path.join(out_dir, f"web_{base_name}_vs_{comp_name}.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-i", out_video, "-vcodec", "libx264", web_video],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        st.markdown(
            success_panel(f"✓ &nbsp; Superimposition created, saved to <code>{out_video}</code>"),
            unsafe_allow_html=True,
        )
        st.markdown(
            section_label("Output"),
            unsafe_allow_html=True,
        )
        st.video(web_video if os.path.exists(web_video) else out_video)

        ### DISPLAY SEQUENCE METADATA
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                metric_card(
                    "Sequence 1",
                    base_pitch,
                ),
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                metric_card(
                    "Sequence 2",
                    compare_pitch,
                ),
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                metric_card(
                    "Highlighted Feature",
                    highlight_target,
                ),
                unsafe_allow_html=True,
            )

