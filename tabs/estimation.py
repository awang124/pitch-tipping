import os
import streamlit as st
import processing.tab1_helper as t1

from utils import load_models
from ui.components import metric_card, info_panel, success_panel, section_label

def render_estimation_tab(base_dir):
    """
    Render Tab 1 (Motion Estimation).

    Provides UI to upload video, run pose & glove extraction,
    display metadata, and save computed keypoint sequences.

    Parameters
    ----------
    base_dir: str
        Root directory for Pitches/ and Keypoints/ folders.
    """

    ### TOP INSTRUCTION PANEL
    st.markdown(
        info_panel(
            "Upload an <code>.MOV</code> or <code>.mp4</code> video of a baseball"
            " pitch (pitcher isolated and in frame): skeletal pose and glove bounding"
            " box sequences (herein called keypoint sequences) will be computed."
        ),
        unsafe_allow_html=True
    )
    col_left, col_right = st.columns([1, 1], gap="large")

    ### BOXES TO ENTER PITCHER NAME / PITCH CLASS, BUTTON TO UPLOAD VIDEO
    with col_left:
        st.markdown(section_label("Video Info"), unsafe_allow_html=True)
        pitcher_name = st.text_input("Pitcher (Folder) Name", "Babe_Ruth")
        pitch_type = st.selectbox(
            "Pitch Class",
            ["Changeup", "Curveball", "Cutter", "Fastball", "Four-Seam Fastball",
             "Slider", "Splitter", "Two-Seam Fastball", "Other"]
        )
        st.markdown(section_label("Video Input"), unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            f"Upload {pitcher_name} · {pitch_type}",
            type=["mov", "mp4"],
            help="Supports .MOV and .mp4 formats."
        )

    ### DISPLAY UPLOADED VIDEO
    with col_right:
        if uploaded_file:
            st.markdown(section_label("Your Video"), unsafe_allow_html=True)
            st.video(uploaded_file)
        else:
            st.markdown(
                """
                <div style='background: #111620; border: 1px dashed #1e2a3a; border-radius: 4px;
                height: 220px; display: flex; flex-direction: column;
                align-items: center; justify-content: center; gap: 8px;'>
                <div style='font-size: 2rem; opacity: 0.15;'>▶</div>
                <div style='font-size: 0.72rem; color: #334155;
                font-family: "IBM Plex Mono", monospace;
                letter-spacing: 0.08em; text-transform: uppercase;'>
                No video uploaded
                </div></div>
                """,
                unsafe_allow_html=True,
            )

    ### BUTTON TO BEGIN KEYPOINT COMPUTATION AFTER UPLOAD
    if uploaded_file:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        run_col, _ = st.columns([1, 3])
        with run_col:
            run_pipeline = st.button("▶  Run Motion Estimation", type="primary", width="stretch")

        ### CREATE PATHS TO STORE UPLOADED VIDEO AND COMPUTED KEYPOINTS
        if run_pipeline:
            pitch_root = os.path.join(base_dir, "Pitches")
            keypoint_root = os.path.join(base_dir, "Keypoints")
            base_filename = os.path.splitext(uploaded_file.name)[0]
            save_name = f"{pitcher_name}/{pitch_type.lower().replace(' ', '_')}"
            save_name = f"{save_name}_{base_filename}"

            ### CREATE PITCHES/ AND KEYPOINTS/ DIRECTORIES
            os.makedirs(pitch_root, exist_ok=True)
            os.makedirs(keypoint_root, exist_ok=True)
            os.makedirs(os.path.join(pitch_root, os.path.dirname(save_name)), exist_ok=True)
            os.makedirs(os.path.join(keypoint_root, os.path.dirname(save_name)), exist_ok=True)

            ### CREATE VIDEO AND KEYPOINT PATHS, SAVE VIDEO
            video_target = os.path.join(pitch_root, f"{save_name}.mov")
            kpt_target = os.path.join(keypoint_root, f"{save_name}.npy")
            with open(video_target, "wb") as f:
                f.write(uploaded_file.getbuffer())

            ### PROGRESS BAR
            st.markdown(
                """
                <div style='margin-top: 1.5rem;'>
                <div style='font-family: "IBM Plex Mono", monospace; font-size: 0.62rem;
                letter-spacing: 0.12em; text-transform: uppercase; color: #64748b;
                margin-bottom: 0.8rem;'>Motion Estimation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            progress_bar = st.progress(0)
            status_text = st.empty()
            with st.spinner("Initializing models:"):
                pose_m, glove_m = load_models()

            ### COMPUTE KEYPOINTS
            success, meta = t1.processVideo(
                video_target, kpt_target, pose_m, glove_m, progress_bar, status_text
            )

            ### DISPLAY VIDEO METADATA
            if success:
                status_text.empty()
                progress_bar.empty()
                if meta:
                    m1, m2, m3, m4 = st.columns(4)
                    for col, label, val in zip(
                        [m1, m2, m3, m4],
                        ["# Total Frames", "# Window Frames",
                         "Frame Rate", "Video Duration (seconds)"],
                        [meta['num_total_frames'], meta['num_extracted_frames'],
                         meta['fps'], meta['duration']]
                    ):
                        with col:
                            st.markdown(metric_card(label, val), unsafe_allow_html=True)
                st.markdown(
                    success_panel(
                        f"✓ &nbsp; Pose and glove sequences stored at <code>{kpt_target}</code>"
                    ),
                    unsafe_allow_html=True,
                )

