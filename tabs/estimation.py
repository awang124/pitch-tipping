import os
import cv2
import streamlit as st
import processing.tab1_helper as t1

from utils import load_pose_landmarker
from ui.components import metric_card, info_panel, success_panel, section_label, warn_panel

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
            with st.spinner("Initializing pose model:"):
                model_path = load_pose_landmarker()

            ### COMPUTE KEYPOINTS
            success, meta = t1.processVideo(
                video_target, kpt_target, model_path, progress_bar, status_text
            )

            ### DISPLAY VIDEO METADATA
            if success:
                status_text.empty()
                progress_bar.empty()
                if meta:
                    m1, m2, m3, m4 = st.columns(4)
                    for col, label, val in zip(
                        [m1, m2, m3, m4],
                        ["# Total Frames", "# Keypoint Frames",
                         "Frame Rate", "Video Duration (seconds)"],
                        [meta['num_total_frames'], meta['num_extracted_frames'],
                         meta['fps'], meta['duration']]
                    ):
                        with col:
                            st.markdown(metric_card(label, val), unsafe_allow_html=True)

                    st.markdown(section_label("Extraction Quality"), unsafe_allow_html=True)
                    q1, q2, q3, q4 = st.columns(4)
                    qa_cards = [
                        (q1, "Valid Joint %", f"{meta.get('avg_joint_pct', '—')}%"),
                        (q2, "Glove Detection %", f"{meta.get('glove_pct', '—')}%"),
                        (q3, "Hip Jitter (px)", meta.get("pitcher_track_jitter", "—")),
                        (q4, "Delivery Type", meta.get("delivery_hint", "—")),
                    ]
                    for col, label, val in qa_cards:
                        with col:
                            st.markdown(metric_card(label, val), unsafe_allow_html=True)

                    if meta.get("retried_tracking"):
                        st.markdown(
                            info_panel(
                                "Tracking was auto-retried using arm-motion selection"
                                " due to high hip jitter on the first pass."
                            ),
                            unsafe_allow_html=True,
                        )

                    if meta.get("pitcher_track_jitter", 0) > 80:
                        st.markdown(
                            warn_panel(
                                "High hip jitter detected — the tracker may be following"
                                " the umpire or another person. Review the preview below."
                            ),
                            unsafe_allow_html=True,
                        )

                    st.markdown(section_label("Skeleton Preview"), unsafe_allow_html=True)
                    show_all = st.checkbox(
                        "Show all detections (debug)",
                        value=False,
                        help="Draw every MediaPipe pose with bbox area and hip-x labels.",
                        key="tab1_show_all_detections",
                    )
                    default_preview = meta.get("peak_knee_frame") or meta.get("windup_end") or 45
                    preview_frame = st.slider(
                        "Preview frame",
                        0,
                        max(meta["num_extracted_frames"] - 1, 0),
                        value=min(
                            default_preview or 45,
                            max(meta["num_extracted_frames"] - 1, 0),
                        ),
                        key="tab1_preview_frame",
                    )
                    if meta.get("peak_knee_frame") is not None:
                        st.caption(
                            f"Peak leg kick: frame {meta['peak_knee_frame']} "
                            f"(Pitch Compare sync landmark)"
                        )
                    if meta.get("windup_end") is not None:
                        st.caption(
                            f"Estimated windup end: frame {meta['windup_end']} "
                            f"(last frame before forward drive — used for tipping analysis)"
                        )
                    if show_all:
                        preview_img = t1.renderAllDetectionsFrame(
                            video_target, preview_frame, model_path=model_path,
                        )
                    else:
                        preview_img = t1.renderPreviewFrame(
                            video_target, kpt_target, preview_frame
                        )
                    if meta.get("dominant_hip_x_frac") is not None:
                        dx = meta["dominant_hip_x_frac"]
                        if meta.get("camera_hint") == "stretch_landscape" and dx < 0.45:
                            st.markdown(
                                warn_panel(
                                    f"Tracked person hip center is left of frame"
                                    f" (x={dx:.2f}). Expected pitcher on right (>0.45)"
                                    f" — may be tracking umpire."
                                ),
                                unsafe_allow_html=True,
                            )
                    if preview_img is not None:
                        st.image(
                            cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {preview_frame + 1} overlay",
                            width="stretch",
                        )
                    else:
                        st.markdown(
                            warn_panel("Could not render preview frame."),
                            unsafe_allow_html=True,
                        )

                st.markdown(
                    success_panel(
                        f"✓ &nbsp; Pose and glove sequences stored at <code>{kpt_target}</code>"
                    ),
                    unsafe_allow_html=True,
                )

