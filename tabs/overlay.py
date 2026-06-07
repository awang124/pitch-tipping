import os
import subprocess
import tempfile
import streamlit as st
import processing.tab5_helper as t5

from ui.components import info_panel, warn_panel, success_panel, section_label, metric_card

def render_overlay_tab(base_dir):
    """
    Render Tab 5 (Skeleton Overlay).

    Upload a source pitch video alongside selecting its corresponding
    keypoint sequence, then renders the extracted skeleton overlaid
    directly onto the real video footage with optional feature highlighting.

    Parameters
    ----------
    base_dir: str
        Root directory for Pitches/ and Keypoints/ folders.
    """

    ### TOP INSTRUCTION PANEL
    st.markdown(
        info_panel(
            "Upload the original source video for a pitch, then select its"
            " corresponding keypoint sequence. <br>"
            "The extracted skeleton will be rendered directly onto the real footage. <br>"
            "Optionally highlight a kinematic feature to inspect it in context."
        ),
        unsafe_allow_html=True,
    )

    ### KEYPOINTS DIRECTORY
    keypoint_root = os.path.join(base_dir, "Keypoints")
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

    if not npy_files:
        st.markdown(
            warn_panel(
                "No keypoint sequences found. Upload videos in Tab 01 first."
            ),
            unsafe_allow_html=True,
        )
        return

    ### INPUT COLUMNS
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown(section_label("Source Video"), unsafe_allow_html=True)
        uploaded_video = st.file_uploader(
            "Upload Original Pitch Video",
            type=["mp4", "mov"],
            help="Upload the original .mp4 or .mov video this keypoint file was extracted from.",
            key="tab5_video_upload",
        )

        ### PREVIEW UPLOADED VIDEO
        if uploaded_video:
            st.video(uploaded_video)

    with right_col:
        st.markdown(section_label("Keypoint Sequence"), unsafe_allow_html=True)
        selected_npy = st.selectbox(
            "Select Keypoint File",
            npy_files,
            help="Choose the .npy keypoint file corresponding to your uploaded video.",
            key="tab5_npy_select",
        )

        ### FEATURE HIGHLIGHT SELECTOR
        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
        st.markdown(section_label("Feature Highlight"), unsafe_allow_html=True)
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
        highlight_target = st.selectbox(
            "Kinematic Feature to Highlight",
            FEATURES,
            key="tab5_highlight",
        )

        ### OPACITY SLIDER
        st.markdown("<div style='margin-top: 1.2rem;'></div>", unsafe_allow_html=True)
        skeleton_opacity = st.slider(
            "Skeleton Opacity",
            min_value=0.3,
            max_value=1.0,
            value=0.85,
            step=0.05,
            help="Controls how strongly the skeleton is blended onto the video.",
            key="tab5_opacity",
        )

    ### GENERATE BUTTON
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    gen_col, _ = st.columns([1, 3])
    with gen_col:
        generate = st.button(
            "▶  Generate Overlay", type="primary", width="stretch", key="tab5_generate"
        )

    if generate:
        if not uploaded_video:
            st.markdown(
                warn_panel("Please upload a source video before generating."),
                unsafe_allow_html=True,
            )
            return

        ### BUILD OUTPUT PATHS
        out_dir = os.path.join(base_dir, "Visualizations")
        os.makedirs(out_dir, exist_ok=True)

        npy_stem = os.path.splitext(selected_npy)[0].replace("/", "_").replace("\\", "_")
        out_video = os.path.join(out_dir, f"overlay_{npy_stem}.mp4")
        web_video = os.path.join(out_dir, f"web_overlay_{npy_stem}.mp4")

        ### SAVE UPLOADED VIDEO TO TEMP FILE
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded_video.getbuffer())
            tmp_video_path = tmp.name

        ### RESOLVE HIGHLIGHT STRING
        highlight = (
            highlight_target.lower().replace(" ", "_")
            if highlight_target != "None"
            else ""
        )

        ### RUN OVERLAY PIPELINE
        with st.spinner("Overlaying skeleton onto video:"):
            success, message = t5.overlaySkeletonOnVideo(
                kpts_path=os.path.join(keypoint_root, selected_npy),
                video_path=tmp_video_path,
                output_path=out_video,
                highlight=highlight,
                skeleton_opacity=skeleton_opacity,
            )

        ### CLEAN UP TEMP FILE
        try:
            os.unlink(tmp_video_path)
        except Exception:
            pass

        if not success:
            st.markdown(
                warn_panel(f"Overlay failed: {message}"),
                unsafe_allow_html=True,
            )
            return

        ### TRANSCODE FOR BROWSER
        with st.spinner("Transcoding for browser playback:"):
            transcode_result = subprocess.run(
                ["ffmpeg", "-y", "-i", out_video, "-vcodec", "libx264", "-crf", "23", web_video],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        ### DISPLAY SUCCESS + VIDEO
        st.markdown(
            success_panel(
                f"✓ &nbsp; Overlay created, saved to <code>{out_video}</code> &nbsp;·&nbsp; {message}"
            ),
            unsafe_allow_html=True,
        )
        st.markdown(section_label("Output"), unsafe_allow_html=True)

        if os.path.exists(web_video):
            st.video(web_video)
        elif os.path.exists(out_video):
            st.video(out_video)
        else:
            st.markdown(
                warn_panel(
                    "Video file could not be located after rendering."
                    " Check that FFmpeg is installed correctly."
                ),
                unsafe_allow_html=True,
            )
            return

        ### METADATA CARDS
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                metric_card("Keypoint File", selected_npy),
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                metric_card("Source Video", uploaded_video.name),
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                metric_card("Highlighted Feature", highlight_target),
                unsafe_allow_html=True,
            )