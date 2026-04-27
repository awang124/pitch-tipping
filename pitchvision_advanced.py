import html
import os
import subprocess
import ultralytics
import pandas as pd
import numpy as np
import streamlit as st

import processing.tab1_helper as t1
import processing.tab2_helper as t2
import processing.tab3_helper as t3

from sklearn.tree import export_text
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="PitchVision | SARG @ JHU",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    /* ── Fonts ─────────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    /* ── Root tokens ────────────────────────────────────────────────────── */
    :root {
        --bg:         #0a0d12;
        --surface:    #111620;
        --surface2:   #181e2c;
        --border:     #1e2a3a;
        --accent:     #2563eb;
        --accent2:    #38bdf8;
        --warn:       #f59e0b;
        --success:    #22c55e;
        --danger:     #ef4444;
        --text:       #e2e8f0;
        --muted:      #64748b;
        --mono:       'IBM Plex Mono', monospace;
        --sans:       'IBM Plex Sans', sans-serif;
    }

    /* ── Global reset ───────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: var(--sans) !important;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    /* ── Hide Streamlit chrome ──────────────────────────────────────────── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1300px; }

    /* ── Sidebar ────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { font-family: var(--sans) !important; }
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.65rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
        margin-bottom: 0.6rem !important;
        font-weight: 600 !important;
    }

    /* ── Top wordmark banner ────────────────────────────────────────────── */
    .wordmark {
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-bottom: 0.2rem;
    }
    .wordmark-primary {
        font-family: var(--mono);
        font-size: 1.7rem;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.02em;
    }
    .wordmark-badge {
        font-family: var(--mono);
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        color: var(--accent2);
        background: rgba(56,189,248,0.08);
        border: 1px solid rgba(56,189,248,0.2);
        padding: 2px 8px;
        border-radius: 2px;
        text-transform: uppercase;
    }
    .wordmark-sub {
        font-size: 0.8rem;
        color: var(--muted);
        letter-spacing: 0.02em;
        margin-bottom: 2rem;
    }
    .divider {
        height: 1px;
        background: linear-gradient(90deg, var(--accent) 0%, transparent 100%);
        margin: 1.5rem 0 2rem;
        opacity: 0.4;
    }

    /* ── Section labels ─────────────────────────────────────────────────── */
    .section-label {
        font-family: var(--mono);
        font-size: 0.6rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-label::before {
        content: '';
        display: inline-block;
        width: 16px;
        height: 1px;
        background: var(--accent);
        opacity: 0.7;
    }

    /* ── Stat cards ─────────────────────────────────────────────────────── */
    .stat-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 4px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
    }
    .stat-card .label {
        font-family: var(--mono);
        font-size: 0.58rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.35rem;
    }
    .stat-card .value {
        font-family: var(--mono);
        font-size: 1.6rem;
        font-weight: 600;
        color: var(--text);
        line-height: 1;
    }
    .stat-card .unit {
        font-size: 0.75rem;
        color: var(--muted);
        margin-left: 4px;
    }

    /* ── Info panel ─────────────────────────────────────────────────────── */
    .info-panel {
        background: var(--surface2);
        border-left: 2px solid var(--accent);
        border-radius: 0 4px 4px 0;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1rem;
        font-size: 0.82rem;
        color: #94a3b8;
        line-height: 1.6;
    }
    .warn-panel {
        background: rgba(245,158,11,0.06);
        border-left: 2px solid var(--warn);
        border-radius: 0 4px 4px 0;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1rem;
        font-size: 0.82rem;
        color: #fcd34d;
        line-height: 1.6;
    }
    .success-panel {
        background: rgba(34,197,94,0.06);
        border-left: 2px solid var(--success);
        border-radius: 0 4px 4px 0;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1rem;
        font-size: 0.82rem;
        color: #86efac;
        line-height: 1.6;
    }

    /* ── ML report / coefficient tables ─────────────────────────────────── */
    .model-report-pre {
        font-family: var(--mono);
        font-size: 0.72rem;
        color: #cbd5e1;
        line-height: 1.5;
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .lr-table-title {
        font-family: var(--mono);
        font-size: 0.58rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 0 0 0.5rem 0;
    }

    /* ── Tabs ───────────────────────────────────────────────────────────── */
    [data-baseweb="tab-list"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border) !important;
        gap: 0 !important;
        margin-bottom: 2rem !important;
    }
    [data-baseweb="tab"] {
        background: transparent !important;
        font-family: var(--mono) !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
        padding: 0.6rem 1.4rem !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
    }
    [aria-selected="true"][data-baseweb="tab"] {
        color: var(--text) !important;
        border-bottom: 2px solid var(--accent) !important;
    }

    /* ── Inputs & selects ───────────────────────────────────────────────── */
    [data-baseweb="input"] input,
    [data-baseweb="select"] div,
    .stTextInput input,
    .stSelectbox div[data-baseweb="select"] > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 3px !important;
        color: var(--text) !important;
        font-family: var(--sans) !important;
        font-size: 0.85rem !important;
    }
    .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(37,99,235,0.2) !important;
    }
    label[data-testid="stWidgetLabel"] p,
    .stSelectbox label p,
    .stTextInput label p {
        font-family: var(--mono) !important;
        font-size: 0.62rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
        font-weight: 600 !important;
    }

    /* ── File uploader ──────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: var(--surface) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 4px !important;
        padding: 1.5rem !important;
    }
    [data-testid="stFileUploader"] label {
        font-family: var(--mono) !important;
        font-size: 0.72rem !important;
    }

    /* ── Buttons ────────────────────────────────────────────────────────── */
    .stButton > button {
        font-family: var(--mono) !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border-radius: 3px !important;
        border: 1px solid var(--border) !important;
        background: var(--surface) !important;
        color: var(--text) !important;
        padding: 0.55rem 1.2rem !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        border-color: var(--accent) !important;
        background: rgba(37,99,235,0.1) !important;
        color: var(--accent2) !important;
    }
    .stButton [kind="primary"] > button,
    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8 !important;
    }

    /* ── Progress bar ───────────────────────────────────────────────────── */
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    }
    [data-testid="stProgressBar"] > div {
        background: var(--surface) !important;
        border-radius: 2px !important;
    }

    /* ── Spinner ─────────────────────────────────────────────────────────── */
    [data-testid="stSpinner"] p {
        font-family: var(--mono) !important;
        font-size: 0.78rem !important;
        color: var(--muted) !important;
    }

    /* ── Columns ─────────────────────────────────────────────────────────── */
    [data-testid="column"] {
        padding: 0 0.75rem !important;
    }

    /* ── Video player ────────────────────────────────────────────────────── */
    video {
        border-radius: 4px !important;
        border: 1px solid var(--border) !important;
    }

    /* ── st.success / st.warning / st.error overrides ───────────────────── */
    [data-testid="stAlert"] {
        border-radius: 3px !important;
        font-family: var(--sans) !important;
        font-size: 0.82rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown(
        """
        <div style='padding: 1.2rem 0 0.5rem;'>
            <div style='font-family: "IBM Plex Mono", monospace; font-size: 1.05rem; font-weight: 600; color: #e2e8f0; letter-spacing: -0.01em;'>PitchVision</div>
            <div style='font-size: 0.68rem; color: #64748b; margin-top: 2px; letter-spacing: 0.04em;'>Sports Analytics Research Group</div>
            <div style='font-size: 0.68rem; color: #475569; letter-spacing: 0.04em;'>Johns Hopkins University</div>
        </div>
        <div style='height: 1px; background: #1e2a3a; margin: 1rem 0 1.5rem;'></div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("### Configuration")
    base_dir = st.text_input("Data Root Directory", ".", help="Root path where Pitches/ and Keypoints/ folders are located.")
    st.markdown(
        """
        <div style='height: 1px; background: #1e2a3a; margin: 1.2rem 0;'></div>
        <div style='font-family: "IBM Plex Mono", monospace; font-size: 0.58rem; letter-spacing: 0.14em; text-transform: uppercase; color: #475569; margin-bottom: 0.6rem;'>Pipeline Modules</div>
        """,
        unsafe_allow_html=True
    )

    pipeline_items = [
        ("YOLOv8x-Pose", "Skeleton extraction"),
        ("Custom Glove YOLO", "Glove localization"),
        ("Motion-Aware Trim", "180-frame windowing"),
        ("RANSAC Alignment", "Superimposition"),
    ]
    for name, desc in pipeline_items:
        st.markdown(
            f"""
            <div style='display:flex; align-items:center; gap:8px; padding: 5px 0;'>
                <div style='width:6px; height:6px; border-radius:50%; background:#22c55e; flex-shrink:0;'></div>
                <div>
                    <div style='font-size:0.75rem; color:#cbd5e1; font-family:"IBM Plex Mono",monospace;'>{name}</div>
                    <div style='font-size:0.65rem; color:#475569;'>{desc}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown(
        """
        <div style='height: 1px; background: #1e2a3a; margin: 1.2rem 0;'></div>
        <div style='font-size: 0.65rem; color: #334155; padding-bottom: 1rem;'>
            Prototype — Spring 2026<br>
            SARG · Dept of Computer Science
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class='wordmark'>
        <span class='wordmark-primary'>PitchVision</span>
        <span class='wordmark-badge'>Research Build</span>
    </div>
    <div class='wordmark-sub'>Pitch Tipping Detection · Pose Estimation · Motion Analysis · SARG @ Johns Hopkins</div>
    <div class='divider'></div>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs([
    "01  /  Extraction & Tracking", "02  /  Comparative Analysis", "03  /  ML Classification", "04  /  Predict Pitches"
])

@st.cache_resource
def load_models():
    return ultralytics.YOLO("yolov8x-pose.pt"), ultralytics.YOLO("NEW_GLOVE_MODEL.pt")


def _lr_coefficients_styler(top_features):
    """Styled pandas Styler for L1 LogReg ranks (signed β, diverging heatmap)."""
    df = pd.DataFrame(
        {
            "#": range(1, len(top_features) + 1),
            "Feature": [t[0] for t in top_features],
            "β (L1 coefficient)": [float(t[1]) for t in top_features],
            "Lean": [
                "Fastball" if float(t[1]) > 0 else "Offspeed" for t in top_features
            ],
        }
    )
    coef = df["β (L1 coefficient)"]
    lim = float(coef.abs().max()) if len(coef) else 1.0
    if lim < 1e-12:
        lim = 1.0
    return (
        df.style.format({"β (L1 coefficient)": "{:+,.4f}"})
        .background_gradient(
            subset=["β (L1 coefficient)"],
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
        )
        .set_properties(**{"text-align": "left"}, subset=["Feature", "Lean"])
        .set_properties(**{"font-family": "'IBM Plex Mono', monospace"}, subset=["#", "β (L1 coefficient)"])
    )


# ──────────────────────────────────────────────────────────────────────────────────────────────────
# TAB 1: EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        """
        <div class='info-panel'>
            Upload a raw <code>.MOV</code> or <code>.mp4</code> clip to run the full extraction pipeline: YOLOv8x skeleton detection,
            custom-trained glove localization, and automated motion-aware 180-frame windowing.
            Output is serialized as a structured <code>.npy</code> array for downstream analysis.
        </div>
        """,
        unsafe_allow_html=True
    )
    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        st.markdown("<div class='section-label'>Subject Metadata</div>", unsafe_allow_html=True)
        pitcher_name = st.text_input("Pitcher Name", "Kieran")
        pitch_type   = st.selectbox(
            "Pitch Classification",
            ["Changeup", "Curveball", "Cutter", "Fastball", "Four-Seam Fastball",
             "Slider", "Splitter", "Two-Seam Fastball", "Other"]
        )
        st.markdown(
            "<div class='section-label' style='margin-top:1.5rem;'>Video Input</div>",
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(
            f"Upload {pitcher_name} · {pitch_type}",
            type=["mov", "mp4"],
            help="Supports .MOV (iPhone/camera) and .mp4 formats."
        )
    with col_right:
        if uploaded_file:
            st.markdown("<div class='section-label'>Preview</div>", unsafe_allow_html=True)
            st.video(uploaded_file)
        else:
            st.markdown(
                """
                <div style='background: #111620; border: 1px dashed #1e2a3a; border-radius: 4px;
                            height: 220px; display: flex; flex-direction: column;
                            align-items: center; justify-content: center; gap: 8px;'>
                    <div style='font-size: 2rem; opacity: 0.15;'>▶</div>
                    <div style='font-size: 0.72rem; color: #334155; font-family: "IBM Plex Mono", monospace;
                                 letter-spacing: 0.08em; text-transform: uppercase;'>No video uploaded</div>
                </div>
                """,
            unsafe_allow_html=True
        )
    if uploaded_file:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        run_col, _ = st.columns([1, 3])
        with run_col:
            run_pipeline = st.button(
                "▶  Run Extraction Pipeline", type="primary", use_container_width=True
            )
        if run_pipeline:
            pitch_root = os.path.join(base_dir, "Pitches")
            keypoint_root = os.path.join(base_dir, "Keypoints")
            os.makedirs(pitch_root, exist_ok=True)
            os.makedirs(keypoint_root, exist_ok=True)
            base_filename = os.path.splitext(uploaded_file.name)[0]
            save_name = f"{pitcher_name.lower()}_{pitch_type.lower().replace(' ', '_')}"
            save_name = f"{save_name}_{base_filename}"
            video_target  = os.path.join(pitch_root, f"{save_name}.mov")
            kpt_target = os.path.join(keypoint_root, f"{save_name}.npy")
            with open(video_target, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.markdown(
                """
                <div style='margin-top: 1.5rem;'>
                    <div style='font-family: "IBM Plex Mono", monospace; font-size: 0.62rem;
                                 letter-spacing: 0.12em; text-transform: uppercase; color: #64748b; margin-bottom: 0.8rem;'>
                        Extraction Log
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            progress_bar = st.progress(0)
            status_text  = st.empty()
            with st.spinner("Initializing pose / glove models:"):
                pose_m, glove_m = load_models()
            success, meta = t1.processVideo(
                video_target, kpt_target, pose_m, glove_m, progress_bar, status_text
            )
            if success:
                status_text.empty()
                progress_bar.empty()
                if meta:
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(
                            f"""<div class='stat-card'>
                            <div class='label'>Total Frames</div>
                            <div class='value'>{meta['total_frames']}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    with m2:
                        st.markdown(
                            f"""<div class='stat-card'>
                            <div class='label'>Extracted Window</div>
                            <div class='value'>{meta['extracted_frames']}<span class='unit'>fr</span></div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    with m3:
                        st.markdown(
                            f"""<div class='stat-card'>
                            <div class='label'>Frame Rate</div>
                            <div class='value'>{meta['fps']}<span class='unit'>fps</span></div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    with m4:
                        st.markdown(
                            f"""<div class='stat-card'>
                            <div class='label'>Clip Duration</div>
                            <div class='value'>{meta['duration_s']}<span class='unit'>s</span></div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                st.markdown(
                    f"""
                    <div class='success-panel'>
                        ✓ &nbsp; Pipeline complete. Keypoints serialized to <code>{kpt_target}</code>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ──────────────────────────────────────────────────────────────────────────────────────────────────
# TAB 2: VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        """
        <div class='info-panel'>
            Select two tracked pitch sequences to generate a RANSAC-aligned superimposition.
            Use the AI anomaly highlighter to isolate the kinematic feature flagged by your
            classification model and visually confirm pitch-tipping behavior.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div class='section-label'>Data Source</div>", unsafe_allow_html=True)
    tab2_data_root = st.text_input("Pitch Sequence Directory", "keypoints/Randy/trimmed_keypoints_normalized", key="tab2_root")
    keypoint_root = os.path.join(base_dir, tab2_data_root)
    if not os.path.exists(keypoint_root):
        st.markdown(
            f"""
            <div class='warn-panel'>
                ⚠ &nbsp; Keypoints directory not found at <code>{keypoint_root}</code>.
                Run the extraction pipeline in Tab 01 first.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        npy_files = []
        for r, _, f_list in os.walk(keypoint_root):
            for f in f_list:
                if f.endswith(".npy"):
                    npy_files.append(os.path.relpath(os.path.join(r, f), keypoint_root))
        if len(npy_files) < 2:
            st.markdown(
                f"""
                <div class='warn-panel'>
                    ⚠ &nbsp; Fewer than two tracked pitches found in <code>{keypoint_root}</code>.
                    At least two <code>.npy</code> sequences are required for comparison.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            if "best_pitch1" not in st.session_state:
                st.session_state.best_pitch1 = npy_files[0]
            if "best_pitch2" not in st.session_state:
                if len(npy_files) > 1:
                    st.session_state.best_pitch2 = npy_files[1]
                else:
                    st.session_state.best_pitch2 = npy_files[0]
            st.markdown(
                "<div class='section-label'>Pitch Sequence Selection</div>",
                unsafe_allow_html=True
            )
            sel_col1, sel_col2 = st.columns(2, gap="large")
            base_idx = npy_files.index(st.session_state.best_pitch1)\
                if st.session_state.best_pitch1 in npy_files else 0
            comp_idx = npy_files.index(st.session_state.best_pitch2)\
                if st.session_state.best_pitch2 in npy_files else 1
            with sel_col1:
                base_pitch = st.selectbox(
                    "Reference Sequence (rendered white)", npy_files, index=base_idx
                )
            with sel_col2:
                compare_pitch = st.selectbox(
                    "Comparison Sequence (rendered blue)",  npy_files, index=comp_idx
                )
            st.session_state.best_pitch1 = base_pitch
            st.session_state.best_pitch2 = compare_pitch
            st.markdown(
                """
                <div style='height: 1px; background: #1e2a3a; margin: 1.5rem 0;'></div>
                <div class='section-label'>Kinematic Anomaly Highlighter</div>
                """,
                unsafe_allow_html=True
            )
            FEATURES = [
                "None",
                "Glove",
                "Left Knee",
                "Right Elbow",
                "Left Shoulder",
                "Right Shoulder",
                "Left Hip",
                "Right Hip"
            ]
            feat_col, btn_col = st.columns([2, 2], gap="large")
            with feat_col:
                highlight_target = st.selectbox("Flagged Kinematic Feature", FEATURES)
            try:
                with btn_col:
                    if highlight_target == "None":
                        raise Exception
                    st.markdown(
                        "<div style='padding-top: 1.65rem;'></div>", unsafe_allow_html=True
                    )
                    if not st.button(
                        f"Auto-Select Extremal Matchup  →  {highlight_target.split(' (')[0]}",
                        use_container_width=True
                    ):
                        raise Exception
                    with st.spinner(
                        f"Scanning {len(npy_files)} sequences for maximum "
                        f"{highlight_target.split(' (')[0]} divergence:"
                    ):
                        best_fb, best_os = t2.findBestPitchPair(keypoint_root, highlight_target)
                        if best_fb and best_os:
                            st.session_state.best_pitch1 = best_fb
                            st.session_state.best_pitch2 = best_os
                            st.rerun()
                            raise Exception
                        st.markdown(
                            """
                            <div class='warn-panel'>
                                Insufficient labeled fastball/offspeed pairs in the keypoints directory to perform auto-selection.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            except:
                pass
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            gen_col, _ = st.columns([1, 3])
            with gen_col:
                generate = st.button(
                    "Generate Superimposition", type="primary", use_container_width=True
            )
            if generate:
                out_dir = os.path.join(base_dir, "Visualizations")
                os.makedirs(out_dir, exist_ok=True)
                base_name = os.path.splitext(base_pitch)[0]
                comp_name = os.path.splitext(compare_pitch)[0]
                out_video = os.path.join(out_dir, f"{base_name}_vs_{comp_name}.mp4")
                with st.spinner("Running RANSAC spatial alignment and frame-synchronised overlay:"):
                    highlight = highlight_target.split(" (")[0].lower().replace(" ", "_")\
                        if highlight_target != "None" else ""
                    t2.visualizeDifferences(
                        os.path.join(keypoint_root, base_pitch),
                        os.path.join(keypoint_root, compare_pitch),
                        out_video,
                        highlight=highlight
                    )
                    web_video = os.path.join(out_dir, f"web_{base_name}_vs_{comp_name}.mp4")
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", out_video, "-vcodec", "libx264", web_video],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                st.markdown(
                    f"""
                    <div class='success-panel'>
                        ✓ &nbsp; Superimposition complete. Output saved to <code>{out_video}</code>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    "<div class='section-label' style='margin-top:1.5rem;'>Output</div>",
                    unsafe_allow_html=True
                )
                print(os.path.exists(web_video))
                st.video(target := web_video if os.path.exists(web_video) else out_video)
                print(target)
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown(
                        f"""<div class='stat-card'>
                        <div class='label'>Reference</div>
                        <div style='font-family:"IBM Plex Mono",monospace; font-size:0.75rem; color:#cbd5e1; margin-top:4px; word-break:break-all;'>{base_pitch}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                with r2:
                    st.markdown(
                        f"""<div class='stat-card'>
                        <div class='label'>Comparison</div>
                        <div style='font-family:"IBM Plex Mono",monospace; font-size:0.75rem; color:#38bdf8; margin-top:4px; word-break:break-all;'>{compare_pitch}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                with r3:
                    st.markdown(
                        f"""<div class='stat-card'>
                        <div class='label'>Highlighted Feature</div>
                        <div style='font-family:"IBM Plex Mono",monospace; font-size:0.75rem; color:#f59e0b; margin-top:4px;'>{highlight_target}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )

# ──────────────────────────────────────────────────────────────────────────────────────────────────
# TAB 3: FEATURE SIGNIFICANCE
# ──────────────────────────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(
        """
        <div class='info-panel'>
            <b>Three analytical tiers:</b><br>
            • <b>Aggregate Feature L1 LogReg</b> isolates core biomechanical tells via sparse regularization (75% accuracy).<br>
            • <b>Per‑frame Decision Tree</b> tracks granular frame-by-frame positional data.<br>
            • <b>Mann‑Whitney U Test</b> statistically validates specific features (e.g. glove_speed_max).
        </div>
        """,
        unsafe_allow_html=True
    )
    keypoints_root = os.path.join(base_dir, "Keypoints")
    if not os.path.exists(keypoints_root):
        st.markdown(f"<div class='warn-panel'> Keypoints directory not found at `{keypoints_root}`. Run Tab 1 first.</div>", unsafe_allow_html=True)
    else:
        with st.spinner("Loading and evaluating cross-session per‑frame sequences (this may take a moment)..."):
            st.markdown("<div class='section-label'>Evaluation Setup</div>", unsafe_allow_html=True)
            eval_mode = st.radio(
                "Evaluation Mode",
                ["Single-Pitcher (80/20 Split & Augment)", "Cross-Session (Train on A, Test on B)"],
                horizontal=True
            )
            
            if eval_mode == "Cross-Session (Train on A, Test on B)":
                c1, c2 = st.columns(2)
                with c1:
                    train_root_rel = st.text_input("Training Sequences (Session 1)", "Keypoints/Kieran")
                with c2:
                    test_root_rel = st.text_input("Testing Sequences (Session 2)", "Keypoints/Randy")
                train_root = os.path.join(base_dir, train_root_rel)
                test_root = os.path.join(base_dir, test_root_rel)
                
                raw_seqs_train, n_frames, pf_feature_names = t3.loadFrameDataset(train_root)
                raw_seqs_test, _, _ = t3.loadFrameDataset(test_root)
            else:
                data_root_rel = st.text_input("Pitcher Sequence Directory", "Keypoints/Randy")
                data_root = os.path.join(base_dir, data_root_rel)
                
                all_seqs, n_frames, pf_feature_names = t3.loadFrameDataset(data_root)
                raw_seqs_train = all_seqs  # We'll split it inside the functions
                raw_seqs_test = []
            
            if eval_mode == "Cross-Session (Train on A, Test on B)":
                if len(raw_seqs_train) < 3 or len(raw_seqs_test) < 1:
                    st.warning(f"Not enough labeled sequences found. Train: {len(raw_seqs_train)}, Test: {len(raw_seqs_test)}.")
                else:
                    n_fastball = sum(1 for s, l in raw_seqs_train if l == 1)
                    n_other = len(raw_seqs_train) - n_fastball
                    st.markdown(f"<div class='success-panel'> Session 1 (Train): {n_fastball} fastball, {n_other} offspeed. Session 2 (Test): {len(raw_seqs_test)} sequences. Features per frame: {t3.N_FRAME_FEATURES}.</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='section-label'> Decision Tree (Cross-Session Validation)</div>", unsafe_allow_html=True)
                    dt_results = t3.evaluate_cross_session(raw_seqs_train, raw_seqs_test, pf_feature_names, model_type="tree", max_depth=10, top_n=10)
                    st.markdown(f"**Augmented Training Set Size:** {dt_results['augmented_train_size']} sequences")
                    col_acc, col_rep = st.columns([1, 2])
                    with col_acc:
                        st.metric("Test Accuracy (Session 2)", f"{dt_results['accuracy']:.3f}")
                    with col_rep:
                        st.text("Classification Report:\n" + dt_results['classification_report'])
                    st.markdown("**Top 10 most important invariant features:**")
                    for feat, imp in dt_results['top_features']:
                        st.write(f"- `{feat}` ({imp:.4f})")
                        
                    st.markdown("<div class='section-label' style='margin-top:2rem;'> ⭐ Aggregate Feature L1 LogReg (Cross-Session Validation)</div>", unsafe_allow_html=True)
                    agg_results = t3.evaluate_cross_session_aggregate_lr(raw_seqs_train, raw_seqs_test, C=1.0, top_n=10)
                    lr_m1, lr_m2, lr_m3 = st.columns(3)
                    with lr_m1:
                        st.metric("Test accuracy (session 2)", f"{agg_results['accuracy']:.3f}")
                    with lr_m2:
                        st.metric("L1 active features", f"{agg_results['n_nonzero_features']} / {agg_results['total_features']}")
                    with lr_m3:
                        st.metric("Augmented train size", f"{agg_results['augmented_train_size']}")
                    lr_coef_col, lr_rep_col = st.columns([1.2, 1])
                    with lr_coef_col:
                        st.markdown(
                            "<div class='lr-table-title'>Top L1 coefficients — hue = direction (fastball vs offspeed), intensity = |β|</div>",
                            unsafe_allow_html=True,
                        )
                        _n_lr = len(agg_results["top_features"])
                        st.dataframe(
                            _lr_coefficients_styler(agg_results["top_features"]),
                            hide_index=True,
                            use_container_width=True,
                            height=min(460, 48 + _n_lr * 38),
                        )
                    with lr_rep_col:
                        st.markdown("<div style='height:1.35rem;'></div>", unsafe_allow_html=True)
                        st.markdown(
                            "<div class='stat-card' style='margin-bottom:0;'>"
                            "<div class='label'>Classification report</div>"
                            f"<pre class='model-report-pre'>{html.escape(agg_results['classification_report'])}</pre>"
                            "</div>",
                            unsafe_allow_html=True,
                        )
            else:
                if len(all_seqs) < 5:
                    st.warning(f"Not enough sequences to split for Single-Pitcher evaluation. Found {len(all_seqs)}.")
                else:
                    n_fastball = sum(1 for s, l in all_seqs if l == 1)
                    n_other = len(all_seqs) - n_fastball
                    st.markdown(f"<div class='success-panel'> Pitcher total: {n_fastball} fastball, {n_other} offspeed sequences. Running stratified 80/20 split and augmenting ONLY training.</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div class='section-label' style='margin-top:2rem;'> Decision Tree (Legacy Per-Frame Model)</div>", unsafe_allow_html=True)
                    dt_results = t3.evaluate_single_session(all_seqs, pf_feature_names, model_type="tree", top_n=10)
                    if dt_results:
                        st.markdown(f"**Augmented Training Set Size:** {dt_results['augmented_train_size']} sequences")
                        c_acc1, c_rep1 = st.columns([1, 2])
                        with c_acc1:
                            st.metric("Test Accuracy (Holdout)", f"{dt_results['accuracy']:.3f}")
                        with c_rep1:
                            st.text("Classification Report:\n" + dt_results['classification_report'])
                            
                    st.markdown("<div class='section-label' style='margin-top:2rem;'> ⭐ Single-Pitcher Aggregate L1 LogReg (80/20 Split)</div>", unsafe_allow_html=True)
                    st.markdown("*Uses ~50 biomechanical summary stats, properly evaluating against a completely pure internal holdout without cross-pitcher leakage.*")
                    
                    agg_results = t3.evaluate_single_session_aggregate_lr(all_seqs, test_size=0.2, C=1.0, top_n=10)
                    if not agg_results:
                        st.error("Error evaluating single session: Not enough samples per class to perform a stratified split.")
                    else:
                        st.session_state["single_pitcher_clf"] = agg_results["clf"]
                        st.session_state["single_pitcher_scaler"] = agg_results["scaler"]
                        st.session_state["single_pitcher_features"] = agg_results["feature_names"]
                        
                        sp_m1, sp_m2, sp_m3 = st.columns(3)
                        with sp_m1:
                            st.metric("Test accuracy (holdout)", f"{agg_results['accuracy']:.3f}")
                        with sp_m2:
                            st.metric("L1 active features", f"{agg_results['n_nonzero_features']} / {agg_results['total_features']}")
                        with sp_m3:
                            st.metric("Held-out pitches", f"{agg_results['test_size']}")
                        st.caption(
                            f"Training: {agg_results['train_base_size']} real pitches → augmented to {agg_results['augmented_train_size']} samples."
                        )
                        sp_lr_tab, sp_lr_rep = st.columns([1.2, 1])
                        with sp_lr_tab:
                            st.markdown(
                                "<div class='lr-table-title'>Top L1 coefficients — hue = direction, intensity = |β|</div>",
                                unsafe_allow_html=True,
                            )
                            _n_splr = len(agg_results["top_features"])
                            st.dataframe(
                                _lr_coefficients_styler(agg_results["top_features"]),
                                hide_index=True,
                                use_container_width=True,
                                height=min(460, 48 + _n_splr * 38),
                            )
                        with sp_lr_rep:
                            st.markdown("<div style='height:1.35rem;'></div>", unsafe_allow_html=True)
                            st.markdown(
                                "<div class='stat-card' style='margin-bottom:0;'>"
                                "<div class='label'>Classification report</div>"
                                f"<pre class='model-report-pre'>{html.escape(agg_results['classification_report'])}</pre>"
                                "</div>",
                                unsafe_allow_html=True,
                            )
    st.markdown("<div class='section-label' style='margin-top:2rem;'> Pairwise Statistical Significance (Fastball vs. Offspeed) – Aggregate Features</div>", unsafe_allow_html=True)
    seqs = t3.loadSequences(keypoints_root)
    relabel = dict.fromkeys(
        ["cutter", "fastball", "four_seam_fastball", "two_seam_fastball"], "fastball"
    )
    relabel.update(dict.fromkeys(
        ["changeup", "curveball", "offspeed", "slider", "splitter"], "offspeed")
    )
    seqs = [(pair[0], relabel[pair[1]]) for pair in seqs if pair[1] in relabel]
    if len(seqs) < 4:
        st.markdown("<div class='warn-panel'>Not enough labeled sequences for statistical testing (need ≥ 4).</div>", unsafe_allow_html=True)
    else:
        raw_features = [t3.extractFeatures(seq, label) for seq, label in seqs]
        # Filter out sequences that returned only {"class": label} (too short)
        raw_features = [f for f in raw_features if len(f) > 1]
        if len(raw_features) < 4:
            st.markdown("<div class='warn-panel'>Not enough sequences with extractable features for statistical testing.</div>", unsafe_allow_html=True)
        else:
            features_df = pd.DataFrame(raw_features)
            mann_table = t3.mannWhitney(features_df, n_show=15)
            st.dataframe(
                mann_table.style.format({"p-value": "{:.4f}", "Effect Size": "{:.3f}"}),
                hide_index=True, use_container_width=True, height=400
            )

# ──────────────────────────────────────────────────────────────────────────────────────────────────
# TAB 4: PREDICT PITCHES
# ──────────────────────────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(
        """
        <div class='info-panel'>
            <b>Live Inference Engine:</b> Upload a raw <code>.mov</code> or <code>.mp4</code> video of an unlabelled pitch.
            The pipeline will automatically extract the skeleton, isolate the throwing motion, and predict the pitch class using the model from Tab 03.
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if "single_pitcher_clf" not in st.session_state or "single_pitcher_scaler" not in st.session_state:
        st.markdown(
            f"""
            <div class='warn-panel'>
                ⚠ &nbsp; No Single-Pitcher model found in memory. Please go to Tab 03, select Single-Pitcher evaluation, and let the model train first.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown("<div class='section-label'>Upload Raw Video for Prediction</div>", unsafe_allow_html=True)
        pred_video = st.file_uploader("Upload an unlabelled pitching video", type=["mov", "mp4"], key="pred_upload")
        
        if pred_video is not None:
            col_pred1, col_pred2 = st.columns([1, 1], gap="large")
            with col_pred1:
                st.video(pred_video)
            
            with col_pred2:
                if st.button("▶ Run Full Inference Pipeline", type="primary", use_container_width=True):
                    pred_vid_target = os.path.join(base_dir, "Pitches", "temp_prediction.mov")
                    pred_kpt_target = os.path.join(base_dir, "Keypoints", "temp_prediction.npy")
                    os.makedirs(os.path.dirname(pred_vid_target), exist_ok=True)
                    
                    with open(pred_vid_target, "wb") as f:
                        f.write(pred_video.getbuffer())
                        
                    progress_bar = st.progress(0)
                    status_text  = st.empty()
                    
                    with st.spinner("Initializing models & tracking pose..."):
                        pose_m, glove_m = load_models()
                        success, meta = t1.processVideo(
                            pred_vid_target, pred_kpt_target, pose_m, glove_m, progress_bar, status_text
                        )
                        
                    if success:
                        progress_bar.empty()
                        status_text.empty()
                        st.markdown("<div class='success-panel'>✓ Skeleton successfully tracked and windowed. Analyzing kinematics...</div>", unsafe_allow_html=True)
                        
                        seq = np.load(pred_kpt_target, allow_pickle=True)
                        if isinstance(seq, np.ndarray) and seq.dtype == object:
                            seq = seq.tolist()
                            
                        # Extract features
                        try:
                            raw_feats = t3.extractFeatures(seq, label="Unknown")
                            if len(raw_feats) <= 1:
                                st.error("Extracted sequence was empty or malformed.")
                            else:
                                if "single_pitcher_clf" not in st.session_state:
                                    st.warning("⚠️ No active model found. Please run the model evaluation in **Tab 3** first to initialize the pitch classifier for this pitcher.")
                                else:
                                    clf = st.session_state["single_pitcher_clf"]
                                    scaler = st.session_state["single_pitcher_scaler"]
                                    feature_names = st.session_state["single_pitcher_features"]
                                
                                raw_df = pd.DataFrame([raw_feats])
                                X_test_raw = raw_df[feature_names].fillna(0).values
                                X_test_scaled = scaler.transform(X_test_raw)
                                
                                pred = clf.predict(X_test_scaled)[0]
                                probs = clf.predict_proba(X_test_scaled)[0]
                                
                                pred_class = "Fastball" if pred == 1 else "Offspeed"
                                max_prob = max(probs) * 100
                                
                                st.markdown(
                                    f"""
                                    <div class='stat-card' style='background: #1e3a8a; border-color: #3b82f6;'>
                                        <div class='label' style='color: #bae6fd;'>AI Pitch Prediction</div>
                                        <div class='value' style='font-size: 2.2rem;'>{pred_class} <span style='font-size: 1rem; color: #93c5fd; font-weight: normal;'>({max_prob:.1f}% Match)</span></div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                st.markdown("<div class='section-label' style='margin-top:2rem;'>Top Contributing Mechanics</div>", unsafe_allow_html=True)
                                
                                contribs = X_test_scaled[0] * clf.coef_[0]
                                contrib_data = []
                                for name, scaled_val, coef, contrib in zip(feature_names, X_test_scaled[0], clf.coef_[0], contribs):
                                    if coef != 0:
                                        direction = "Pushed toward Fastball" if contrib > 0 else "Pushed toward Offspeed"
                                        contrib_data.append({
                                            "Feature": name,
                                            "Standardized Size": scaled_val,
                                            "LogReg β Weight": coef,
                                            "Net Contribution": contrib,
                                            "Effect": direction
                                        })
                                        
                                if contrib_data:
                                    contrib_df = pd.DataFrame(contrib_data).sort_values(by="Net Contribution", key=abs, ascending=False).head(8)
                                    st.dataframe(
                                        contrib_df.style.format({
                                            "Standardized Size": "{:+.2f}", 
                                            "LogReg β Weight": "{:+.4f}", 
                                            "Net Contribution": "{:+.4f}"
                                        }).background_gradient(subset=["Net Contribution"], cmap="RdBu_r").set_properties(**{"font-family": "monospace"}, subset=["Feature", "Net Contribution"]),
                                        hide_index=True,
                                        use_container_width=True
                                    )
                        except Exception as e:
                            st.error(f"Error analyzing kinematics: {e}")
