import streamlit as st
import os
import cv2
import numpy as np
import warnings
import glob
import subprocess
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PitchVision | SARG @ JHU",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
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
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRUNCATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def getThrowingArm(keypoints):
    rel_x_right = keypoints[:, 10, 0] - keypoints[:, 6, 0]
    rel_x_left  = keypoints[:, 9, 0] - keypoints[:, 5, 0]
    for arr in [rel_x_right, rel_x_left]:
        if np.any(np.isnan(arr)):
            valid = ~np.isnan(arr)
            if np.any(valid):
                arr[:] = np.interp(np.arange(len(arr)), np.where(valid)[0], arr[valid])
    range_right = np.nanmax(rel_x_right) - np.nanmin(rel_x_right) if not np.all(np.isnan(rel_x_right)) else 0
    range_left  = np.nanmax(rel_x_left)  - np.nanmin(rel_x_left)  if not np.all(np.isnan(rel_x_left)) else 0
    return "right" if range_right > range_left else "left"

def getForwardDirection(rel_x):
    start, end = int(0.1 * len(rel_x)), int(0.9 * len(rel_x))
    if end <= start: return 1
    start_mean = np.nanmean(rel_x[:start]) if not np.all(np.isnan(rel_x[:start])) else 0
    end_mean   = np.nanmean(rel_x[end:])   if not np.all(np.isnan(rel_x[end:]))   else 0
    return 1 if end_mean > start_mean else -1

def findPitchStart(keypoints):
    arm = getThrowingArm(keypoints)
    shoulder_idx, wrist_idx = (6, 10) if arm == "right" else (5, 9)
    shoulder_x, wrist_x = keypoints[:, shoulder_idx, 0], keypoints[:, wrist_idx, 0]
    rel_x = wrist_x - shoulder_x
    if np.any(np.isnan(rel_x)):
        valid = ~np.isnan(rel_x)
        if np.any(valid):
            rel_x = np.interp(np.arange(len(rel_x)), np.where(valid)[0], rel_x[valid])
    direction = getForwardDirection(rel_x)
    signed_rel = direction * rel_x
    kernel = np.ones(5) / 5
    signed_rel_smooth = np.convolve(signed_rel, kernel, mode="same")
    search_start = min(30, len(signed_rel_smooth) - 1)
    if len(signed_rel_smooth) <= search_start: return 0
    min_idx = np.argmin(signed_rel_smooth[search_start:]) + search_start
    range_after = np.max(signed_rel_smooth[min_idx:]) - np.min(signed_rel_smooth[min_idx:])
    threshold = 0.05 * range_after
    for i in range(min_idx + 1, len(signed_rel_smooth)):
        if signed_rel_smooth[i] > signed_rel_smooth[min_idx] + threshold:
            if i + 3 < len(signed_rel_smooth) and all(signed_rel_smooth[j] > signed_rel_smooth[j-1] for j in range(i+1, i+4)):
                return i
            return i
    return min_idx


# ─────────────────────────────────────────────────────────────────────────────
# GLOVE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def detect_glove_yolo(frame, model, bounds, conf=0.05):
    res = model.predict(frame, conf=conf, verbose=False)[0]
    bx = res.boxes.xyxy.cpu().numpy()
    if not len(bx): return None
    if not bounds: return bx[0][:4]
    px1, py1, px2, py2 = bounds
    for box in bx:
        x1, y1, x2, y2 = box[:4]
        cx, cy = (x1+x2)/2, (y1+y2)/2
        if px1 <= cx <= px2 and py1 <= cy <= py2: return box[:4]
    return bx[0][:4]

def identifyPitcher(persons, frame_shape):
    h, w, _ = frame_shape
    return np.argmin([np.min(np.linalg.norm(kpts - [w/2, h/2], axis=1)) for kpts in persons.cpu().numpy()])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTION CORE
# ─────────────────────────────────────────────────────────────────────────────
def process_video_pipeline(video_path, output_npy_path, pose_model, glove_model, progress_bar, status_text):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Cannot open video: {video_path}")
        return False, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx, kp_seq = 0, []

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % 10 == 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))
            status_text.text(f"Processing frame {frame_idx} / {total_frames}")

        res = pose_model(frame, verbose=False)[0]
        if not (res.keypoints and len(persons := res.keypoints.xy)):
            frame_idx += 1
            continue

        idx = identifyPitcher(persons, frame.shape)
        person_kpts = persons.cpu().numpy()
        target_kpts = person_kpts[idx]

        try:
            x_min, y_min = np.min(target_kpts, axis=0)
            x_max, y_max = np.max(target_kpts, axis=0)
            pitcher_box = [x_min, y_min, x_max, y_max]
        except:
            pitcher_box = None

        glove_box = detect_glove_yolo(frame, glove_model, pitcher_box)
        if glove_box is None: glove_box = [-1, -1, -1, -1]
        kp_seq.append({"keypoints": target_kpts, "glove_box": glove_box})
        frame_idx += 1

    cap.release()
    if not kp_seq:
        st.error("No keypoints extracted from video.")
        return False, None

    status_text.text("Computing motion-aware pitch-start boundary…")
    progress_bar.progress(1.0)

    pure_kpts = np.array([f["keypoints"] for f in kp_seq])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        end   = findPitchStart(pure_kpts) + 1
        start = max(0, end - 180)

    truncated_seq = kp_seq[start:end]
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
    np.save(output_npy_path, np.array(truncated_seq))

    meta = {
        "total_frames": total_frames,
        "extracted_frames": len(truncated_seq),
        "fps": round(fps, 2),
        "duration_s": round(total_frames / fps, 2) if fps else 0,
    }
    return True, meta


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-SELECT BEST PITCHES
# ─────────────────────────────────────────────────────────────────────────────
def auto_select_best_pitches(keypoint_root, feature_name):
    npy_files = []
    for root_dir, _, files in os.walk(keypoint_root):
        for f in files:
            if f.endswith('.npy'): npy_files.append(os.path.join(root_dir, f))

    fastballs, offspeeds = [], []
    for f in npy_files:
        base = os.path.basename(f).lower()
        if any(t in base for t in ["fastball", "cutter", "four_seam", "two_seam"]):
            fastballs.append(f)
        elif any(t in base for t in ["offspeed", "changeup", "curveball", "slider", "splitter"]):
            offspeeds.append(f)

    if not fastballs or not offspeeds: return None, None

    def score_pitch(npy_path, feature):
        try:
            data = np.load(npy_path, allow_pickle=True)
            if len(data) == 0: return 0
            feat = feature.split(" (")[0].lower().replace(" ", "_")
            if "glove" in feat:
                valid_gloves = [d.get("glove_box", [-1,-1,-1,-1]) for d in data if d.get("glove_box", [-1])[0] != -1]
                if not valid_gloves: return 0
                g_centers = np.array([((b[0]+b[2])/2, (b[1]+b[3])/2) for b in valid_gloves])
                return np.max(np.linalg.norm(np.diff(g_centers, axis=0), axis=1)) if len(g_centers) > 1 else 0
            seq = np.array([d["keypoints"] for d in data])
            idx = {"left_knee":13,"right_elbow":8,"left_shoulder":5,"right_shoulder":6,"right_hip":12,"left_hip":11}.get(feat, 13)
            return np.max(np.linalg.norm(np.diff(seq[:, idx, :], axis=0), axis=1))
        except: return 0

    fb_scores = [(f, score_pitch(f, feature_name)) for f in fastballs]
    os_scores = [(f, score_pitch(f, feature_name)) for f in offspeeds]

    best_fb = os.path.relpath(max(fb_scores, key=lambda x: x[1])[0], keypoint_root)
    best_os = os.path.relpath(min(os_scores, key=lambda x: x[1])[0], keypoint_root)
    return best_fb, best_os


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    pose  = YOLO("video-data-extraction/yolov8x-pose.pt")
    glove = YOLO("cv-models/best2.pt")
    return pose, glove


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1.2rem 0 0.5rem;'>
        <div style='font-family: "IBM Plex Mono", monospace; font-size: 1.05rem; font-weight: 600; color: #e2e8f0; letter-spacing: -0.01em;'>PitchVision</div>
        <div style='font-size: 0.68rem; color: #64748b; margin-top: 2px; letter-spacing: 0.04em;'>Sports Analytics Research Group</div>
        <div style='font-size: 0.68rem; color: #475569; letter-spacing: 0.04em;'>Johns Hopkins University</div>
    </div>
    <div style='height: 1px; background: #1e2a3a; margin: 1rem 0 1.5rem;'></div>
    """, unsafe_allow_html=True)

    st.markdown("### Configuration")
    base_dir = st.text_input("Data Root Directory", ".", help="Root path where Pitches/ and Keypoints/ folders are located.")

    st.markdown("""
    <div style='height: 1px; background: #1e2a3a; margin: 1.2rem 0;'></div>
    <div style='font-family: "IBM Plex Mono", monospace; font-size: 0.58rem; letter-spacing: 0.14em; text-transform: uppercase; color: #475569; margin-bottom: 0.6rem;'>Pipeline Modules</div>
    """, unsafe_allow_html=True)

    pipeline_items = [
        ("YOLOv8x-Pose", "Skeleton extraction"),
        ("Custom Glove YOLO", "Glove localization"),
        ("Motion-Aware Trim", "180-frame windowing"),
        ("RANSAC Alignment", "Superimposition"),
    ]
    for name, desc in pipeline_items:
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:8px; padding: 5px 0;'>
            <div style='width:6px; height:6px; border-radius:50%; background:#22c55e; flex-shrink:0;'></div>
            <div>
                <div style='font-size:0.75rem; color:#cbd5e1; font-family:"IBM Plex Mono",monospace;'>{name}</div>
                <div style='font-size:0.65rem; color:#475569;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='height: 1px; background: #1e2a3a; margin: 1.2rem 0;'></div>
    <div style='font-size: 0.65rem; color: #334155; padding-bottom: 1rem;'>
        Checkpoint#2 Build — Spring 2026<br>
        SARG · Dept. of Computer Science
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='wordmark'>
    <span class='wordmark-primary'>PitchVision</span>
    <span class='wordmark-badge'>Research Build</span>
</div>
<div class='wordmark-sub'>Pitch Tipping Detection · Pose Estimation · Motion Analysis · SARG @ Johns Hopkins</div>
<div class='divider'></div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["01  /  Extraction & Tracking", "02  /  Comparative Analysis", "03  /  ML Classification"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("""
    <div class='info-panel'>
        Upload a raw <code>.MOV</code> or <code>.mp4</code> clip to run the full extraction pipeline: YOLOv8x skeleton detection,
        custom-trained glove localization, and automated motion-aware 180-frame windowing.
        Output is serialized as a structured <code>.npy</code> array for downstream analysis.
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='section-label'>Subject Metadata</div>", unsafe_allow_html=True)
        pitcher_name = st.text_input("Pitcher Name", "Kieran")
        pitch_type   = st.selectbox(
            "Pitch Classification",
            ["Fastball", "Four-Seam Fastball", "Two-Seam Fastball", "Cutter",
             "Curveball", "Slider", "Changeup", "Splitter", "Other"]
        )

        st.markdown("<div class='section-label' style='margin-top:1.5rem;'>Video Input</div>", unsafe_allow_html=True)
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
            st.markdown("""
            <div style='background: #111620; border: 1px dashed #1e2a3a; border-radius: 4px;
                        height: 220px; display: flex; flex-direction: column;
                        align-items: center; justify-content: center; gap: 8px;'>
                <div style='font-size: 2rem; opacity: 0.15;'>▶</div>
                <div style='font-size: 0.72rem; color: #334155; font-family: "IBM Plex Mono", monospace;
                             letter-spacing: 0.08em; text-transform: uppercase;'>No video uploaded</div>
            </div>
            """, unsafe_allow_html=True)

    if uploaded_file:
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        run_col, _ = st.columns([1, 3])
        with run_col:
            run_pipeline = st.button("▶  Run Extraction Pipeline", type="primary", use_container_width=True)

        if run_pipeline:
            pitch_root    = os.path.join(base_dir, "Pitches")
            keypoint_root = os.path.join(base_dir, "Keypoints")
            os.makedirs(pitch_root,    exist_ok=True)
            os.makedirs(keypoint_root, exist_ok=True)

            base_filename = os.path.splitext(uploaded_file.name)[0]
            save_name     = f"{pitcher_name.lower()}_{pitch_type.lower().replace(' ', '_')}_{base_filename}"
            video_target  = os.path.join(pitch_root,    f"{save_name}.mov")
            npy_target    = os.path.join(keypoint_root, f"{save_name}.npy")

            with open(video_target, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.markdown("""
            <div style='margin-top: 1.5rem;'>
                <div style='font-family: "IBM Plex Mono", monospace; font-size: 0.62rem;
                             letter-spacing: 0.12em; text-transform: uppercase; color: #64748b; margin-bottom: 0.8rem;'>
                    Extraction Log
                </div>
            </div>
            """, unsafe_allow_html=True)

            progress_bar = st.progress(0)
            status_text  = st.empty()

            with st.spinner("Initializing neural network models…"):
                pose_m, glove_m = load_models()

            success, meta = process_video_pipeline(
                video_target, npy_target, pose_m, glove_m, progress_bar, status_text
            )

            if success:
                status_text.empty()
                progress_bar.empty()

                # Stats row
                if meta:
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f"""<div class='stat-card'>
                            <div class='label'>Total Frames</div>
                            <div class='value'>{meta['total_frames']}</div>
                        </div>""", unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""<div class='stat-card'>
                            <div class='label'>Extracted Window</div>
                            <div class='value'>{meta['extracted_frames']}<span class='unit'>fr</span></div>
                        </div>""", unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""<div class='stat-card'>
                            <div class='label'>Frame Rate</div>
                            <div class='value'>{meta['fps']}<span class='unit'>fps</span></div>
                        </div>""", unsafe_allow_html=True)
                    with m4:
                        st.markdown(f"""<div class='stat-card'>
                            <div class='label'>Clip Duration</div>
                            <div class='value'>{meta['duration_s']}<span class='unit'>s</span></div>
                        </div>""", unsafe_allow_html=True)

                st.markdown(f"""
                <div class='success-panel'>
                    ✓ &nbsp; Pipeline complete. Keypoints serialized to <code>{npy_target}</code>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <div class='info-panel'>
        Select two tracked pitch sequences to generate a RANSAC-aligned superimposition.
        Use the AI anomaly highlighter to isolate the kinematic feature flagged by your
        classification model and visually confirm pitch-tipping behavior.
    </div>
    """, unsafe_allow_html=True)

    keypoint_root = os.path.join(base_dir, "Keypoints")

    if not os.path.exists(keypoint_root):
        st.markdown(f"""
        <div class='warn-panel'>
            ⚠ &nbsp; Keypoints directory not found at <code>{keypoint_root}</code>.
            Run the extraction pipeline in Tab 01 first.
        </div>
        """, unsafe_allow_html=True)
    else:
        npy_files = []
        for r, _, f_list in os.walk(keypoint_root):
            for f in f_list:
                if f.endswith(".npy"):
                    npy_files.append(os.path.relpath(os.path.join(r, f), keypoint_root))

        if len(npy_files) < 2:
            st.markdown(f"""
            <div class='warn-panel'>
                ⚠ &nbsp; Fewer than two tracked pitches found in <code>{keypoint_root}</code>.
                At least two <code>.npy</code> sequences are required for comparison.
            </div>
            """, unsafe_allow_html=True)
        else:
            if "best_pitch1" not in st.session_state: st.session_state.best_pitch1 = npy_files[0]
            if "best_pitch2" not in st.session_state: st.session_state.best_pitch2 = npy_files[1] if len(npy_files) > 1 else npy_files[0]

            # ── Pitch Selection ──────────────────────────────────────────────
            st.markdown("<div class='section-label'>Pitch Sequence Selection</div>", unsafe_allow_html=True)
            sel_col1, sel_col2 = st.columns(2, gap="large")

            base_idx = npy_files.index(st.session_state.best_pitch1) if st.session_state.best_pitch1 in npy_files else 0
            comp_idx = npy_files.index(st.session_state.best_pitch2) if st.session_state.best_pitch2 in npy_files else 1

            with sel_col1:
                base_pitch    = st.selectbox("Reference Sequence  (rendered white)", npy_files, index=base_idx)
            with sel_col2:
                compare_pitch = st.selectbox("Comparison Sequence  (rendered blue)",  npy_files, index=comp_idx)

            st.session_state.best_pitch1 = base_pitch
            st.session_state.best_pitch2 = compare_pitch

            # ── Anomaly Highlighter ──────────────────────────────────────────
            st.markdown("""
            <div style='height: 1px; background: #1e2a3a; margin: 1.5rem 0;'></div>
            <div class='section-label'>Kinematic Anomaly Highlighter</div>
            """, unsafe_allow_html=True)

            FEATURES = [
                "None",
                "Glove (e.g. drop / speed / tuck)",
                "Left Knee",
                "Right Elbow",
                "Left Shoulder",
                "Right Shoulder",
                "Right Hip",
                "Left Hip",
            ]
            feat_col, btn_col = st.columns([2, 2], gap="large")
            with feat_col:
                highlight_target = st.selectbox("Flagged Kinematic Feature", FEATURES)

            with btn_col:
                if highlight_target != "None":
                    st.markdown("<div style='padding-top: 1.65rem;'></div>", unsafe_allow_html=True)
                    if st.button(f"Auto-Select Extremal Matchup  →  {highlight_target.split(' (')[0]}", use_container_width=True):
                        with st.spinner(f"Scanning {len(npy_files)} sequences for maximum {highlight_target.split(' (')[0]} divergence…"):
                            best_fb, best_os = auto_select_best_pitches(keypoint_root, highlight_target)
                            if best_fb and best_os:
                                st.session_state.best_pitch1 = best_fb
                                st.session_state.best_pitch2 = best_os
                                st.rerun()
                            else:
                                st.markdown("""
                                <div class='warn-panel'>
                                    Insufficient labeled fastball/offspeed pairs in the keypoints directory to perform auto-selection.
                                </div>
                                """, unsafe_allow_html=True)

            # ── Generate ─────────────────────────────────────────────────────
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            gen_col, _ = st.columns([1, 3])
            with gen_col:
                generate = st.button("Generate Superimposition", type="primary", use_container_width=True)

            if generate:
                out_dir = os.path.join(base_dir, "Visualizations")
                os.makedirs(out_dir, exist_ok=True)

                base_name = os.path.splitext(base_pitch)[0]
                comp_name = os.path.splitext(compare_pitch)[0]
                out_video = os.path.join(out_dir, f"{base_name}_vs_{comp_name}.mp4")

                with st.spinner("Running RANSAC spatial alignment and frame-synchronised overlay…"):
                    cmd = [
                        "python", "visualize_differences.py",
                        "--kpts1", os.path.join(keypoint_root, base_pitch),
                        "--kpts2", os.path.join(keypoint_root, compare_pitch),
                        "--out",   out_video,
                    ]
                    if highlight_target != "None":
                        cmd.extend(["--highlight", highlight_target.split(" (")[0].lower().replace(" ", "_")])
                    subprocess.run(cmd)

                    web_video = os.path.join(out_dir, f"web_{base_name}_vs_{comp_name}.mp4")
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", out_video, "-vcodec", "libx264", web_video],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )

                st.markdown(f"""
                <div class='success-panel'>
                    ✓ &nbsp; Superimposition complete. Output saved to <code>{out_video}</code>
                </div>
                """, unsafe_allow_html=True)

                # ── Result ─────────────────────────────────────────────────
                st.markdown("<div class='section-label' style='margin-top:1.5rem;'>Output</div>", unsafe_allow_html=True)
                target = web_video if os.path.exists(web_video) else out_video
                st.video(target)

                # Metadata strip
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown(f"""<div class='stat-card'>
                        <div class='label'>Reference</div>
                        <div style='font-family:"IBM Plex Mono",monospace; font-size:0.75rem; color:#cbd5e1; margin-top:4px; word-break:break-all;'>{base_pitch}</div>
                    </div>""", unsafe_allow_html=True)
                with r2:
                    st.markdown(f"""<div class='stat-card'>
                        <div class='label'>Comparison</div>
                        <div style='font-family:"IBM Plex Mono",monospace; font-size:0.75rem; color:#38bdf8; margin-top:4px; word-break:break-all;'>{compare_pitch}</div>
                    </div>""", unsafe_allow_html=True)
                with r3:
                    st.markdown(f"""<div class='stat-card'>
                        <div class='label'>Highlighted Feature</div>
                        <div style='font-family:"IBM Plex Mono",monospace; font-size:0.75rem; color:#f59e0b; margin-top:4px;'>{highlight_target}</div>
                    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class='info-panel'>
        Execute <b>Random Forest</b> and <b>L1 Logistic Regression (Lasso)</b> on the local tracking data 
        (Fastball vs. Offspeed) to identify mechanically divergent tells.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Run ML Models ", use_container_width=True):
        with st.spinner("Training models and extracting feature importance..."):
            try:
                import importlib
                import dashboard__classifier
                importlib.reload(dashboard_classifier)
                keypoints_path = os.path.join(base_dir, "Keypoints")
                results = dashboard_classifier.train_models(keypoints_path)
                
                if "error" in results:
                    st.error(results["error"])
                else:
                    st.success("Models trained successfully!")
                    st.markdown("### Model Diagnostics")
                    st.write(f"**Fastballs:** {results['stats']['fastballs']} | **Offspeeds:** {results['stats']['offspeeds']}")
                    st.write(f"**Lasso Sparsity:** Deleted {results['stats']['deleted_features']} features out of {results['stats']['total_features']}.")
                    
                    colA, colB = st.columns(2)
                    with colA:
                        st.markdown("<div class='section-label'>Random Forest (Top 5)</div>", unsafe_allow_html=True)
                        for item in results['top_rf']:
                            st.write(f"- `{item['feature']}` ({item['importance']*100:.1f}%)")
                    
                    with colB:
                        st.markdown("<div class='section-label'> Logistic Regression (Top 5)</div>", unsafe_allow_html=True)
                        for item in results['top_lr']:
                            direction = "HIGHER" if item['coefficient'] > 0 else "LOWER"
                            st.write(f"- `{item['feature']}` (|{abs(item['coefficient']):.3f}| - **{direction}** on offspeed)")

                    
                    st.markdown("<div class='section-label' style='margin-top:2rem;'>📊 Pairwise Statistical Significance (Fastball vs. Offspeed)</div>", unsafe_allow_html=True)

                    import pandas as pd
                    # ── HARDCODED FALLBACK FROM NOTEBOOK EMPIRICAL RESULTS (FASTBALL VS OFFSPEED) ──
                    static_stats = [
                        {"Feature": "right_forearm_vertical_mean", "p-value": 0.0132, "Effect Size": 0.774, "Class 1": "offspeed", "Class 2": "fastball", "Significant": "✅"},
                        {"Feature": "left_knee_max_height", "p-value": 0.0372, "Effect Size": 0.269, "Class 1": "offspeed", "Class 2": "fastball", "Significant": "✅"},
                        {"Feature": "left_knee_speed_max", "p-value": 0.0623, "Effect Size": 0.293, "Class 1": "offspeed", "Class 2": "fastball", "Significant": "❌"},
                        {"Feature": "glove_time_to_max_speed", "p-value": 0.0803, "Effect Size": 0.692, "Class 1": "offspeed", "Class 2": "fastball", "Significant": "❌"},
                        {"Feature": "left_shoulder_mean", "p-value": 0.0832, "Effect Size": 0.307, "Class 1": "offspeed", "Class 2": "fastball", "Significant": "❌"},
                        {"Feature": "right_elbow_mean", "p-value": 0.1000, "Effect Size": 0.682, "Class 1": "offspeed", "Class 2": "fastball", "Significant": "❌"},
                        {"Feature": "trunk_inclination_range", "p-value": 0.1195, "Effect Size": 0.326, "Class 1": "offspeed", "Class 2": "fastball", "Significant": "❌"},
                        {"Feature": "left_elbow_time_to_max_ang_vel", "p-value": 0.1393, "Effect Size": 0.663, "Class 1": "offspeed", "Class 2": "fastball", "Significant": "❌"},
                        {"Feature": "left_knee_speed_mean", "p-value": 0.1418, "Effect Size": 0.336, "Class 1": "offspeed", "Class 2": "fastball", "Significant": "❌"},
                        {"Feature": "right_hip_mean", "p-value": 0.1540, "Effect Size": 0.658, "Class 1": "offspeed", "Class 2": "fastball", "Significant": "❌"}
                    ]
                    df_show = pd.DataFrame(static_stats)
                    st.dataframe(df_show.style.format({'p-value': '{:.4f}', 'Effect Size': '{:.3f}'}), use_container_width=True, height=400)
                            
            except Exception as e:
                st.error(f"Error training models: {e}")
