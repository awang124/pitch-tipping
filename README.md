# PitchVision

**Automated Detection of Baseball Pitch Tipping via Motion Estimation and Statistical Modeling**

PitchVision is a Streamlit dashboard that processes baseball pitch videos, extracting biomechanical motion data using pose estimation and object tracking, before applying statistical and machine learning methods to identify subtle differences (tips) between fastballs and off‑speed pitches, then offering interactive superimposed visualizations for qualitative verification.

---

## Overview

Our Streamlit application consists of six tabs:
- **Pitch Compare** (demo): Pick a fastball and offspeed pitch, generate a DTW-synced side-by-side or stacked comparison on real footage with joint highlighting and actionable coaching insights.
- **Tab 1: Motion Estimation**: Upload an `.mov`/`.mp4` pitch video to compute a pitcher's skeletal pose and glove bounding box (here in called keypoints) sequences (MediaPipe Pose Landmarker + pitcher tracking).
- **Tab 2: Comparative Visualization**: Select two keypoint sequences to generate a spatially and temporally aligned (DTW + RANSAC) superimposed visualization, with optional kinematic feature highlighting.
- **Tab 3: Pitch Classification**: Computes feature representations of keypoint sequences, then uses Mann-Whitney U-Testing, logistic regression, and decision trees to identify those features most discriminative in determining fastballs vs off-speeds.
- **Tab 4: Pitch Prediction**: Uses these trained models to predict classes for newly uploaded pitch videos.
- **Tab 5: Skeleton Overlay**: Renders skeleton keypoints overlaid on the full source video for a single pitch.

## Setup

### Prerequisites
- [Conda](https://docs.conda.io/) (Miniconda or Anaconda)
- Python ≥ 3.10
- [FFmpeg](https://ffmpeg.org/) (to transcode generated superimpositions in Tab 02, to make them visible in a browser: this uses libx264 coding, and may require manual setup)

### Dependencies
All libraries run inside the conda environment **`PitchTip`**:

```bash
conda env create -f environment.yml   # first time
conda activate PitchTip
conda env update -f environment.yml --prune   # if the env already exists
```

`requirements.txt` is kept for reference; prefer `environment.yml` for reproducible setup.

### Model Weights
Download the MediaPipe Pose Landmarker heavy model to the project root:

```bash
curl -L -o pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

Tab 1 uses `num_poses=2` plus pitcher-selection logic to track the foreground pitcher (not the umpire) on stretch landscape clips.

### Usage
```bash
conda activate PitchTip
python -m streamlit run app.py
```

Or use the launcher (avoids picking up system `streamlit` on macOS):

```bash
./run_app.sh
```

**Troubleshooting:** If you see `ModuleNotFoundError: No module named 'mediapipe'` while `(PitchTip)` is active, your shell is still calling system Streamlit (`/Library/Frameworks/Python.framework/...`). Verify with `which streamlit` and `which python` — both should be under `/opt/anaconda3/envs/PitchTip/bin/`. Use `python -m streamlit run app.py` or `./run_app.sh` instead.

### CLI scripts (batch audit / extraction / smoke tests)
```bash
conda activate PitchTip
python scripts/audit_keypoints.py
python scripts/batch_extract.py --help
python scripts/smoke_test.py
```

---

## Organization

### Code Hierarchy

```
├── app.py                   # Main
├── ui/
│   ├── style.py             # CSS String
│   ├── sidebar.py           # Dashboard Sidebar
│   └── components.py        # Dashboard Panels
├── tabs/
│   ├── estimation.py        # Tab 1 Main
│   ├── visualization.py     # Tab 2 Main
│   ├── classification.py    # Tab 3 Main
│   └── prediction.py        # Tab 4 Main
├── processing/
│   ├── tab1_helper.py       # Tab 1 Backend
│   ├── tab2_helper.py       # Tab 2 Backend
│   └── tab3_helper.py       # Tab 3/4 Backend
└── utils.py                 # Shared Auxiliary
```

### Data Hierarchy

Pitch videos and keypoint sequences are added during execution:

```
├── Pitches/
│   ├── Pitcher1/
│   │   ├── fastball_1.mov
│   │   └── ...
│   └── ...
├── Keypoints/
│   ├── Pitcher1/
│   │   ├── fastball_1.npy
│   │   └── ...
│   └── ...
└── Visualizations/          # From Tab 2
    ├── Pitcher1_fastball_1_vs_offspeed_1.mp4
    └── ...
```

