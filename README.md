# PitchVision

**Automated Detection of Baseball Pitch Tipping via Motion Estimation and Statistical Modeling**

PitchVision is a Streamlit dashboard that processes baseball pitch videos, extracting biomechanical motion data using pose estimation and object tracking, before applying statistical and machine learning methods to identify subtle differences (tips) between fastballs and off‑speed pitches, then offering interactive superimposed visualizations for qualitative verification.

---

## Overview

Our Streamlit application consists of four tabs:
- **Tab 1: Motion Estimation**: Upload an `.mov`/`.mp4` pitch video to compute a pitcher's skeletal pose and glove bounding box (here in called keypoints) sequences (YOLOv8x‑Pose + custom glove YOLO + pitch windup truncation).
- **Tab 2: Comparative Visualization**: Select two keypoint sequences to generate a spatially and temporally aligned (DTW + RANSAC) superimposed visualization, with optional kinematic feature highlighting.
- **Tab 3: Pitch Classification**: Computes feature representations of keypoint sequences, then uses Mann-Whitney U-Testing, logistic regression, and decision trees to identify those features most discriminative in determining fastballs vs off-speeds.
- **Tab 4: Pitch Prediction**: Uses these trained models to predict classes for newly uploaded pitch videos.

## Setup

### Prerequisites
- Python ≥ 3.10
- [FFmpeg](https://ffmpeg.org/) (to transcode generated superimpositions in Tab 02, to make them visible in a browser: this uses libx264 coding, and may require manual setup)

### Dependencies
Create a virtual environment and install required packages:

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Model Weights
- `yolov8x-pose.pt`: will automatically be downloaded by Ultralytics during execution.
- `GLOVE_MODEL.pt`: our custom glove detection YOLO model.

### Usage
Simply run:
```bash
streamlit run pitchvision.py
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

