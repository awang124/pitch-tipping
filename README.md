# PitchVision

**Automated Detection of Baseball Pitch Tipping via Pose Estimation and Statistical Modeling**

PitchVision is a Streamlit dashboard that processes baseball pitch videos, extracting biomechanical motion data using pose estimation and object tracking, before applying statistical and machine learning methods to identify subtle differences (tips) between fastballs and off‑speed pitches, then offering interactive superimposed visualizations for qualitative verification.

---

## Overview

The repository contains the following core Python files:

### `pitchvision.py`  
Main Streamlit application with four tabs:
- **Tab 01: Motion Estimation**: Upload an `.mov`/`.mp4` pitch video to compute a pitcher's skeletal pose and glove bounding box (here in called keypoints) sequences (YOLOv8x‑Pose + custom glove YOLO + pitch windup truncation).
- **Tab 02: Comparative Analysis**: Select two keypoint sequences to generate a spatially and temporally aligned (DTW + RANSAC) superimposed visualization, with optional kinematic feature highlighting.
- **Tab 03: Pitch Classification**: Computes feature representations of keypoint sequences, then uses Mann-Whitney U-Testing, logistic regression, and decision trees to identify those features most discriminative in determining fastballs vs off-speeds.
- **Tab 04: Pitch Prediction**: Uses these trained models to predict classes for newly uploaded pitch videos.

## Setup

### Prerequisites
- Python ≥ 3.10
- [FFmpeg](https://ffmpeg.org/) (for video transcoding in Tab 02 only, which uses libx264 coding)

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

Directory structure after usage:

```
├── Pitches/
│   ├── Pitcher1/
│   │   ├── fastball1.mov
│   │   └── ...
│   └── Pitcher2/
│   │   ├── fastball1.mov
│   │   └── ...
├── Keypoints/
│   ├── Pitcher1/
│   │   ├── fastball1.npy
│   │   └── ...
│   └── Pitcher2/
│   │   ├── fastball1.npy
│   │   └── ...
├── Visualizations/       # Generated Superimpositions
└── (*.py, *.pt, etc.)
```

