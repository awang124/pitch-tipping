# SARG Pitch Tipping Project

### Overview
Pitch tipping occurs when a pitcher inadvertently reveals the type of pitch through a consistent change in their windup or release motion. Coaches and analysts spend hours manually reviewing footage to spot these tells. We automate this process by:  
- Extracting 17‑keypoint skeletons (COCO format) from pitching videos using YOLOv8x‑pose.
- Detecting the pitcher’s glove (mitt) with a custom YOLOv8 model.
- Automatically truncating each video to the critical 3‑second window before release.
- Computing over 100 kinematic features (angles, velocities, symmetries, glove motion).
- Using Random Forest and L1‑regularised logistic regression to find the most discriminating tells.
- Generating side‑by‑side DTW‑aligned animations that highlight mechanical differences between pitch types.

### Pipeline
1. Pose Estimation (`pose_estimation.py`): Runs YOLOv8x‑pose on each frame, identifies the pitcher, detects glove box, and saves raw keypoints.
2. Sequence Truncation (`pose_estimation.py`): Finds pitch release frame using wrist‑shoulder horizontal motion, then keeps the last 180 frames (~3 sec at 60 fps).
3. Feature Extraction (`pitch_classification.py`): Extracts 100+ kinematic features: joint angles, angular velocities, linear speeds, symmetry ratios, glove trajectories, etc.
4. Statistical Testing (`pitch_classification.py`): Mann‑Whitney U tests (Bonferroni corrected) to find features that differ significantly between pitch types.
5. Machine Learning (`pitch_classification.py`): Trains Random Forest and L1‑logistic regression to classify fastball vs. off‑speed; reports feature importances.
6. Visualization (`pose_visualization.py`): Aligns two pitch sequences via DTW, spatially registers them using RANSAC, and renders a superimposed video.
7. Dashboard (IN PROGRESS) (`dashboard/`): Streamlit app for uploading videos, running extraction, comparing pitches, and viewing ML results.
8. Decision Tree (IN PROGRESS) (`experimental/`): Per‑frame feature extraction (15 per frame) and interpretable decision tree for fastball/other classification.

### Setup
1. Clone Repostory:
 ```
git clone https://github.com/awang124/pitch-tipping
cd pitch-tipping
```
2. Create Virtual Environment: 
```
python -m venv venv
source venv/bin/activate
```
3. Install Dependencies: 
```
pip install -r requirements.txt
```
4. Get your Pitch Videos

### Usage
1. Pose Estimation & Truncation

Standalone:
 ```
python Pose-Estimation.py
```
Modify paths at the bottom of script to point to your video folder and output directory.

Inside Dashboard (Recommended):
 ```
streamlit run dashboard.py
```
Go to video upload tab, fill in pitcher name and pitch type, and click "Run Extraction Pipeline". The processed .npy file will be saved in Keypoints/.

2. Feature Extration & Classification

Standalone (from Keypoints directory): 
```
python pitch_classification.py
```

Inside Dashboard (Recommended):  
Go to ML modeling tab, dashboard will call `dashboard_classifier.py`, and display top features & model results.

3. Visualization

Standalone: 
```
python pitch_visualization.py
```

Inside Dashboard (Recommended):  
Go to visualization tab. Choose a feature group (e.g., “Right Elbow”) and click “Auto‑Select Extremal Matchup” – dashboard will automatically pick fastball and off‑speed pitch showing largest difference in that feature. 

It then creates a DTW‑aligned, RANSAC‑registered overlay video, showing reference skeleton in white and compared skeleton in blue, with option to highlight specific joints.

