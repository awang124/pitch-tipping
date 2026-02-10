import cv2
import os
import numpy as np
import argparse
from ultralytics import YOLO
from dotenv import load_dotenv


#TO RUN: video-data-extraction/extract-keypoints --pitcher <pitcher>


# Program that takes in video from data directory (for now) 
# Current capacity is one pitcher, will have to figure how to store/use multiple pitcher's data 
# Converts the video into a numpy array of these elements at each frame in the video
        # Nose
        # Eyes (left, right)
        # Ears (left, right)
        # Shoulders (left, right)
        # Elbows (left, right)
        # Wrists (left, right)
        # Hips (left, right)
        # Knees (left, right)
        # Ankles (left, right)
# Numpy array easier for CNN to process for the most part I believe, might have to do something 
# else if getting precise with wrist rotation, or clothes or something
# Finally, outputs into data/keypoints





#input and output directories from .env
load_dotenv()
DATA_DIR = os.getenv("EXTRACTION_DATA_DIR")
if DATA_DIR is None:
    raise ValueError("Missing EXTRACTION_DATA_DIR in .env")

#get pitcher name from arguments
parser = argparse.ArgumentParser(description="Extract keypoints from videos.")
parser.add_argument("--pitcher", required=True, help="Pitcher name")
args = parser.parse_args()
PITCHER_NAME = args.pitcher

#final directories to a per pitcher basis
VIDEO_DIR = DATA_DIR + PITCHER_NAME
OUTPUT_DIR = DATA_DIR + PITCHER_NAME + "/keypoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#sample model for now
POSE_MODEL = "yolov8x-pose.pt"

#change based on frame rate, sample for now
FRAME_SKIP = 1


#load YOLO model ------------------------- might change!
model = YOLO(POSE_MODEL)
os.makedirs(OUTPUT_DIR, exist_ok=True)

#extract keypoints from a single video
def process_video(video_path, label):

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    keypoint_sequence = []

    #run through every frame of video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #skip frames if needed
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue

        #run pose model
        results = model(frame, verbose=False)
        kpts = results[0].keypoints

        #extract pitcher position into keypoint_sequence
        if kpts is not None and len(kpts.xy) > 0:
            person_kpts = kpts.xy[0].cpu().numpy()
            keypoint_sequence.append(person_kpts)

        #go next frame
        frame_idx += 1

    cap.release()

    #convert pose data to numpy array: (num_frames, num_keypoints, 2)
    #num_frames is total frames in video
    #num keypoints contain x and y coordinates of each body part in each frame
    keypoint_sequence = np.array(keypoint_sequence)

    #create output file to store with correct name as npy
    base = os.path.basename(video_path)
    name = os.path.splitext(base)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{label}_{name}.npy")
    np.save(out_path, keypoint_sequence)

    #confirmation message
    print(f"Saved keypoints: {out_path}  shape={keypoint_sequence.shape}")


def main():
    #loop through each VIDEO_DIR subfolder of different pitch types
    if not os.path.exists(VIDEO_DIR):
        raise FileNotFoundError(f"Data directory not found: {VIDEO_DIR}")
    for label in os.listdir(VIDEO_DIR):
        label_dir = os.path.join(VIDEO_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        #status message
        print(f"\nProcessing label: {label}")

        #for each video in each subfolder
        for filename in os.listdir(label_dir):

            #check that is valid video file
            if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue

            #get correct video path
            video_path = os.path.join(label_dir, filename)

            #status message
            print(f"  - {filename}")

            #process the video, should produce output file of data
            process_video(video_path, label)


if __name__ == "__main__":
    main()
