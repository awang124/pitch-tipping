import os
import cv2
import numpy as np

def detectGlove(model, frame, bounds, conf=0.05):
    """
    Detect the glove bounding box in a frame.

    Parameters
    ----------
    model: YOLO model for glove detection.
    frame: Image frame (numpy array).
    bounds: Optional (x1, y1, x2, y2) bounding box of the pitcher.
    conf: Confidence threshold.

    Returns
    -------
    Bounding box (x1, y1, x2, y2) or None if not found.
    """
    res = model.predict(frame, conf=conf, verbose=False)[0]
    if res.boxes is None:
        return None
    if not len(boxes := res.boxes.xyxy.cpu().numpy()):
        return None
    if not bounds:
        return boxes[0][:4]
    px1, py1, px2, py2 = bounds
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if px1 <= cx <= px2 and py1 <= cy <= py2:
            return x1, y1, x2, y2
    return boxes[0][:4]

def getThrowingArm(kpts):
    """
    Determine whether the pitcher is right‑handed by comparing
    horizontal ranges of right vs left wrist relative to shoulders.

    Parameters
    ----------
    kpts: Array of keypoints (T, 17, 2).

    Returns
    -------
    True if right‑handed, False otherwise.
    """
    relxr = kpts[:, 10, 0] - kpts[:, 6, 0]
    relxl = kpts[:,  9, 0] - kpts[:, 5, 0]
    for arr in (relxr, relxl):
        if np.any(np.isnan(arr)) and np.any(valid := ~np.isnan(arr)):
            arr[:] = np.interp(np.arange(len(arr)), np.where(valid)[0], arr[valid])
    ranger = np.nanmax(relxr) - np.nanmin(relxr) if not np.all(np.isnan(relxr)) else 0
    rangel = np.nanmax(relxl) - np.nanmin(relxl) if not np.all(np.isnan(relxl)) else 0
    return ranger > rangel

def getForwardDirection(relx):
    """
    Determine the forward direction of motion by comparing
    mean relative x‑positions in early vs late frames.

    Parameters
    ----------
    relx: Array of wrist‑shoulder x offsets over time.

    Returns
    -------
    1 if moving left‑to‑right, -1 if moving right‑to‑left.
    """
    if (end := int(0.9 * len(relx))) <= (start := int(0.1 * len(relx))):
        return 1
    smean = np.nanmean(relx[:start]) if not np.all(np.isnan(relx[:start])) else 0
    emean = np.nanmean(relx[end:])   if not np.all(np.isnan(relx[end:]))   else 0
    return 2 * (emean > smean) - 1

def findWindupEnd(kpts):
    """
    Find frame index where windup ends / release phase begins
    using throwing‑side wrist‑shoulder horizontal displacement.

    Parameters
    ----------
    kpts: Array of keypoints (T, 17, 2).

    Returns
    -------
    Frame index corresponding to the end of the windup.
    """
    arm = getThrowingArm(kpts)
    shoulder, wrist = (6, 10) if arm else (5, 9)
    shoulder, wrist = kpts[:, shoulder, 0], kpts[:, wrist, 0]
    relx = wrist - shoulder
    if np.any(np.isnan(relx)) and np.any(valid := ~np.isnan(relx)):
        relx = np.interp(np.arange(len(relx)), np.where(valid)[0], relx[valid])
    srel = np.convolve(relx * getForwardDirection(relx), np.ones(5) / 5, mode="same")
    if len(srel) <= (search_start := min(30, len(srel) - 1)):
        return 0
    minidx = np.argmin(srel[search_start:]) + search_start
    range_after = np.max(srel[minidx:]) - np.min(srel[minidx:])
    for i in range(minidx + 1, len(srel)):
        if srel[i] > srel[minidx] + 0.05 * range_after:
            return i
    return minidx

def findPitcher(persons, frame_shape):
    """
    Identify pitcher among detected person
    as person closest to center of frame.

    Parameters
    ----------
    persons: Array of detected person keypoints.
    frame_shape: (height, width, channels).

    Returns
    -------
    Index of the chosen person.
    """
    h, w, _ = frame_shape
    return np.argmin([
        np.min(np.linalg.norm(kpts - [w/2, h/2], axis=1))
        for kpts in persons.cpu().numpy()
    ])

def detectKeypoints(cap, pose_model, glove_model, progress, status):
    """
    Process a video frame‑by‑frame to extract
    pitcher keypoints and glove boxes.

    Parameters
    ----------
    cap: OpenCV VideoCapture object.
    pose_model: YOLO pose model.
    glove_model: YOLO glove model.
    progress: Streamlit progress bar.
    status: Streamlit status element.

    Returns
    -------
    Tuple (list of dicts with 'keypoints' and 'glove_box', total frame count).
    """
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fridx, kpts = 0, []
    while True:
        if not (frame := cap.read())[0]:
            break
        if not fridx % 10:
            progress.progress(min((fridx + 1) / nframes, 1.0))
            status.text(f"Process frame {fridx + 1} / {nframes}")

        ### COMPUTE POSES
        res = pose_model(frame[1], verbose=False)[0]
        if not (res.keypoints and len(persons := res.keypoints.xy)):
            fridx += 1
            continue

        ### ISOLATE PITCHER
        idx = findPitcher(persons, frame[1].shape)
        target_kpts = persons.cpu().numpy()[idx]
        try:
            xmin, ymin = np.min(target_kpts, axis=0)
            xmax, ymax = np.max(target_kpts, axis=0)
            pitcher_box = [xmin, ymin, xmax, ymax]
        except:
            pitcher_box = None

        ### TRACK GLOVE
        if (glove_box := detectGlove(glove_model, frame[1], pitcher_box)) is None:
            glove_box = [-1] * 4
        kpts.append({"keypoints": target_kpts, "glove_box": glove_box})
        fridx += 1
    return kpts, nframes

def processVideo(video_path, output_path, pose_model, glove_model, progress, status):
    """
    Run full extraction pipeline on a video file.

    Parameters
    ----------
    video_path: Path to input video.
    output_path: Path for the output .npy file.
    pose_model: YOLO pose model.
    glove_model: YOLO glove model.
    progress: Streamlit progress bar.
    status: Streamlit status text.

    Returns
    -------
    Tuple (success boolean, metadata dict).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    ### COMPUTE POSE AND GLOVE SEQUENCES
    kpts, nframes = detectKeypoints(cap, pose_model, glove_model, progress, status)
    cap.release()

    ### TRUNCATE TO UP TO END OF WINDUP
    status.text("Truncating pitch to windup")
    progress.progress(1.0)
    only_kpts = np.array([f["keypoints"] for f in kpts])
    end = findWindupEnd(only_kpts) + 1
    zero_f = {
        k: (np.zeros_like(v) if hasattr(v, "__len__") else 0) for k, v in kpts[0].items()
    }
    kpts = ([zero_f] * max(0, 180 - end)) + kpts[max(0, end - 180):end]

    ### SAVE RESULT AND METADATA
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.array(kpts))
    meta = {
        "num_total_frames": nframes,
        "num_extracted_frames": len(kpts),
        "fps": fps,
        "duration": int(nframes / fps)
    }
    return True, meta

