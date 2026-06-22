import os
import cv2
import numpy as np

from processing.kpt_utils import (
    HYSTERESIS_MARGIN,
    JITTER_RETRY_THRESHOLD,
    TRACK_SWITCH_MAX_PX,
    WARMUP_FRAMES,
    computeArmMotionDelta,
    computeQualityMetrics,
    findDeliverySyncFrame,
    isStretchLandscapePath,
    resolvePeakKneeFrame,
    resolveWindupEnd,
    saveKeypointMeta,
)
from processing.mediapipe_pose import detect_image_frame, detect_video_frame, image_landmarker
from processing.pose_adapter import glove_box_from_landmarks, poses_to_coco_batch

MAX_HOLD_FRAMES = 3
SLOT_MATCH_MAX_PX = 120
NUM_POSES = 2


def _personAnchor(kpts):
    """Hip midpoint, else shoulder midpoint, else keypoint mean."""
    if (kpts[11, 0] != 0 or kpts[11, 1] != 0) and (kpts[12, 0] != 0 or kpts[12, 1] != 0):
        return (kpts[11] + kpts[12]) / 2
    if (kpts[5, 0] != 0 or kpts[5, 1] != 0) and (kpts[6, 0] != 0 or kpts[6, 1] != 0):
        return (kpts[5] + kpts[6]) / 2
    valid = kpts[(kpts[:, 0] != 0) | (kpts[:, 1] != 0)]
    return np.mean(valid, axis=0) if len(valid) else np.array([0.0, 0.0])


def _personBBox(kpts):
    valid = kpts[(kpts[:, 0] != 0) | (kpts[:, 1] != 0)]
    if len(valid) == 0:
        return 0.0, 0.0, 0.0, 0.0
    xmin, ymin = np.min(valid, axis=0)
    xmax, ymax = np.max(valid, axis=0)
    return float(xmin), float(ymin), float(xmax), float(ymax)


def _bboxArea(kpts):
    xmin, ymin, xmax, ymax = _personBBox(kpts)
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    return (xmax - xmin) * (ymax - ymin)


def scorePitcherCandidate(
    kpts, frame_shape, prev_anchor=None, conf=None,
    camera_hint="default", motion_only=False, area_relative=None,
):
    """
    Score a detected person as pitcher candidate.
    Higher = more likely pitcher (not umpire/catcher).
    """
    h, w, _ = frame_shape
    anchor = _personAnchor(kpts)
    xmin, ymin, xmax, ymax = _personBBox(kpts)
    if xmax <= xmin or ymax <= ymin:
        return -1.0

    if motion_only:
        return 0.0

    area = (xmax - xmin) * (ymax - ymin)
    area_frac = area / max(w * h, 1)
    if camera_hint == "stretch_landscape" and area_frac < 0.015:
        return -2.0

    area_weight = 4.0 if camera_hint == "stretch_landscape" else 3.0
    area_score = area_frac * area_weight
    if camera_hint == "stretch_landscape" and area_relative is not None:
        area_score += float(area_relative) * 6.0

    bbox_h = ymax - ymin
    bbox_w = xmax - xmin
    aspect = bbox_h / max(bbox_w, 1)
    aspect_score = 0.3 if 1.0 < aspect < 4.5 else -0.4

    cx, cy = float(anchor[0]), float(anchor[1])
    mound_score = 0.0
    center_penalty = 0.0

    if camera_hint == "stretch_landscape":
        if cx > 0.45 * w:
            mound_score = 0.8 * min(1.0, (cx - 0.45 * w) / (0.25 * w))
        if cx < 0.40 * w:
            mound_score -= 1.0 * (1.0 - cx / max(0.40 * w, 1))
    elif w >= h:
        edge_dist = min(cx, w - cx) / max(w / 2, 1)
        mound_score = edge_dist * 0.5
        center_dist = np.linalg.norm(anchor - np.array([w / 2, h / 2]))
        max_dist = np.sqrt(w * w + h * h) / 2
        center_penalty = -0.25 * (1.0 - center_dist / max(max_dist, 1))
    else:
        mound_score = max(0.0, (h * 0.55 - cy) / max(h * 0.55, 1)) * 0.5

    temp_score = 0.0
    if prev_anchor is not None:
        dist = float(np.linalg.norm(anchor - prev_anchor))
        temp_score = max(0.0, 1.0 - dist / 250.0) * 0.8

    conf_score = float(conf) * 0.15 if conf is not None else 0.0
    return area_score + aspect_score + mound_score + temp_score + center_penalty + conf_score


def _updateMotionSlots(slots, persons_np, frame_idx):
    """Match detections to persistent slots and accumulate arm motion."""
    used_slots = set()
    person_slot_map = {}

    for i, kpts in enumerate(persons_np):
        anchor = _personAnchor(kpts)
        best_slot, best_dist = None, SLOT_MATCH_MAX_PX
        for slot in slots:
            if slot["id"] in used_slots:
                continue
            dist = float(np.linalg.norm(anchor - slot["anchor"]))
            if dist < best_dist:
                best_slot, best_dist = slot, dist

        if best_slot is None:
            best_slot = {
                "id": len(slots),
                "anchor": anchor.copy(),
                "last_kpts": kpts.copy(),
                "motion": 0.0,
                "last_seen": frame_idx,
                "areas": [_bboxArea(kpts)],
            }
            slots.append(best_slot)
        else:
            best_slot["motion"] += computeArmMotionDelta(best_slot["last_kpts"], kpts)
            best_slot["anchor"] = anchor.copy()
            best_slot["last_kpts"] = kpts.copy()
            best_slot["last_seen"] = frame_idx
            best_slot.setdefault("areas", []).append(_bboxArea(kpts))
            used_slots.add(best_slot["id"])

        person_slot_map[i] = best_slot["id"]

    return person_slot_map


def _pickLockedSlot(slots, frame_shape, camera_hint):
    """After warmup, lock onto largest right-side slot (pitcher foreground)."""
    if not slots:
        return None
    h, w, _ = frame_shape

    if camera_hint == "stretch_landscape":
        right_slots = [s for s in slots if s["anchor"][0] > 0.40 * w]
        pool = right_slots if right_slots else slots
        return max(
            pool,
            key=lambda s: (
                np.median(s.get("areas", [_bboxArea(s["last_kpts"])])),
                scorePitcherCandidate(
                    s["last_kpts"], frame_shape, camera_hint=camera_hint,
                ),
            ),
        )

    viable = []
    for slot in slots:
        pos = scorePitcherCandidate(
            slot["last_kpts"], frame_shape, camera_hint=camera_hint,
        )
        if pos >= 0.2:
            viable.append((slot, pos, slot["motion"]))
    if not viable:
        return max(
            slots,
            key=lambda s: scorePitcherCandidate(
                s["last_kpts"], frame_shape, camera_hint=camera_hint,
            ),
        )
    return max(viable, key=lambda x: x[1] * 100.0 + x[2])[0]


def _countMultiPersonFrames(cap, landmarker, max_frames=30, fps=30.0):
    """Count warmup frames with 2+ detected poses."""
    count = 0
    checked = 0
    while checked < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        ts_ms = int(checked * 1000.0 / max(fps, 1))
        poses = detect_video_frame(landmarker, frame, ts_ms)
        if len(poses) >= 2:
            count += 1
        checked += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return count


def warmupMultiPoseCount(model_path, cap, fps=30.0, max_frames=WARMUP_FRAMES):
    """Run warmup multi-pose check in a separate landmarker session."""
    from processing.mediapipe_pose import video_landmarker

    with video_landmarker(model_path, num_poses=NUM_POSES) as landmarker:
        return _countMultiPersonFrames(cap, landmarker, max_frames, fps=fps)


def findPitcherTracked(
    persons_np, confs_np, frame_shape, state,
    person_slot_map=None,
):
    """Pick pitcher using scoring, motion lock, and hysteresis."""
    if len(persons_np) == 1:
        state["anchor"] = _personAnchor(persons_np[0])
        state["idx"] = 0
        return 0

    camera_hint = state.get("camera_hint", "default")
    motion_only = state.get("motion_only", False)
    prev_anchor = state.get("anchor")
    locked_slot_id = state.get("locked_slot_id")

    person_areas = [_bboxArea(kpts) for kpts in persons_np]
    max_area = max(person_areas) if person_areas else 1.0

    scores = []
    anchors = []
    for i, kpts in enumerate(persons_np):
        conf = float(np.mean(confs_np[i])) if confs_np is not None and len(confs_np[i]) else None
        area_rel = person_areas[i] / max_area if max_area > 0 else 1.0
        pos_score = scorePitcherCandidate(
            kpts, frame_shape, prev_anchor, conf,
            camera_hint=camera_hint, motion_only=motion_only,
            area_relative=area_rel,
        )
        motion_bonus = 0.0
        if person_slot_map and i in person_slot_map and state.get("slots"):
            slot = next(
                (s for s in state["slots"] if s["id"] == person_slot_map[i]), None
            )
            if slot:
                motion_bonus = slot["motion"] * (0.01 if motion_only else 0.002)
        scores.append(pos_score + motion_bonus)
        anchors.append(_personAnchor(kpts))

    if motion_only and state.get("slots"):
        h, w, _ = frame_shape
        for i, kpts in enumerate(persons_np):
            if person_slot_map and i in person_slot_map:
                slot = next(
                    (s for s in state["slots"] if s["id"] == person_slot_map[i]), None
                )
                if slot:
                    anchor = slot["anchor"]
                    median_area = float(
                        np.median(slot.get("areas", [person_areas[i]]))
                    )
                    scores[i] = slot["motion"] + median_area * 0.001
                    if camera_hint == "stretch_landscape" and anchor[0] < 0.40 * w:
                        scores[i] *= 0.1

    best_idx = int(np.argmax(scores))

    if camera_hint == "stretch_landscape" and not motion_only:
        h, w, _ = frame_shape
        right_valid = [
            i for i, s in enumerate(scores)
            if s > 0 and anchors[i][0] > 0.40 * w
        ]
        if right_valid:
            best_idx = max(right_valid, key=lambda i: scores[i])
        else:
            valid = [i for i, s in enumerate(scores) if s > 0]
            if valid:
                best_idx = max(valid, key=lambda i: scores[i])

    if locked_slot_id is not None and person_slot_map and state.get("use_motion_lock"):
        locked_indices = [
            i for i, sid in person_slot_map.items() if sid == locked_slot_id
        ]
        if locked_indices:
            locked_idx = locked_indices[0]
            locked_score = scores[locked_idx]
            if best_idx != locked_idx:
                if scores[best_idx] < locked_score + HYSTERESIS_MARGIN:
                    best_idx = locked_idx
                elif float(np.linalg.norm(anchors[best_idx] - anchors[locked_idx])) > TRACK_SWITCH_MAX_PX:
                    best_idx = locked_idx

    state["anchor"] = anchors[best_idx]
    state["idx"] = best_idx
    return best_idx


def _interpolateFrames(prev_entry, next_entry, alpha):
    """Linear blend between two frame dicts."""
    prev_k, next_k = prev_entry["keypoints"], next_entry["keypoints"]
    blended = prev_k * (1 - alpha) + next_k * alpha
    glove = prev_entry["glove_box"]
    if next_entry["glove_box"][0] != -1 and alpha > 0.5:
        glove = next_entry["glove_box"]
    prev_vf = prev_entry.get("video_frame", -1)
    next_vf = next_entry.get("video_frame", -1)
    if prev_vf >= 0 and next_vf >= 0:
        video_frame = int(round(prev_vf * (1 - alpha) + next_vf * alpha))
    elif next_vf >= 0:
        video_frame = next_vf
    else:
        video_frame = prev_vf
    return {
        "keypoints": blended,
        "glove_box": glove,
        "interpolated": True,
        "video_frame": video_frame,
    }


def _hipJitterFromKpts(kpts_list):
    """Quick hip-center jitter on a list of frame dicts."""
    anchors = []
    for entry in kpts_list:
        k = entry["keypoints"]
        if (k[11, 0] != 0 or k[11, 1] != 0) and (k[12, 0] != 0 or k[12, 1] != 0):
            anchors.append((k[11] + k[12]) / 2)
    if len(anchors) < 6:
        return 999.0
    anchors = np.array(anchors)
    return float(np.std(anchors, axis=0).mean())


def detectKeypoints(
    cap, landmarker, progress, status,
    video_path=None, track_options=None, fps=30.0,
    use_motion_lock=False,
):
    """Process video frame-by-frame with MediaPipe pose + pitcher tracking."""
    track_options = track_options or {}
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stretch_landscape = isStretchLandscapePath(
        video_path, frame_width, frame_height
    )
    camera_hint = (
        "stretch_landscape" if stretch_landscape else "default"
    )

    fridx, kpts = 0, []
    multi_pose_frames = 0
    track_state = {
        "camera_hint": camera_hint,
        "motion_only": track_options.get("motion_only", False),
        "slots": [],
        "locked_slot_id": None,
        "use_motion_lock": use_motion_lock,
    }
    miss_streak = 0
    last_valid = None
    gap_start = None

    while True:
        if not (frame := cap.read())[0]:
            break
        if not fridx % 10:
            progress.progress(min((fridx + 1) / nframes, 1.0))
            status.text(f"Process frame {fridx + 1} / {nframes}")

        ts_ms = int(fridx * 1000.0 / max(fps, 1))

        mp_poses = detect_video_frame(landmarker, frame[1], ts_ms)
        if len(mp_poses) >= 2:
            multi_pose_frames += 1

        persons_np, confs_np = poses_to_coco_batch(
            mp_poses, frame_width, frame_height
        )

        if len(persons_np) == 0:
            if last_valid is not None and miss_streak < MAX_HOLD_FRAMES:
                entry = {
                    "keypoints": last_valid["keypoints"].copy(),
                    "glove_box": last_valid["glove_box"],
                    "interpolated": True,
                    "video_frame": fridx,
                }
                kpts.append(entry)
                miss_streak += 1
            elif last_valid is not None:
                gap_start = len(kpts)
            fridx += 1
            continue

        person_slot_map = _updateMotionSlots(
            track_state["slots"], persons_np, fridx
        )

        if (
            track_state["use_motion_lock"]
            and fridx >= WARMUP_FRAMES
            and track_state["locked_slot_id"] is None
            and not track_state["motion_only"]
        ):
            locked = _pickLockedSlot(
                track_state["slots"], frame[1].shape, camera_hint
            )
            if locked is not None:
                track_state["locked_slot_id"] = locked["id"]

        idx = findPitcherTracked(
            persons_np, confs_np, frame[1].shape, track_state,
            person_slot_map=person_slot_map,
        )
        target_kpts = persons_np[idx]
        glove_box = glove_box_from_landmarks(
            mp_poses[idx], frame_width, frame_height
        )

        entry = {
            "keypoints": target_kpts,
            "glove_box": glove_box,
            "interpolated": False,
            "video_frame": fridx,
        }
        if gap_start is not None and last_valid is not None and len(kpts) > gap_start:
            n_gap = len(kpts) - gap_start
            for g in range(n_gap):
                alpha = (g + 1) / (n_gap + 1)
                kpts[gap_start + g] = _interpolateFrames(last_valid, entry, alpha)
            gap_start = None

        kpts.append(entry)
        last_valid = entry
        miss_streak = 0
        fridx += 1

    return kpts, nframes, multi_pose_frames


def processVideo(video_path, output_path, model_path, progress, status, num_poses=None):
    """Run full MediaPipe extraction pipeline on a video file."""
    from processing.mediapipe_pose import video_landmarker

    num_poses = NUM_POSES if num_poses is None else num_poses

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    stretch_landscape = isStretchLandscapePath(
        video_path, frame_width, frame_height
    )
    use_motion_lock = False
    if stretch_landscape:
        use_motion_lock = warmupMultiPoseCount(model_path, cap, fps=fps) >= 8

    with video_landmarker(model_path, num_poses=num_poses) as landmarker:
        kpts, nframes, multi_pose_frames = detectKeypoints(
            cap, landmarker, progress, status,
            video_path=video_path, fps=fps,
            use_motion_lock=use_motion_lock,
        )
    cap.release()

    retried_tracking = False
    if stretch_landscape:
        jitter = _hipJitterFromKpts(kpts)
        if jitter > JITTER_RETRY_THRESHOLD:
            cap = cv2.VideoCapture(video_path)
            with video_landmarker(model_path, num_poses=num_poses) as landmarker:
                retry_kpts, _, retry_multi = detectKeypoints(
                    cap, landmarker, progress, status,
                    video_path=video_path,
                    track_options={"motion_only": True},
                    fps=fps,
                    use_motion_lock=use_motion_lock,
                )
            cap.release()
            retry_jitter = _hipJitterFromKpts(retry_kpts)
            if retry_jitter < jitter:
                kpts = retry_kpts
                multi_pose_frames = retry_multi
                retried_tracking = True

    extraction_failed = not any(
        np.max(f["keypoints"]) > 0 for f in kpts
    )

    only_kpts = np.array([f["keypoints"] for f in kpts], dtype=np.float32)
    gloves = np.array([f.get("glove_box", [-1] * 4) for f in kpts])
    sync_frame = findDeliverySyncFrame(only_kpts) if len(only_kpts) else 0
    windup_end = resolveWindupEnd(only_kpts) if len(only_kpts) else None
    peak_knee_frame = resolvePeakKneeFrame(only_kpts) if len(only_kpts) else None
    video_align_start = next(
        (int(f["video_frame"]) for f in kpts if f.get("video_frame", -1) >= 0),
        0,
    )

    status.text("Saving keypoint sequence")
    progress.progress(1.0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.array(kpts))

    qa = computeQualityMetrics(only_kpts, gloves, frame_width, frame_height)
    dominant_x = _dominantHipX(kpts, frame_width)

    meta = {
        "num_total_frames": nframes,
        "num_extracted_frames": len(kpts),
        "fps": fps,
        "duration": int(nframes / fps) if fps else 0,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "sync_frame": sync_frame,
        "windup_end": windup_end,
        "peak_knee_frame": peak_knee_frame,
        "video_align_start": video_align_start,
        "delivery_hint": qa["delivery_hint"],
        "pitcher_track_jitter": qa["hip_jitter_px"],
        "retried_tracking": retried_tracking,
        "camera_hint": "stretch_landscape" if stretch_landscape else "default",
        "extraction_failed": extraction_failed,
        "dominant_hip_x_frac": dominant_x,
        "pose_backend": "mediapipe",
        "num_poses_detected": multi_pose_frames,
        "num_poses_setting": num_poses,
        **qa,
    }
    saveKeypointMeta(output_path, meta)
    if extraction_failed:
        return False, meta
    return True, meta


def renderPreviewFrame(video_path, kpts_path, frame_idx=0):
    """Render one video frame with skeleton overlay for QA preview."""
    from processing.kpt_utils import parseKeypointSequence, resolveVideoAlignStart
    from processing.tab5_helper import drawSkeleton

    kpts_raw = np.load(kpts_path, allow_pickle=True)
    kpts_seq, glove_seq, video_frames, _ = parseKeypointSequence(kpts_raw)
    if kpts_seq is None or len(kpts_seq) == 0:
        return None

    idx = min(frame_idx, len(kpts_seq) - 1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_kpt = len(kpts_seq)
    if video_frames is not None and video_frames[idx] >= 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(video_frames[idx]))
    else:
        video_start = resolveVideoAlignStart(kpts_path, n_kpt, total)
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_start + idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    overlay = frame.copy()
    drawSkeleton(
        overlay, kpts_seq[idx], color=(0, 220, 255), thickness=2,
        frame_width=fw, frame_height=fh,
    )
    glove = glove_seq[idx]
    if glove[0] != -1:
        cv2.rectangle(
            overlay,
            (int(glove[0]), int(glove[1])),
            (int(glove[2]), int(glove[3])),
            (0, 220, 255),
            2,
        )
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


def renderAllDetectionsFrame(video_path, frame_idx=0, model_path=None):
    """
    Render one video frame with all MediaPipe pose detections for QA debug.
    Each person is drawn in a different color with bbox area label.
    """
    from processing.pose_adapter import mp_landmarks_to_coco17
    from processing.tab5_helper import drawSkeleton

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = min(frame_idx, max(total - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None

    if model_path is None:
        from utils import load_pose_landmarker
        model_path = load_pose_landmarker()

    with image_landmarker(model_path, num_poses=NUM_POSES) as landmarker:
        mp_poses = detect_image_frame(landmarker, frame)

    if not mp_poses:
        return frame

    colors = [
        (0, 220, 255), (255, 100, 50), (50, 255, 100),
        (255, 50, 200), (200, 200, 50),
    ]
    overlay = frame.copy()
    for i, landmarks in enumerate(mp_poses):
        kpts = mp_landmarks_to_coco17(landmarks, fw, fh)
        color = colors[i % len(colors)]
        drawSkeleton(
            overlay, kpts, color=color, thickness=2,
            frame_width=fw, frame_height=fh,
        )
        xmin, ymin, xmax, ymax = _personBBox(kpts)
        area = int((xmax - xmin) * (ymax - ymin))
        anchor = _personAnchor(kpts)
        label = f"P{i} area={area} x={int(anchor[0])}"
        cv2.putText(
            overlay, label, (int(anchor[0]), int(anchor[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    return frame


def _dominantHipX(kpts_list, frame_width):
    """Median hip x as fraction of frame width — pitcher should be right (>0.45)."""
    xs = []
    for entry in kpts_list:
        k = entry["keypoints"]
        if (k[11, 0] != 0 or k[11, 1] != 0) and (k[12, 0] != 0 or k[12, 1] != 0):
            xs.append((k[11, 0] + k[12, 0]) / 2)
    if not xs or not frame_width:
        return None
    return round(float(np.median(xs)) / frame_width, 3)
