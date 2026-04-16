import numpy as np
from scipy import interpolate
import copy

def add_noise(frames, sigma=0.005):
    """Adds zero-mean Gaussian noise to keypoints."""
    aug_frames = []
    for frame in frames:
        new_frame = copy.deepcopy(frame)
        noise = np.random.normal(0, sigma, new_frame["keypoints"].shape).astype(np.float32)
        new_frame["keypoints"] += noise
        aug_frames.append(new_frame)
    return aug_frames

def time_warp(frames, factor):
    """
    Resamples the frame sequence by `factor`.
    If factor < 1, the pitch is faster (compressed). We pad it to maintain length.
    If factor > 1, the pitch is slower (stretched). We crop it to maintain length.
    """
    orig_len = len(frames)
    new_len = int(orig_len * factor)
    
    # We will interpolate each keypoint and glove_box coordinate
    # The original time axis is 0 to orig_len-1
    orig_t = np.linspace(0, 1, orig_len)
    new_t = np.linspace(0, 1, new_len)
    
    # We need to collect all scalar sequences
    seq_kpt = np.array([f["keypoints"] for f in frames]) # (T, 17, 2)
    has_glove = "glove_box" in frames[0]
    if has_glove:
        seq_glove = np.array([f["glove_box"] for f in frames]) # (T, 4)
        
    # Interpolate keypoints
    f_kpt = interpolate.interp1d(orig_t, seq_kpt, axis=0, kind='linear')
    resampled_kpt = f_kpt(new_t)
    
    if has_glove:
        f_glove = interpolate.interp1d(orig_t, seq_glove, axis=0, kind='linear')
        resampled_glove = f_glove(new_t)
        
    # Reconstruct frames
    resampled_frames = []
    for i in range(new_len):
        frame = {"keypoints": resampled_kpt[i].astype(np.float32)}
        if has_glove:
            frame["glove_box"] = resampled_glove[i].tolist()
        resampled_frames.append(frame)
        
    # Maintain length (anchor at the end since windup truncation stops at release point)
    if new_len < orig_len:
        # Pad at the beginning with the first frame
        pad_len = orig_len - new_len
        pad_frames = [copy.deepcopy(resampled_frames[0]) for _ in range(pad_len)]
        final_frames = pad_frames + resampled_frames
    elif new_len > orig_len:
        # Crop from the beginning
        crop_len = new_len - orig_len
        final_frames = resampled_frames[crop_len:]
    else:
        final_frames = resampled_frames
        
    return final_frames

def augment_sequence(frames, label):
    """Generates synthetic variants of a single sequence."""
    variants = []
    
    # 1. Add noise
    variants.append((add_noise(frames, sigma=0.005), label))
    variants.append((add_noise(frames, sigma=0.01), label))
    
    # 2. Time warping
    variants.append((time_warp(frames, factor=0.9), label))
    variants.append((time_warp(frames, factor=0.95), label))
    variants.append((time_warp(frames, factor=1.05), label))
    variants.append((time_warp(frames, factor=1.1), label))
    
    # 3. Combinations
    variants.append((add_noise(time_warp(frames, factor=0.95), sigma=0.005), label))
    variants.append((add_noise(time_warp(frames, factor=1.05), sigma=0.005), label))
    
    return variants
