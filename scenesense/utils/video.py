import cv2
import os
from typing import List
import numpy as np


def extract_frames(video_path: str, sample_rate: float = 1.0) -> List[np.ndarray]:
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        sample_rate: How many frames to extract per second (default: 1.0)
    
    Returns:
        List of frames as numpy arrays (BGR format)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # how many frames to skip between each sample
    frame_interval = max(1, int(fps / sample_rate))

    frames = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_interval == 0:
            frames.append(frame)
        frame_index += 1

    cap.release()

    print(f"[video] Extracted {len(frames)} frames from {os.path.basename(video_path)}")
    print(f"[video] Duration: {duration:.1f}s | FPS: {fps:.1f} | Sample rate: {sample_rate}/s")

    return frames


def get_video_info(video_path: str) -> dict:
    """
    Get basic metadata about a video file.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": 0.0
    }
    if info["fps"] > 0:
        info["duration"] = info["total_frames"] / info["fps"]
    cap.release()
    return info


def estimate_camera_motion(frames: List[np.ndarray]) -> float:
    """
    Estimate how much the camera moved across a sequence of frames.
    Uses optical flow between consecutive frame pairs.

    Returns:
        Average motion score (0.0 = fully static, higher = more movement)
    """
    if len(frames) < 2:
        return 0.0

    motion_scores = []

    for i in range(min(len(frames) - 1, 20)):  # sample up to 20 pairs
        f1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        f2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            f1, f2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        motion_scores.append(float(np.mean(magnitude)))

    return round(sum(motion_scores) / len(motion_scores), 4)