import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import cv2
import tempfile
import pytest
from scenesense.utils.video import extract_frames, get_video_info


def create_test_video(path: str, num_frames: int = 30, fps: int = 30):
    """Helper to create a small synthetic test video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (320, 240))
    for i in range(num_frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:] = (i * 8 % 255, 100, 200)  # varying color per frame
        writer.write(frame)
    writer.release()


def test_extract_frames_returns_list():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    create_test_video(path, num_frames=30, fps=30)
    frames = extract_frames(path, sample_rate=1.0)
    assert isinstance(frames, list)
    assert len(frames) > 0
    os.unlink(path)


def test_extract_frames_are_numpy_arrays():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    create_test_video(path, num_frames=30, fps=30)
    frames = extract_frames(path, sample_rate=1.0)
    for frame in frames:
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3  # height x width x channels
    os.unlink(path)


def test_extract_frames_sample_rate():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    create_test_video(path, num_frames=60, fps=30)  # 2 second video
    frames_1fps = extract_frames(path, sample_rate=1.0)
    frames_2fps = extract_frames(path, sample_rate=2.0)
    assert len(frames_2fps) >= len(frames_1fps)
    os.unlink(path)


def test_get_video_info():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    create_test_video(path, num_frames=30, fps=30)
    info = get_video_info(path)
    assert "fps" in info
    assert "width" in info
    assert "height" in info
    assert "duration" in info
    assert info["width"] == 320
    assert info["height"] == 240
    os.unlink(path)


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        extract_frames("nonexistent_video.mp4")