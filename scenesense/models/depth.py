import numpy as np
import cv2
from typing import Tuple


class DepthEstimator:
    """
    Estimates relative depth for each pixel in a frame.
    Uses MiDaS via OpenCV DNN - no extra install needed.
    
    Note: this gives RELATIVE depth (not metric metres).
    Higher value = further away. Good enough for 3D positioning.
    """

    def __init__(self):
        print("[depth] Loading MiDaS depth model...")
        self.model = cv2.dnn.readNet(
            cv2.samples.findFile("dnn/MiDaS_small.onnx", False) or self._download_model()
        )
        self.input_size = (256, 256)
        print("[depth] Ready.")

    def _download_model(self) -> str:
        import urllib.request
        import os
        model_path = "MiDaS_small.onnx"
        if not os.path.exists(model_path):
            print("[depth] Downloading MiDaS model (~25MB)...")
            url = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
            urllib.request.urlretrieve(url, model_path)
            print("[depth] Download complete.")
        return model_path

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth map for a frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            Depth map as float32 array, same spatial size as input.
            Values are relative (not metres). Higher = further.
        """
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=self.input_size,
            mean=(0.485, 0.456, 0.406),
            swapRB=True,
            crop=False
        )

        self.model.setInput(blob)
        output = self.model.forward()

        depth_map = output[0, 0]
        depth_map = cv2.resize(depth_map, (w, h))

        # normalise to 0-1 range
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)

        return depth_map.astype(np.float32)

    def get_depth_at(self, depth_map: np.ndarray, cx: int, cy: int) -> float:
        """
        Get the depth value at a specific pixel coordinate.

        Args:
            depth_map: Output from estimate()
            cx, cy: Pixel coordinates (center of bounding box)

        Returns:
            Depth value between 0.0 (closest) and 1.0 (furthest)
        """
        h, w = depth_map.shape
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))
        return float(depth_map[cy, cx])