import numpy as np
from typing import List, Dict
from scenesense.models.detector import ObjectDetector
from scenesense.models.depth import DepthEstimator
from scenesense.utils.spatial import project_to_3d


class PerceptionPipeline:
    """
    Runs object detection + depth estimation on a frame
    and returns detected objects with 3D positions.
    """

    def __init__(self, model_size: str = "n", confidence: float = 0.4):
        self.detector = ObjectDetector(model_size=model_size, confidence=confidence)
        self.depth_estimator = DepthEstimator()

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Process a single frame through the full perception pipeline.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detected objects, each as:
            {
                "class": "chair",
                "confidence": 0.87,
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy],
                "position": [x, y, z]   # 3D position in camera space
            }
        """
        h, w = frame.shape[:2]

        # Step 1: detect objects in 2D
        detections = self.detector.detect(frame)

        if not detections:
            return []

        # Step 2: estimate depth map for the whole frame
        depth_map = self.depth_estimator.estimate(frame)

        # Step 3: project each detection into 3D
        results = []
        for det in detections:
            cx, cy = det["center"]
            depth = self.depth_estimator.get_depth_at(depth_map, cx, cy)
            position = project_to_3d(cx, cy, depth, w, h)

            results.append({
                "class": det["class"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "center": det["center"],
                "position": position
            })

        return results

    def process_frames(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Process a list of frames. Returns per-frame detections.
        """
        all_detections = []
        for i, frame in enumerate(frames):
            dets = self.process_frame(frame)
            all_detections.append(dets)
            print(f"[perception] Frame {i+1}/{len(frames)} — {len(dets)} objects detected")
        return all_detections