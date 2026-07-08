from ultralytics import YOLO
import numpy as np
from typing import List, Dict


class ObjectDetector:
    """
    Wraps YOLOv8 for object detection on individual frames.
    Downloads the model automatically on first use.
    """

    def __init__(self, model_size: str = "n", confidence: float = 0.4):
        """
        Args:
            model_size: YOLO model size - 'n' (nano), 's' (small), 'm' (medium)
                        Start with 'n' - fastest, good enough for most scenes
            confidence: Minimum confidence threshold to accept a detection
        """
        model_name = f"yolov8{model_size}.pt"
        print(f"[detector] Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.confidence = confidence
        print(f"[detector] Ready. Confidence threshold: {confidence}")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detections, each as:
            {
                "class": "chair",
                "confidence": 0.87,
                "bbox": [x1, y1, x2, y2],   # pixel coordinates
                "center": [cx, cy]            # center of bounding box
            }
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.confidence:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            detections.append({
                "class": class_name,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "center": [cx, cy]
            })

        return detections