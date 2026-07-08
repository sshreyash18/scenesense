import json
import uuid
import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from scenesense.utils.spatial import cluster_detections, normalize_depth_across_frames


class SceneGraph:

    def __init__(self):
        self.scene_id: str = str(uuid.uuid4())[:8]
        self.created_at: str = datetime.utcnow().isoformat()
        self.objects: List[Dict] = []
        self.representative_frames: List[np.ndarray] = []

    def build(self, all_frame_detections: List[List[Dict]], frames: List = None) -> None:
        all_frame_detections = normalize_depth_across_frames(all_frame_detections)

        flat = []
        for frame_dets in all_frame_detections:
            flat.extend(frame_dets)

        if not flat:
            print("[scene_graph] Warning: no detections found across all frames.")
            self.objects = []
            return

        strategy = "class"
        if frames is not None:
            from scenesense.utils.video import estimate_camera_motion
            motion_score = estimate_camera_motion(frames)
            strategy = "position" if motion_score < 1.5 else "class"
            print(f"[scene_graph] Camera motion score: {motion_score:.4f} → strategy: {strategy}")

            # store 5 evenly spaced representative frames
            indices = np.linspace(0, len(frames) - 1, min(5, len(frames)), dtype=int)
            self.representative_frames = [frames[i] for i in indices]

        if strategy == "class":
            best = {}
            for det in flat:
                cls = det["class"]
                if cls not in best or det["confidence"] > best[cls]["confidence"]:
                    best[cls] = det
            clustered = list(best.values())
        else:
            clustered = cluster_detections(flat, distance_threshold=0.25)

        self.objects = []
        for i, obj in enumerate(clustered):
            self.objects.append({
                "id": f"obj_{i+1}",
                "class": obj["class"],
                "position": obj["position"],
                "confidence": round(obj["confidence"], 3)
            })

        print(f"[scene_graph] Built scene with {len(self.objects)} unique objects.")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

        data = {
            "scene_id": self.scene_id,
            "created_at": self.created_at,
            "objects": self.objects
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        # save representative frames alongside the JSON
        frames_dir = path.replace(".json", "_frames")
        if self.representative_frames:
            os.makedirs(frames_dir, exist_ok=True)
            for i, frame in enumerate(self.representative_frames):
                cv2.imwrite(os.path.join(frames_dir, f"frame_{i}.jpg"), frame)
            print(f"[scene_graph] Saved {len(self.representative_frames)} reference frames to {frames_dir}/")

        print(f"[scene_graph] Saved to {path}")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scene graph not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        self.scene_id = data["scene_id"]
        self.created_at = data["created_at"]
        self.objects = data["objects"]

        # load representative frames if they exist
        frames_dir = path.replace(".json", "_frames")
        self.representative_frames = []
        if os.path.exists(frames_dir):
            frame_files = sorted([
                f for f in os.listdir(frames_dir) if f.endswith(".jpg")
            ])
            for fname in frame_files:
                frame = cv2.imread(os.path.join(frames_dir, fname))
                if frame is not None:
                    self.representative_frames.append(frame)
            print(f"[scene_graph] Loaded {len(self.representative_frames)} reference frames.")

        print(f"[scene_graph] Loaded scene with {len(self.objects)} objects.")

    def summary(self) -> str:
        lines = [f"Scene ID: {self.scene_id}", f"Created: {self.created_at}", "Objects:"]
        for obj in self.objects:
            lines.append(f"  [{obj['id']}] {obj['class']} @ {obj['position']} (conf: {obj['confidence']})")
        return "\n".join(lines)