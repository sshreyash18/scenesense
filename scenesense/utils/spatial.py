import numpy as np
from typing import List, Tuple


def project_to_3d(cx: int, cy: int, depth: float, frame_width: int, frame_height: int) -> List[float]:
    """
    Convert a 2D pixel coordinate + depth value into a 3D position.
    Uses a simple pinhole camera model with estimated focal length.
    """
    focal_length = frame_width / (2 * np.tan(np.radians(30)))
    cx_0 = frame_width / 2
    cy_0 = frame_height / 2

    z = depth
    x = (cx - cx_0) * z / focal_length
    y = (cy - cy_0) * z / focal_length

    return [round(float(x), 4), round(float(y), 4), round(float(z), 4)]


def normalize_depth_across_frames(all_frame_detections: List[List[dict]]) -> List[List[dict]]:
    """
    Normalize depth values across all frames so the same object
    gets consistent Z coordinates regardless of per-frame MiDaS scale drift.

    Strategy:
    - Collect all depth values across all frames
    - Compute global min/max
    - Re-normalize every detection's Z to that global scale
    - Recompute X, Y accordingly

    Args:
        all_frame_detections: Raw per-frame detections with position [x, y, z]

    Returns:
        Same structure with depth-normalized positions
    """
    # collect all z values
    all_z = []
    for frame_dets in all_frame_detections:
        for det in frame_dets:
            all_z.append(det["position"][2])

    if not all_z:
        return all_frame_detections

    z_min = min(all_z)
    z_max = max(all_z)
    z_range = z_max - z_min

    if z_range < 1e-6:
        return all_frame_detections

    normalized = []
    for frame_dets in all_frame_detections:
        norm_frame = []
        for det in frame_dets:
            det_copy = dict(det)
            old_z = det_copy["position"][2]
            new_z = (old_z - z_min) / z_range

            # scale x, y proportionally
            if old_z > 1e-6:
                scale = new_z / old_z
                new_x = round(det_copy["position"][0] * scale, 4)
                new_y = round(det_copy["position"][1] * scale, 4)
            else:
                new_x = det_copy["position"][0]
                new_y = det_copy["position"][1]

            det_copy["position"] = [new_x, new_y, round(new_z, 4)]
            norm_frame.append(det_copy)
        normalized.append(norm_frame)

    return normalized


def euclidean_distance(pos1: List[float], pos2: List[float]) -> float:
    """Euclidean distance between two 3D positions."""
    return float(np.linalg.norm(np.array(pos1) - np.array(pos2)))


def cluster_detections(detections: List[dict], distance_threshold: float = 0.25) -> List[dict]:
    """
    Collapse multiple detections of the same object across frames
    into a single stable entry by clustering nearby same-class detections.
    """
    if not detections:
        return []

    clusters = []

    for det in detections:
        matched = False
        for cluster in clusters:
            if cluster["class"] != det["class"]:
                continue
            dist = euclidean_distance(cluster["position"], det["position"])
            if dist < distance_threshold:
                n = cluster["count"]
                cluster["position"] = [
                    (cluster["position"][i] * n + det["position"][i]) / (n + 1)
                    for i in range(3)
                ]
                cluster["confidence"] = max(cluster["confidence"], det["confidence"])
                cluster["count"] += 1
                matched = True
                break
        if not matched:
            clusters.append({
                "class": det["class"],
                "position": det["position"][:],
                "confidence": det["confidence"],
                "count": 1
            })

    for c in clusters:
        c.pop("count")
        c["position"] = [round(float(v), 4) for v in c["position"]]

    return clusters