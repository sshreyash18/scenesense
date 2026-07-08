from typing import List, Dict
from scenesense.utils.spatial import euclidean_distance


def compute_delta(
    baseline: List[Dict],
    current: List[Dict],
    move_threshold: float = 0.3,
    match_threshold: float = 0.5,
    class_match: bool = True
) -> List[Dict]:
    """
    Compare a current scene against a baseline and return what changed.

    Args:
        baseline: List of objects from the baseline scene graph
        current: List of objects from the current scene graph
        move_threshold: Min distance to consider an object has moved
        match_threshold: Max distance to consider two objects the same
        class_match: If True, match objects by class name first before
                     checking position. Best for panning video scenes.

    Returns:
        List of change events with status: 'missing', 'moved', or 'new'
    """
    changes = []
    matched_current_ids = set()

    for base_obj in baseline:
        best_match = None
        best_dist = float("inf")

        for i, cur_obj in enumerate(current):
            if cur_obj["class"] != base_obj["class"]:
                continue

            if class_match:
                # class matched — accept it, use distance only for moved detection
                dist = euclidean_distance(base_obj["position"], cur_obj["position"])
                if best_match is None or dist < best_dist:
                    best_dist = dist
                    best_match = (i, cur_obj)
            else:
                dist = euclidean_distance(base_obj["position"], cur_obj["position"])
                if dist < best_dist:
                    best_dist = dist
                    best_match = (i, cur_obj)

        if best_match is None or (not class_match and best_dist > match_threshold):
            changes.append({
                "status": "missing",
                "object": base_obj["class"],
                "id": base_obj["id"],
                "last_position": base_obj["position"]
            })
        else:
            matched_current_ids.add(best_match[0])
            if best_dist > move_threshold:
                changes.append({
                    "status": "moved",
                    "object": base_obj["class"],
                    "id": base_obj["id"],
                    "from": base_obj["position"],
                    "to": best_match[1]["position"],
                    "distance": round(best_dist, 4)
                })

    for i, cur_obj in enumerate(current):
        if i not in matched_current_ids:
            changes.append({
                "status": "new",
                "object": cur_obj["class"],
                "position": cur_obj["position"]
            })

    return changes