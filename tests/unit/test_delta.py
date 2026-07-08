import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scenesense.core.delta import compute_delta


def make_obj(id, cls, position):
    return {"id": id, "class": cls, "position": position, "confidence": 0.9}


def test_missing_object():
    baseline = [make_obj("obj_1", "chair", [1.0, 0.5, 0.8])]
    current = []
    changes = compute_delta(baseline, current)
    assert len(changes) == 1
    assert changes[0]["status"] == "missing"
    assert changes[0]["object"] == "chair"


def test_new_object():
    baseline = []
    current = [make_obj("obj_1", "bottle", [0.5, 0.2, 0.6])]
    changes = compute_delta(baseline, current)
    assert len(changes) == 1
    assert changes[0]["status"] == "new"
    assert changes[0]["object"] == "bottle"


def test_moved_object():
    baseline = [make_obj("obj_1", "bag", [1.0, 0.5, 0.8])]
    current  = [make_obj("obj_1", "bag", [1.5, 0.5, 0.8])]  # moved 0.5 units
    changes = compute_delta(baseline, current, move_threshold=0.3)
    assert len(changes) == 1
    assert changes[0]["status"] == "moved"
    assert changes[0]["object"] == "bag"


def test_no_change():
    baseline = [make_obj("obj_1", "chair", [1.0, 0.5, 0.8])]
    current  = [make_obj("obj_1", "chair", [1.01, 0.5, 0.8])]  # tiny drift
    changes = compute_delta(baseline, current, move_threshold=0.3)
    assert len(changes) == 0


def test_multiple_changes():
    baseline = [
        make_obj("obj_1", "chair",  [1.0, 0.5, 0.8]),
        make_obj("obj_2", "bottle", [0.5, 0.2, 0.6]),
    ]
    current = [
        make_obj("obj_1", "chair",  [1.0, 0.5, 0.8]),  # unchanged
        make_obj("obj_3", "bag",    [2.0, 0.5, 0.9]),  # new
        # bottle is missing
    ]
    changes = compute_delta(baseline, current)
    statuses = [c["status"] for c in changes]
    assert "missing" in statuses
    assert "new" in statuses
    assert "moved" not in statuses