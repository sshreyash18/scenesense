# scenesense

A Python library for 3D spatial scene monitoring. Point a camera at any space, build a baseline, and get notified when objects are added, removed, or moved — with timestamps.

## Install

```bash
pip install .
```

Or for development:

```bash
pip install -e .
```

## Quick Start

### Compare two videos

```python
from scenesense import Scene

scene = Scene()

# Build baseline from first video
scene.baseline("room_before.mp4", save_path="baseline.json")

# Compare against second video
changes = scene.compare("room_after.mp4")

for c in changes:
    print(c)
# {"status": "missing", "object": "cup", "last_position": [...]}
# {"status": "new",     "object": "bag", "position": [...]}
# {"status": "moved",   "object": "chair", "from": [...], "to": [...]}
```

### Load a saved baseline

```python
scene = Scene()
scene.load_baseline("baseline.json")
changes = scene.compare("room_after.mp4")
```

### Live camera monitoring

```python
from scenesense import Scene

def on_event(event):
    print(f"[{event['timestamp']}] {event['event'].upper()} — {event['object']}")
    # event keys: event, object, timestamp, stream_offset_seconds
    # + last_position (missing), position (new), from/to (moved)

scene = Scene()

scene.watch(
    source=0,                  # webcam index or RTSP URL
    baseline_duration=15,      # seconds to build baseline
    interval=30,               # seconds between checks
    on_event=on_event
)
```

## Configuration

```python
scene = Scene(
    model_size="n",        # yolo model: 'n' (fast), 's', 'm' (accurate)
    confidence=0.4,        # detection confidence threshold
    sample_rate=1.0,       # frames per second to sample from video
    move_threshold=0.3,    # distance to consider an object moved
    match_threshold=0.5    # distance to consider two detections the same object
)
```

## How it works

1. **Frame extraction** — samples frames from video at a controlled rate
2. **Object detection** — runs YOLOv8 on each frame to find objects and bounding boxes
3. **Depth estimation** — runs MiDaS to estimate per-pixel depth
4. **3D projection** — combines detection center + depth to get XYZ coordinates
5. **Scene graph** — collapses per-frame detections into a stable list of unique objects
6. **Registration** — aligns new video coordinate frame to baseline using ORB feature matching
7. **Delta detection** — compares scene graphs to find missing, moved, and new objects

## Use cases

- Warehouse shelf auditing
- Museum artifact monitoring  
- Security and facility management
- Construction site progress tracking
- Retail planogram compliance
- Manufacturing FOD detection

## Requirements

- Python 3.10+
- opencv-python
- numpy
- ultralytics (YOLOv8)

Models are downloaded automatically on first use (~30MB total).

## Testing

```bash
pytest tests/unit/ -v
```

## Project status

SceneSense is an early-stage (`v0.1`) open-source library. The core pipeline works
end-to-end, but there's a lot we'd love help with — better accuracy, metric depth,
a CLI, GPU support, and more. See the [roadmap](CONTRIBUTING.md#roadmap--good-places-to-jump-in).

## Contributing

Contributions are very welcome — whether it's a bug report, a doc fix, or a new
feature. Start with **[CONTRIBUTING.md](CONTRIBUTING.md)**, which explains how the
pipeline fits together and lists concrete tasks (with "good first issue" picks).

1. Fork the repo and create a branch
2. `pip install -e .` and run `pytest tests/unit/ -v`
3. Make your change, add tests, open a PR

## License

Released under the [MIT License](LICENSE) — free to use, modify, and distribute.