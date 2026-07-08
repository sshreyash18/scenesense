# Contributing to SceneSense

Thanks for your interest in improving SceneSense! This project turns any camera
into a 3D spatial change detector, and there's plenty of room to make it more
accurate, faster, and easier to use. Contributions of every size are welcome —
from fixing a typo to adding a whole new capability.

## Ways to contribute

- 🐛 **Report bugs** — open an issue with steps to reproduce, your OS, and Python version.
- 💡 **Suggest features** — open an issue describing the use case.
- 📖 **Improve docs** — clearer docstrings, better examples, tutorials.
- 🔧 **Write code** — pick something from the roadmap below or an open issue.
- 🎥 **Share test scenes** — short before/after videos help us test edge cases.

## Getting set up

```bash
# 1. Fork this repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/scenesense.git
cd scenesense

# 2. Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install in editable mode with dependencies
pip install -e .
pip install -r requirements.txt

# 4. Run the tests
pytest tests/unit/ -v
```

The YOLOv8 and MiDaS model weights (~72MB) download automatically on first run —
they are **not** stored in the repo.

## How the pipeline fits together

Read [README.md](README.md#how-it-works) for the high-level flow. In code:

| Stage | File |
|-------|------|
| Frame extraction | [scenesense/utils/video.py](scenesense/utils/video.py) |
| Object detection (YOLOv8) | [scenesense/models/detector.py](scenesense/models/detector.py) |
| Depth estimation (MiDaS) | [scenesense/models/depth.py](scenesense/models/depth.py) |
| 2D→3D projection | [scenesense/utils/spatial.py](scenesense/utils/spatial.py) |
| Scene graph building | [scenesense/core/scene_graph.py](scenesense/core/scene_graph.py) |
| Coordinate registration (ORB) | [scenesense/core/registration.py](scenesense/core/registration.py) |
| Change / delta detection | [scenesense/core/delta.py](scenesense/core/delta.py) |
| Live event log | [scenesense/core/event_log.py](scenesense/core/event_log.py) |
| Public API | [scenesense/scene.py](scenesense/scene.py) |

## Roadmap — good places to jump in

These are real, open problems in the current codebase. Each links roughly to the
area you'd work in. Comment on (or open) an issue before starting a big one so we
don't duplicate effort.

### Good first issues
- **Fill in the integration test suite.** `tests/integration/test_pipeline.py` is
  currently empty — add an end-to-end test using the sample videos in
  `tests/integration/videos/`.
- **Migrate off deprecated `datetime.utcnow()`** (used in `scene.py`,
  `scene_graph.py`, `event_log.py`) to timezone-aware `datetime.now(timezone.utc)`.
- **Replace `print()` calls with the `logging` module** so library users can
  control verbosity.
- **Add a `Scene.compare_image()` / single-image path** — today everything assumes
  a video; support still images too.

### Core accuracy
- **Multi-instance same-class objects.** In `SceneGraph.build`, the `"class"`
  strategy keeps only the single highest-confidence detection per class, so two
  chairs collapse into one. Add proper per-instance tracking.
- **Metric depth.** MiDaS returns *relative* depth only. Add optional camera-intrinsic
  calibration so positions can be reported in real-world units.
- **Configurable camera intrinsics.** `project_to_3d` in `utils/spatial.py` hardcodes
  a 30° field of view. Let users pass focal length / FOV.
- **Stronger registration.** `registration.py` uses 2D homography (ORB). Explore
  full 3D registration for scenes shot from very different angles.

### Performance & usability
- **GPU acceleration** for YOLO + MiDaS, with automatic device selection.
- **A command-line interface** (`scenesense baseline video.mp4`, `scenesense compare ...`)
  so it's usable without writing Python.
- **Event outputs / integrations** — write events to a file, POST to a webhook, or
  push to a message queue from `watch()`.
- **Publish to PyPI** and add a GitHub Actions CI workflow that runs the tests.

## Pull request checklist

1. Create a branch: `git checkout -b feature/short-description`
2. Keep changes focused — one logical change per PR.
3. Match the existing code style (type hints, docstrings on public methods).
4. Add or update tests; make sure `pytest tests/unit/ -v` passes.
5. Update the README if you changed public behavior.
6. Write a clear PR description: what changed and why.

## Code style

- Python 3.10+, standard library `typing` hints on public functions.
- Docstrings on all public classes and methods (see existing files for the style).
- Keep dependencies minimal — discuss in an issue before adding a new one.

## Questions?

Open an issue with the `question` label. Happy to help you find a good first task.
