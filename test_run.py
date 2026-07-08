from scenesense import Scene

scene = Scene(
    model_size="n",
    confidence=0.4,
    sample_rate=1.0,
    move_threshold=0.3,
    match_threshold=0.5
)

# Phase 1 — build and save baseline (includes reference frames now)
scene.baseline(
    "tests/integration/videos/baseline.mp4",
    save_path="tests/integration/videos/baseline_scene.json"
)

print("\n" + "="*50 + "\n")

# Phase 2 — load baseline and compare
scene2 = Scene(
    model_size="n",
    confidence=0.4,
    sample_rate=1.0,
    move_threshold=0.3,
    match_threshold=0.5
)
scene2.load_baseline("tests/integration/videos/baseline_scene.json")
changes = scene2.compare("tests/integration/videos/compare.mp4")

print("\n--- CHANGES ---")
for c in changes:
    print(c)