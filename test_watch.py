from scenesense import Scene

def handle_event(event):
    print(f"\n*** EVENT ***")
    print(f"  What    : {event['event'].upper()}")
    print(f"  Object  : {event['object']}")
    print(f"  When    : {event['timestamp']}")
    print(f"  Offset  : +{event['stream_offset_seconds']}s")
    if "last_position" in event:
        print(f"  Last pos: {event['last_position']}")
    if "position" in event:
        print(f"  Position: {event['position']}")
    if "from" in event:
        print(f"  Moved   : {event['from']} → {event['to']}")
    print()

scene = Scene(
    model_size="n",
    confidence=0.4,
    move_threshold=0.3
)

scene.watch(
    source=0,                  # 0 = your webcam
    baseline_duration=10,      # 10 seconds to build baseline
    interval=10,               # check every 10 seconds
    on_event=handle_event,
    save_baseline_path="tests/integration/videos/watch_baseline.json"
)