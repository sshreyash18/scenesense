import os
import json
from datetime import datetime
from typing import List, Dict, Callable, Optional

from scenesense.utils.video import extract_frames
from scenesense.core.perception import PerceptionPipeline
from scenesense.core.scene_graph import SceneGraph
from scenesense.core.delta import compute_delta


class Scene:
    """
    Main public API for scenesense.

    Usage:
        scene = Scene()
        scene.baseline("room_video.mp4")
        changes = scene.compare("room_video_later.mp4")
    """

    def __init__(
        self,
        model_size: str = "n",
        confidence: float = 0.4,
        sample_rate: float = 1.0,
        move_threshold: float = 0.3,
        match_threshold: float = 0.5
    ):
        """
        Args:
            model_size: YOLO model size — 'n' (fastest), 's', 'm' (most accurate)
            confidence: Minimum detection confidence (0.0 to 1.0)
            sample_rate: Frames to sample per second from video
            move_threshold: Distance an object must move to be flagged as moved
            match_threshold: Max distance to consider two detections the same object
        """
        self.pipeline = PerceptionPipeline(model_size=model_size, confidence=confidence)
        self.sample_rate = sample_rate
        self.move_threshold = move_threshold
        self.match_threshold = match_threshold
        self._baseline_graph: Optional[SceneGraph] = None
        self._baseline_frames: Optional[List] = None

    def baseline(self, video_path: str, save_path: Optional[str] = None) -> SceneGraph:
        """
        Build the reference scene graph from a video.

        Args:
            video_path: Path to the baseline video
            save_path: Optional path to save the scene graph as JSON

        Returns:
            The baseline SceneGraph
        """
        print(f"\n[scene] Building baseline from: {video_path}")

        frames = extract_frames(video_path, sample_rate=self.sample_rate)
        self._baseline_frames = frames
        all_detections = self.pipeline.process_frames(frames)

        graph = SceneGraph()
        graph.build(all_detections, frames=frames)

        if save_path:
            graph.save(save_path)

        self._baseline_graph = graph

        print(f"\n[scene] Baseline ready.")
        print(graph.summary())
        return graph
    
    def load_baseline(self, path: str) -> None:
        """
        Load a previously saved baseline scene graph from JSON.
        Also loads reference frames for registration if available.
        """
        graph = SceneGraph()
        graph.load(path)
        self._baseline_graph = graph
        self._baseline_frames = graph.representative_frames if graph.representative_frames else None
        if self._baseline_frames:
            print(f"[scene] Registration enabled — {len(self._baseline_frames)} reference frames loaded.")
        else:
            print(f"[scene] No reference frames found — registration disabled.")
        print(graph.summary())

    def compare(self, video_path: str, use_registration: bool = True) -> List[Dict]:
        """
        Compare a new video against the baseline and return what changed.

        Args:
            video_path: Path to the new video to compare
            use_registration: Whether to align coordinate frames before comparing

        Returns:
            List of change events
        """
        if self._baseline_graph is None:
            raise RuntimeError("No baseline set. Call scene.baseline() or scene.load_baseline() first.")

        print(f"\n[scene] Comparing: {video_path}")

        frames = extract_frames(video_path, sample_rate=self.sample_rate)
        all_detections = self.pipeline.process_frames(frames)

        # apply registration if baseline frames are available
        if use_registration and self._baseline_frames is not None:
            from scenesense.core.registration import SceneRegistrar
            registrar = SceneRegistrar()
            H = registrar.estimate_transform(self._baseline_frames, frames)
            if H is not None:
                h, w = frames[0].shape[:2]
                all_detections = [
                    [registrar.apply_transform_to_detections(det_list, H, w, h)]
                    for det_list in all_detections
                ]
                all_detections = [d[0] for d in all_detections]

        current_graph = SceneGraph()
        current_graph.build(all_detections, frames=frames)

        changes = compute_delta(
            baseline=self._baseline_graph.objects,
            current=current_graph.objects,
            move_threshold=self.move_threshold,
            match_threshold=self.match_threshold,
            class_match=True
        )

        print(f"\n[scene] {len(changes)} change(s) detected.")
        for c in changes:
            if c["status"] == "missing":
                print(f"  MISSING  — {c['object']} (last seen at {c['last_position']})")
            elif c["status"] == "moved":
                print(f"  MOVED    — {c['object']} from {c['from']} to {c['to']}")
            elif c["status"] == "new":
                print(f"  NEW      — {c['object']} at {c['position']}")

        return changes

    def watch(
        self,
        source,
        baseline_duration: int = 15,
        interval: int = 30,
        on_event=None,
        save_baseline_path: Optional[str] = None
    ) -> None:
        """
        Monitor a camera or stream continuously.
        Builds a baseline automatically from the first N seconds,
        then checks for changes every interval seconds.
        Emits timestamped events only when state actually changes.

        Args:
            source: Camera index (0 for webcam) or stream URL string
            baseline_duration: Seconds to observe before building baseline
            interval: Seconds between each comparison cycle
            on_event: Callback fired for each new event — receives event dict
            save_baseline_path: Optional path to save the auto-built baseline
        """
        import cv2
        import time
        from scenesense.core.event_log import EventLog

        print(f"\n[scene] Starting watch on source: {source}")
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")

        event_log = EventLog()

        try:
            # ── Phase 1: build baseline ──────────────────────────────
            print(f"[scene] Building baseline — observing for {baseline_duration} seconds...")
            baseline_frames = []
            start = time.time()

            while time.time() - start < baseline_duration:
                ret, frame = cap.read()
                if ret:
                    baseline_frames.append(frame)

            if not baseline_frames:
                raise RuntimeError("No frames received during baseline phase.")

            # sample evenly
            step = max(1, len(baseline_frames) // 37)
            sampled = baseline_frames[::step]

            print(f"[scene] Baseline phase complete — {len(sampled)} frames sampled.")

            all_detections = self.pipeline.process_frames(sampled)
            graph = SceneGraph()
            graph.build(all_detections, frames=sampled)
            self._baseline_graph = graph
            self._baseline_frames = sampled[:5]

            if save_baseline_path:
                graph.save(save_baseline_path)

            baseline_classes = [obj["class"] for obj in graph.objects]
            print(f"\n[scene] Baseline ready — monitoring for changes.")
            print(graph.summary())
            print(f"\n[scene] Checking every {interval} seconds. Press Ctrl+C to stop.\n")

            # ── Phase 2: continuous monitoring ───────────────────────
            event_log.start()
            cycle = 0

            while True:
                cycle_frames = []
                cycle_start = time.time()

                while time.time() - cycle_start < interval:
                    ret, frame = cap.read()
                    if ret:
                        cycle_frames.append(frame)

                if not cycle_frames:
                    print("[scene] No frames in this cycle — skipping.")
                    continue

                cycle += 1
                step = max(1, len(cycle_frames) // 10)
                sampled_cycle = cycle_frames[::step]

                all_detections = self.pipeline.process_frames(sampled_cycle)

                # registration against baseline
                if self._baseline_frames:
                    from scenesense.core.registration import SceneRegistrar
                    registrar = SceneRegistrar()
                    H = registrar.estimate_transform(self._baseline_frames, sampled_cycle)
                    if H is not None:
                        h, w = sampled_cycle[0].shape[:2]
                        all_detections = [
                            registrar.apply_transform_to_detections(det_list, H, w, h)
                            for det_list in all_detections
                        ]

                current_graph = SceneGraph()
                current_graph.build(all_detections, frames=sampled_cycle)

                from scenesense.core.delta import compute_delta
                changes = compute_delta(
                    baseline=self._baseline_graph.objects,
                    current=current_graph.objects,
                    move_threshold=self.move_threshold,
                    match_threshold=self.match_threshold,
                    class_match=True
                )

                new_events = event_log.process_changes(
                    changes=changes,
                    baseline_classes=baseline_classes,
                    on_event=on_event
                )

                timestamp = datetime.utcnow().strftime("%H:%M:%S")
                if new_events:
                    for e in new_events:
                        print(f"  [{timestamp}] {e['event'].upper():10} — {e['object']}")
                else:
                    print(f"  [{timestamp}] Cycle {cycle} — no changes.")

        except KeyboardInterrupt:
            print("\n[scene] Stopped.")
            print("\n" + event_log.summary())

        finally:
            cap.release()