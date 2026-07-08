from datetime import datetime
from typing import List, Dict, Optional, Callable


class EventLog:
    """
    Tracks scene state over time and emits timestamped events
    only when something actually changes.

    Prevents duplicate events — if a cup is already missing,
    it won't fire the same event again until the cup returns
    and goes missing again.
    """

    def __init__(self):
        self.events: List[Dict] = []
        self.current_state: Dict[str, str] = {}
        # tracks what each object's last known status was
        # e.g. {"cup": "missing", "keyboard": "present", "mouse": "new"}
        self.stream_start: Optional[datetime] = None

    def start(self):
        """Call when monitoring begins."""
        self.stream_start = datetime.utcnow()
        self.current_state = {}
        self.events = []
        self._first_cycle_done = False

    def process_changes(
        self,
        changes: List[Dict],
        baseline_classes: List[str],
        on_event: Optional[Callable] = None
    ) -> List[Dict]:
        """
        Takes raw delta output, compares against last known state,
        and emits events only for genuinely new state transitions.

        Args:
            changes: Output from compute_delta()
            baseline_classes: List of object classes in the baseline
                              (so we know what "present" looks like)
            on_event: Optional callback fired for each new event

        Returns:
            List of new events emitted this cycle (may be empty)
        """
        now = datetime.utcnow()
        stream_offset = (now - self.stream_start).total_seconds() if self.stream_start else 0.0

        # build current snapshot from delta output
        # start by assuming everything in baseline is present
        snapshot = {cls: "present" for cls in baseline_classes}

        for change in changes:
            obj = change["object"]
            if change["status"] == "missing":
                snapshot[obj] = "missing"
            elif change["status"] == "moved":
                snapshot[obj] = "moved"
            elif change["status"] == "new":
                snapshot[obj] = "new"

        # compare snapshot against last known state — emit only transitions
        new_events = []

        for obj, status in snapshot.items():
            last_status = self.current_state.get(obj)

            if status != last_status:
                # suppress PRESENT events on first cycle — not useful noise
                if status == "present" and not self._first_cycle_done:
                    pass
                else:
                    event = self._build_event(
                        obj=obj,
                        status=status,
                        change=self._find_change(changes, obj),
                        timestamp=now,
                        stream_offset=stream_offset
                    )
                    self.events.append(event)
                    new_events.append(event)

                    if on_event:
                        on_event(event)

        # also catch objects that were missing/moved and have now returned
        for obj in list(self.current_state.keys()):
            if obj not in snapshot and self.current_state[obj] != "present":
                event = self._build_event(
                    obj=obj,
                    status="returned",
                    change=None,
                    timestamp=now,
                    stream_offset=stream_offset
                )
                self.events.append(event)
                new_events.append(event)
                if on_event:
                    on_event(event)

        # update state
        self.current_state = snapshot
        self._first_cycle_done = True

        return new_events

    def _build_event(
        self,
        obj: str,
        status: str,
        change: Optional[Dict],
        timestamp: datetime,
        stream_offset: float
    ) -> Dict:
        event = {
            "event": status,
            "object": obj,
            "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "stream_offset_seconds": round(stream_offset, 2)
        }

        if change:
            if status == "missing":
                event["last_position"] = change.get("last_position")
            elif status == "moved":
                event["from"] = change.get("from")
                event["to"] = change.get("to")
                event["distance"] = change.get("distance")
            elif status == "new":
                event["position"] = change.get("position")

        return event

    def _find_change(self, changes: List[Dict], obj: str) -> Optional[Dict]:
        for c in changes:
            if c["object"] == obj:
                return c
        return None

    def get_history(self, object_class: Optional[str] = None) -> List[Dict]:
        """
        Get full event history, optionally filtered by object class.

        Args:
            object_class: If provided, only return events for this class

        Returns:
            List of events in chronological order
        """
        if object_class:
            return [e for e in self.events if e["object"] == object_class]
        return self.events

    def summary(self) -> str:
        """Print a readable summary of all events so far."""
        if not self.events:
            return "No events recorded."
        lines = [f"Event log ({len(self.events)} events):"]
        for e in self.events:
            offset = f"+{e['stream_offset_seconds']:.1f}s"
            lines.append(f"  [{e['timestamp']}] ({offset}) {e['event'].upper():10} — {e['object']}")
        return "\n".join(lines)