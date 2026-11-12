
"""Confirm behavior events using a minimum duration rule."""


class BehaviorConfirmer:
    def __init__(self, minimum_duration=60.0, gap_tolerance=3.0):
        self.minimum_duration = minimum_duration
        self.gap_tolerance = gap_tolerance

        self.candidate_behavior = None
        self.candidate_start = None
        self.last_seen_timestamp = None
        self.gap_start = None

        self.current_event = None
        self.confirmed_events = []

    def reset_candidate(self):
        """Reset the current behavior candidate."""
        self.candidate_behavior = None
        self.candidate_start = None
        self.last_seen_timestamp = None
        self.gap_start = None

    def reset(self):
        """Reset all confirmation state before processing a new video."""
        self.reset_candidate()
        self.current_event = None
        self.confirmed_events.clear()

    def _close_current_event(self, end_timestamp=None):
        """Close the current confirmed event."""

        if self.current_event is None:
            return

        if end_timestamp is None:
            end_timestamp = self.last_seen_timestamp

        if end_timestamp is None:
            end_timestamp = self.current_event["start_time"]

        # Prevent an event from having a negative duration.
        end_timestamp = max(
            self.current_event["start_time"],
            end_timestamp,
        )

        self.current_event["end_time"] = end_timestamp
        self.current_event["duration_seconds"] = (
            end_timestamp - self.current_event["start_time"]
        )

        self.current_event = None

    def update(
        self,
        behavior,
        timestamp,
        behavior_start_timestamp=None,
    ):
        """
        Update behavior confirmation.

        Args:
            behavior: Filtered behavior prediction, or None if unknown.
            timestamp: Current video timestamp in seconds.
            behavior_start_timestamp: Original start timestamp supplied
                by TemporalBehaviorFilter. If omitted, uses timestamp.

        Returns:
            Dictionary containing confirmation status and current event.
        """

        # Handle unknown predictions.
        if behavior is None or behavior == "unknown":
            if self.candidate_behavior is not None:
                if self.gap_start is None:
                    self.gap_start = timestamp

                gap_duration = timestamp - self.gap_start

                if gap_duration > self.gap_tolerance:
                    self._close_current_event()
                    self.reset_candidate()

            return {
                "confirmed": self.current_event is not None,
                "newly_confirmed": False,
                "event": self.current_event,
            }

        # Use the original behavior start time when provided.
        if behavior_start_timestamp is None:
            behavior_start_timestamp = timestamp

        # Start a new candidate.
        if self.candidate_behavior is None:
            self.candidate_behavior = behavior
            self.candidate_start = behavior_start_timestamp

        # A different behavior has been accepted by the temporal filter.
        elif behavior != self.candidate_behavior:
            # End the previous event at the actual transition time,
            # not when the transition was accepted.
            self._close_current_event(
                end_timestamp=behavior_start_timestamp
            )

            self.reset_candidate()

            self.candidate_behavior = behavior
            self.candidate_start = behavior_start_timestamp

        # Same behavior continues.
        else:
            # A short unknown gap is tolerated.
            if self.gap_start is not None:
                gap_duration = timestamp - self.gap_start

                if gap_duration > self.gap_tolerance:
                    self._close_current_event()
                    self.reset_candidate()

                    self.candidate_behavior = behavior
                    self.candidate_start = behavior_start_timestamp

                self.gap_start = None

        self.last_seen_timestamp = timestamp

        candidate_duration = timestamp - self.candidate_start
        newly_confirmed = False

        # Confirm once the behavior has lasted long enough.
        if (
            self.current_event is None
            and candidate_duration >= self.minimum_duration
        ):
            self.current_event = {
                "behavior": self.candidate_behavior,
                "start_time": self.candidate_start,
                "end_time": timestamp,
                "duration_seconds": candidate_duration,
            }

            self.confirmed_events.append(self.current_event)
            newly_confirmed = True

        # Update the active event's end time.
        if self.current_event is not None:
            self.current_event["end_time"] = timestamp
            self.current_event["duration_seconds"] = (
                timestamp - self.current_event["start_time"]
            )

        return {
            "confirmed": self.current_event is not None,
            "newly_confirmed": newly_confirmed,
            "event": self.current_event,
        }

    def finalize(self):
        """Close the final event and return all confirmed events."""

        self._close_current_event()

        events = [
            event.copy()
            for event in self.confirmed_events
        ]

        self.reset()

        return events
