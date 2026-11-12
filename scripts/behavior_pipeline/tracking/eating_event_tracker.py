
"""Track individual eating episodes and their durations."""


class EatingEventTracker:
    def __init__(
        self,
        minimum_duration=5.0,
        gap_tolerance=3.0,
    ):
        self.minimum_duration = minimum_duration
        self.gap_tolerance = gap_tolerance

        self.candidate_start = None
        self.last_eating_timestamp = None
        self.gap_start = None

        self.current_event = None
        self.confirmed_events = []

        self.next_event_id = 1

    def reset(self):
        """Reset the tracker before processing another video."""
        self.candidate_start = None
        self.last_eating_timestamp = None
        self.gap_start = None

        self.current_event = None
        self.confirmed_events.clear()

        self.next_event_id = 1

    def _start_candidate(self, timestamp):
        self.candidate_start = timestamp
        self.last_eating_timestamp = timestamp
        self.gap_start = None

    def _clear_candidate(self):
        self.candidate_start = None
        self.last_eating_timestamp = None
        self.gap_start = None

    def _confirm_event(self, timestamp):
        """Confirm an eating episode after the minimum duration."""
        if self.current_event is not None:
            return False

        if self.candidate_start is None:
            return False

        duration = timestamp - self.candidate_start

        if duration < self.minimum_duration:
            return False

        self.current_event = {
            "event_id": self.next_event_id,
            "behavior": "eating",
            "start_time": self.candidate_start,
            "end_time": timestamp,
            "duration_seconds": duration,
        }

        self.confirmed_events.append(self.current_event)
        self.next_event_id += 1

        return True

    def _close_event(self, end_timestamp):
        """Close the current confirmed eating episode."""
        if self.current_event is None:
            return

        end_timestamp = max(
            self.current_event["start_time"],
            end_timestamp,
        )

        self.current_event["end_time"] = end_timestamp
        self.current_event["duration_seconds"] = (
            end_timestamp - self.current_event["start_time"]
        )

        self.current_event = None

    def update(self, behavior, timestamp):
        """
        Update the eating tracker.

        Args:
            behavior:
                "eating" when eating is detected.
                Any other behavior, None, or "unknown" means
                eating is not currently confirmed by the classifier.
            timestamp:
                Video timestamp in seconds.

        Returns:
            Dictionary containing confirmation status and current event.
        """
        if (
            self.last_eating_timestamp is not None
            and timestamp < self.last_eating_timestamp
        ):
            raise ValueError("Timestamps must be non-decreasing.")

        is_eating = behavior == "eating"

        # -------------------------------------------------
        # Eating prediction
        # -------------------------------------------------
        if is_eating:

            # Start a new eating candidate.
            if self.candidate_start is None:
                self._start_candidate(timestamp)

            else:
                # Check whether an interruption was too long.
                if self.gap_start is not None:
                    gap_duration = timestamp - self.gap_start

                    if gap_duration > self.gap_tolerance:
                        # Previous episode ended at the beginning
                        # of the interruption.
                        self._close_event(self.gap_start)
                        self._clear_candidate()
                        self._start_candidate(timestamp)

                    else:
                        # Brief interruption: resume the same episode.
                        self.gap_start = None

                self.last_eating_timestamp = timestamp

            newly_confirmed = self._confirm_event(timestamp)

        # -------------------------------------------------
        # Non-eating or uncertain prediction
        # -------------------------------------------------
        else:
            newly_confirmed = False

            if self.candidate_start is not None:

                if self.gap_start is None:
                    self.gap_start = timestamp

                gap_duration = timestamp - self.gap_start

                if gap_duration > self.gap_tolerance:
                    self._close_event(self.gap_start)
                    self._clear_candidate()

        # Update the current confirmed event.
        if self.current_event is not None:
            if is_eating:
                self.current_event["end_time"] = timestamp
                self.current_event["duration_seconds"] = (
                    timestamp - self.current_event["start_time"]
                )

        return {
            "confirmed": self.current_event is not None,
            "newly_confirmed": newly_confirmed,
            "event": (
                self.current_event.copy()
                if self.current_event is not None
                else None
            ),
        }

    def finalize(self):
        """Close the final episode and return confirmed eating events."""

        if self.candidate_start is not None:

            if self.current_event is not None:
                # If the video ends during a brief interruption,
                # end at the last actual eating prediction.
                self._close_event(self.last_eating_timestamp)

        events = [
            event.copy()
            for event in self.confirmed_events
        ]

        self.reset()

        return events
