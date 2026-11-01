
"""Track overall activity-state intervals for the civet."""


class ActivityStateManager:
    VALID_STATES = {"active", "inactive", "not_visible"}

    def __init__(
        self,
        transition_duration=3.0,
        unknown_tolerance=3.0,
        not_visible_tolerance=3.0,
    ):
        self.transition_duration = transition_duration
        self.unknown_tolerance = unknown_tolerance
        self.not_visible_tolerance = not_visible_tolerance

        self.current_state = None
        self.current_start = None

        self.pending_state = None
        self.pending_start = None

        self.unknown_since = None
        self.not_visible_since = None
        self.last_timestamp = None

        self.records = []

    def reset(self):
        """Reset the manager before processing another video."""
        self.current_state = None
        self.current_start = None

        self.pending_state = None
        self.pending_start = None

        self.unknown_since = None
        self.not_visible_since = None
        self.last_timestamp = None

        self.records.clear()

    def _clear_pending(self):
        self.pending_state = None
        self.pending_start = None

    def _close_current(self, end_timestamp):
        """Close the current interval."""
        if self.current_state is None:
            return

        end_timestamp = max(self.current_start, end_timestamp)

        self.records.append({
            "state": self.current_state,
            "start_time": self.current_start,
            "end_time": end_timestamp,
            "duration_seconds": end_timestamp - self.current_start,
        })

        self.current_state = None
        self.current_start = None

    def _start_state(self, state, start_timestamp):
        self.current_state = state
        self.current_start = start_timestamp

    def _accept_state(self, state, start_timestamp):
        """Immediately accept a state, preserving its original start time."""
        if state == self.current_state:
            return

        self._close_current(start_timestamp)
        self._start_state(state, start_timestamp)

    def _update_known_state(self, state, timestamp):
        """Process a known active/inactive prediction."""
        self.unknown_since = None
        self.not_visible_since = None

        if state == self.current_state:
            self._clear_pending()
            return

        if state != self.pending_state:
            self.pending_state = state
            self.pending_start = timestamp

        pending_duration = timestamp - self.pending_start

        if pending_duration >= self.transition_duration:
            self._accept_state(state, self.pending_start)
            self._clear_pending()

    def _update_unknown(self, timestamp, reason):
        """Handle uncertainty without incorrectly recording inactivity."""
        self._clear_pending()

        if reason == "detection":
            self.unknown_since = None

            if self.not_visible_since is None:
                self.not_visible_since = timestamp

            missing_duration = timestamp - self.not_visible_since

            if missing_duration >= self.not_visible_tolerance:
                self._accept_state(
                    "not_visible",
                    self.not_visible_since,
                )

        elif reason == "classification":
            self.not_visible_since = None

            if self.unknown_since is None:
                self.unknown_since = timestamp

            unknown_duration = timestamp - self.unknown_since

            if unknown_duration >= self.unknown_tolerance:
                # Do not label an uncertain period as inactive.
                # Close the previous interval and leave a gap.
                self._close_current(self.unknown_since)

        else:
            raise ValueError(
                "reason must be 'detection' or 'classification'"
            )

    def update(self, state, timestamp, unknown_reason=None):
        """
        Update the overall activity state.

        Args:
            state:
                "active", "inactive", or None/"unknown".
            timestamp:
                Video timestamp in seconds.
            unknown_reason:
                "detection" when the detector misses the civet.
                "classification" when its behavior is uncertain.

        Returns:
            A dictionary describing the current state and pending state.
        """
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("Timestamps must be non-decreasing.")

        self.last_timestamp = timestamp

        if state in (None, "unknown"):
            if unknown_reason is None:
                raise ValueError(
                    "unknown_reason is required for unknown predictions"
                )

            self._update_unknown(timestamp, unknown_reason)

        else:
            if state not in {"active", "inactive"}:
                raise ValueError(
                    "Input state must be 'active', 'inactive', "
                    "or 'unknown'."
                )

            self._update_known_state(state, timestamp)

        return {
            "state": self.current_state,
            "pending_state": self.pending_state,
            "pending_start": self.pending_start,
        }

    def finalize(self):
        """Close the final interval and return all completed records."""
        if self.current_state is not None and self.last_timestamp is not None:
            self._close_current(self.last_timestamp)

        completed_records = [record.copy() for record in self.records]

        self.reset()

        return completed_records
