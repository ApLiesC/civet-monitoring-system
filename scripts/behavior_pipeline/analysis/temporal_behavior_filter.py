
"""Filter short behavior fluctuations using transition grace periods."""


class TemporalBehaviorFilter:
    def __init__(
        self,
        transition_duration=3.0,
        unknown_tolerance=3.0,
        pending_reversion_tolerance=0.75,
    ):
        self.transition_duration = transition_duration
        self.unknown_tolerance = unknown_tolerance
        self.pending_reversion_tolerance = pending_reversion_tolerance

        self.current_behavior = None
        self.current_start_timestamp = None

        self.pending_behavior = None
        self.pending_start_timestamp = None
        self.pending_interruption_since = None

        self.unknown_since = None

    def reset(self):
        """Reset all temporal filtering state."""
        self.current_behavior = None
        self.current_start_timestamp = None

        self.pending_behavior = None
        self.pending_start_timestamp = None
        self.pending_interruption_since = None

        self.unknown_since = None

    def _clear_pending(self):
        self.pending_behavior = None
        self.pending_start_timestamp = None
        self.pending_interruption_since = None

    def _start_pending(self, behavior, timestamp):
        self.pending_behavior = behavior
        self.pending_start_timestamp = timestamp
        self.pending_interruption_since = None

    def update(self, behavior, timestamp):
        """
        Filter a raw behavior prediction.

        Args:
            behavior: Raw prediction such as locomotion, Eating, Resting,
                      or None for unknown.
            timestamp: Current video timestamp in seconds.

        Returns:
            Dictionary containing the filtered behavior and its original
            start timestamp.
        """

        is_unknown = behavior is None or behavior == "unknown"

        # Handle unknown predictions.
        if is_unknown:
            if self.current_behavior is None:
                self._clear_pending()
                return self._result(None, None, False)

            if self.unknown_since is None:
                self.unknown_since = timestamp

            unknown_duration = timestamp - self.unknown_since

            # Preserve the current behavior temporarily.
            if unknown_duration <= self.unknown_tolerance:
                if self.pending_behavior is not None:
                    if self.pending_interruption_since is None:
                        self.pending_interruption_since = timestamp

                    interruption_duration = (
                        timestamp - self.pending_interruption_since
                    )

                    if (
                        interruption_duration
                        > self.pending_reversion_tolerance
                    ):
                        self._clear_pending()

                return self._result(
                    self.current_behavior,
                    self.current_start_timestamp,
                    False,
                )

            # Unknown for too long: discard both current and pending behavior.
            self.current_behavior = None
            self.current_start_timestamp = None
            self._clear_pending()
            self.unknown_since = None

            return self._result(None, None, False)

        # A known prediction has returned.
        self.unknown_since = None

        # No current behavior: establish the first behavior.
        if self.current_behavior is None:
            if self.pending_behavior != behavior:
                self._start_pending(behavior, timestamp)

            return self._check_transition(timestamp)

        # Prediction matches the current behavior.
        if behavior == self.current_behavior:
            if self.pending_behavior is not None:
                if self.pending_interruption_since is None:
                    self.pending_interruption_since = timestamp

                interruption_duration = (
                    timestamp - self.pending_interruption_since
                )

                # A brief return to the current behavior does not reset
                # the pending transition timer.
                if (
                    interruption_duration
                    > self.pending_reversion_tolerance
                ):
                    self._clear_pending()

            return self._result(
                self.current_behavior,
                self.current_start_timestamp,
                False,
            )

        # Prediction matches the pending behavior.
        if behavior == self.pending_behavior:
            self.pending_interruption_since = None
            return self._check_transition(timestamp)

        # A third behavior appeared: start a new pending transition.
        self._start_pending(behavior, timestamp)

        return self._check_transition(timestamp)

    def _check_transition(self, timestamp):
        """Accept a pending behavior once it survives the grace period."""

        if self.pending_behavior is None:
            return self._result(
                self.current_behavior,
                self.current_start_timestamp,
                False,
            )

        pending_duration = timestamp - self.pending_start_timestamp

        if pending_duration >= self.transition_duration:
            self.current_behavior = self.pending_behavior

            # Preserve the original beginning of the new behavior.
            self.current_start_timestamp = self.pending_start_timestamp

            self._clear_pending()

            return self._result(
                self.current_behavior,
                self.current_start_timestamp,
                True,
            )

        return self._result(
            self.current_behavior,
            self.current_start_timestamp,
            False,
        )

    def _result(self, behavior, start_timestamp, transition_accepted):
        return {
            "behavior": behavior,
            "behavior_start_timestamp": start_timestamp,
            "transition_accepted": transition_accepted,
            "pending_behavior": self.pending_behavior,
        }
