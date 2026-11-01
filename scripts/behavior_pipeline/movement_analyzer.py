
"""Analyze civet movement using bounding-box center displacement."""

from collections import deque
from math import hypot


class MovementAnalyzer:
    """Determine whether a civet is moving or stationary."""

    def __init__(
        self,
        movement_threshold=70.0,
        movement_window=3,
    ):
        self.movement_threshold = movement_threshold
        self.movement_window = movement_window

        self.center_history = deque(maxlen=movement_window)

    def reset(self):
        """Clear movement history."""
        self.center_history.clear()

    def update(self, bbox, timestamp):
        """
        Update movement history and determine movement state.

        Args:
            bbox: Bounding box (x1, y1, x2, y2), or None.
            timestamp: Current frame timestamp in seconds.

        Returns:
            dict containing movement state and estimated speed.
        """

        # Reset history if the civet is not detected.
        if bbox is None:
            self.reset()

            return {
                "state": "unknown",
                "speed": None,
            }

        x1, y1, x2, y2 = bbox

        # Calculate bounding-box center.
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        self.center_history.append(
            (center_x, center_y, timestamp)
        )

        # Not enough samples to calculate movement yet.
        if len(self.center_history) < self.movement_window:
            return {
                "state": "unknown",
                "speed": None,
            }

        first = self.center_history[0]
        last = self.center_history[-1]

        elapsed_time = last[2] - first[2]

        if elapsed_time <= 0:
            return {
                "state": "unknown",
                "speed": None,
            }

        # Calculate net displacement between the two centers.
        displacement = hypot(
            last[0] - first[0],
            last[1] - first[1],
        )

        # Estimate movement speed in pixels per second.
        speed = displacement / elapsed_time

        if speed >= self.movement_threshold:
            state = "locomotion"
        else:
            state = "stationary"

        return {
            "state": state,
            "speed": speed,
        }
