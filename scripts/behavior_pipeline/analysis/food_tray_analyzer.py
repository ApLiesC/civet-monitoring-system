
"""Analyze the proximity between a detected civet and food trays."""


class FoodTrayAnalyzer:
    """Determine whether a civet is sufficiently close to a food tray."""

    def __init__(self, proximity_threshold_ratio: float = 0.05):
        """
        Args:
            proximity_threshold_ratio:
                Maximum distance between the civet and tray, expressed
                as a fraction of the civet bounding-box height.

                For example, 0.05 means the civet must be within a
                distance of 5% of its bounding-box height.
        """
        if proximity_threshold_ratio < 0:
            raise ValueError("proximity_threshold_ratio must be >= 0.")

        self.proximity_threshold_ratio = proximity_threshold_ratio

    @staticmethod
    def _validate_bbox(bbox):
        """Check and normalize a bounding box."""
        if bbox is None or len(bbox) != 4:
            raise ValueError("A bounding box must contain four coordinates.")

        x1, y1, x2, y2 = map(float, bbox)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bounding box: {bbox}")

        return x1, y1, x2, y2

    @staticmethod
    def _bbox_distance(bbox_a, bbox_b):
        """
        Calculate the shortest distance between two bounding boxes.

        The distance is zero when the boxes overlap or touch.
        """
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        gap_x = max(bx1 - ax2, ax1 - bx2, 0.0)
        gap_y = max(by1 - ay2, ay1 - by2, 0.0)

        return (gap_x ** 2 + gap_y ** 2) ** 0.5

    def analyze(self, civet_detection, food_tray_detections):
        """
        Analyze the civet's proximity to all detected food trays.

        Args:
            civet_detection:
                Civet detection dictionary from CivetDetector:
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float,
                    "crop": ...
                }

            food_tray_detections:
                List of food tray detection dictionaries:
                [
                    {"bbox": (x1, y1, x2, y2), "confidence": float},
                    ...
                ]

        Returns:
            Dictionary containing proximity information.
        """

        if civet_detection is None:
            return {
                "is_near_tray": False,
                "nearest_tray": None,
                "distance_px": None,
                "normalized_distance": None,
                "threshold": self.proximity_threshold_ratio,
            }

        civet_bbox = self._validate_bbox(civet_detection["bbox"])
        civet_height = civet_bbox[3] - civet_bbox[1]

        if not food_tray_detections:
            return {
                "is_near_tray": False,
                "nearest_tray": None,
                "distance_px": None,
                "normalized_distance": None,
                "threshold": self.proximity_threshold_ratio,
            }

        nearest_tray = None
        nearest_distance = float("inf")
        nearest_index = None

        for index, tray_detection in enumerate(food_tray_detections):
            tray_bbox = self._validate_bbox(tray_detection["bbox"])

            distance = self._bbox_distance(civet_bbox, tray_bbox)

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_tray = tray_detection
                nearest_index = index

        normalized_distance = nearest_distance / civet_height

        is_near_tray = (
            normalized_distance <= self.proximity_threshold_ratio
        )

        return {
            "is_near_tray": is_near_tray,
            "nearest_tray": {
                "index": nearest_index,
                "bbox": nearest_tray["bbox"],
                "confidence": nearest_tray.get("confidence"),
            },
            "distance_px": round(nearest_distance, 2),
            "normalized_distance": round(normalized_distance, 4),
            "threshold": self.proximity_threshold_ratio,
        }
