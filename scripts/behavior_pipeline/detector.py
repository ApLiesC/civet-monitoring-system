
"""Civet detection and bounding box cropping."""

from pathlib import Path

import cv2
from ultralytics import YOLO


class CivetDetector:
    """Detect a civet and extract its bounding box crop."""


    def __init__(
        self,
        model_path: Path,
        confidence: float = 0.25,
        image_size: int = 640,
        padding_ratio: float = 0.10,
    ):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Detection model not found:\n{model_path}"
            )

        self.confidence = confidence
        self.image_size = image_size
        self.padding_ratio = padding_ratio

        print("Loading civet detection model...")
        self.model = YOLO(str(model_path))

        print(f"Detection model loaded: {model_path}")


    def detect(self, frame):
        """
        Detect the civet in a frame.

        Returns:
            dict: Detection information, or None if no civet is detected.

            {
                "bbox": (x1, y1, x2, y2),
                "confidence": float,
                "crop": numpy.ndarray
            }
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None

        # Select the highest-confidence detection.
        best_index = int(result.boxes.conf.argmax())
        box = result.boxes[best_index]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf.item())

        # Get frame dimensions.
        height, width = frame.shape[:2]

        # Calculate padding around the bounding box.
        box_width = x2 - x1
        box_height = y2 - y1

        pad_x = box_width * self.padding_ratio
        pad_y = box_height * self.padding_ratio

        # Apply padding and keep coordinates inside the frame.
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(width, int(x2 + pad_x))
        y2 = min(height, int(y2 + pad_y))

        # Extract civet crop.
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return {
            "bbox": (x1, y1, x2, y2),
            "confidence": confidence,
            "crop": crop,
        }