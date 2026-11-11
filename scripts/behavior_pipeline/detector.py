
"""Civet and food tray detection with civet bounding box cropping."""

from pathlib import Path

import cv2
from ultralytics import YOLO


class CivetDetector:
    """Detect civets and food trays in a frame."""

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

        # Map class names to their class IDs.
        self.class_names = self.model.names

        self.civet_class_id = self._get_class_id("civet")
        self.food_tray_class_id = self._get_class_id("food_tray")

        print(f"Detection model loaded: {model_path}")
        print(f"Classes: {self.class_names}")

    def _get_class_id(self, class_name):
        """Get class ID from model class names."""

        for class_id, name in self.class_names.items():
            if name.lower() == class_name.lower():
                return class_id

        raise ValueError(
            f"Class '{class_name}' not found in detection model. "
            f"Available classes: {self.class_names}"
        )

    def _get_bbox(self, box):
        """Extract bounding box coordinates."""

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        return (
            int(x1),
            int(y1),
            int(x2),
            int(y2),
        )

    def _crop_civet(self, frame, bbox):
        """Extract a padded civet crop."""

        x1, y1, x2, y2 = bbox

        height, width = frame.shape[:2]

        box_width = x2 - x1
        box_height = y2 - y1

        pad_x = box_width * self.padding_ratio
        pad_y = box_height * self.padding_ratio

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(width, int(x2 + pad_x))
        y2 = min(height, int(y2 + pad_y))

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return {
            "bbox": (x1, y1, x2, y2),
            "crop": crop,
        }

    def detect(self, frame):
        """
        Detect civets and food trays in a frame.

        Returns:
            dict: Detection information.

            {
                "civet": {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float,
                    "crop": numpy.ndarray
                } or None,

                "food_trays": [
                    {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": float
                    }
                ]
            }
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )

        result = results[0]

        civet_detections = []
        food_trays = []

        if result.boxes is None or len(result.boxes) == 0:
            return {
                "civet": None,
                "food_trays": [],
            }

        # Process each detection according to its class.
        for box in result.boxes:

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            bbox = self._get_bbox(box)

            # Civet detection.
            if class_id == self.civet_class_id:

                crop_data = self._crop_civet(frame, bbox)

                if crop_data is None:
                    continue

                civet_detections.append({
                    "bbox": crop_data["bbox"],
                    "confidence": confidence,
                    "crop": crop_data["crop"],
                })

            # Food tray detection.
            elif class_id == self.food_tray_class_id:

                food_trays.append({
                    "bbox": bbox,
                    "confidence": confidence,
                })

        # Select the highest-confidence civet detection.
        civet = (
            max(
                civet_detections,
                key=lambda detection: detection["confidence"],
            )
            if civet_detections
            else None
        )

        return {
            "civet": civet,
            "food_trays": food_trays,
        }