
"""Civet behavior classification using cropped images."""

from pathlib import Path

from ultralytics import YOLO


class CivetClassifier:
    """Classify a cropped civet image into a behavior category."""


    def __init__(
        self,
        model_path: Path,
        image_size: int = 224,
    ):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Classification model not found:\n{model_path}"
            )

        self.image_size = image_size

        print("Loading civet classification model...")
        self.model = YOLO(str(model_path))

        # Read class names directly from the trained model.
        self.class_names = self.model.names

        print(f"Classification model loaded: {model_path}")
        print(f"Available classes: {self.class_names}")


    def classify(self, crop):
        """
        Classify a cropped civet image.

        Args:
            crop: Cropped civet image (NumPy array).

        Returns:
            dict: Classification information, or None if invalid.

            {
                "behavior": str,
                "confidence": float,
                "class_id": int
            }
        """

        # Validate input image.
        if crop is None or crop.size == 0:
            return None

        if crop.ndim != 3 or crop.shape[2] != 3:
            return None

        # Run image classification.
        results = self.model.predict(
            source=crop,
            imgsz=self.image_size,
            verbose=False,
        )

        result = results[0]

        if result.probs is None:
            return None

        # Get the highest-probability class.
        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)

        # Get the corresponding behavior name.
        behavior = self.class_names[class_id]

        return {
            "behavior": behavior,
            "confidence": confidence,
            "class_id": class_id,
        }