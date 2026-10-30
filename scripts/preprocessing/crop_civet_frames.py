
from pathlib import Path
import csv

import cv2
from ultralytics import YOLO


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "civet_detector_yolo11n"
    / "weights"
    / "best.pt"
)

# Change this to the folder containing your enhanced frames.
INPUT_DIR = PROJECT_DIR / "data" / "enhanced_frames"

# Crops will be saved here.
OUTPUT_DIR = PROJECT_DIR / "data" / "behavior_crops" / "unlabeled"

CONFIDENCE_THRESHOLD = 0.35

# Extra space around the detected bounding box.
# 0.20 means 20% of the box width/height on each side.
CROP_PADDING = 0.20

# Set to True to save grayscale crops.
# Set to False to preserve the input image's channels.
SAVE_GRAYSCALE = True

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}


# ==================================================
# MAIN
# ==================================================

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Enhanced frame folder not found: {INPUT_DIR}\n"
            "Update INPUT_DIR to your actual folder."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in INPUT_DIR.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(f"No images found in {INPUT_DIR}")
        return

    print(f"Model: {MODEL_PATH}")
    print(f"Input: {INPUT_DIR}")
    print(f"Images found: {len(image_paths)}")

    model = YOLO(str(MODEL_PATH))

    manifest_path = OUTPUT_DIR / "manifest.csv"

    saved = 0
    no_detection = 0
    failed = 0

    with manifest_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "source_image",
                "confidence",
                "x1", "y1", "x2", "y2",
                "crop_path",
                "behavior_label",
            ],
        )
        writer.writeheader()

        for index, image_path in enumerate(
            image_paths, start=1
        ):
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"[SKIP] Cannot read: {image_path}")
                failed += 1
                continue

            # The frames are already enhanced.
            # Do not apply the enhancement pipeline again.
            result = model.predict(
                source=image,
                conf=CONFIDENCE_THRESHOLD,
                max_det=10,
                verbose=False,
            )[0]

            if result.boxes is None or len(result.boxes) == 0:
                no_detection += 1
                continue

            # Assumption: only one civet is present.
            # Keep the highest-confidence detection.
            best_index = int(
                result.boxes.conf.argmax().item()
            )

            box = result.boxes[best_index]
            x1, y1, x2, y2 = (
                box.xyxy[0].cpu().numpy()
            )
            confidence = float(
                box.conf[0].cpu().item()
            )

            height, width = image.shape[:2]

            box_width = x2 - x1
            box_height = y2 - y1

            pad_x = box_width * CROP_PADDING
            pad_y = box_height * CROP_PADDING

            # Expand the box and clamp it to image boundaries.
            x1 = max(0, int(x1 - pad_x))
            y1 = max(0, int(y1 - pad_y))
            x2 = min(width, int(x2 + pad_x))
            y2 = min(height, int(y2 + pad_y))

            if x2 <= x1 or y2 <= y1:
                no_detection += 1
                continue

            crop = image[y1:y2, x1:x2].copy()

            if SAVE_GRAYSCALE:
                crop = cv2.cvtColor(
                    crop, cv2.COLOR_BGR2GRAY
                )

            # Preserve the relative input folder structure
            # to avoid filename collisions.
            relative_path = image_path.relative_to(INPUT_DIR)
            crop_name = (
                f"{relative_path.stem}_civet.jpg"
            )
            crop_path = (
                OUTPUT_DIR
                / relative_path.parent
                / crop_name
            )
            crop_path.parent.mkdir(
                parents=True, exist_ok=True
            )

            if not cv2.imwrite(str(crop_path), crop):
                print(f"[SKIP] Cannot save: {crop_path}")
                failed += 1
                continue

            writer.writerow({
                "source_image": str(image_path),
                "confidence": round(confidence, 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop_path": str(crop_path),
                "behavior_label": "",
            })

            saved += 1

            if index % 100 == 0:
                print(
                    f"Processed {index}/{len(image_paths)} "
                    f"| crops saved: {saved}"
                )

    print("\n========== COMPLETE ==========")
    print(f"Images scanned: {len(image_paths)}")
    print(f"Crops saved: {saved}")
    print(f"No detection: {no_detection}")
    print(f"Failed reads/saves: {failed}")
    print(f"Crop folder: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
