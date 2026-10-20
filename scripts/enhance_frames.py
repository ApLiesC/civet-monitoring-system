
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_DIR / "data" / "selected_frames"
OUTPUT_DIR = PROJECT_DIR / "data" / "enhanced_frames"

# Enhancement settings from the testing configuration
CLAHE_CLIP = 2.0
CLAHE_GRID = 8
GAMMA = 0.85
GAUSSIAN_KERNEL = (3, 3)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def enhance_frame(frame):
    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Apply mild smoothing
    gray = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)

    # 3. Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP,
        tileGridSize=(CLAHE_GRID, CLAHE_GRID)
    )
    enhanced = clahe.apply(gray)

    # 4. Apply gamma correction
    lookup_table = np.clip(
        np.rint(
            255 * ((np.arange(256) / 255.0) ** GAMMA)
        ),
        0,
        255
    ).astype(np.uint8)

    enhanced = cv2.LUT(enhanced, lookup_table)

    return enhanced


def main():
    if not INPUT_DIR.exists():
        print(f"Input folder not found: {INPUT_DIR}")
        return

    images = sorted(
        path for path in INPUT_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"No selected images found in {INPUT_DIR}")
        return

    processed = 0

    for image_path in images:
        relative_path = image_path.relative_to(INPUT_DIR)
        output_path = (OUTPUT_DIR / relative_path).with_suffix(".jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"Could not read: {image_path}")
            continue

        enhanced = enhance_frame(frame)

        if cv2.imwrite(str(output_path), enhanced):
            processed += 1

    print(f"Enhancement complete: {processed}/{len(images)} images")


if __name__ == "__main__":
    main()
