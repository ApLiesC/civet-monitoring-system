
import json
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config" / "config.json"

INPUT_DIR = PROJECT_DIR / "data" / "selected_frames"
OUTPUT_DIR = PROJECT_DIR / "data" / "enhanced_frames"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def enhance_frame(frame, settings, gamma_table):
    enhanced = frame

    # 1. Convert to grayscale if enabled
    if settings["grayscale"]:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    # 2. Apply denoising if enabled
    if settings["denoise"]:
        strength = float(settings["denoise_strength"])

        if enhanced.ndim == 2:
            # Grayscale denoising
            enhanced = cv2.fastNlMeansDenoising(
                enhanced,
                None,
                h=strength
            )
        else:
            # Color denoising
            enhanced = cv2.fastNlMeansDenoisingColored(
                enhanced,
                None,
                h=strength,
                hColor=strength
            )

    # 3. Apply CLAHE if enabled
    if settings["clahe"]:
        clahe = cv2.createCLAHE(
            clipLimit=float(settings["clahe_clip"]),
            tileGridSize=(
                int(settings["clahe_grid"]),
                int(settings["clahe_grid"])
            )
        )

        if enhanced.ndim == 2:
            enhanced = clahe.apply(enhanced)
        else:
            # Apply CLAHE to brightness while preserving color
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            lightness, a, b = cv2.split(lab)
            lightness = clahe.apply(lightness)
            lab = cv2.merge((lightness, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 4. Apply gamma correction if enabled
    if gamma_table is not None:
        enhanced = cv2.LUT(enhanced, gamma_table)

    return enhanced


def main():
    config = load_config()
    settings = config["enhancement"]

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

    # Build the gamma lookup table once
    gamma_table = None

    if settings["gamma"] is not None:
        gamma = float(settings["gamma"])

        if gamma <= 0:
            raise ValueError("gamma must be greater than 0")

        gamma_table = np.clip(
            np.rint(
                255 * (
                    (np.arange(256, dtype=np.float32) / 255.0)
                    ** gamma
                )
            ),
            0,
            255
        ).astype(np.uint8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Grayscale: {settings['grayscale']}")
    print(f"Denoise: {settings['denoise']}")
    print(f"CLAHE: {settings['clahe']}")
    print(f"Gamma: {settings['gamma']}")

    processed = 0

    for image_path in images:
        relative_path = image_path.relative_to(INPUT_DIR)
        output_path = (OUTPUT_DIR / relative_path).with_suffix(".jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"Could not read: {image_path}")
            continue

        enhanced = enhance_frame(frame, settings, gamma_table)

        if cv2.imwrite(str(output_path), enhanced):
            processed += 1

    print(f"Enhancement complete: {processed}/{len(images)} images")


if __name__ == "__main__":
    main()
