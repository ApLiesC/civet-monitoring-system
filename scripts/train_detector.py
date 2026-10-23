
from pathlib import Path

from ultralytics import YOLO


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = (
    PROJECT_ROOT
    / "data"
    / "civet_dataset"
    / "data.yaml"
)

# Training configuration
MODEL_NAME = "yolo11n.pt"
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 4
PATIENCE = 12


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    print("Loading pretrained YOLO11 nano model...")
    model = YOLO(MODEL_NAME)

    print(f"Dataset: {DATA_YAML}")
    print("Starting civet detection training...")

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        project=str(PROJECT_ROOT / "runs"),
        name="civet_detector_yolo11n",
        seed=42,
        plots=True,
        save=True,
        device="cpu",
    )

    print("Training finished.")
    print(
        "Best weights: "
        f"{PROJECT_ROOT / 'runs' / 'civet_detector_yolo11n' / 'weights' / 'best.pt'}"
    )


if __name__ == "__main__":
    main()
