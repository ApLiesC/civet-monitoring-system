from pathlib import Path
from ultralytics import YOLO

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "civet_detector_yolo11n-2"
    / "weights"
    / "best.pt"
)

VIDEO_PATH = (
    PROJECT_DIR
    / "data"
    / "videos"
    / "locomotion11.mp4"
)

model = YOLO(str(MODEL_PATH))

model.predict(
    source=str(VIDEO_PATH),
    conf=0.05,
    iou=0.5,
    save=True,
    show=True,
    verbose=False,
)