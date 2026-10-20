
import json
from pathlib import Path
import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_DIR / "config" / "config.json"

VIDEO_DIR = PROJECT_DIR / "data" / "videos"
OUTPUT_DIR = PROJECT_DIR / "data" / "raw_frames"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_frames(video_path, interval_seconds):
    relative_path = video_path.relative_to(VIDEO_DIR)
    output_folder = (OUTPUT_DIR / relative_path).with_suffix("")
    output_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        print(f"Invalid video information: {video_path}")
        cap.release()
        return

    duration = total_frames / fps
    timestamp = 0.0
    extracted = 0

    while timestamp < duration:
        frame_index = round(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = cap.read()

        if success:
            filename = (
                f"{video_path.stem}_frame_{timestamp:07.2f}s.jpg"
            )
            output_path = output_folder / filename

            if cv2.imwrite(str(output_path), frame):
                extracted += 1

        timestamp += interval_seconds

    cap.release()
    print(f"{video_path.name}: extracted {extracted} frames")


def main():
    config = load_config()
    interval_seconds = float(config["frame_interval_seconds"])

    if interval_seconds <= 0:
        raise ValueError("frame_interval_seconds must be greater than 0")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        path for path in VIDEO_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        print(f"No videos found in {VIDEO_DIR}")
        return

    print(f"Frame interval: {interval_seconds} seconds")

    for video_path in videos:
        extract_frames(video_path, interval_seconds)


if __name__ == "__main__":
    main()
