
from pathlib import Path
import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_DIR / "data" / "videos"
OUTPUT_DIR = PROJECT_DIR / "data" / "raw_frames"

INTERVAL_SECONDS = 2.5
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def extract_frames(video_path):
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
            video_name = video_path.stem
            filename = f"{video_name}_frame_{timestamp:07.2f}s.jpg"
            output_path = output_folder / filename

            if cv2.imwrite(str(output_path), frame):
                extracted += 1

        timestamp += INTERVAL_SECONDS

    cap.release()
    print(f"{video_path.name}: extracted {extracted} frames")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        path for path in VIDEO_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        print(f"No videos found in {VIDEO_DIR}")
        return

    for video_path in videos:
        extract_frames(video_path)


if __name__ == "__main__":
    main()
