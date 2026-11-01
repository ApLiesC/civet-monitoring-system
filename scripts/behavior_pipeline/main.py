
"""Run the civet behavior monitoring pipeline."""

import json
from pathlib import Path

from detector import CivetDetector
from classifier import CivetClassifier
from movement_analyzer import MovementAnalyzer
from temporal_behavior_filter import TemporalBehaviorFilter
from behavior_confirmer import BehaviorConfirmer
from video_processor import BehaviorVideoProcessor
from output import BehaviorOutput


PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_DIR / "configs" / "behavior_pipeline_config.json"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_DIR / path


def format_timestamp(seconds):
    seconds = max(0, int(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def main():
    config = load_config()

    # ---------------------------------------------------------
    # 1. Load models
    # ---------------------------------------------------------

    detector = CivetDetector(
        model_path=resolve_path(config["models"]["detector"]),
        confidence=config["detection"]["confidence"],
        image_size=config["detection"]["image_size"],
    )

    classifier = CivetClassifier(
        model_path=resolve_path(config["models"]["classifier"]),
        image_size=config["classification"]["image_size"],
    )

    # ---------------------------------------------------------
    # 2. Initialize movement analyzer
    # ---------------------------------------------------------

    movement_analyzer = MovementAnalyzer(
        movement_threshold=50.0,
        movement_window=3,
    )

    # ---------------------------------------------------------
    # 3. Initialize temporal behavior filter
    # ---------------------------------------------------------

    temporal_behavior_filter = TemporalBehaviorFilter(
        transition_duration=3.0,
        unknown_tolerance=3.0,
    )

    # ---------------------------------------------------------
    # 4. Initialize behavior confirmer
    # ---------------------------------------------------------

    behavior_confirmer = BehaviorConfirmer(
        minimum_duration=60.0,
        gap_tolerance=0.0,
    )

    # ---------------------------------------------------------
    # 5. Initialize video processor
    # ---------------------------------------------------------

    processor = BehaviorVideoProcessor(
        detector=detector,
        classifier=classifier,
        movement_analyzer=movement_analyzer,
        temporal_behavior_filter=temporal_behavior_filter,
        behavior_confirmer=behavior_confirmer,
        sample_fps=config["video"]["sample_fps"],
    )

    # ---------------------------------------------------------
    # 6. Initialize output handler
    # ---------------------------------------------------------

    output_config = config["output"]

    output = BehaviorOutput(
        output_dir=resolve_path(output_config["directory"]),
        video_filename=output_config["video_filename"],
        csv_filename=output_config["csv_filename"],
        event_csv_filename=output_config.get(
            "event_csv_filename",
            "confirmed_behavior_records.csv",
        ),
    )

    input_video = resolve_path(config["video"]["input"])

    output_started = False

    # Processing statistics
    stats = {
        "total": 0,
        "detected": 0,
        "locomotion": 0,
        "stationary_wait": 0,
        "movement_unknown": 0,
        "classified": 0,
        "classification_failed": 0,
        "not_detected": 0,
    }

    try:
        # -----------------------------------------------------
        # 7. Process video
        # -----------------------------------------------------

        for result in processor.process_video(input_video):

            # Start output after the processor initializes video metadata.
            if not output_started:
                output.start(
                    source_fps=processor.source_fps,
                    sample_interval=processor.sample_interval,
                )
                output_started = True

            output.write(result)

            stats["total"] += 1

            if result["detection"] is not None:
                stats["detected"] += 1

            status = result["status"]

            if status in stats:
                stats[status] += 1

        # -----------------------------------------------------
        # 8. Save confirmed behavior events
        # -----------------------------------------------------

        confirmed_events = processor.confirmed_events

        if output_started:
            output.write_confirmed_events(confirmed_events)

        # -----------------------------------------------------
        # 9. Print processing summary
        # -----------------------------------------------------

        print("\n========== PROCESSING SUMMARY ==========")

        print(f"Processed frames: {stats['total']}")
        print(f"Frames with detection: {stats['detected']}")

        print(f"Locomotion frames: {stats['locomotion']}")
        print(f"Stationary wait frames: {stats['stationary_wait']}")
        print(f"Movement unknown frames: {stats['movement_unknown']}")

        print(f"Classified frames: {stats['classified']}")
        print(
            f"Classification failed frames: "
            f"{stats['classification_failed']}"
        )
        print(f"Frames without detection: {stats['not_detected']}")

        print(f"\nConfirmed behavior events: {len(confirmed_events)}")

        for index, event in enumerate(confirmed_events, start=1):
            print(f"\nEvent {index}")
            print(f"Behavior: {event['behavior']}")
            print(
                f"Start: {format_timestamp(event['start_time'])}"
            )
            print(
                f"End: {format_timestamp(event['end_time'])}"
            )
            print(
                f"Duration: {event['duration_seconds']:.2f} seconds"
            )

        if output_started:
            print("\nOutput files saved successfully.")
        else:
            print("\nNo video frames were processed.")

    finally:
        output.close()


if __name__ == "__main__":
    main()
