
"""Run the civet behavior monitoring pipeline."""

import json
from pathlib import Path

from detector import CivetDetector
from classifier import CivetClassifier
from movement_analyzer import MovementAnalyzer
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

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )


def format_cctv_timestamp(timestamp):
    """Format a CCTV datetime as HH:MM:SS."""
    if timestamp is None:
        return "Unavailable"

    return timestamp.strftime("%H:%M:%S")


def print_records(title, records):
    print(f"\n========== {title} ==========")

    if not records:
        print("No records.")
        return

    for index, record in enumerate(records, start=1):
        print(f"\nRecord {index}")

        if "event_id" in record:
            print(f"Event ID: {record['event_id']}")

        label = record.get("behavior", record.get("state"))
        print(f"State/Behavior: {label}")

        print(
            f"CCTV Start: "
            f"{format_cctv_timestamp(record.get('cctv_start_time'))}"
        )

        print(
            f"CCTV End: "
            f"{format_cctv_timestamp(record.get('cctv_end_time'))}"
        )

        print(
            f"Elapsed Start: "
            f"{format_timestamp(record['start_time'])}"
        )

        print(
            f"Elapsed End: "
            f"{format_timestamp(record['end_time'])}"
        )

        print(
            f"Duration: {record['duration_seconds']:.2f} seconds"
        )


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

    movement_config = config.get("movement", {})

    movement_analyzer = MovementAnalyzer(
        movement_threshold=movement_config.get(
            "movement_threshold",
            70.0,
        ),
        movement_window=movement_config.get(
            "movement_window",
            3,
        ),
    )

    # ---------------------------------------------------------
    # 3. Initialize behavior pipeline
    # ---------------------------------------------------------

    activity_config = config.get("activity", {})
    eating_config = config.get("eating", {})
    classification_config = config["classification"]

    processor = BehaviorVideoProcessor(
        detector=detector,
        classifier=classifier,
        movement_analyzer=movement_analyzer,

        sample_fps=config["video"]["sample_fps"],

        minimum_classification_confidence=classification_config.get(
            "minimum_confidence",
            0.70,
        ),

        activity_transition_duration=activity_config.get(
            "transition_duration",
            3.0,
        ),

        unknown_tolerance=activity_config.get(
            "unknown_tolerance",
            3.0,
        ),

        not_visible_tolerance=activity_config.get(
            "not_visible_tolerance",
            3.0,
        ),

        eating_minimum_duration=eating_config.get(
            "minimum_duration",
            5.0,
        ),

        eating_gap_tolerance=eating_config.get(
            "gap_tolerance",
            3.0,
        ),
    )

    # ---------------------------------------------------------
    # 4. Initialize output handler
    # ---------------------------------------------------------

    output_config = config["output"]

    output = BehaviorOutput(
        output_dir=resolve_path(output_config["directory"]),
        video_filename=output_config["video_filename"],
        csv_filename=output_config["csv_filename"],

        activity_csv_filename=output_config.get(
            "activity_csv_filename",
            "activity_records.csv",
        ),

        eating_csv_filename=output_config.get(
            "eating_csv_filename",
            "eating_records.csv",
        ),

        frequency_csv_filename=output_config.get(
            "frequency_csv_filename",
            "eating_frequency.csv",
        ),

        metadata_csv_filename=output_config.get(
            "metadata_csv_filename",
            "session_metadata.csv",
        ),

        frequency_interval_seconds=output_config.get(
            "frequency_interval_seconds",
            3600,
        ),
    )

    input_video = resolve_path(config["video"]["input"])

    output_started = False

    # ---------------------------------------------------------
    # 5. Processing statistics
    # ---------------------------------------------------------

    stats = {
        "total": 0,
        "detected": 0,
        "locomotion": 0,
        "classified": 0,
        "classification_failed": 0,
        "movement_unknown": 0,
        "not_detected": 0,
    }

    try:
        # -----------------------------------------------------
        # 6. Process video
        # -----------------------------------------------------

        for result in processor.process_video(input_video):

            # Start output after video metadata is initialized.
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

            if status == "locomotion":
                stats["locomotion"] += 1

            elif status == "classified":
                stats["classified"] += 1

            elif status == "classification_failed":
                stats["classification_failed"] += 1

            elif status == "movement_unknown":
                stats["movement_unknown"] += 1

            elif status == "not_detected":
                stats["not_detected"] += 1

        # -----------------------------------------------------
        # 7. Save activity, eating, and session records
        # -----------------------------------------------------

        activity_records = processor.activity_records
        eating_events = processor.eating_events

        if output_started:

            output.write_activity_records(activity_records)

            output.write_eating_records(eating_events)

            output.write_eating_frequency(
                events=eating_events,
                video_duration=processor.duration_seconds,
                recording_start_timestamp=(
                    processor.recording_start_timestamp
                ),
            )

            output.write_session_metadata(
                recording_date=processor.recording_date,
                recording_start_timestamp=(
                    processor.recording_start_timestamp
                ),
                video_duration=processor.duration_seconds,
            )

        # -----------------------------------------------------
        # 8. Print processing summary
        # -----------------------------------------------------

        print("\n========== PROCESSING SUMMARY ==========")

        print(f"Processed frames: {stats['total']}")
        print(f"Frames with detection: {stats['detected']}")

        print(f"Locomotion frames: {stats['locomotion']}")
        print(f"Classified frames: {stats['classified']}")

        print(
            f"Classification failed frames: "
            f"{stats['classification_failed']}"
        )

        print(
            f"Movement unknown frames: "
            f"{stats['movement_unknown']}"
        )

        print(f"Frames without detection: {stats['not_detected']}")

        print(
            f"\nTotal activity intervals: {len(activity_records)}"
        )

        print(
            f"Total confirmed eating episodes: {len(eating_events)}"
        )

        print(f"Recording date: {processor.recording_date or 'Unavailable'}")

        print(
            "Recording start: "
            f"{processor.recording_start_timestamp or 'Unavailable'}"
        )

        print_records("ACTIVITY RECORDS", activity_records)
        print_records("EATING RECORDS", eating_events)

        if output_started:
            print("\nOutput files saved successfully.")
        else:
            print("\nNo video frames were processed.")

    finally:
        output.close()


if __name__ == "__main__":
    main()
