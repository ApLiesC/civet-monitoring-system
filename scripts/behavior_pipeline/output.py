
"""Save annotated video and civet behavior records."""

import csv
import math
from pathlib import Path

import cv2


class BehaviorOutput:
    def __init__(
        self,
        output_dir,
        video_filename,
        csv_filename,
        activity_csv_filename="activity_records.csv",
        eating_csv_filename="eating_records.csv",
        frequency_csv_filename="eating_frequency.csv",
        frequency_interval_seconds=3600,
    ):
        self.output_dir = Path(output_dir)

        self.video_filename = video_filename
        self.csv_filename = csv_filename

        self.activity_csv_filename = activity_csv_filename
        self.eating_csv_filename = eating_csv_filename
        self.frequency_csv_filename = frequency_csv_filename

        self.frequency_interval_seconds = frequency_interval_seconds

        self.video_writer = None

        self.frame_csv_file = None
        self.frame_csv_writer = None

        self.source_fps = None
        self.output_fps = None

        self.video_path = None
        self.csv_path = None
        self.activity_csv_path = None
        self.eating_csv_path = None
        self.frequency_csv_path = None

    def start(self, source_fps, sample_interval):
        """Initialize output directories and CSV files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.video_path = self.output_dir / self.video_filename
        self.csv_path = self.output_dir / self.csv_filename

        self.activity_csv_path = (
            self.output_dir / self.activity_csv_filename
        )

        self.eating_csv_path = (
            self.output_dir / self.eating_csv_filename
        )

        self.frequency_csv_path = (
            self.output_dir / self.frequency_csv_filename
        )

        self.source_fps = source_fps
        self.output_fps = source_fps / sample_interval

        # Frame-level CSV.
        self.frame_csv_file = open(
            self.csv_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        self.frame_csv_writer = csv.writer(self.frame_csv_file)

        self.frame_csv_writer.writerow([
            "timestamp_seconds",
            "frame_index",
            "status",
            "detected",
            "detection_confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "movement_state",
            "movement_speed",
            "behavior",
            "activity_state",
            "pending_activity_state",
            "classification_confidence",
        ])

        print(f"Output video: {self.video_path}")
        print(f"Frame CSV: {self.csv_path}")
        print(f"Activity CSV: {self.activity_csv_path}")
        print(f"Eating CSV: {self.eating_csv_path}")
        print(f"Eating frequency CSV: {self.frequency_csv_path}")

    @staticmethod
    def format_timestamp(seconds):
        """Convert seconds into HH:MM:SS."""
        seconds = max(0, int(seconds))

        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{remaining_seconds:02d}"
        )

    @staticmethod
    def get_behavior_label(behavior, status):
        """Create a readable activity and behavior label."""

        if behavior == "locomotion":
            return "ACTIVE"

        if behavior == "eating":
            return "ACTIVE - Eating"

        if behavior == "resting":
            return "INACTIVE - Resting"

        status_labels = {
            "not_detected": "CIVET NOT DETECTED",
            "movement_unknown": "MOVEMENT UNKNOWN",
            "classification_failed": "CLASSIFICATION UNCERTAIN",
        }

        return status_labels.get(status, "UNKNOWN")

    @staticmethod
    def draw_label(frame, label, bbox=None):
        """Draw a readable label above the bounding box."""
        height, width = frame.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        thickness = 2
        padding = 8

        text_width, text_height = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )[0]

        label_width = text_width + padding * 2
        label_height = text_height + padding * 2

        if bbox is not None:
            x1, y1, _, _ = bbox

            label_x = max(0, int(x1))
            label_y = int(y1) - label_height

            # If there is insufficient space above the bounding box,
            # place the label inside the top of the bounding box.
            if label_y < 0:
                label_y = int(y1)

            label_x = min(
                label_x,
                max(0, width - label_width),
            )

            label_y = min(
                label_y,
                max(0, height - label_height),
            )

        else:
            label_x = 20
            label_y = 20

        # Draw label background.
        cv2.rectangle(
            frame,
            (label_x, label_y),
            (
                min(width - 1, label_x + label_width),
                min(height - 1, label_y + label_height),
            ),
            (0, 255, 255),
            -1,
        )

        # Draw label text.
        cv2.putText(
            frame,
            label,
            (
                label_x + padding,
                label_y + padding + text_height,
            ),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    def write(self, result):
        """Write one processed frame to video and frame-level CSV."""
        frame = result["frame"].copy()

        detection = result["detection"]
        movement = result["movement"]
        classification = result["classification"]

        behavior = result["behavior"]
        activity_state = result["activity_state"]
        pending_state = result["pending_activity_state"]

        # -------------------------------------------------
        # 1. Detection information
        # -------------------------------------------------

        detected = detection is not None
        detection_confidence = None

        bbox = None

        if detected:
            detection_confidence = detection["confidence"]

            bbox = [
                int(value)
                for value in detection["bbox"]
            ]

            x1, y1, x2, y2 = bbox

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

        # -------------------------------------------------
        # 2. Movement information
        # -------------------------------------------------

        movement_state = movement["state"]
        movement_speed = movement["speed"]

        # -------------------------------------------------
        # 3. Classification information
        # -------------------------------------------------

        classification_confidence = None

        if classification is not None:
            classification_confidence = classification.get(
                "confidence"
            )

        # -------------------------------------------------
        # 4. Write frame-level CSV
        # -------------------------------------------------

        self.frame_csv_writer.writerow([
            round(result["timestamp"], 3),
            result["frame_index"],
            result["status"],
            detected,
            detection_confidence,
            *(bbox if bbox is not None else [None] * 4),
            movement_state,
            movement_speed,
            behavior,
            activity_state,
            pending_state,
            classification_confidence,
        ])

        # -------------------------------------------------
        # 5. Display activity and behavior label
        # -------------------------------------------------

        label = self.get_behavior_label(
            behavior=behavior,
            status=result["status"],
        )

        self.draw_label(
            frame,
            label,
            bbox=bbox,
        )

        # -------------------------------------------------
        # 6. Display debugging information
        # -------------------------------------------------

        if movement_speed is not None:
            movement_text = (
                f"Movement: {movement_speed:.1f} px/sec"
            )
        else:
            movement_text = f"Movement: {movement_state}"

        cv2.putText(
            frame,
            movement_text,
            (20, frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if classification_confidence is not None:
            confidence_text = (
                f"Classification confidence: "
                f"{classification_confidence:.2f}"
            )

            cv2.putText(
                frame,
                confidence_text,
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # -------------------------------------------------
        # 7. Initialize and write annotated video
        # -------------------------------------------------

        if self.video_writer is None:
            height, width = frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            self.video_writer = cv2.VideoWriter(
                str(self.video_path),
                fourcc,
                self.output_fps,
                (width, height),
            )

            if not self.video_writer.isOpened():
                raise RuntimeError(
                    f"Could not create output video: {self.video_path}"
                )

        self.video_writer.write(frame)

    def write_activity_records(self, records):
        """Save overall activity-state intervals."""
        with open(
            self.activity_csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "state",
                "start_time_seconds",
                "end_time_seconds",
                "duration_seconds",
                "start_time",
                "end_time",
            ])

            for record in records:
                start = record["start_time"]
                end = record["end_time"]

                writer.writerow([
                    record["state"],
                    round(start, 3),
                    round(end, 3),
                    round(record["duration_seconds"], 3),
                    self.format_timestamp(start),
                    self.format_timestamp(end),
                ])

    def write_eating_records(self, events):
        """Save confirmed individual eating episodes."""
        with open(
            self.eating_csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "event_id",
                "start_time_seconds",
                "end_time_seconds",
                "duration_seconds",
                "start_time",
                "end_time",
            ])

            for event in events:
                start = event["start_time"]
                end = event["end_time"]

                writer.writerow([
                    event["event_id"],
                    round(start, 3),
                    round(end, 3),
                    round(event["duration_seconds"], 3),
                    self.format_timestamp(start),
                    self.format_timestamp(end),
                ])

    def write_eating_frequency(
        self,
        events,
        video_duration,
    ):
        """
        Count confirmed eating episodes by their start time.

        Intervals use [start, end) boundaries.
        """
        interval = self.frequency_interval_seconds

        if interval <= 0:
            raise ValueError(
                "frequency_interval_seconds must be greater than zero."
            )

        number_of_intervals = math.ceil(video_duration / interval)

        counts = [0] * number_of_intervals

        for event in events:
            start_time = event["start_time"]

            interval_index = int(start_time // interval)

            if 0 <= interval_index < number_of_intervals:
                counts[interval_index] += 1

        with open(
            self.frequency_csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "interval_start_seconds",
                "interval_end_seconds",
                "interval_start",
                "interval_end",
                "eating_episode_count",
            ])

            for index, count in enumerate(counts):
                start = index * interval
                end = min(
                    (index + 1) * interval,
                    video_duration,
                )

                writer.writerow([
                    round(start, 3),
                    round(end, 3),
                    self.format_timestamp(start),
                    self.format_timestamp(end),
                    count,
                ])

    def close(self):
        """Release video writer and close the frame CSV."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.frame_csv_file is not None:
            self.frame_csv_file.close()
            self.frame_csv_file = None
