
"""Save annotated video and civet behavior records."""

import csv
from datetime import datetime, time, timedelta
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
        metadata_csv_filename="session_metadata.csv",
        frequency_interval_seconds=3600,
    ):
        self.output_dir = Path(output_dir)

        self.video_filename = video_filename
        self.csv_filename = csv_filename

        self.activity_csv_filename = activity_csv_filename
        self.eating_csv_filename = eating_csv_filename
        self.frequency_csv_filename = frequency_csv_filename
        self.metadata_csv_filename = metadata_csv_filename

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
        self.metadata_csv_path = None

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

        self.metadata_csv_path = (
            self.output_dir / self.metadata_csv_filename
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
            "cctv_timestamp",
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
        print(f"Session metadata CSV: {self.metadata_csv_path}")

    @staticmethod
    def format_timestamp(seconds):
        """Convert elapsed seconds into HH:MM:SS."""
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
    def format_cctv_time(timestamp):
        """Convert a CCTV datetime or time into HH:MM:SS."""
        if timestamp is None:
            return ""

        return timestamp.strftime("%H:%M:%S")

    @staticmethod
    def format_cctv_timestamp(timestamp):
        """
        Format a CCTV timestamp for CSV output.

        datetime -> YYYY-MM-DD HH:MM:SS
        time     -> HH:MM:SS
        None     -> empty string
        """
        if timestamp is None:
            return ""

        if isinstance(timestamp, datetime):
            return timestamp.isoformat(
                sep=" ",
                timespec="seconds",
            )

        if isinstance(timestamp, time):
            return timestamp.isoformat(
                timespec="seconds",
            )

        return str(timestamp)

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

        # 1. Detection information.
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

        # 2. Movement information.
        movement_state = movement["state"]
        movement_speed = movement["speed"]

        # 3. Classification information.
        classification_confidence = None

        if classification is not None:
            classification_confidence = classification.get(
                "confidence"
            )

        # 4. Write frame-level CSV.
        cctv_timestamp = self.format_cctv_timestamp(
            result.get("cctv_timestamp")
        )

        self.frame_csv_writer.writerow([
            round(result["timestamp"], 3),
            cctv_timestamp,
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

        # 5. Display activity and behavior label.
        label = self.get_behavior_label(
            behavior=behavior,
            status=result["status"],
        )

        self.draw_label(
            frame,
            label,
            bbox=bbox,
        )

        # 6. Display debugging information.
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

        # 7. Initialize and write annotated video.
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

    def write_session_metadata(
        self,
        recording_date,
        recording_start_timestamp,
        video_duration,
    ):
        """Save recording information once for the processing session."""

        recording_end_timestamp = None

        if recording_start_timestamp is not None:
            recording_end_timestamp = (
                recording_start_timestamp
                + timedelta(seconds=video_duration)
            )

        with open(
            self.metadata_csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(["metadata", "value"])

            writer.writerow([
                "recording_date",
                recording_date or "",
            ])

            writer.writerow([
                "recording_start_timestamp",
                self.format_cctv_timestamp(
                    recording_start_timestamp
                ),
            ])

            writer.writerow([
                "recording_end_timestamp",
                self.format_cctv_timestamp(
                    recording_end_timestamp
                ),
            ])

            writer.writerow([
                "video_duration_seconds",
                round(video_duration, 3),
            ])

            writer.writerow([
                "source_fps",
                self.source_fps,
            ])

            writer.writerow([
                "output_fps",
                self.output_fps,
            ])

    
    def write_activity_records(self, records):
        """
        Save activity intervals as paired active/inactive cycles,
        with separate columns for not_visible intervals.
        """

        rows = []
        pending_active_row = None

        for record in records:
            state = record["state"]

            start = self.format_cctv_time(
                record.get("cctv_start_time")
            )

            end = self.format_cctv_time(
                record.get("cctv_end_time")
            )

            if state == "active":
                if pending_active_row is not None:
                    rows.append(pending_active_row)

                pending_active_row = [
                    start, end,
                    "", "",
                    "", "",
                ]

            elif state == "inactive":
                if pending_active_row is not None:
                    pending_active_row[2] = start
                    pending_active_row[3] = end

                    rows.append(pending_active_row)
                    pending_active_row = None

                else:
                    rows.append([
                        "", "",
                        start, end,
                        "", "",
                    ])

            elif state == "not_visible":
                if pending_active_row is not None:
                    rows.append(pending_active_row)
                    pending_active_row = None

                rows.append([
                    "", "",
                    "", "",
                    start, end,
                ])

        if pending_active_row is not None:
            rows.append(pending_active_row)

        with open(
            self.activity_csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "No.",
                "Active Start",
                "Active Stop",
                "Inactive Start",
                "Inactive Stop",
                "Not Visible Start",
                "Not Visible Stop",
            ])

            for index, row in enumerate(rows, start=1):
                writer.writerow([index, *row])


    def write_eating_records(self, events):
        """Save confirmed individual eating episodes using CCTV time."""

        with open(
            self.eating_csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "No.",
                "Behavior",
                "Start Time",
                "End Time",
                "Duration (seconds)",
            ])

            for index, event in enumerate(events, start=1):
                writer.writerow([
                    index,
                    event.get("behavior", "eating").capitalize(),
                    self.format_cctv_time(
                        event.get("cctv_start_time")
                    ),
                    self.format_cctv_time(
                        event.get("cctv_end_time")
                    ),
                    round(event["duration_seconds"], 3),
                ])

    def write_eating_frequency(
        self,
        events,
        video_duration,
        recording_start_timestamp=None,
    ):
        """
        Count confirmed eating episodes by their CCTV start time.

        Hourly intervals use [start, end) boundaries.
        Hours with zero eating episodes are included.
        """

        interval = self.frequency_interval_seconds

        if interval <= 0:
            raise ValueError(
                "frequency_interval_seconds must be greater than zero."
            )

        if recording_start_timestamp is None:
            print(
                "[WARNING] CCTV recording timestamp unavailable. "
                "Eating frequency cannot be grouped by CCTV hour."
            )

            with open(
                self.frequency_csv_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.writer(file)
                writer.writerow([
                    "No.",
                    "Duration",
                    "Behavior",
                    "Frequency",
                ])

            return

        recording_end_timestamp = (
            recording_start_timestamp
            + timedelta(seconds=video_duration)
        )

        first_interval_start = recording_start_timestamp.replace(
            minute=0,
            second=0,
            microsecond=0,
        )

        intervals = []
        current_start = first_interval_start

        while current_start < recording_end_timestamp:
            current_end = current_start + timedelta(
                seconds=interval
            )

            intervals.append({
                "start": current_start,
                "end": current_end,
                "count": 0,
            })

            current_start = current_end

        for event in events:
            event_start = event.get("cctv_start_time")

            if event_start is None:
                continue

            for time_interval in intervals:
                if (
                    time_interval["start"]
                    <= event_start
                    < time_interval["end"]
                ):
                    time_interval["count"] += 1
                    break

        with open(
            self.frequency_csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "No.",
                "Duration",
                "Behavior",
                "Frequency",
            ])

            for index, time_interval in enumerate(
                intervals,
                start=1,
            ):
                start_label = time_interval["start"].strftime(
                    "%H:%M"
                )

                end_label = time_interval["end"].strftime(
                    "%H:%M"
                )

                writer.writerow([
                    index,
                    f"{start_label}–{end_label}",
                    "Eating",
                    time_interval["count"],
                ])

    def close(self):
        """Release video writer and close the frame CSV."""

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.frame_csv_file is not None:
            self.frame_csv_file.close()
            self.frame_csv_file = None
