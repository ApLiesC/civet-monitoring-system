
"""Save annotated video, frame-level predictions and confirmed behavior records."""

import csv
from pathlib import Path

import cv2


class BehaviorOutput:
    def __init__(
        self,
        output_dir,
        video_filename,
        csv_filename,
        event_csv_filename="confirmed_behavior_records.csv",
    ):
        self.output_dir = Path(output_dir)
        self.video_filename = video_filename
        self.csv_filename = csv_filename
        self.event_csv_filename = event_csv_filename

        self.video_writer = None

        self.frame_csv_file = None
        self.frame_csv_writer = None

        self.event_csv_file = None
        self.event_csv_writer = None

    def start(self, source_fps, sample_interval):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.video_path = self.output_dir / self.video_filename

        # Frame-level CSV
        self.csv_path = self.output_dir / self.csv_filename

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
            "raw_behavior",
            "filtered_behavior",
            "behavior_start_timestamp",
            "classification_confidence",
            "behavior_confirmed",
        ])

        # Confirmed event-level CSV
        self.event_csv_path = (
            self.output_dir / self.event_csv_filename
        )

        self.event_csv_file = open(
            self.event_csv_path,
            "w",
            newline="",
            encoding="utf-8",
        )

        self.event_csv_writer = csv.writer(self.event_csv_file)

        self.event_csv_writer.writerow([
            "behavior",
            "start_time_seconds",
            "end_time_seconds",
            "duration_seconds",
            "start_time",
            "end_time",
        ])

        self.source_fps = source_fps
        self.output_fps = source_fps / sample_interval

        print(f"Output video: {self.video_path}")
        print(f"Frame-level CSV: {self.csv_path}")
        print(f"Confirmed events CSV: {self.event_csv_path}")

    @staticmethod
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

    @staticmethod
    def draw_label(frame, label, bbox=None):
        """Draw a label above the bounding box."""

        height, width = frame.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2
        padding = 6

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

            label_x = max(0, x1)
            label_y = y1 - label_height

            # If there is not enough space above the box,
            # place the label inside the top of the box.
            if label_y < 0:
                label_y = y1

            label_x = min(
                label_x,
                max(0, width - label_width),
            )

            label_y = min(
                label_y,
                max(0, height - label_height),
            )

        else:
            # No bounding box available.
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
        frame = result["frame"].copy()

        detection = result["detection"]
        movement = result["movement"]
        classification = result["classification"]

        raw_behavior = result["predicted_behavior"]
        filtered_behavior = result["filtered_behavior"]

        behavior_start_timestamp = (
            result["behavior_start_timestamp"]
        )

        confirmation = result["confirmation"]

        # -----------------------------------------------------
        # 1. Detection information
        # -----------------------------------------------------

        detected = detection is not None
        detection_confidence = None

        bbox = [None, None, None, None]

        if detected:
            detection_confidence = detection["confidence"]
            bbox = list(detection["bbox"])

            x1, y1, x2, y2 = bbox

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

        # -----------------------------------------------------
        # 2. Movement information
        # -----------------------------------------------------

        movement_state = movement["state"]
        movement_speed = movement["speed"]

        # -----------------------------------------------------
        # 3. Classification information
        # -----------------------------------------------------

        classification_confidence = None

        if classification is not None:
            classification_confidence = classification["confidence"]

        # -----------------------------------------------------
        # 4. Write frame-level CSV
        # -----------------------------------------------------

        self.frame_csv_writer.writerow([
            round(result["timestamp"], 3),
            result["frame_index"],
            result["status"],
            detected,
            detection_confidence,
            *bbox,
            movement_state,
            movement_speed,
            raw_behavior,
            filtered_behavior,
            behavior_start_timestamp,
            classification_confidence,
            confirmation["confirmed"],
        ])

        # -----------------------------------------------------
        # 5. Determine the displayed behavior label
        # -----------------------------------------------------

        if filtered_behavior is not None:
            label = filtered_behavior.upper()

        else:
            status_labels = {
                "not_detected": "CIVET NOT DETECTED",
                "stationary_wait": "STATIONARY WAIT",
                "movement_unknown": "MOVEMENT UNKNOWN",
                "classification_failed": "CLASSIFICATION FAILED",
            }

            label = status_labels.get(
                result["status"],
                result["status"].upper(),
            )

        # Draw the filtered behavior above the bounding box.
        self.draw_label(
            frame,
            label,
            bbox=bbox if detected else None,
        )

        # -----------------------------------------------------
        # 6. Display debugging information
        # -----------------------------------------------------

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

        raw_text = (
            f"Raw: {raw_behavior.upper()}"
            if raw_behavior is not None
            else "Raw: UNKNOWN"
        )

        cv2.putText(
            frame,
            raw_text,
            (20, frame.shape[0] - 20),
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
                (20, frame.shape[0] - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # -----------------------------------------------------
        # 7. Initialize and write annotated video
        # -----------------------------------------------------

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

    def write_confirmed_events(self, events):
        """Write confirmed behavior events to the event-level CSV."""

        for event in events:
            start_time = event["start_time"]
            end_time = event["end_time"]

            self.event_csv_writer.writerow([
                event["behavior"],
                round(start_time, 3),
                round(end_time, 3),
                round(event["duration_seconds"], 3),
                self.format_timestamp(start_time),
                self.format_timestamp(end_time),
            ])

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.frame_csv_file is not None:
            self.frame_csv_file.close()
            self.frame_csv_file = None

        if self.event_csv_file is not None:
            self.event_csv_file.close()
            self.event_csv_file = None
