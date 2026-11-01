
"""Process video through detection, movement analysis and behavior filtering."""

import cv2


class BehaviorVideoProcessor:
    def __init__(
        self,
        detector,
        classifier,
        movement_analyzer,
        temporal_behavior_filter,
        behavior_confirmer,
        sample_fps=8,
    ):
        self.detector = detector
        self.classifier = classifier
        self.movement_analyzer = movement_analyzer
        self.temporal_behavior_filter = temporal_behavior_filter
        self.behavior_confirmer = behavior_confirmer

        self.sample_fps = sample_fps

        self.source_fps = None
        self.output_fps = None
        self.total_frames = None
        self.duration_seconds = None
        self.sample_interval = None

        self.confirmed_events = []

    def process_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            self.source_fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if self.source_fps <= 0:
                raise ValueError("Invalid video FPS.")

            self.duration_seconds = self.total_frames / self.source_fps

            effective_sample_fps = min(
                self.sample_fps,
                self.source_fps,
            )

            self.sample_interval = max(
                1,
                round(self.source_fps / effective_sample_fps),
            )

            self.output_fps = self.source_fps / self.sample_interval

            # Reset pipeline components before processing a new video.
            self.movement_analyzer.reset()
            self.temporal_behavior_filter.reset()
            self.behavior_confirmer.reset()

            self.confirmed_events = []

            frame_index = 0

            print(f"Input video: {video_path}")
            print(f"Source FPS: {self.source_fps:.2f}")
            print(f"Sampling FPS: {self.output_fps:.2f}")
            print(f"Total frames: {self.total_frames}")

            while True:
                success, frame = cap.read()

                if not success:
                    break

                if frame_index % self.sample_interval == 0:
                    timestamp = frame_index / self.source_fps

                    detection = self.detector.detect(frame)
                    classification = None

                    if detection is None:
                        movement = self.movement_analyzer.update(
                            bbox=None,
                            timestamp=timestamp,
                        )
                        status = "not_detected"

                    else:
                        movement = self.movement_analyzer.update(
                            bbox=detection["bbox"],
                            timestamp=timestamp,
                        )

                        movement_state = movement["state"]

                        if movement_state == "locomotion":
                            status = "locomotion"

                        elif movement_state == "stationary":
                            # Classify immediately once stationary movement
                            # has been identified.
                            classification = self.classifier.classify(
                                detection["crop"]
                            )

                            status = (
                                "classification_failed"
                                if classification is None
                                else "classified"
                            )

                        else:
                            status = "movement_unknown"

                    # Determine raw behavior prediction.
                    if status == "locomotion":
                        predicted_behavior = "locomotion"

                    elif status == "classified":
                        predicted_behavior = classification["behavior"]

                    else:
                        predicted_behavior = None

                    # Apply temporal behavior filtering.
                    filtered = self.temporal_behavior_filter.update(
                        behavior=predicted_behavior,
                        timestamp=timestamp,
                    )

                    print(
                        f"[{timestamp:7.2f}s] "
                        f"Raw: {str(predicted_behavior):12s} | "
                        f"Filtered: {str(filtered['behavior']):12s} | "
                        f"Pending: {str(filtered['pending_behavior']):12s} | "
                        f"Status: {status}"
                    )

                    filtered_behavior = filtered["behavior"]

                    behavior_start_timestamp = (
                        filtered["behavior_start_timestamp"]
                    )

                    # Confirm behaviors that last at least 60 seconds.
                    confirmation = self.behavior_confirmer.update(
                        behavior=filtered_behavior,
                        timestamp=timestamp,
                        behavior_start_timestamp=behavior_start_timestamp,
                    )

                    yield {
                        "frame": frame,
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "status": status,
                        "detection": detection,
                        "movement": movement,
                        "classification": classification,
                        "predicted_behavior": predicted_behavior,
                        "filtered_behavior": filtered_behavior,
                        "behavior_start_timestamp": behavior_start_timestamp,
                        "confirmation": confirmation,
                    }

                frame_index += 1

            self.confirmed_events = self.behavior_confirmer.finalize()

        finally:
            cap.release()
            print("Video processing finished.")
