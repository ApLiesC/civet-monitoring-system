
"""Process video through detection, movement analysis and behavior tracking."""

import cv2
from datetime import datetime, date, time, timedelta

from activity_state_manager import ActivityStateManager
from eating_event_tracker import EatingEventTracker
from cctv_timestamp_reader import CCTVTimestampReader


class BehaviorVideoProcessor:
    def __init__(
        self,
        detector,
        classifier,
        movement_analyzer,
        sample_fps=8,
        minimum_classification_confidence=0.70,
        activity_transition_duration=3.0,
        unknown_tolerance=3.0,
        not_visible_tolerance=3.0,
        eating_minimum_duration=5.0,
        eating_gap_tolerance=3.0,
    ):
        self.detector = detector
        self.classifier = classifier
        self.movement_analyzer = movement_analyzer
        self.sample_fps = sample_fps
        self.minimum_classification_confidence = (
            minimum_classification_confidence
        )

        self.activity_manager = ActivityStateManager(
            transition_duration=activity_transition_duration,
            unknown_tolerance=unknown_tolerance,
            not_visible_tolerance=not_visible_tolerance,
        )

        self.eating_tracker = EatingEventTracker(
            minimum_duration=eating_minimum_duration,
            gap_tolerance=eating_gap_tolerance,
        )

        # CCTV timestamp reader.
        self.timestamp_reader = CCTVTimestampReader(debug=False)
        self.timestamp_read_interval = 0.5
        self.timestamp_max_retry_duration = 5.0

        self.source_fps = None
        self.output_fps = None
        self.total_frames = None
        self.duration_seconds = None
        self.sample_interval = None

        self.activity_records = []
        self.eating_events = []

        # CCTV recording information.
        # recording_start_timestamp represents estimated video time 0.
        self.recording_start_timestamp = None
        self.recording_start_time = None
        self.recording_start_elapsed = None
        self.recording_date = None

        # Internal datetime anchor at the first successful OCR reading.
        # A placeholder date is used only when OCR recognizes time alone.
        self._recording_time_anchor = None
        self._has_real_recording_date = False

        # OCR retry information.
        self.last_timestamp_attempt_elapsed = None
        self.timestamp_reading_finished = False

    def _normalize_classification(self, classification):
        """
        Validate and normalize the classifier prediction.

        Returns:
            "eating", "resting", or None if uncertain.
        """
        if classification is None:
            return None

        confidence = classification.get("confidence")

        if confidence is None:
            return None

        if confidence < self.minimum_classification_confidence:
            return None

        predicted_class = classification.get("behavior")

        if predicted_class is None:
            return None

        predicted_class = str(predicted_class).strip().lower()

        if predicted_class == "eating":
            return "eating"

        resting_classes = {
            "resting",
            "standing",
            "sitting",
            "lying",
            "sleeping",
            "inactive",
        }

        if predicted_class in resting_classes:
            return "resting"

        return None

    def _update_trackers(
        self,
        behavior,
        timestamp,
        unknown_reason=None,
    ):
        """Update activity and eating trackers independently."""

        # Determine overall activity state.
        if behavior in {"locomotion", "eating"}:
            activity_state = "active"
        elif behavior == "resting":
            activity_state = "inactive"
        else:
            activity_state = None

        # Update the overall activity-state manager.
        if activity_state is None:
            activity_result = self.activity_manager.update(
                None,
                timestamp,
                unknown_reason=unknown_reason,
            )
        else:
            activity_result = self.activity_manager.update(
                activity_state,
                timestamp,
            )

        # Update individual eating episodes.
        eating_result = self.eating_tracker.update(
            behavior,
            timestamp,
        )

        return activity_state, activity_result, eating_result

    def _convert_elapsed_to_cctv_time(self, elapsed_timestamp):
        """
        Convert elapsed video seconds into CCTV date/time.

        Uses the first successful OCR reading as the reference point.
        """
        if (
            self._recording_time_anchor is None
            or self.recording_start_elapsed is None
            or elapsed_timestamp is None
        ):
            return None

        elapsed_from_ocr = (
            elapsed_timestamp - self.recording_start_elapsed
        )

        return self._recording_time_anchor + timedelta(
            seconds=elapsed_from_ocr
        )

    def _set_recording_timestamp(
        self,
        recognized_timestamp,
        elapsed_timestamp,
    ):
        """
        Store a recognized CCTV timestamp.

        Supports:
        - datetime: date and time were recognized.
        - time: only time was recognized.

        Calculates the estimated CCTV timestamp at video elapsed time 0.
        """
        if isinstance(recognized_timestamp, datetime):
            self._recording_time_anchor = recognized_timestamp
            self._has_real_recording_date = True

        elif isinstance(recognized_timestamp, time):
            # Use a placeholder date only for internal datetime arithmetic.
            self._recording_time_anchor = datetime.combine(
                date(2000, 1, 1),
                recognized_timestamp,
            )

            self._has_real_recording_date = False

        else:
            return False

        self.recording_start_elapsed = elapsed_timestamp

        # Calculate the estimated CCTV timestamp at video elapsed time 0.
        self.recording_start_timestamp = (
            self._convert_elapsed_to_cctv_time(0.0)
        )

        if self._has_real_recording_date:
            self.recording_date = (
                self.recording_start_timestamp.date().isoformat()
            )
        else:
            self.recording_date = None

        self.recording_start_time = (
            self.recording_start_timestamp.time()
        )

        return True

    def _read_cctv_timestamp(self, frame, elapsed_timestamp):
        """
        Read CCTV time near the beginning of the video.

        OCR is retried every timestamp_read_interval seconds until:
        - A valid time is extracted, or
        - timestamp_max_retry_duration is reached.

        Date recognition is optional.
        After the first successful time reading, no more OCR is performed.
        """

        if not self.timestamp_reading_finished:
            should_read = (
                self.last_timestamp_attempt_elapsed is None
                or (
                    elapsed_timestamp - self.last_timestamp_attempt_elapsed
                    >= self.timestamp_read_interval
                )
            )

            if should_read:
                self.last_timestamp_attempt_elapsed = elapsed_timestamp

                recognized_timestamp = self.timestamp_reader.read(frame)

                if recognized_timestamp is not None:
                    self._set_recording_timestamp(
                        recognized_timestamp,
                        elapsed_timestamp,
                    )

                    self.timestamp_reading_finished = True

                    print(
                        "[CCTV TIME] Successfully read OCR time: "
                        f"{recognized_timestamp.strftime('%H:%M:%S')}"
                    )

                    print(
                        "[CCTV TIME] Estimated video start time: "
                        f"{self.recording_start_time}"
                    )

                    if self.recording_date is not None:
                        print(
                            f"[CCTV DATE] Estimated recording date: "
                            f"{self.recording_date}"
                        )
                    else:
                        print(
                            "[CCTV DATE] Date unavailable. "
                            "Continuing with time only."
                        )

                else:
                    print(
                        f"[CCTV TIME] OCR failed at "
                        f"{elapsed_timestamp:.2f}s."
                    )

            # Stop OCR retries after the allowed time.
            if (
                not self.timestamp_reading_finished
                and elapsed_timestamp >= self.timestamp_max_retry_duration
            ):
                self.timestamp_reading_finished = True

                print(
                    "[CCTV TIME] Could not read CCTV time within "
                    f"{self.timestamp_max_retry_duration:.1f} seconds. "
                    "CCTV timestamps will be unavailable."
                )

        if self._recording_time_anchor is None:
            return None

        estimated_timestamp = self._convert_elapsed_to_cctv_time(
            elapsed_timestamp
        )

        if estimated_timestamp is None:
            return None

        # If the actual date was unavailable, expose only the time.
        if not self._has_real_recording_date:
            return estimated_timestamp.time()

        return estimated_timestamp

    def _convert_activity_records_to_cctv_time(self):
        """
        Add CCTV timestamps to finalized activity records.

        Original elapsed-time fields are preserved.
        """
        for record in self.activity_records:
            record["cctv_start_time"] = (
                self._convert_elapsed_to_cctv_time(
                    record["start_time"]
                )
            )

            record["cctv_end_time"] = (
                self._convert_elapsed_to_cctv_time(
                    record["end_time"]
                )
            )

    def _convert_eating_events_to_cctv_time(self):
        """
        Add CCTV timestamps to finalized eating events.

        Original elapsed-time fields are preserved.
        """
        for event in self.eating_events:
            event["cctv_start_time"] = (
                self._convert_elapsed_to_cctv_time(
                    event["start_time"]
                )
            )

            event["cctv_end_time"] = (
                self._convert_elapsed_to_cctv_time(
                    event["end_time"]
                )
            )

    def process_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            self.source_fps = cap.get(cv2.CAP_PROP_FPS)

            self.total_frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            if self.source_fps <= 0:
                raise ValueError("Invalid video FPS.")

            self.duration_seconds = (
                self.total_frames / self.source_fps
            )

            effective_sample_fps = min(
                self.sample_fps,
                self.source_fps,
            )

            self.sample_interval = max(
                1,
                round(self.source_fps / effective_sample_fps),
            )

            self.output_fps = (
                self.source_fps / self.sample_interval
            )

            # Reset all components before processing a new video.
            self.movement_analyzer.reset()
            self.activity_manager.reset()
            self.eating_tracker.reset()

            self.activity_records = []
            self.eating_events = []

            self.recording_start_timestamp = None
            self.recording_start_time = None
            self.recording_start_elapsed = None
            self.recording_date = None

            self._recording_time_anchor = None
            self._has_real_recording_date = False

            self.last_timestamp_attempt_elapsed = None
            self.timestamp_reading_finished = False

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

                    # Elapsed video time.
                    # Keep this for all existing calculations.
                    timestamp = frame_index / self.source_fps

                    # Read CCTV timestamp near the beginning,
                    # then estimate it using elapsed video time.
                    cctv_timestamp = self._read_cctv_timestamp(
                        frame,
                        timestamp,
                    )

                    detection = self.detector.detect(frame)

                    classification = None
                    behavior = None
                    unknown_reason = None

                    # -----------------------------------------
                    # 1. Detection
                    # -----------------------------------------

                    if detection is None:
                        movement = self.movement_analyzer.update(
                            bbox=None,
                            timestamp=timestamp,
                        )

                        status = "not_detected"
                        unknown_reason = "detection"

                    else:
                        # -------------------------------------
                        # 2. Movement analysis
                        # -------------------------------------

                        movement = self.movement_analyzer.update(
                            bbox=detection["bbox"],
                            timestamp=timestamp,
                        )

                        movement_state = movement["state"]

                        if movement_state == "locomotion":
                            behavior = "locomotion"
                            status = "locomotion"

                        elif movement_state == "stationary":
                            # 3. Behavior classification
                            classification = self.classifier.classify(
                                detection["crop"]
                            )

                            behavior = self._normalize_classification(
                                classification
                            )

                            if behavior is None:
                                status = "classification_failed"
                                unknown_reason = "classification"
                            else:
                                status = "classified"

                        else:
                            # Movement needs more samples.
                            status = "movement_unknown"
                            unknown_reason = "classification"

                    # -----------------------------------------
                    # 4. Update both trackers
                    # -----------------------------------------

                    (
                        activity_state,
                        activity_result,
                        eating_result,
                    ) = self._update_trackers(
                        behavior=behavior,
                        timestamp=timestamp,
                        unknown_reason=unknown_reason,
                    )

                    print(
                        f"[{timestamp:7.2f}s] "
                        f"Behavior: {str(behavior):12s} | "
                        f"Activity: {str(activity_result['state']):12s} | "
                        f"Pending: {str(activity_result['pending_state']):12s} | "
                        f"Status: {status}"
                    )

                    yield {
                        "frame": frame,
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "cctv_timestamp": cctv_timestamp,
                        "status": status,
                        "detection": detection,
                        "movement": movement,
                        "classification": classification,
                        "behavior": behavior,
                        "activity_state": activity_result["state"],
                        "pending_activity_state": (
                            activity_result["pending_state"]
                        ),
                        "activity_update": activity_result,
                        "eating_update": eating_result,
                    }

                frame_index += 1

            # ---------------------------------------------
            # 5. Finalize records
            # ---------------------------------------------

            self.activity_records = (
                self.activity_manager.finalize()
            )

            self.eating_events = (
                self.eating_tracker.finalize()
            )

            # Add CCTV timestamps to finalized records.
            self._convert_activity_records_to_cctv_time()
            self._convert_eating_events_to_cctv_time()

            print("\n========== TRACKING SUMMARY ==========")
            print(
                f"Activity intervals: {len(self.activity_records)}"
            )
            print(
                f"Eating episodes: {len(self.eating_events)}"
            )
            print(
                f"Recording date: {self.recording_date}"
            )
            print(
                f"Recording start time: {self.recording_start_time}"
            )

            if self.recording_start_timestamp is None:
                print("CCTV timestamp: unavailable")
            elif self.recording_date is None:
                print(
                    "CCTV timestamp mode: time only "
                    "(date unavailable)"
                )
            else:
                print("CCTV timestamp mode: date and time")

        finally:
            cap.release()
            print("Video processing finished.")
