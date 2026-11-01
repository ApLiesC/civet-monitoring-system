
"""Process video through detection, movement analysis and behavior tracking."""

import cv2

from activity_state_manager import ActivityStateManager
from eating_event_tracker import EatingEventTracker


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

        self.source_fps = None
        self.output_fps = None
        self.total_frames = None
        self.duration_seconds = None
        self.sample_interval = None

        self.activity_records = []
        self.eating_events = []

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
                            # ---------------------------------
                            # 3. Behavior classification
                            # ---------------------------------

                            classification = (
                                self.classifier.classify(
                                    detection["crop"]
                                )
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

            print("\n========== TRACKING SUMMARY ==========")
            print(
                f"Activity intervals: {len(self.activity_records)}"
            )
            print(
                f"Eating episodes: {len(self.eating_events)}"
            )

        finally:
            cap.release()
            print("Video processing finished.")
