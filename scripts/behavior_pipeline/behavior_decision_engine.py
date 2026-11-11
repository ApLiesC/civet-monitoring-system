
"""Evidence-based civet behavior decision logic."""


class BehaviorDecisionEngine:
    """
    Combines movement, behavior classification, and food tray
    proximity to determine the current behavior candidate.

    Possible outputs:
        locomotion
        eating
        resting
        unknown
    """

    def __init__(self, classifier_confidence_threshold=0.60):
        if not 0.0 <= classifier_confidence_threshold <= 1.0:
            raise ValueError(
                "classifier_confidence_threshold must be between 0 and 1."
            )

        self.classifier_confidence_threshold = (
            classifier_confidence_threshold
        )

    def _get_classification(self, classification):
        """
        Extract the predicted behavior and confidence.

        Expected input:
            {
                "behavior": "eating",
                "confidence": 0.90
            }

        Returns:
            (behavior, confidence)
        """
        if classification is None:
            return "unknown", 0.0

        behavior = classification.get("behavior", "unknown")
        confidence = float(classification.get("confidence", 0.0))

        if confidence < self.classifier_confidence_threshold:
            return "unknown", confidence

        behavior = str(behavior).strip().lower()

        if behavior not in ("eating", "resting"):
            return "unknown", confidence

        return behavior, confidence

    def decide(self, movement_state, classification, proximity):
        """
        Determine the behavior candidate using available evidence.

        Args:
            movement_state:
                "locomotion", "stationary", or "unknown"

            classification:
                {
                    "behavior": "eating" or "resting",
                    "confidence": float
                }

            proximity:
                Output dictionary from FoodTrayAnalyzer.

        Returns:
            Dictionary containing the behavior decision and evidence.
        """

        movement_state = (
            str(movement_state).strip().lower()
            if movement_state is not None
            else "unknown"
        )

        classifier_behavior, classifier_confidence = (
            self._get_classification(classification)
        )

        is_near_tray = (
            proximity is not None
            and proximity.get("is_near_tray", False)
        )

        # Movement evidence takes priority over classification.
        if movement_state == "locomotion":
            behavior = "locomotion"
            reason = "Civet movement detected."

        # Tray proximity takes priority over the classifier
        # when the civet is stationary.
        elif movement_state == "stationary":
            if is_near_tray:
                behavior = "eating"
                reason = (
                    "Civet is stationary and near a food tray. "
                    "Eating is confirmed by tray proximity."
                )

            elif classifier_behavior == "resting":
                behavior = "resting"
                reason = (
                    "Civet is stationary, classifier predicts resting, "
                    "and civet is not near a food tray."
                )

            else:
                behavior = "unknown"
                reason = (
                    "Insufficient evidence to determine "
                    "stationary behavior."
                )

        else:
            behavior = "unknown"
            reason = "Movement state is unknown."

        return {
            "behavior": behavior,
            "movement_state": movement_state,
            "classifier_behavior": classifier_behavior,
            "classifier_confidence": classifier_confidence,
            "is_near_tray": bool(is_near_tray),
            "reason": reason,
        }
