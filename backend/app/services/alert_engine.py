from __future__ import annotations


class AlertEngine:
    def __init__(self):
        self.level_order = ["safe", "soft", "warning", "danger"]

    def _level_from_score(self, score: float) -> str:
        # Better aligned with new hybrid score logic
        if score < 35:
            return "safe"
        if score < 55:
            return "soft"
        if score < 78:
            return "warning"
        return "danger"

    def evaluate(self, score: float, previous_level: str = "safe") -> dict:
        desired = self._level_from_score(score)

        # Hysteresis logic:
        # harder to escalate instantly,
        # easier to recover after waking up.

        if previous_level == "safe":
            if score >= 38:
                level = "soft"
            else:
                level = "safe"

        elif previous_level == "soft":
            if score < 30:
                level = "safe"
            elif score >= 58:
                level = "warning"
            else:
                level = "soft"

        elif previous_level == "warning":
            if score < 48:
                level = "soft"
            elif score >= 80:
                level = "danger"
            else:
                level = "warning"

        else:  # previous danger
            if score < 68:
                level = "warning"
            else:
                level = "danger"

        messages = {
            "safe": "Driver appears alert.",
            "soft": "Mild fatigue signs detected. Stay attentive.",
            "warning": "Warning: drowsiness increasing. Consider a break.",
            "danger": "Critical alert: possible microsleep risk. Wake immediately.",
        }

        return {
            "level": level,
            "desired_level": desired,
            "message": messages[level],
        }