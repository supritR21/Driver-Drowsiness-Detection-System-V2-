from __future__ import annotations


class AlertEngine:
    def __init__(self):
        self.level_order = ["safe", "soft", "warning", "danger"]

    def _level_from_score(self, score: float) -> str:
        if score < 30:
            return "safe"
        if score < 50:
            return "soft"
        if score < 75:
            return "warning"
        return "danger"

    def evaluate(self, score: float, previous_level: str = "safe") -> dict:
        desired = self._level_from_score(score)

        # Hysteresis rules
        if previous_level == "safe":
            if score >= 30:
                level = "soft"
            else:
                level = "safe"
        elif previous_level == "soft":
            if score < 25:
                level = "safe"
            elif score >= 50:
                level = "warning"
            else:
                level = "soft"
        elif previous_level == "warning":
            if score < 45:
                level = "soft"
            elif score >= 75:
                level = "danger"
            else:
                level = "warning"
        else:  # danger
            if score < 70:
                level = "warning"
            else:
                level = "danger"

        messages = {
            "safe": "Driver appears alert.",
            "soft": "Gentle warning: signs of fatigue detected.",
            "warning": "Warning: driver may be getting drowsy.",
            "danger": "Danger: immediate attention required.",
        }

        return {
            "level": level,
            "desired_level": desired,
            "message": messages[level],
        }