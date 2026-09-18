from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class DecisionState:
    ema_probability: float | None = None

    votes: deque[int] = field(
        default_factory=lambda: deque(
            maxlen=5
        )
    )

    stable_drowsy: bool = False

    level: str = "safe"


class TransformerDecisionStore:

    def __init__(
        self,
        vote_window: int = 5,
        min_votes: int = 3,
        enter_vote_ratio: float = 0.60,
        exit_vote_ratio: float = 0.20,
        ema_rise_alpha: float = 0.50,
        ema_fall_alpha: float = 0.70,
    ):

        self.vote_window = vote_window
        self.min_votes = min_votes

        self.enter_vote_ratio = (
            enter_vote_ratio
        )

        self.exit_vote_ratio = (
            exit_vote_ratio
        )

        self.ema_rise_alpha = (
            ema_rise_alpha
        )

        self.ema_fall_alpha = (
            ema_fall_alpha
        )

        self._states: dict[
            str,
            DecisionState,
        ] = {}

        self._lock = Lock()

    def _get_state(
        self,
        session_id: str,
    ) -> DecisionState:

        state = self._states.get(
            session_id
        )

        if state is None:

            state = DecisionState(
                votes=deque(
                    maxlen=self.vote_window
                )
            )

            self._states[
                session_id
            ] = state

        return state

    def update(
        self,
        session_id: str,
        raw_probability: float,
        threshold: float,
    ) -> dict:

        raw_probability = float(
            max(
                0.0,
                min(
                    1.0,
                    raw_probability,
                ),
            )
        )

        threshold = float(
            threshold
        )

        with self._lock:

            state = self._get_state(
                session_id
            )

            previous_ema = (
                state.ema_probability
            )

            # ---------------------------------
            # EMA
            #
            # Rise moderately fast.
            # Recover faster when probability
            # falls after the driver wakes.
            # ---------------------------------

            if previous_ema is None:

                ema = raw_probability

            else:

                if (
                    raw_probability
                    >= previous_ema
                ):

                    alpha = (
                        self.ema_rise_alpha
                    )

                else:

                    alpha = (
                        self.ema_fall_alpha
                    )

                ema = (
                    alpha
                    * raw_probability
                    + (
                        1.0
                        - alpha
                    )
                    * previous_ema
                )

            state.ema_probability = (
                float(
                    ema
                )
            )

            # ---------------------------------
            # Binary temporal vote
            # ---------------------------------

            vote = int(
                raw_probability
                >= threshold
            )

            state.votes.append(
                vote
            )

            vote_count = len(
                state.votes
            )

            vote_ratio = (
                sum(
                    state.votes
                )
                / vote_count
                if vote_count
                else 0.0
            )

            # ---------------------------------
            # Stable binary decision
            # ---------------------------------

            if (
                vote_count
                >= self.min_votes
            ):

                if not state.stable_drowsy:

                    if (
                        vote_ratio
                        >= self.enter_vote_ratio

                        and ema
                        >= threshold
                    ):

                        state.stable_drowsy = (
                            True
                        )

                else:

                    if (
                        vote_ratio
                        <= self.exit_vote_ratio

                        and ema
                        < threshold
                    ):

                        state.stable_drowsy = (
                            False
                        )

            # ---------------------------------
            # Severity
            #
            # Binary ML prediction remains
            # alert/drowsy.
            #
            # soft/warning/danger are UI/alarm
            # severity states.
            # ---------------------------------

            if state.stable_drowsy:

                if (
                    ema >= 0.80
                    and vote_ratio
                    >= 0.80
                ):

                    level = "danger"

                else:

                    level = "warning"

            else:

                if (
                    vote_count
                    >= self.min_votes

                    and (
                        vote_ratio >= 0.40
                        or ema >= threshold
                    )
                ):

                    level = "soft"

                else:

                    level = "safe"

            state.level = level

            prediction = (
                "drowsy"
                if state.stable_drowsy
                else "alert"
            )

            messages = {
                "safe":
                    "Driver appears alert.",

                "soft":
                    (
                        "Possible fatigue signal "
                        "detected. Monitoring..."
                    ),

                "warning":
                    (
                        "Drowsiness detected. "
                        "Consider taking a break."
                    ),

                "danger":
                    (
                        "Critical drowsiness alert. "
                        "Wake up and stop safely."
                    ),
            }

            return {
                "raw_probability":
                    raw_probability,

                "ema_probability":
                    float(
                        ema
                    ),

                "score":
                    float(
                        ema * 100.0
                    ),

                "vote":
                    vote,

                "vote_count":
                    vote_count,

                "vote_ratio":
                    float(
                        vote_ratio
                    ),

                "stable_drowsy":
                    bool(
                        state.stable_drowsy
                    ),

                "prediction":
                    prediction,

                "level":
                    level,

                "message":
                    messages[
                        level
                    ],
            }

    def clear_session(
        self,
        session_id: str,
    ) -> None:

        with self._lock:

            self._states.pop(
                session_id,
                None,
            )


transformer_decision_store = (
    TransformerDecisionStore()
)