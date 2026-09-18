from __future__ import annotations

import time

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque

import numpy as np

from app.services.transformer_feature_extractor import (
    TransformerBaseFeatures,
)


SEQ_LEN = 60

WINDOW_SECONDS = 2.0

MAX_POSE_STEP_DEGREES = 45.0

MIN_SOURCE_SAMPLES = 12


FEATURE_NAMES = [
    "ear_left",
    "ear_right",
    "ear_mean",
    "mar",
    "nose_x",
    "nose_y",
    "yaw",
    "pitch",
    "roll",
    "delta_ear",
    "delta_mar",
    "delta_yaw",
    "delta_pitch",
    "delta_roll",
]


def wrap_angle_deg(
    angle,
):

    return (
        np.asarray(
            angle
        )
        + 180.0
    ) % 360.0 - 180.0


def circular_delta_deg(
    current: float,
    previous: float,
) -> float:

    return float(
        (
            current
            - previous
            + 180.0
        )
        % 360.0
        - 180.0
    )


def stabilize_pose_angles(
    values: np.ndarray,
) -> np.ndarray:

    values = values.astype(
        np.float64,
        copy=True,
    )

    # yaw, pitch, roll
    for index in (
        6,
        7,
        8,
    ):

        raw = wrap_angle_deg(
            values[
                :,
                index,
            ]
        )

        stable = np.empty_like(
            raw,
            dtype=np.float64,
        )

        stable[0] = raw[0]

        for i in range(
            1,
            len(raw),
        ):

            previous_wrapped = float(
                wrap_angle_deg(
                    stable[
                        i - 1
                    ]
                )
            )

            delta = (
                circular_delta_deg(
                    float(
                        raw[i]
                    ),
                    previous_wrapped,
                )
            )

            if (
                abs(delta)
                > MAX_POSE_STEP_DEGREES
            ):

                stable[i] = (
                    stable[
                        i - 1
                    ]
                )

            else:

                stable[i] = (
                    stable[
                        i - 1
                    ]
                    + delta
                )

        values[
            :,
            index,
        ] = stable

    return values


@dataclass
class TimedBaseFeature:

    timestamp: float

    values: np.ndarray


class TransformerSessionState:

    def __init__(
        self,
    ):

        self._samples: dict[
            str,
            Deque[TimedBaseFeature],
        ] = {}

        self._lock = Lock()

        # Enough for several seconds even
        # if capture temporarily runs fast.
        self.max_raw_samples = 300

    def _get_buffer(
        self,
        session_id: str,
    ) -> Deque[
        TimedBaseFeature
    ]:

        if session_id not in self._samples:

            self._samples[
                session_id
            ] = deque(
                maxlen=self.max_raw_samples
            )

        return self._samples[
            session_id
        ]

    def append(
        self,
        session_id: str,
        features: TransformerBaseFeatures,
        timestamp: float | None = None,
    ) -> None:

        if timestamp is None:

            timestamp = (
                time.monotonic()
            )

        values = (
            features.to_array()
            .astype(
                np.float32
            )
        )

        if values.shape != (
            9,
        ):

            raise ValueError(
                "Expected 9 base features, "
                f"got {values.shape}"
            )

        if not np.isfinite(
            values
        ).all():

            return

        with self._lock:

            buffer = self._get_buffer(
                session_id
            )

            # Avoid invalid time ordering.
            if (
                buffer
                and timestamp
                <= buffer[-1].timestamp
            ):

                timestamp = (
                    buffer[-1].timestamp
                    + 1e-6
                )

            buffer.append(
                TimedBaseFeature(
                    timestamp=float(
                        timestamp
                    ),
                    values=values,
                )
            )

    def reset_temporal_context(
        self,
        session_id: str,
    ) -> None:

        # This mirrors training:
        # when the face disappears,
        # current temporal context is lost.

        with self._lock:

            self._samples.pop(
                session_id,
                None,
            )

    def clear_session(
        self,
        session_id: str,
    ) -> None:

        self.reset_temporal_context(
            session_id
        )

    def get_progress(
        self,
        session_id: str,
    ) -> dict:

        with self._lock:

            buffer = list(
                self._samples.get(
                    session_id,
                    [],
                )
            )

        if not buffer:

            return {
                "source_samples": 0,
                "coverage_seconds": 0.0,
                "ready": False,
            }

        coverage = float(
            buffer[-1].timestamp
            - buffer[0].timestamp
        )

        ready = (
            len(buffer)
            >= MIN_SOURCE_SAMPLES
            and coverage
            >= 1.85
        )

        return {
            "source_samples":
                len(
                    buffer
                ),

            "coverage_seconds":
                coverage,

            "ready":
                ready,
        }

    def build_sequence(
        self,
        session_id: str,
    ) -> np.ndarray | None:

        with self._lock:

            buffer = list(
                self._samples.get(
                    session_id,
                    [],
                )
            )

        if (
            len(buffer)
            < MIN_SOURCE_SAMPLES
        ):

            return None

        timestamps = np.asarray(
            [
                item.timestamp
                for item
                in buffer
            ],
            dtype=np.float64,
        )

        raw_values = np.stack(
            [
                item.values
                for item
                in buffer
            ],
            axis=0,
        ).astype(
            np.float64
        )

        end_time = float(
            timestamps[-1]
        )

        start_time = (
            end_time
            - WINDOW_SECONDS
        )

        # Need approximately a complete
        # two-second history.
        if (
            end_time
            - timestamps[0]
            < 1.85
        ):

            return None

        start_index = int(
            np.searchsorted(
                timestamps,
                start_time,
                side="left",
            )
        )

        # Include one point before the
        # exact boundary for interpolation.
        if start_index > 0:
            start_index -= 1

        timestamps = timestamps[
            start_index:
        ]

        raw_values = raw_values[
            start_index:
        ]

        if (
            len(timestamps)
            < MIN_SOURCE_SAMPLES
        ):

            return None

        if timestamps[0] > start_time:

            return None

        # ---------------------------------
        # Pose stabilization
        # ---------------------------------

        raw_values = (
            stabilize_pose_angles(
                raw_values
            )
        )

        # ---------------------------------
        # Resample actual elapsed time
        # into exactly 60 temporal steps.
        # ---------------------------------

        target_times = np.linspace(
            start_time,
            end_time,
            SEQ_LEN,
            dtype=np.float64,
        )

        resampled = np.empty(
            (
                SEQ_LEN,
                9,
            ),
            dtype=np.float32,
        )

        for feature_index in range(
            9
        ):

            resampled[
                :,
                feature_index,
            ] = np.interp(
                target_times,
                timestamps,
                raw_values[
                    :,
                    feature_index,
                ],
            )

        # Continuous version is needed
        # for correct angular deltas.
        continuous_pose = (
            resampled[
                :,
                6:9,
            ].copy()
        )

        # Absolute head angles stored
        # as [-180, +180].
        resampled[
            :,
            6:9,
        ] = wrap_angle_deg(
            resampled[
                :,
                6:9,
            ]
        ).astype(
            np.float32
        )

        # ---------------------------------
        # Same sanity clipping used
        # during training preprocessing.
        # ---------------------------------

        resampled[
            :,
            0,
        ] = np.clip(
            resampled[
                :,
                0,
            ],
            0.0,
            1.0,
        )

        resampled[
            :,
            1,
        ] = np.clip(
            resampled[
                :,
                1,
            ],
            0.0,
            1.0,
        )

        resampled[
            :,
            2,
        ] = np.clip(
            resampled[
                :,
                2,
            ],
            0.0,
            1.0,
        )

        resampled[
            :,
            3,
        ] = np.clip(
            resampled[
                :,
                3,
            ],
            0.0,
            2.0,
        )

        resampled[
            :,
            4,
        ] = np.clip(
            resampled[
                :,
                4,
            ],
            0.0,
            1.0,
        )

        resampled[
            :,
            5,
        ] = np.clip(
            resampled[
                :,
                5,
            ],
            0.0,
            1.0,
        )

        # ---------------------------------
        # Final 14-feature representation
        # ---------------------------------

        features = np.zeros(
            (
                SEQ_LEN,
                len(
                    FEATURE_NAMES
                ),
            ),
            dtype=np.float32,
        )

        features[
            :,
            :9,
        ] = resampled

        # ΔEAR
        features[
            1:,
            9,
        ] = np.diff(
            resampled[
                :,
                2,
            ]
        )

        # ΔMAR
        features[
            1:,
            10,
        ] = np.diff(
            resampled[
                :,
                3,
            ]
        )

        # ΔYaw
        features[
            1:,
            11,
        ] = np.diff(
            continuous_pose[
                :,
                0,
            ]
        )

        # ΔPitch
        features[
            1:,
            12,
        ] = np.diff(
            continuous_pose[
                :,
                1,
            ]
        )

        # ΔRoll
        features[
            1:,
            13,
        ] = np.diff(
            continuous_pose[
                :,
                2,
            ]
        )

        if features.shape != (
            60,
            14,
        ):

            raise RuntimeError(
                "Unexpected sequence "
                f"shape: {features.shape}"
            )

        if not np.isfinite(
            features
        ).all():

            raise RuntimeError(
                "Runtime sequence contains "
                "NaN or infinity."
            )

        return features