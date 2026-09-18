from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


BASE_FEATURE_NAMES = [
    "ear_left",
    "ear_right",
    "ear_mean",
    "mar",
    "nose_x",
    "nose_y",
    "yaw",
    "pitch",
    "roll",
]


def _dist(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(
            a - b
        )
    )


def _eye_aspect_ratio(
    pts: np.ndarray,
) -> float:

    a = _dist(
        pts[1],
        pts[5],
    )

    b = _dist(
        pts[2],
        pts[4],
    )

    c = _dist(
        pts[0],
        pts[3],
    )

    return (
        (a + b)
        / (2.0 * c + 1e-6)
    )


def _mouth_aspect_ratio(
    pts: np.ndarray,
) -> float:

    a = _dist(
        pts[0],
        pts[1],
    )

    b = _dist(
        pts[4],
        pts[5],
    )

    c = _dist(
        pts[2],
        pts[3],
    )

    return (
        (a + b)
        / (2.0 * c + 1e-6)
    )


@dataclass(frozen=True)
class TransformerBaseFeatures:

    ear_left: float
    ear_right: float
    ear_mean: float

    mar: float

    nose_x: float
    nose_y: float

    yaw: float
    pitch: float
    roll: float

    def to_array(
        self,
    ) -> np.ndarray:

        return np.asarray(
            [
                self.ear_left,
                self.ear_right,
                self.ear_mean,
                self.mar,

                self.nose_x,
                self.nose_y,

                self.yaw,
                self.pitch,
                self.roll,
            ],
            dtype=np.float32,
        )


class TransformerFeatureExtractor:

    LEFT_EYE = [
        33,
        160,
        158,
        133,
        153,
        144,
    ]

    RIGHT_EYE = [
        362,
        385,
        387,
        263,
        373,
        380,
    ]

    MOUTH = [
        13,
        14,
        61,
        291,
        78,
        308,
    ]

    NOSE_INDEX = 1

    HEAD_POSE_INDICES = [
        33,
        263,
        1,
        61,
        291,
        199,
    ]

    HEAD_POSE_3D = np.asarray(
        [
            (-30.0, -30.0, -30.0),
            (30.0, -30.0, -30.0),
            (0.0, 0.0, 0.0),
            (-25.0, 25.0, -25.0),
            (25.0, 25.0, -25.0),
            (0.0, 50.0, -20.0),
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
    ) -> None:

        self.face_mesh = (
            mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

    def close(
        self,
    ) -> None:

        self.face_mesh.close()

    @staticmethod
    def _to_pixels(
        face_landmarks,
        width: int,
        height: int,
    ) -> np.ndarray:

        return np.asarray(
            [
                [
                    landmark.x * width,
                    landmark.y * height,
                ]
                for landmark
                in face_landmarks.landmark
            ],
            dtype=np.float32,
        )

    def _estimate_head_pose(
        self,
        points: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[
        float,
        float,
        float,
    ]:

        image_points = np.asarray(
            [
                tuple(
                    points[index]
                )
                for index
                in self.HEAD_POSE_INDICES
            ],
            dtype=np.float64,
        )

        focal_length = float(
            width
        )

        center = (
            width / 2.0,
            height / 2.0,
        )

        camera_matrix = np.asarray(
            [
                [
                    focal_length,
                    0.0,
                    center[0],
                ],
                [
                    0.0,
                    focal_length,
                    center[1],
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                ],
            ],
            dtype=np.float64,
        )

        dist_coeffs = np.zeros(
            (4, 1),
            dtype=np.float64,
        )

        (
            success,
            rvec,
            tvec,
        ) = cv2.solvePnP(
            self.HEAD_POSE_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return (
                0.0,
                0.0,
                0.0,
            )

        rotation_matrix, _ = (
            cv2.Rodrigues(
                rvec
            )
        )

        pose_matrix = cv2.hconcat(
            (
                rotation_matrix,
                tvec,
            )
        )

        (
            _,
            _,
            _,
            _,
            _,
            _,
            euler_angles,
        ) = (
            cv2.decomposeProjectionMatrix(
                pose_matrix
            )
        )

        angles = (
            euler_angles
            .reshape(-1)
        )

        # OpenCV order here:
        # pitch, yaw, roll
        pitch = float(
            angles[0]
        )

        yaw = float(
            angles[1]
        )

        roll = float(
            angles[2]
        )

        # IMPORTANT:
        # training feature order is
        # yaw, pitch, roll

        return (
            yaw,
            pitch,
            roll,
        )

    def extract(
        self,
        frame_bgr: np.ndarray,
    ) -> Optional[
        TransformerBaseFeatures
    ]:

        if frame_bgr is None:
            return None

        if frame_bgr.size == 0:
            return None

        height, width = (
            frame_bgr.shape[:2]
        )

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        results = self.face_mesh.process(
            frame_rgb
        )

        if not results.multi_face_landmarks:
            return None

        face = (
            results.multi_face_landmarks[
                0
            ]
        )

        points = self._to_pixels(
            face,
            width,
            height,
        )

        left_eye = points[
            self.LEFT_EYE
        ]

        right_eye = points[
            self.RIGHT_EYE
        ]

        mouth = points[
            self.MOUTH
        ]

        ear_left = (
            _eye_aspect_ratio(
                left_eye
            )
        )

        ear_right = (
            _eye_aspect_ratio(
                right_eye
            )
        )

        ear_mean = (
            ear_left
            + ear_right
        ) / 2.0

        mar = (
            _mouth_aspect_ratio(
                mouth
            )
        )

        nose = face.landmark[
            self.NOSE_INDEX
        ]

        nose_x = float(
            nose.x
        )

        nose_y = float(
            nose.y
        )

        (
            yaw,
            pitch,
            roll,
        ) = self._estimate_head_pose(
            points,
            width,
            height,
        )

        values = np.asarray(
            [
                ear_left,
                ear_right,
                ear_mean,
                mar,
                nose_x,
                nose_y,
                yaw,
                pitch,
                roll,
            ],
            dtype=np.float32,
        )

        if not np.isfinite(
            values
        ).all():

            return None

        return TransformerBaseFeatures(
            ear_left=float(
                ear_left
            ),

            ear_right=float(
                ear_right
            ),

            ear_mean=float(
                ear_mean
            ),

            mar=float(
                mar
            ),

            nose_x=nose_x,
            nose_y=nose_y,

            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )