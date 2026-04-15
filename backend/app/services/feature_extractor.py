from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _eye_aspect_ratio(pts: np.ndarray) -> float:
    A = _dist(pts[1], pts[5])
    B = _dist(pts[2], pts[4])
    C = _dist(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def _mouth_aspect_ratio(pts: np.ndarray) -> float:
    A = _dist(pts[0], pts[1])
    B = _dist(pts[4], pts[5])
    C = _dist(pts[2], pts[3])
    return (A + B) / (2.0 * C + 1e-6)


@dataclass
class FrameFeatures:
    ear_left: float
    ear_right: float
    ear_mean: float
    mar: float
    head_pitch: float
    head_yaw: float
    head_roll: float
    blink_flag: float
    yawn_flag: float
    gaze_dev: float

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.ear_left,
                self.ear_right,
                self.ear_mean,
                self.mar,
                self.head_pitch,
                self.head_yaw,
                self.head_roll,
                self.blink_flag,
                self.yawn_flag,
                self.gaze_dev,
            ],
            dtype=np.float32,
        )


class FeatureExtractor:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH = [13, 14, 61, 291, 78, 308]

    HEAD_POSE_2D = [33, 263, 1, 61, 291, 199]
    HEAD_POSE_3D = [
        (-30.0, -30.0, -30.0),
        (30.0, -30.0, -30.0),
        (0.0, 0.0, 0.0),
        (-25.0, 25.0, -25.0),
        (25.0, 25.0, -25.0),
        (0.0, 50.0, -20.0),
    ]

    def __init__(self) -> None:
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _landmarks_to_np(self, face_landmarks, w: int, h: int) -> np.ndarray:
        pts = []
        for lm in face_landmarks.landmark:
            pts.append(np.array([lm.x * w, lm.y * h], dtype=np.float32))
        return np.array(pts)

    def extract(self, frame_bgr: np.ndarray) -> Optional[FrameFeatures]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        pts = self._landmarks_to_np(results.multi_face_landmarks[0], w, h)

        left_eye = pts[self.LEFT_EYE]
        right_eye = pts[self.RIGHT_EYE]
        mouth = pts[self.MOUTH]

        ear_left = _eye_aspect_ratio(left_eye)
        ear_right = _eye_aspect_ratio(right_eye)
        ear_mean = (ear_left + ear_right) / 2.0
        mar = _mouth_aspect_ratio(mouth)

        blink_flag = float(ear_mean < 0.20)
        yawn_flag = float(mar > 0.60)

        image_points = np.array(
            [
                tuple(pts[33]),
                tuple(pts[263]),
                tuple(pts[1]),
                tuple(pts[61]),
                tuple(pts[291]),
                tuple(pts[199]),
            ],
            dtype=np.float64,
        )

        model_points = np.array(self.HEAD_POSE_3D, dtype=np.float64)
        focal_length = w
        center = (w / 2.0, h / 2.0)

        camera_matrix = np.array(
            [
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if success:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            pose_matrix = cv2.hconcat((rotation_matrix, tvec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)

            # Fix: convert each angle to a Python float using .item()
            angles = euler_angles.flatten()
            head_pitch = angles[0].item()
            head_yaw = angles[1].item()
            head_roll = angles[2].item()
        else:
            head_pitch = 0.0
            head_yaw = 0.0
            head_roll = 0.0

        gaze_dev = 0.0  # placeholder for now

        return FrameFeatures(
            ear_left=ear_left,
            ear_right=ear_right,
            ear_mean=ear_mean,
            mar=mar,
            head_pitch=head_pitch,
            head_yaw=head_yaw,
            head_roll=head_roll,
            blink_flag=blink_flag,
            yawn_flag=yawn_flag,
            gaze_dev=gaze_dev,
        )