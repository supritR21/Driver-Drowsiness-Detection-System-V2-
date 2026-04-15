from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm


SEQ_LEN = 30
BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "datasets" / "raw"
OUTPUT_DIR = BASE_DIR / "datasets" / "processed"
CLASS_TO_IDX = {"alert": 0, "drowsy": 1, "microsleep": 2}

def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts: np.ndarray) -> float:
    a = dist(pts[1], pts[5])
    b = dist(pts[2], pts[4])
    c = dist(pts[0], pts[3])
    return (a + b) / (2.0 * c + 1e-6)


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    a = dist(pts[0], pts[1])
    b = dist(pts[4], pts[5])
    c = dist(pts[2], pts[3])
    return (a + b) / (2.0 * c + 1e-6)


class FeatureExtractor:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH = [13, 14, 61, 291, 78, 308]

    HEAD_POSE_3D = np.array(
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

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        pts = self._landmarks_to_np(results.multi_face_landmarks[0], w, h)

        left_eye = pts[self.LEFT_EYE]
        right_eye = pts[self.RIGHT_EYE]
        mouth = pts[self.MOUTH]

        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear_mean = (ear_left + ear_right) / 2.0
        mar = mouth_aspect_ratio(mouth)

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
            self.HEAD_POSE_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if success:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            pose_matrix = cv2.hconcat((rotation_matrix, tvec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
            angles = euler_angles.flatten()
            head_pitch = float(angles[0])
            head_yaw = float(angles[1])
            head_roll = float(angles[2])
        else:
            head_pitch = 0.0
            head_yaw = 0.0
            head_roll = 0.0

        gaze_dev = 0.0

        return np.array(
            [
                ear_left,
                ear_right,
                ear_mean,
                mar,
                head_pitch,
                head_yaw,
                head_roll,
                blink_flag,
                yawn_flag,
                gaze_dev,
            ],
            dtype=np.float32,
        )


def extract_sequences_from_video(video_path: Path, extractor: FeatureExtractor):
    cap = cv2.VideoCapture(str(video_path))
    sequence = []
    sequences = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        feat = extractor.extract(frame)
        if feat is None:
            continue

        sequence.append(feat)

        if len(sequence) == SEQ_LEN:
            sequences.append(np.stack(sequence, axis=0))
            sequence = sequence[1:]  # sliding window

    cap.release()
    return sequences


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X = []
    y = []
    meta = []

    extractor = FeatureExtractor()

    for class_name, class_idx in CLASS_TO_IDX.items():
        class_dir = INPUT_DIR / class_name
        if not class_dir.exists():
            print(f"Skipping missing folder: {class_dir}")
            continue

        for video_file in tqdm(list(class_dir.glob("*.*")), desc=f"Processing {class_name}"):
            if video_file.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
                continue

            try:
                seqs = extract_sequences_from_video(video_file, extractor)
                for seq in seqs:
                    X.append(seq)
                    y.append(class_idx)
                    meta.append(
                        {
                            "video": str(video_file),
                            "class_name": class_name,
                            "class_idx": class_idx,
                        }
                    )
            except Exception as e:
                print(f"Failed on {video_file}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)

    with open(OUTPUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved:", X.shape, y.shape)


if __name__ == "__main__":
    main()