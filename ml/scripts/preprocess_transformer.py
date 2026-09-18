from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from feature_v2 import (
    FEATURE_NAMES,
    FaceFeatureExtractorV2,
)


# =========================================================
# CONFIGURATION
# =========================================================

TARGET_FPS = 30.0

SEQ_LEN = 60

WINDOW_SECONDS = (
    SEQ_LEN / TARGET_FPS
)

# Engineering choice:
# generate one training sample every 0.2 sec.
# This avoids tens of thousands of nearly identical
# overlapping sequences.
WINDOW_STEP_SECONDS = 0.20


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(
    __file__
).resolve().parents[1]


INPUT_DIR = (
    BASE_DIR
    / "datasets"
    / "raw"
)


OUTPUT_DIR = (
    BASE_DIR
    / "datasets"
    / "processed"
)


# =========================================================
# LABELS
# =========================================================

CLASS_TO_LABEL = {
    "alert": 0,
    "drowsy": 1,
    "microsleep": 1,
}


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


# =========================================================
# BASE FEATURE CONVERSION
# =========================================================

def base_to_array(base) -> np.ndarray:

    return np.asarray(
        [
            base.ear_left,
            base.ear_right,
            base.ear_mean,
            base.mar,

            base.nose_x,
            base.nose_y,

            base.yaw,
            base.pitch,
            base.roll,
        ],
        dtype=np.float32,
    )


# =========================================================
# ANGLE UNWRAPPING
# =========================================================

def wrap_angle_deg(
    angle: np.ndarray | float,
):
    return (
        np.asarray(angle) + 180.0
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
    max_step_degrees: float = 45.0,
) -> np.ndarray:

    values = values.astype(
        np.float64,
        copy=True,
    )

    # 6 = yaw
    # 7 = pitch
    # 8 = roll

    for index in (
        6,
        7,
        8,
    ):

        raw = wrap_angle_deg(
            values[:, index]
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
                    stable[i - 1]
                )
            )

            delta = circular_delta_deg(
                float(raw[i]),
                previous_wrapped,
            )

            # A change above 45 degrees in
            # ~33-40 ms is treated as a
            # pose-estimation flip/glitch.
            if abs(delta) > max_step_degrees:

                stable[i] = (
                    stable[i - 1]
                )

            else:

                stable[i] = (
                    stable[i - 1]
                    + delta
                )

        values[
            :,
            index,
        ] = stable

    return values


# =========================================================
# RESAMPLE A 2-SECOND WINDOW TO 60 STEPS
# =========================================================

def resample_window(
    base_window: list[np.ndarray],
) -> np.ndarray:

    raw = np.stack(
        base_window,
        axis=0,
    )

    if raw.ndim != 2:

        raise RuntimeError(
            f"Expected 2D raw window, got {raw.shape}"
        )

    if raw.shape[1] != 9:

        raise RuntimeError(
            "Expected 9 base features, "
            f"got {raw.shape}"
        )

    # ---------------------------------------------
    # Make yaw/pitch/roll temporally continuous
    # before interpolation.
    # ---------------------------------------------

    raw = stabilize_pose_angles(
        raw
    )


    source_steps = raw.shape[0]


    old_axis = np.linspace(
        0.0,
        1.0,
        source_steps,
        dtype=np.float64,
    )


    new_axis = np.linspace(
        0.0,
        1.0,
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


    for feature_index in range(9):

        resampled[
            :,
            feature_index,
        ] = np.interp(
            new_axis,
            old_axis,
            raw[
                :,
                feature_index,
            ],
        )

    # Keep a continuous version specifically
    # for correct angular delta calculation.
    continuous_pose = resampled[
        :,
        6:9,
    ].copy()


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


    # =====================================================
    # BASIC SANITY LIMITS
    #
    # These are defensive limits against occasional
    # MediaPipe landmark spikes.
    # =====================================================

    # EAR values
    resampled[:, 0] = np.clip(
        resampled[:, 0],
        0.0,
        1.0,
    )

    resampled[:, 1] = np.clip(
        resampled[:, 1],
        0.0,
        1.0,
    )

    resampled[:, 2] = np.clip(
        resampled[:, 2],
        0.0,
        1.0,
    )


    # MAR
    resampled[:, 3] = np.clip(
        resampled[:, 3],
        0.0,
        2.0,
    )


    # normalized nose coordinates
    resampled[:, 4] = np.clip(
        resampled[:, 4],
        0.0,
        1.0,
    )

    resampled[:, 5] = np.clip(
        resampled[:, 5],
        0.0,
        1.0,
    )

    


    # =====================================================
    # BUILD FINAL 14-FEATURE SEQUENCE
    # =====================================================

    features = np.zeros(
        (
            SEQ_LEN,
            len(FEATURE_NAMES),
        ),
        dtype=np.float32,
    )


    # absolute/base features
    features[
        :,
        :9,
    ] = resampled


    # =====================================================
    # TEMPORAL DELTAS
    # =====================================================

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


    # First temporal step has no previous step,
    # therefore delta remains zero.


    if not np.isfinite(
        features
    ).all():

        raise RuntimeError(
            "Resampled sequence contains "
            "NaN or infinity"
        )


    return features


# =========================================================
# PROCESS ONE VIDEO
# =========================================================

def extract_sequences_from_video(
    video_path: Path,
    extractor: FaceFeatureExtractorV2,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )


    source_fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 0.0
    )


    if source_fps <= 0:

        print(
            f"\nWARNING: FPS unavailable for "
            f"{video_path.name}. "
            f"Assuming {TARGET_FPS} FPS."
        )

        source_fps = TARGET_FPS


    # =====================================================
    # HOW MANY SOURCE FRAMES REPRESENT THE SAME
    # 60 / 30 = 2 SECOND TEMPORAL WINDOW?
    # =====================================================

    source_window_size = max(
        2,
        int(
            round(
                source_fps
                * WINDOW_SECONDS
            )
        ),
    )


    step_frames = max(
        1,
        int(
            round(
                source_fps
                * WINDOW_STEP_SECONDS
            )
        ),
    )


    raw_window = deque(
        maxlen=source_window_size
    )


    sequences = []

    end_frames = []


    frame_index = -1

    valid_segment_frames = 0


    while True:

        ok, frame = cap.read()


        if not ok:
            break


        frame_index += 1


        # =================================================
        # EXTRACT 9 ABSOLUTE FEATURES
        # =================================================

        base = extractor.extract_base(
            frame
        )


        # =================================================
        # FACE LOST
        # =================================================

        if base is None:

            raw_window.clear()

            valid_segment_frames = 0

            continue


        base_array = base_to_array(
            base
        )


        if not np.isfinite(
            base_array
        ).all():

            raw_window.clear()

            valid_segment_frames = 0

            continue


        raw_window.append(
            base_array
        )


        valid_segment_frames += 1


        # =================================================
        # WAIT UNTIL ~2 SEC OF CONTIGUOUS FACE DATA EXISTS
        # =================================================

        if len(
            raw_window
        ) != source_window_size:

            continue


        offset = (
            valid_segment_frames
            - source_window_size
        )


        # =================================================
        # FIXED TIME-BASED WINDOW STRIDE
        # =================================================

        if (
            offset
            % step_frames
            != 0
        ):

            continue


        sequence = resample_window(
            list(
                raw_window
            )
        )


        sequences.append(
            sequence
        )


        end_frames.append(
            frame_index
        )


    cap.release()


    return (
        sequences,
        end_frames,
        source_fps,
        source_window_size,
        step_frames,
    )


# =========================================================
# MAIN DATASET GENERATION
# =========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    X = []

    y = []

    meta = []


    extractor = (
        FaceFeatureExtractorV2()
    )


    try:

        for (
            class_name,
            label,
        ) in CLASS_TO_LABEL.items():

            class_dir = (
                INPUT_DIR
                / class_name
            )


            if not class_dir.exists():

                print(
                    "Skipping missing folder: "
                    f"{class_dir}"
                )

                continue


            video_files = sorted(
                [
                    path

                    for path
                    in class_dir.iterdir()

                    if (
                        path.is_file()

                        and path.suffix.lower()
                        in VIDEO_EXTENSIONS
                    )
                ]
            )


            print(
                f"\n{class_name}: "
                f"{len(video_files)} videos"
            )


            for video_file in tqdm(
                video_files,
                desc=(
                    f"Processing "
                    f"{class_name}"
                ),
            ):

                try:

                    (
                        sequences,
                        end_frames,
                        source_fps,
                        source_window_size,
                        step_frames,
                    ) = (
                        extract_sequences_from_video(
                            video_file,
                            extractor,
                        )
                    )


                except Exception as exc:

                    print(
                        f"\nFailed on "
                        f"{video_file.name}: "
                        f"{exc}"
                    )

                    continue


                for (
                    sequence,
                    end_frame,
                ) in zip(
                    sequences,
                    end_frames,
                ):

                    X.append(
                        sequence
                    )


                    y.append(
                        label
                    )


                    meta.append(
                        {
                            "video":
                                str(
                                    video_file
                                ),

                            "original_class":
                                class_name,

                            "label":
                                label,

                            "source_fps":
                                source_fps,

                            "source_window_frames":
                                source_window_size,

                            "target_fps":
                                TARGET_FPS,

                            "target_sequence_length":
                                SEQ_LEN,

                            "window_seconds":
                                WINDOW_SECONDS,

                            "window_step_seconds":
                                WINDOW_STEP_SECONDS,

                            "step_frames":
                                step_frames,

                            "end_frame":
                                end_frame,
                        }
                    )


    finally:

        extractor.close()


    # =====================================================
    # VALIDATION
    # =====================================================

    if not X:

        raise RuntimeError(
            "No Transformer sequences "
            "were generated."
        )


    X = np.asarray(
        X,
        dtype=np.float32,
    )


    y = np.asarray(
        y,
        dtype=np.float32,
    )


    expected_shape = (
        SEQ_LEN,
        len(
            FEATURE_NAMES
        ),
    )


    if (
        X.ndim != 3

        or tuple(
            X.shape[1:]
        ) != expected_shape
    ):

        raise RuntimeError(
            "Expected X shape "
            f"(N, {SEQ_LEN}, "
            f"{len(FEATURE_NAMES)}), "
            f"got {X.shape}"
        )


    if not np.isfinite(
        X
    ).all():

        raise RuntimeError(
            "X contains NaN or infinity."
        )


    unique_labels = set(
        np.unique(
            y
        ).tolist()
    )


    if not unique_labels.issubset(
        {
            0.0,
            1.0,
        }
    ):

        raise RuntimeError(
            "Expected binary labels. "
            f"Got {unique_labels}"
        )


    # =====================================================
    # SAVE
    # =====================================================

    x_file = (
        OUTPUT_DIR
        / "X_transformer.npy"
    )


    y_file = (
        OUTPUT_DIR
        / "y_transformer.npy"
    )


    meta_file = (
        OUTPUT_DIR
        / "meta_transformer.json"
    )


    np.save(
        x_file,
        X,
    )


    np.save(
        y_file,
        y,
    )


    with open(
        meta_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "feature_names":
                    FEATURE_NAMES,

                "sequence_length":
                    SEQ_LEN,

                "target_fps":
                    TARGET_FPS,

                "window_seconds":
                    WINDOW_SECONDS,

                "window_step_seconds":
                    WINDOW_STEP_SECONDS,

                "class_to_label":
                    CLASS_TO_LABEL,

                "samples":
                    meta,
            },

            file,

            indent=2,
        )


    # =====================================================
    # SUMMARY
    # =====================================================

    binary_counts = Counter(
        int(value)
        for value
        in y.tolist()
    )


    original_counts = Counter(
        item[
            "original_class"
        ]
        for item
        in meta
    )


    fps_counts = Counter(
        round(
            float(
                item[
                    "source_fps"
                ]
            ),
            2,
        )
        for item
        in meta
    )


    print("\n")

    print(
        "=================================="
    )

    print(
        "TRANSFORMER DATASET V2 CREATED"
    )

    print(
        "=================================="
    )


    print(
        "X:",
        X.shape,
        X.dtype,
    )


    print(
        "y:",
        y.shape,
        y.dtype,
    )


    print(
        "\nBinary labels:"
    )

    print(
        dict(
            sorted(
                binary_counts.items()
            )
        )
    )


    print(
        "\nOriginal class sample counts:"
    )

    print(
        dict(
            sorted(
                original_counts.items()
            )
        )
    )


    print(
        "\nSource FPS sample counts:"
    )

    print(
        dict(
            sorted(
                fps_counts.items()
            )
        )
    )


    print(
        "\nAll samples represent "
        f"approximately "
        f"{WINDOW_SECONDS:.2f} sec "
        f"resampled to "
        f"{SEQ_LEN} steps."
    )


    print(
        "\nOutput:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()