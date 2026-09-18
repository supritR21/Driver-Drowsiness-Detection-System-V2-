from pathlib import Path

import cv2
import numpy as np

from app.services.transformer_feature_extractor import (
    TransformerFeatureExtractor,
)

from app.services.transformer_session_state import (
    TransformerSessionState,
)

from app.services.transformer_model_service import (
    transformer_model_service,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


RAW_DIR = (
    ROOT
    / "ml"
    / "datasets"
    / "raw"
)


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
}


def find_first_video(
    class_name: str,
):

    folder = (
        RAW_DIR
        / class_name
    )

    videos = sorted(
        [
            path
            for path
            in folder.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in VIDEO_EXTENSIONS
            )
        ]
    )

    if not videos:

        raise RuntimeError(
            f"No video found in "
            f"{folder}"
        )

    return videos[0]


def test_video(
    video_path: Path,
    session_id: str,
    extractor,
    session_store,
):

    cap = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open "
            f"{video_path}"
        )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 0.0
    )

    if fps <= 0:
        fps = 30.0

    print(
        "\n================================"
    )

    print(
        "VIDEO:",
        video_path.name,
    )

    print(
        "FPS:",
        fps,
    )

    print(
        "================================"
    )

    frame_index = -1

    sequence = None

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame_index += 1

        base = extractor.extract(
            frame
        )

        if base is None:

            session_store.reset_temporal_context(
                session_id
            )

            continue

        timestamp = (
            frame_index / fps
        )

        session_store.append(
            session_id,
            base,
            timestamp=timestamp,
        )

        sequence = (
            session_store.build_sequence(
                session_id
            )
        )

        if sequence is not None:
            break

    cap.release()

    if sequence is None:

        raise RuntimeError(
            "Could not build a complete "
            "2-second sequence."
        )

    print(
        "Sequence shape:",
        sequence.shape,
    )

    print(
        "Finite:",
        bool(
            np.isfinite(
                sequence
            ).all()
        ),
    )

    print(
        "EAR mean range:",
        float(
            sequence[
                :,
                2,
            ].min()
        ),
        "to",
        float(
            sequence[
                :,
                2,
            ].max()
        ),
    )

    print(
        "ΔYaw range:",
        float(
            sequence[
                :,
                11,
            ].min()
        ),
        "to",
        float(
            sequence[
                :,
                11,
            ].max()
        ),
    )

    print(
        "ΔPitch range:",
        float(
            sequence[
                :,
                12,
            ].min()
        ),
        "to",
        float(
            sequence[
                :,
                12,
            ].max()
        ),
    )

    print(
        "ΔRoll range:",
        float(
            sequence[
                :,
                13,
            ].min()
        ),
        "to",
        float(
            sequence[
                :,
                13,
            ].max()
        ),
    )

    result = (
        transformer_model_service.predict(
            sequence
        )
    )

    print(
        "Prediction:",
        result,
    )

    session_store.clear_session(
        session_id
    )


def main():

    print(
        "================================"
    )

    print(
        "LIVE TRANSFORMER PIPELINE TEST"
    )

    print(
        "================================"
    )

    print(
        "Model loaded:",
        transformer_model_service.loaded,
    )

    print(
        "Device:",
        transformer_model_service.device,
    )

    if not (
        transformer_model_service.loaded
    ):

        raise RuntimeError(
            transformer_model_service.load_error
        )

    extractor = (
        TransformerFeatureExtractor()
    )

    session_store = (
        TransformerSessionState()
    )

    try:

        alert_video = (
            find_first_video(
                "alert"
            )
        )

        drowsy_video = (
            find_first_video(
                "drowsy"
            )
        )

        test_video(
            alert_video,
            "runtime-alert-test",
            extractor,
            session_store,
        )

        test_video(
            drowsy_video,
            "runtime-drowsy-test",
            extractor,
            session_store,
        )

    finally:

        extractor.close()


if __name__ == "__main__":
    main()