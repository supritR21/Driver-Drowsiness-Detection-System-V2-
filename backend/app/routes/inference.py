from __future__ import annotations

import base64

import cv2
import numpy as np

from fastapi import (
    APIRouter,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)

from app.schemas.inference import (
    FrameInferenceRequest,
    FrameInferenceResponse,
)

# ==========================================
# OLD PIPELINE
# Keep during migration
# ==========================================

from app.services.alert_engine import (
    AlertEngine,
)

from app.services.feature_extractor import (
    FeatureExtractor,
)

from app.services.model_service import (
    model_service,
)

from app.services.session_state import (
    SessionStateStore,
)


# ==========================================
# NEW TRANSFORMER PIPELINE
# ==========================================

from app.services.transformer_feature_extractor import (
    TransformerFeatureExtractor,
)

from app.services.transformer_session_state import (
    TransformerSessionState,
)

from app.services.transformer_model_service import (
    transformer_model_service,
)

from app.services.transformer_decision_state import (
    transformer_decision_store,
)


router = APIRouter(
    prefix="/inference",
    tags=["Inference"],
)


# ==========================================
# OLD GLOBAL SERVICES
# ==========================================

feature_extractor = (
    FeatureExtractor()
)

session_store = (
    SessionStateStore(
        seq_len=45
    )
)

alert_engine = (
    AlertEngine()
)


# ==========================================
# TRANSFORMER GLOBAL SERVICES
# ==========================================

transformer_feature_extractor = (
    TransformerFeatureExtractor()
)

transformer_session_store = (
    TransformerSessionState()
)


# ==========================================
# IMAGE DECODING
# ==========================================

def decode_frame_base64(
    frame_base64: str,
) -> np.ndarray:

    if "," in frame_base64:

        frame_base64 = (
            frame_base64.split(
                ",",
                1,
            )[1]
        )

    try:

        raw = base64.b64decode(
            frame_base64
        )

        arr = np.frombuffer(
            raw,
            dtype=np.uint8,
        )

        frame = cv2.imdecode(
            arr,
            cv2.IMREAD_COLOR,
        )

        if frame is None:

            raise ValueError(
                "Could not decode image"
            )

        return frame

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid frame data: "
                f"{exc}"
            ),
        ) from exc


# =========================================================
# OLD PIPELINE
# =========================================================

def process_frame(
    session_id: str,
    frame_bgr: np.ndarray,
) -> FrameInferenceResponse:

    features = (
        feature_extractor.extract(
            frame_bgr
        )
    )

    if features is None:

        session_store.set_eye_closed_count(
            session_id,
            0,
        )

        return FrameInferenceResponse(
            status="no_face_detected",

            session_id=session_id,

            sequence_length=len(
                session_store.get_sequence(
                    session_id
                )
            ),

            message=(
                "No face detected in frame."
            ),
        )

    feature_vec = (
        features
        .to_array()
        .astype(
            np.float32
        )
    )

    session_store.append_features(
        session_id,
        feature_vec,
    )

    sequence = (
        session_store.get_sequence(
            session_id
        )
    )

    if (
        len(sequence)
        < model_service.seq_len
    ):

        session_store.set_eye_closed_count(
            session_id,
            0,
        )

        return FrameInferenceResponse(
            status="collecting",

            session_id=session_id,

            sequence_length=len(
                sequence
            ),

            message=(
                "Collecting temporal "
                f"window: "
                f"{len(sequence)}/"
                f"{model_service.seq_len}"
            ),
        )

    sequence = sequence[
        -model_service.seq_len:
    ]

    result = (
        model_service.predict(
            sequence
        )
    )

    probs = np.asarray(
        result[
            "probabilities"
        ],
        dtype=np.float32,
    )

    prediction_idx = int(
        result[
            "prediction"
        ]
    )

    labels = [
        "alert",
        "drowsy",
        "microsleep",
    ]

    prediction = labels[
        prediction_idx
    ]

    ear_mean = float(
        feature_vec[
            2
        ]
    )

    mar = float(
        feature_vec[
            3
        ]
    )

    blink_flag = float(
        feature_vec[
            7
        ]
    )

    yawn_flag = float(
        feature_vec[
            8
        ]
    )

    eye_closed_count = (
        session_store
        .get_eye_closed_count(
            session_id
        )
    )

    if ear_mean < 0.22:

        eye_closed_count += 1

    else:

        eye_closed_count = 0

    session_store.set_eye_closed_count(
        session_id,
        eye_closed_count,
    )

    prob_score = float(
        probs[1] * 25.0
        + probs[2] * 50.0
    )

    eye_score = float(
        np.clip(
            (
                0.24
                - ear_mean
            )
            / 0.12,

            0.0,
            1.0,
        )
        * 35.0
    )

    mouth_score = float(
        np.clip(
            (
                mar
                - 0.42
            )
            / 0.25,

            0.0,
            1.0,
        )
        * 10.0
    )

    duration_bonus = float(
        min(
            30.0,
            eye_closed_count
            * 4.5,
        )
    )

    blink_bonus = (
        2.0
        if blink_flag > 0
        else 0.0
    )

    yawn_bonus = (
        4.0
        if yawn_flag > 0
        else 0.0
    )

    raw_score = (
        prob_score
        + eye_score
        + mouth_score
        + duration_bonus
        + blink_bonus
        + yawn_bonus
    )

    raw_score = float(
        np.clip(
            raw_score,
            0.0,
            100.0,
        )
    )

    previous_score = (
        session_store
        .get_last_score(
            session_id
        )
    )

    if (
        raw_score
        >= previous_score
    ):

        smoothed_score = (
            0.65
            * previous_score
            + 0.35
            * raw_score
        )

    else:

        smoothed_score = (
            0.82
            * previous_score
            + 0.18
            * raw_score
        )

    session_store.set_last_score(
        session_id,
        smoothed_score,
    )

    previous_level = (
        session_store
        .get_last_level(
            session_id
        )
    )

    alert_result = (
        alert_engine.evaluate(
            smoothed_score,
            previous_level=previous_level,
        )
    )

    session_store.set_last_level(
        session_id,
        alert_result[
            "level"
        ],
    )

    return FrameInferenceResponse(
        status="ok",

        session_id=session_id,

        sequence_length=len(
            sequence
        ),

        score=round(
            smoothed_score,
            2,
        ),

        level=alert_result[
            "level"
        ],

        prediction=prediction,

        message=alert_result[
            "message"
        ],

        source=result[
            "source"
        ],
    )


# =========================================================
# NEW TRANSFORMER PIPELINE
# =========================================================

def process_transformer_frame(
    session_id: str,
    frame_bgr: np.ndarray,
) -> FrameInferenceResponse:

    base_features = (
        transformer_feature_extractor
        .extract(
            frame_bgr
        )
    )

    # -----------------------------------------------------
    # Face lost
    # -----------------------------------------------------

    if base_features is None:

        transformer_session_store\
            .reset_temporal_context(
                session_id
            )

        transformer_decision_store\
            .clear_session(
                session_id
            )

        return FrameInferenceResponse(
            status="no_face_detected",

            session_id=session_id,

            sequence_length=0,

            sequence_ready=False,

            source_samples=0,

            coverage_seconds=0.0,

            level=None,

            prediction=None,

            message=(
                "No face detected. "
                "Temporal context reset."
            ),

            source="transformer",
        )

    # -----------------------------------------------------
    # Store base features with real timestamp
    # -----------------------------------------------------

    transformer_session_store.append(
        session_id,
        base_features,
    )

    progress = (
        transformer_session_store
        .get_progress(
            session_id
        )
    )

    sequence = (
        transformer_session_store
        .build_sequence(
            session_id
        )
    )

    # -----------------------------------------------------
    # Convert elapsed temporal progress into
    # a 0..60 UI progress indicator.
    # -----------------------------------------------------

    coverage_seconds = float(
        progress[
            "coverage_seconds"
        ]
    )

    temporal_progress = int(
        round(
            min(
                coverage_seconds
                / 1.85,

                1.0,
            )
            * 60.0
        )
    )

    temporal_progress = max(
        0,
        min(
            60,
            temporal_progress,
        ),
    )

    # -----------------------------------------------------
    # Not enough history yet
    # -----------------------------------------------------

    if sequence is None:

        return FrameInferenceResponse(
            status="collecting",

            session_id=session_id,

            sequence_length=temporal_progress,

            sequence_ready=False,

            source_samples=int(
                progress[
                    "source_samples"
                ]
            ),

            coverage_seconds=round(
                coverage_seconds,
                3,
            ),

            message=(
                "Collecting approximately "
                "2 seconds of temporal "
                "face features."
            ),

            source="transformer",
        )

    # -----------------------------------------------------
    # Transformer inference
    # -----------------------------------------------------

    model_result = (
        transformer_model_service
        .predict(
            sequence
        )
    )

    raw_probability = float(
        model_result[
            "probability"
        ]
    )

    threshold = float(
        model_result[
            "threshold"
        ]
    )

    # -----------------------------------------------------
    # EMA + temporal voting
    # -----------------------------------------------------

    decision = (
        transformer_decision_store
        .update(
            session_id=session_id,

            raw_probability=(
                raw_probability
            ),

            threshold=threshold,
        )
    )

    return FrameInferenceResponse(
        status="ok",

        session_id=session_id,

        sequence_length=60,

        sequence_ready=True,

        source_samples=int(
            progress[
                "source_samples"
            ]
        ),

        coverage_seconds=round(
            coverage_seconds,
            3,
        ),

        score=round(
            decision[
                "score"
            ],
            2,
        ),

        level=decision[
            "level"
        ],

        prediction=decision[
            "prediction"
        ],

        message=decision[
            "message"
        ],

        source="transformer",

        raw_probability=round(
            raw_probability,
            6,
        ),

        smoothed_probability=round(
            decision[
                "ema_probability"
            ],
            6,
        ),

        decision_threshold=(
            threshold
        ),

        vote_ratio=round(
            decision[
                "vote_ratio"
            ],
            3,
        ),
    )


# =========================================================
# OLD REST ENDPOINT
# =========================================================

@router.post(
    "/frame",
    response_model=FrameInferenceResponse,
)
def infer_frame(
    payload: FrameInferenceRequest,
):

    frame_bgr = decode_frame_base64(
        payload.frame_base64
    )

    return process_frame(
        payload.session_id,
        frame_bgr,
    )


# =========================================================
# NEW TRANSFORMER REST ENDPOINT
# =========================================================

@router.post(
    "/transformer/frame",
    response_model=FrameInferenceResponse,
)
def infer_transformer_frame(
    payload: FrameInferenceRequest,
):

    frame_bgr = decode_frame_base64(
        payload.frame_base64
    )

    return process_transformer_frame(
        payload.session_id,
        frame_bgr,
    )


# =========================================================
# OLD WEBSOCKET
# =========================================================

@router.websocket(
    "/ws/live/{session_id}"
)
async def live_inference_socket(
    websocket: WebSocket,
    session_id: str,
):

    await websocket.accept()

    try:

        while True:

            data = (
                await websocket.receive_json()
            )

            frame_base64 = data.get(
                "frame_base64"
            )

            if not frame_base64:

                await websocket.send_json(
                    {
                        "status":
                            "error",

                        "session_id":
                            session_id,

                        "message":
                            (
                                "frame_base64 "
                                "is required"
                            ),
                    }
                )

                continue

            frame_bgr = (
                decode_frame_base64(
                    frame_base64
                )
            )

            result = process_frame(
                session_id,
                frame_bgr,
            )

            await websocket.send_json(
                result.model_dump()
            )

    except WebSocketDisconnect:

        pass


# =========================================================
# NEW TRANSFORMER WEBSOCKET
# =========================================================

@router.websocket(
    "/ws/transformer/{session_id}"
)
async def transformer_live_socket(
    websocket: WebSocket,
    session_id: str,
):

    await websocket.accept()

    try:

        while True:

            data = (
                await websocket.receive_json()
            )

            frame_base64 = data.get(
                "frame_base64"
            )

            if not frame_base64:

                await websocket.send_json(
                    {
                        "status":
                            "error",

                        "session_id":
                            session_id,

                        "message":
                            (
                                "frame_base64 "
                                "is required"
                            ),
                    }
                )

                continue

            try:

                frame_bgr = (
                    decode_frame_base64(
                        frame_base64
                    )
                )

                result = (
                    process_transformer_frame(
                        session_id,
                        frame_bgr,
                    )
                )

                await websocket.send_json(
                    result.model_dump()
                )

            except Exception as exc:

                await websocket.send_json(
                    {
                        "status":
                            "error",

                        "session_id":
                            session_id,

                        "message":
                            str(
                                exc
                            ),
                    }
                )

    except WebSocketDisconnect:

        pass

    finally:

        transformer_session_store\
            .clear_session(
                session_id
            )

        transformer_decision_store\
            .clear_session(
                session_id
            )