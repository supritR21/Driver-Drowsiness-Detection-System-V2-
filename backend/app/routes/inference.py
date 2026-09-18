from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.schemas.inference import FrameInferenceRequest, FrameInferenceResponse
from app.services.alert_engine import AlertEngine
from app.services.feature_extractor import FeatureExtractor
from app.services.model_service import model_service
from app.services.session_state import SessionStateStore

router = APIRouter(prefix="/inference", tags=["Inference"])

feature_extractor = FeatureExtractor()
session_store = SessionStateStore(seq_len=45)
alert_engine = AlertEngine()


def decode_frame_base64(frame_base64: str) -> np.ndarray:
    if "," in frame_base64:
        frame_base64 = frame_base64.split(",", 1)[1]

    try:
        raw = base64.b64decode(frame_base64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")
        return frame
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid frame data: {exc}") from exc


def process_frame(session_id: str, frame_bgr: np.ndarray) -> FrameInferenceResponse:
    features = feature_extractor.extract(frame_bgr)

    if features is None:
        session_store.set_eye_closed_count(session_id, 0)
        return FrameInferenceResponse(
            status="no_face_detected",
            session_id=session_id,
            sequence_length=len(session_store.get_sequence(session_id)),
            message="No face detected in frame.",
        )

    feature_vec = features.to_array().astype(np.float32)

    session_store.append_features(session_id, feature_vec)
    sequence = session_store.get_sequence(session_id)

    if len(sequence) < model_service.seq_len:
        session_store.set_eye_closed_count(session_id, 0)
        return FrameInferenceResponse(
            status="collecting",
            session_id=session_id,
            sequence_length=len(sequence),
            message=f"Collecting temporal window: {len(sequence)}/{model_service.seq_len}",
        )

    sequence = sequence[-model_service.seq_len :]
    result = model_service.predict(sequence)

    probs = np.asarray(result["probabilities"], dtype=np.float32)
    prediction_idx = int(result["prediction"])

    labels = ["alert", "drowsy", "microsleep"]
    prediction = labels[prediction_idx]

    # Feature layout from feature extractor:
    # [0] ear_left, [1] ear_right, [2] ear_mean, [3] mar,
    # [4] head_pitch, [5] head_yaw, [6] head_roll,
    # [7] blink_flag, [8] yawn_flag, [9] gaze_dev
    ear_mean = float(feature_vec[2])
    mar = float(feature_vec[3])
    blink_flag = float(feature_vec[7])
    yawn_flag = float(feature_vec[8])

    # Duration-based eye closure logic
    eye_closed_count = session_store.get_eye_closed_count(session_id)
    if ear_mean < 0.22:
        eye_closed_count += 1
    else:
        eye_closed_count = 0
    session_store.set_eye_closed_count(session_id, eye_closed_count)

    # Base model contribution
    prob_score = float(probs[1] * 25.0 + probs[2] * 50.0)

    # Direct physiological contribution
    eye_score = float(np.clip((0.24 - ear_mean) / 0.12, 0.0, 1.0) * 35.0)
    mouth_score = float(np.clip((mar - 0.42) / 0.25, 0.0, 1.0) * 10.0)

    # Duration bonus for sustained eye closure
    duration_bonus = float(min(30.0, eye_closed_count * 4.5))

    # Small cue bonuses
    blink_bonus = 2.0 if blink_flag > 0 else 0.0
    yawn_bonus = 4.0 if yawn_flag > 0 else 0.0

    raw_score = prob_score + eye_score + mouth_score + duration_bonus + blink_bonus + yawn_bonus
    raw_score = float(np.clip(raw_score, 0.0, 100.0))

    # Asymmetric smoothing:
    # rise slowly, recover faster when the driver wakes up
    previous_score = session_store.get_last_score(session_id)
    if raw_score >= previous_score:
        smoothed_score = 0.65 * previous_score + 0.35 * raw_score
    else:
        smoothed_score = 0.82 * previous_score + 0.18 * raw_score

    session_store.set_last_score(session_id, smoothed_score)

    previous_level = session_store.get_last_level(session_id)
    alert_result = alert_engine.evaluate(smoothed_score, previous_level=previous_level)
    session_store.set_last_level(session_id, alert_result["level"])

    return FrameInferenceResponse(
        status="ok",
        session_id=session_id,
        sequence_length=len(sequence),
        score=round(smoothed_score, 2),
        level=alert_result["level"],
        prediction=prediction,
        message=alert_result["message"],
        source=result["source"],
    )


@router.post("/frame", response_model=FrameInferenceResponse)
def infer_frame(payload: FrameInferenceRequest):
    frame_bgr = decode_frame_base64(payload.frame_base64)
    return process_frame(payload.session_id, frame_bgr)


@router.websocket("/ws/live/{session_id}")
async def live_inference_socket(websocket: WebSocket, session_id: str):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            frame_base64 = data.get("frame_base64")
            if not frame_base64:
                await websocket.send_json(
                    {
                        "status": "error",
                        "session_id": session_id,
                        "message": "frame_base64 is required",
                    }
                )
                continue

            frame_bgr = decode_frame_base64(frame_base64)
            result = process_frame(session_id, frame_bgr)
            await websocket.send_json(result.model_dump())
    except WebSocketDisconnect:
        pass