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
session_store = SessionStateStore(seq_len=30)
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
        return FrameInferenceResponse(
            status="no_face_detected",
            session_id=session_id,
            sequence_length=len(session_store.get_sequence(session_id)),
            message="No face detected in frame.",
        )

    session_store.append_features(session_id, features.to_array())
    sequence = session_store.get_sequence(session_id)

    if len(sequence) < model_service.seq_len:
        return FrameInferenceResponse(
            status="collecting",
            session_id=session_id,
            sequence_length=len(sequence),
            message=f"Collecting temporal window: {len(sequence)}/{model_service.seq_len}",
        )

    sequence = sequence[-model_service.seq_len :]
    result = model_service.predict(sequence)

    probs = result["probabilities"]
    prediction_idx = result["prediction"]

    labels = ["alert", "drowsy", "microsleep"]
    prediction = labels[prediction_idx]

    fatigue_score = float(probs[1] * 50.0 + probs[2] * 100.0)

    previous_level = session_store.get_last_level(session_id)
    alert_result = alert_engine.evaluate(fatigue_score, previous_level=previous_level)
    session_store.set_last_level(session_id, alert_result["level"])

    return FrameInferenceResponse(
        status="ok",
        session_id=session_id,
        sequence_length=len(sequence),
        score=round(fatigue_score, 2),
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