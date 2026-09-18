from pydantic import BaseModel, Field


class FrameInferenceRequest(BaseModel):
    session_id: str = Field(default="demo-session")
    frame_base64: str


class FrameInferenceResponse(BaseModel):
    status: str
    session_id: str
    sequence_length: int = 0
    score: float | None = None
    level: str | None = None
    prediction: str | None = None
    message: str | None = None
    source: str | None = None

    # =====================================
    # Transformer debug/runtime information
    # =====================================

    raw_probability: float | None = None

    smoothed_probability: (
        float | None
    ) = None

    decision_threshold: (
        float | None
    ) = None

    vote_ratio: float | None = None

    sequence_ready: bool = False

    source_samples: int = 0

    coverage_seconds: float = 0.0