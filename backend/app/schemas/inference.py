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