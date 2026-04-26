from datetime import datetime
from pydantic import BaseModel, EmailStr


class DashboardUser(BaseModel):
    id: int
    email: EmailStr
    full_name: str | None = None
    role: str

    model_config = {"from_attributes": True}


class DashboardSession(BaseModel):
    id: int
    session_name: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    avg_score: float | None = None
    max_score: float | None = None
    notes: str | None = None

    model_config = {"from_attributes": True}


class DashboardAlert(BaseModel):
    id: int
    alert_level: str
    prediction: str
    fatigue_score: float
    message: str | None = None
    frame_index: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DashboardSummary(BaseModel):
    total_sessions: int
    total_alerts: int
    average_score: float | None = None
    max_score: float | None = None
    alert_counts: dict[str, int]


class DashboardResponse(BaseModel):
    user: DashboardUser
    summary: DashboardSummary
    recent_sessions: list[DashboardSession]
    recent_alerts: list[DashboardAlert]