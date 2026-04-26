from sqlalchemy import func
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends

from app.core.deps import get_current_user
from app.db.session import get_db
from app.models.alert_event import AlertEvent
from app.models.driving_session import DrivingSession
from app.models.user import User
from app.schemas.dashboard import (
    DashboardAlert,
    DashboardResponse,
    DashboardSession,
    DashboardSummary,
    DashboardUser,
)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/me", response_model=DashboardResponse)
def get_my_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sessions_query = (
        db.query(DrivingSession)
        .filter(DrivingSession.user_id == current_user.id)
        .order_by(DrivingSession.started_at.desc())
    )
    recent_sessions = sessions_query.limit(10).all()

    alerts_query = (
        db.query(AlertEvent)
        .join(DrivingSession, AlertEvent.session_id == DrivingSession.id)
        .filter(DrivingSession.user_id == current_user.id)
        .order_by(AlertEvent.created_at.desc())
    )
    recent_alerts = alerts_query.limit(20).all()

    total_sessions = db.query(func.count(DrivingSession.id)).filter(
        DrivingSession.user_id == current_user.id
    ).scalar() or 0

    total_alerts = db.query(func.count(AlertEvent.id)).join(
        DrivingSession, AlertEvent.session_id == DrivingSession.id
    ).filter(
        DrivingSession.user_id == current_user.id
    ).scalar() or 0

    avg_score = db.query(func.avg(DrivingSession.avg_score)).filter(
        DrivingSession.user_id == current_user.id
    ).scalar()

    max_score = db.query(func.max(DrivingSession.max_score)).filter(
        DrivingSession.user_id == current_user.id
    ).scalar()

    alert_rows = db.query(
        AlertEvent.alert_level,
        func.count(AlertEvent.id),
    ).join(
        DrivingSession, AlertEvent.session_id == DrivingSession.id
    ).filter(
        DrivingSession.user_id == current_user.id
    ).group_by(AlertEvent.alert_level).all()

    alert_counts = {level: count for level, count in alert_rows}

    return DashboardResponse(
        user=DashboardUser.model_validate(current_user),
        summary=DashboardSummary(
            total_sessions=total_sessions,
            total_alerts=total_alerts,
            average_score=float(avg_score) if avg_score is not None else None,
            max_score=float(max_score) if max_score is not None else None,
            alert_counts=alert_counts,
        ),
        recent_sessions=[DashboardSession.model_validate(s) for s in recent_sessions],
        recent_alerts=[DashboardAlert.model_validate(a) for a in recent_alerts],
    )