"""
Session management endpoints for privacy budget tracking.
"""

from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from ..database import get_db
from ..services.session_service import SessionService
from ..schemas import SessionCreate, SessionResponse, SessionStatus

router = APIRouter()


async def get_session_service(db: AsyncSession = Depends(get_db)) -> SessionService:
    """Dependency for session service."""
    return SessionService(db)


@router.post("/", response_model=SessionResponse)
async def create_session(
    request: SessionCreate = SessionCreate(),
    service: SessionService = Depends(get_session_service)
):
    """
    Create a new privacy session.
    
    Each session has a fixed privacy budget that gets depleted
    as queries are made. Once exhausted, no more queries are allowed.
    
    Args:
        request: Session creation parameters including initial budget
        
    Returns:
        Session details including the token for authentication
    """
    session = await service.create_session(request.initial_budget)
    
    return SessionResponse(
        session_token=session.session_token,
        initial_budget=session.initial_budget,
        remaining_budget=session.remaining_budget,
        created_at=session.created_at,
        last_query_at=session.last_query_at,
        is_active=session.is_active,
        query_count=len(session.query_history) if session.query_history else 0,
    )


@router.get("/status", response_model=SessionStatus)
async def get_session_status(
    x_session_token: str = Header(..., description="Session token from create_session"),
    service: SessionService = Depends(get_session_service)
):
    """
    Get current session status and remaining budget.
    
    Use this to check how much privacy budget remains before
    making a query.
    """
    status = await service.get_session_status(x_session_token)
    
    if not status:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionStatus(**status)


@router.get("/{token}", response_model=SessionResponse)
async def get_session(
    token: str,
    service: SessionService = Depends(get_session_service)
):
    """Get session details by token."""
    session = await service.get_session(token)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        session_token=session.session_token,
        initial_budget=session.initial_budget,
        remaining_budget=session.remaining_budget,
        created_at=session.created_at,
        last_query_at=session.last_query_at,
        is_active=session.is_active,
        query_count=len(session.query_history) if session.query_history else 0,
    )


@router.delete("/{token}")
async def deactivate_session(
    token: str,
    service: SessionService = Depends(get_session_service)
):
    """
    Deactivate a session.
    
    After deactivation, no more queries can be made with this session.
    This is useful for security when you're done analyzing data.
    """
    success = await service.deactivate_session(token)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session deactivated successfully"}

