"""
Health check endpoints.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..database import get_db
from ..config import get_settings
from ..schemas import HealthResponse

settings = get_settings()
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Check API and database health.
    
    Returns:
        Health status including database connectivity
    """
    db_status = "healthy"
    
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        database=db_status,
    )


@router.get("/health/simple")
async def simple_health():
    """Simple health check without database."""
    return {"status": "ok"}

