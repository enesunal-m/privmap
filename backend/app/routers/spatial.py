"""
Spatial query endpoints with differential privacy.
"""

from fastapi import APIRouter, Depends, HTTPException, Header, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from ..database import get_db
from ..services.session_service import SessionService
from ..services.spatial_service import SpatialService
from ..privacy.privtree import BoundingBox
from ..schemas import (
    DecompositionRequest,
    DecompositionResponse,
    DecompositionStatistics,
    HeatmapRequest,
    HeatmapResponse,
    HeatmapCell,
    BoundsParams,
)
from ..config import get_settings

settings = get_settings()
router = APIRouter()


async def get_spatial_service(db: AsyncSession = Depends(get_db)) -> SpatialService:
    """Dependency for spatial service with database connection."""
    return SpatialService(db)


async def get_session_service(db: AsyncSession = Depends(get_db)) -> SessionService:
    """Dependency for session service."""
    return SessionService(db)


def bounds_params_to_box(params: Optional[BoundsParams]) -> Optional[BoundingBox]:
    """Convert Pydantic BoundsParams to PrivTree BoundingBox."""
    if params is None:
        return None
    return BoundingBox(
        min_lon=params.min_lon,
        max_lon=params.max_lon,
        min_lat=params.min_lat,
        max_lat=params.max_lat,
    )


@router.post("/decomposition", response_model=DecompositionResponse)
async def create_decomposition(
    request: DecompositionRequest,
    x_session_token: Optional[str] = Header(None, description="Session token for budget tracking"),
    spatial: SpatialService = Depends(get_spatial_service),
    session_svc: SessionService = Depends(get_session_service),
):
    """
    Create a PrivTree spatial decomposition.
    
    This endpoint applies the PrivTree algorithm to create an adaptive
    hierarchical decomposition of the taxi pickup data. The result is
    a GeoJSON structure where:
    
    - Dense areas have more detailed (smaller) cells
    - Sparse areas have larger cells with more noise
    - All counts are protected with differential privacy
    
    **Privacy Cost**: This query consumes ε from your session budget.
    
    Args:
        request: Query parameters including epsilon and bounds
        x_session_token: Optional session token for budget tracking
        
    Returns:
        GeoJSON decomposition with noisy counts and statistics
    """
    remaining_budget = float('inf')
    
    # If session provided, check and deduct budget
    if x_session_token:
        status = await session_svc.get_session_status(x_session_token)
        
        if not status:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if not status["can_query"]:
            raise HTTPException(
                status_code=403,
                detail="Session is inactive or budget exhausted"
            )
        
        if status["remaining_budget"] < request.epsilon:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient budget. Remaining: {status['remaining_budget']:.4f}, Required: {request.epsilon}"
            )
        
        # Deduct budget
        session = await session_svc.spend_budget(
            x_session_token,
            request.epsilon,
            {"query_type": "decomposition"}
        )
        remaining_budget = session.remaining_budget
    
    # Create decomposition
    bounds = bounds_params_to_box(request.bounds)
    result = await spatial.create_decomposition(
        epsilon=request.epsilon,
        bounds=bounds,
    )
    
    return DecompositionResponse(
        geojson=result["geojson"],
        statistics=DecompositionStatistics(**result["statistics"]),
        epsilon_spent=request.epsilon,
        remaining_budget=remaining_budget,
    )


@router.get("/decomposition/quick")
async def quick_decomposition(
    epsilon: float = Query(1.0, ge=0.01, le=10.0, description="Privacy budget"),
    min_lon: float = Query(settings.map_min_lon, ge=-180, le=180),
    max_lon: float = Query(settings.map_max_lon, ge=-180, le=180),
    min_lat: float = Query(settings.map_min_lat, ge=-90, le=90),
    max_lat: float = Query(settings.map_max_lat, ge=-90, le=90),
    spatial: SpatialService = Depends(get_spatial_service),
):
    """
    Quick decomposition without session tracking.
    
    Use this for demos and testing. For production use with
    budget tracking, use POST /decomposition with a session token.
    """
    bounds = BoundingBox(
        min_lon=min_lon,
        max_lon=max_lon,
        min_lat=min_lat,
        max_lat=max_lat,
    )
    
    result = await spatial.create_decomposition(epsilon=epsilon, bounds=bounds)
    
    return {
        "geojson": result["geojson"],
        "statistics": result["statistics"],
        "epsilon_used": epsilon,
    }


@router.post("/heatmap", response_model=HeatmapResponse)
async def create_heatmap(
    request: HeatmapRequest,
    x_session_token: Optional[str] = Header(None),
    spatial: SpatialService = Depends(get_spatial_service),
    session_svc: SessionService = Depends(get_session_service),
):
    """
    Create a privacy-preserving heatmap.
    
    Uses a uniform grid with Laplace noise. This is a simpler
    alternative to PrivTree, useful for basic visualizations.
    
    **Privacy Cost**: This query consumes ε from your session budget.
    """
    remaining_budget = float('inf')
    
    if x_session_token:
        status = await session_svc.get_session_status(x_session_token)
        
        if not status or not status["can_query"]:
            raise HTTPException(status_code=403, detail="Cannot query")
        
        if status["remaining_budget"] < request.epsilon:
            raise HTTPException(status_code=400, detail="Insufficient budget")
        
        session = await session_svc.spend_budget(
            x_session_token,
            request.epsilon,
            {"query_type": "heatmap"}
        )
        remaining_budget = session.remaining_budget
    
    bounds = bounds_params_to_box(request.bounds)
    result = await spatial.create_heatmap(
        epsilon=request.epsilon,
        bounds=bounds,
        resolution=request.resolution,
    )
    
    return HeatmapResponse(
        cells=[HeatmapCell(**c) for c in result["cells"]],
        max_count=result["max_count"],
        epsilon_spent=request.epsilon,
        remaining_budget=remaining_budget,
    )


@router.get("/statistics")
async def get_data_statistics(
    spatial: SpatialService = Depends(get_spatial_service),
):
    """
    Get non-private statistics about the dataset.
    
    Returns basic bounds and counts. Note: This endpoint
    does not consume privacy budget as it returns aggregate
    statistics that don't reveal individual records.
    """
    return await spatial.get_data_statistics()


@router.get("/bounds")
async def get_default_bounds():
    """Get default map bounds for the Porto area."""
    return {
        "min_lon": settings.map_min_lon,
        "max_lon": settings.map_max_lon,
        "min_lat": settings.map_min_lat,
        "max_lat": settings.map_max_lat,
        "center": {
            "lon": (settings.map_min_lon + settings.map_max_lon) / 2,
            "lat": (settings.map_min_lat + settings.map_max_lat) / 2,
        }
    }
