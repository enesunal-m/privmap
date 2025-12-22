"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Any
from datetime import datetime


# --- Privacy Parameters ---

class PrivacyParams(BaseModel):
    """Parameters for privacy-preserving queries."""
    epsilon: float = Field(
        default=1.0,
        ge=0.01,
        le=10.0,
        description="Privacy budget (smaller = more private, more noise)"
    )
    
    @field_validator('epsilon')
    @classmethod
    def validate_epsilon(cls, v):
        if v <= 0:
            raise ValueError("Epsilon must be positive")
        return round(v, 4)  # Limit precision


class BoundsParams(BaseModel):
    """Geographic bounding box parameters."""
    min_lon: float = Field(default=-8.7, ge=-180, le=180)
    max_lon: float = Field(default=-8.5, ge=-180, le=180)
    min_lat: float = Field(default=41.1, ge=-90, le=90)
    max_lat: float = Field(default=41.25, ge=-90, le=90)
    
    @field_validator('max_lon')
    @classmethod
    def validate_lon_range(cls, v, info):
        if 'min_lon' in info.data and v <= info.data['min_lon']:
            raise ValueError("max_lon must be greater than min_lon")
        return v
    
    @field_validator('max_lat')
    @classmethod
    def validate_lat_range(cls, v, info):
        if 'min_lat' in info.data and v <= info.data['min_lat']:
            raise ValueError("max_lat must be greater than min_lat")
        return v


# --- Session Management ---

class SessionCreate(BaseModel):
    """Request to create a new privacy session."""
    initial_budget: float = Field(
        default=5.0,
        ge=0.1,
        le=50.0,
        description="Total privacy budget for this session"
    )


class SessionResponse(BaseModel):
    """Response with session information."""
    session_token: str
    initial_budget: float
    remaining_budget: float
    created_at: datetime
    last_query_at: Optional[datetime] = None
    is_active: bool
    query_count: int = 0


class SessionStatus(BaseModel):
    """Current session status."""
    remaining_budget: float
    queries_made: int
    is_active: bool
    can_query: bool


# --- Decomposition Queries ---

class DecompositionRequest(BaseModel):
    """Request for PrivTree spatial decomposition."""
    epsilon: float = Field(
        default=1.0,
        ge=0.01,
        le=10.0,
        description="Privacy budget for this query"
    )
    bounds: Optional[BoundsParams] = None
    use_cache: bool = Field(
        default=True,
        description="Whether to use cached results if available"
    )


class DecompositionStatistics(BaseModel):
    """Statistics about a PrivTree decomposition."""
    total_leaves: int
    max_depth: int
    min_depth: int
    avg_depth: float
    total_noisy_count: float
    epsilon_used: float
    noise_scale: float
    delta: float


class DecompositionResponse(BaseModel):
    """Response containing PrivTree decomposition result."""
    geojson: dict
    statistics: DecompositionStatistics
    epsilon_spent: float
    remaining_budget: float


# --- Heatmap Generation ---

class HeatmapRequest(BaseModel):
    """Request for heatmap visualization data."""
    epsilon: float = Field(default=1.0, ge=0.01, le=10.0)
    bounds: Optional[BoundsParams] = None
    resolution: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Grid resolution for heatmap"
    )


class HeatmapCell(BaseModel):
    """Single cell in a heatmap grid."""
    lon: float
    lat: float
    count: float
    normalized: float


class HeatmapResponse(BaseModel):
    """Response containing heatmap data."""
    cells: List[HeatmapCell]
    max_count: float
    epsilon_spent: float
    remaining_budget: float


# --- General Responses ---

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database: str


class ErrorResponse(BaseModel):
    """Error response format."""
    detail: str
    error_code: Optional[str] = None


class PrivacyExhaustedError(BaseModel):
    """Error when privacy budget is exhausted."""
    detail: str = "Privacy budget exhausted for this session"
    remaining_budget: float = 0.0
    required_budget: float

