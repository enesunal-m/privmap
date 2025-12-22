"""
PrivMap - Differentially Private Spatial Analytics Platform

Main FastAPI application entry point.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from .config import get_settings
from .database import init_db
from .routers import sessions, spatial, health

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting PrivMap API...")
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down PrivMap API...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="""
    ## PrivMap - Adaptive Differentially Private Spatial Analytics
    
    A privacy-preserving platform for exploring sensitive spatial data.
    Uses the **PrivTree** algorithm to generate adaptive heatmaps that
    provide high resolution in dense areas while protecting sparse regions.
    
    ### Key Features
    
    - **Differential Privacy**: Mathematically proven privacy guarantees
    - **Adaptive Resolution**: Automatically adjusts detail based on data density
    - **Budget Tracking**: Monitor and manage your privacy budget per session
    - **Consistency Guarantee**: Hierarchical counts are always consistent
    
    ### Privacy Budget (ε)
    
    - Smaller ε = More privacy, more noise
    - Larger ε = Less privacy, less noise
    - Typical range: 0.1 to 2.0
    """,
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["Sessions"])
app.include_router(spatial.router, prefix="/api/spatial", tags=["Spatial Queries"])


# Root endpoint
@app.get("/")
async def root():
    """API root - returns basic info."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Differentially Private Spatial Analytics Platform",
        "docs_url": "/docs",
        "health_url": "/health",
    }

