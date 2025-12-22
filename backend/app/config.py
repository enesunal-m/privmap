"""
Application configuration module.
Handles environment variables and app settings.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "PrivMap"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/privmap"
    
    # Data path (for taxi CSV file)
    data_path: str = "/app/data/train.csv"
    
    # Privacy defaults
    default_epsilon: float = 1.0
    min_epsilon: float = 0.01
    max_epsilon: float = 10.0
    default_session_budget: float = 5.0
    
    # PrivTree algorithm parameters
    privtree_fanout: int = 4  # Quadtree splits into 4 children
    privtree_theta: float = 0.0  # Threshold for splitting
    
    # Map bounds (Porto, Portugal area for taxi dataset)
    map_min_lon: float = -8.7
    map_max_lon: float = -8.5
    map_min_lat: float = 41.1
    map_max_lat: float = 41.25
    
    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

