"""
Database models for PrivMap.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text, JSON
from sqlalchemy.sql import func
from geoalchemy2 import Geometry
from datetime import datetime

from .database import Base


class TaxiPickup(Base):
    """
    Stores taxi pickup locations from the Porto dataset.
    
    Each record represents a single pickup point extracted from
    the trajectory polylines in the original CSV.
    """
    __tablename__ = "taxi_pickups"
    
    id = Column(Integer, primary_key=True, index=True)
    trip_id = Column(String(50), index=True)
    longitude = Column(Float, nullable=False)
    latitude = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=True)
    
    # PostGIS geometry column for efficient spatial queries
    location = Column(
        Geometry(geometry_type="POINT", srid=4326),
        nullable=True
    )
    
    def __repr__(self):
        return f"<TaxiPickup(id={self.id}, lon={self.longitude}, lat={self.latitude})>"


class PrivacySession(Base):
    """
    Tracks privacy budget usage per analyst session.
    
    Each analyst gets a fixed privacy budget for their session.
    Every query deducts from this budget, and queries are rejected
    when the budget is exhausted.
    """
    __tablename__ = "privacy_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_token = Column(String(64), unique=True, index=True)
    
    # Budget tracking
    initial_budget = Column(Float, default=5.0)
    remaining_budget = Column(Float, default=5.0)
    
    # Session metadata
    created_at = Column(DateTime, default=func.now())
    last_query_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Query history (JSON array)
    query_history = Column(JSON, default=list)
    
    def can_spend(self, epsilon: float) -> bool:
        """Check if session has enough budget for a query."""
        return self.is_active and self.remaining_budget >= epsilon
    
    def spend(self, epsilon: float, query_info: dict = None):
        """Deduct privacy budget for a query."""
        if not self.can_spend(epsilon):
            raise ValueError("Insufficient privacy budget")
        
        self.remaining_budget -= epsilon
        self.last_query_at = datetime.utcnow()
        
        if query_info:
            if self.query_history is None:
                self.query_history = []
            self.query_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "epsilon": epsilon,
                **query_info
            })


class CachedDecomposition(Base):
    """
    Caches PrivTree decomposition results.
    
    Since PrivTree output depends on epsilon and data bounds,
    we can cache results for common parameter combinations to
    improve response time (the cached result is already private).
    """
    __tablename__ = "cached_decompositions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Cache key parameters
    epsilon = Column(Float, nullable=False)
    min_lon = Column(Float, nullable=False)
    max_lon = Column(Float, nullable=False)
    min_lat = Column(Float, nullable=False)
    max_lat = Column(Float, nullable=False)
    
    # Cached result (GeoJSON)
    geojson = Column(JSON, nullable=False)
    statistics = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    point_count = Column(Integer, nullable=True)

