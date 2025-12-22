"""
Spatial query service using PrivTree for differential privacy.
"""

import numpy as np
from typing import Optional, Tuple
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..privacy.privtree import PrivTree, BoundingBox
from ..config import get_settings

settings = get_settings()


class SpatialService:
    """
    Handles spatial queries with differential privacy guarantees.
    
    Uses the PrivTree algorithm to create adaptive hierarchical
    decompositions that provide high resolution in dense areas
    while protecting sparse regions.
    """
    
    def __init__(self, db: AsyncSession):
        """
        Initialize the spatial service.
        
        Args:
            db: Async database session
        """
        self.db = db
        self._cached_points: Optional[np.ndarray] = None
        self._data_bounds: Optional[Tuple[float, float, float, float]] = None
    
    def _get_default_bounds(self) -> BoundingBox:
        """Get default bounds from settings."""
        return BoundingBox(
            min_lon=settings.map_min_lon,
            max_lon=settings.map_max_lon,
            min_lat=settings.map_min_lat,
            max_lat=settings.map_max_lat,
        )
    
    async def load_data(
        self,
        limit: Optional[int] = None,
        bounds: Optional[BoundingBox] = None,
        sample_size: int = 1_700_000  # Default sample size for performance
    ) -> np.ndarray:
        """
        Load taxi pickup data points from PostgreSQL with sampling.
        
        Uses random sampling for large datasets to maintain performance
        while preserving statistical properties for privacy algorithms.
        """
        if bounds is None:
            bounds = self._get_default_bounds()
        
        # First, get count to determine if sampling needed
        count_result = await self.db.execute(
            text("""
                SELECT COUNT(*) FROM taxi_pickups 
                WHERE longitude >= :min_lon 
                  AND longitude < :max_lon 
                  AND latitude >= :min_lat 
                  AND latitude < :max_lat
            """),
            {
                "min_lon": bounds.min_lon,
                "max_lon": bounds.max_lon,
                "min_lat": bounds.min_lat,
                "max_lat": bounds.max_lat,
            }
        )
        total_count = count_result.scalar() or 0
        
        if total_count == 0:
            # Return demo data if no data in database
            return self._generate_demo_data(bounds)
        
        # Determine effective limit
        effective_limit = limit or sample_size
        
        # Use TABLESAMPLE for large datasets, or direct query for smaller ones
        if total_count > effective_limit * 2:
            # Calculate sample percentage to get approximately the desired sample size
            sample_pct = min(100, (effective_limit / total_count) * 100 * 1.2)  # 20% buffer
            
            query = text("""
                SELECT longitude, latitude 
                FROM taxi_pickups TABLESAMPLE BERNOULLI(:sample_pct)
                WHERE longitude >= :min_lon 
                  AND longitude < :max_lon 
                  AND latitude >= :min_lat 
                  AND latitude < :max_lat
                LIMIT :limit
            """)
            
            result = await self.db.execute(
                query,
                {
                    "sample_pct": sample_pct,
                    "min_lon": bounds.min_lon,
                    "max_lon": bounds.max_lon,
                    "min_lat": bounds.min_lat,
                    "max_lat": bounds.max_lat,
                    "limit": effective_limit,
                }
            )
        else:
            # Small enough dataset - just fetch with limit
            query = text("""
                SELECT longitude, latitude 
                FROM taxi_pickups 
                WHERE longitude >= :min_lon 
                  AND longitude < :max_lon 
                  AND latitude >= :min_lat 
                  AND latitude < :max_lat
                LIMIT :limit
            """)
            
            result = await self.db.execute(
                query,
                {
                    "min_lon": bounds.min_lon,
                    "max_lon": bounds.max_lon,
                    "min_lat": bounds.min_lat,
                    "max_lat": bounds.max_lat,
                    "limit": effective_limit,
                }
            )
        
        rows = result.fetchall()
        
        if not rows:
            return self._generate_demo_data(bounds)
        
        # Convert to numpy array
        points = np.array([[row[0], row[1]] for row in rows], dtype=np.float64)
        return points
    
    def _generate_demo_data(self, bounds: BoundingBox, n_points: int = 50000) -> np.ndarray:
        """
        Generate synthetic demo data with realistic clustering.
        
        Creates a mixture of Gaussians to simulate taxi pickup hotspots.
        """
        np.random.seed(42)  # Reproducible for demos
        
        # Define cluster centers (simulating city hotspots)
        centers = [
            # Downtown/central area (highest density)
            (
                (bounds.min_lon + bounds.max_lon) / 2,
                (bounds.min_lat + bounds.max_lat) / 2,
                0.4
            ),
            # Train station area
            (
                bounds.min_lon + bounds.width * 0.3,
                bounds.min_lat + bounds.height * 0.6,
                0.2
            ),
            # Airport area
            (
                bounds.min_lon + bounds.width * 0.8,
                bounds.min_lat + bounds.height * 0.3,
                0.15
            ),
            # University district
            (
                bounds.min_lon + bounds.width * 0.25,
                bounds.min_lat + bounds.height * 0.75,
                0.15
            ),
            # Business district
            (
                bounds.min_lon + bounds.width * 0.65,
                bounds.min_lat + bounds.height * 0.7,
                0.1
            ),
        ]
        
        points = []
        
        for center_lon, center_lat, weight in centers:
            n_cluster = int(n_points * weight)
            
            # Variance inversely related to density
            lon_std = bounds.width * 0.08
            lat_std = bounds.height * 0.08
            
            cluster_lons = np.random.normal(center_lon, lon_std, n_cluster)
            cluster_lats = np.random.normal(center_lat, lat_std, n_cluster)
            
            cluster_points = np.column_stack([cluster_lons, cluster_lats])
            points.append(cluster_points)
        
        all_points = np.vstack(points)
        
        # Clip to bounds
        all_points[:, 0] = np.clip(all_points[:, 0], bounds.min_lon, bounds.max_lon - 0.0001)
        all_points[:, 1] = np.clip(all_points[:, 1], bounds.min_lat, bounds.max_lat - 0.0001)
        
        return all_points
    
    async def create_decomposition(
        self,
        epsilon: float,
        bounds: Optional[BoundingBox] = None,
        max_points: Optional[int] = None
    ) -> dict:
        """
        Create a PrivTree decomposition of the spatial data.
        
        Args:
            epsilon: Privacy budget for this query
            bounds: Optional bounding box (defaults to full dataset)
            max_points: Optional limit on points to process
            
        Returns:
            Dictionary containing GeoJSON and statistics
        """
        if bounds is None:
            bounds = self._get_default_bounds()
        
        # Load data points from PostgreSQL
        points = await self.load_data(limit=max_points, bounds=bounds)
        
        # Create and build PrivTree
        privtree = PrivTree(
            epsilon=epsilon,
            bounds=bounds,
            fanout=settings.privtree_fanout,
            theta=settings.privtree_theta,
        )
        
        privtree.build(points)
        
        # Get results
        geojson = privtree.to_geojson()
        statistics = privtree.get_statistics()
        
        return {
            "geojson": geojson,
            "statistics": statistics,
            "point_count": len(points),
        }
    
    async def create_heatmap(
        self,
        epsilon: float,
        bounds: Optional[BoundingBox] = None,
        resolution: int = 50
    ) -> dict:
        """
        Create a privacy-preserving heatmap using uniform grid.
        
        This is a simpler alternative to PrivTree, using a uniform
        grid with Laplace noise added to each cell count.
        
        Args:
            epsilon: Privacy budget
            bounds: Bounding box for the heatmap
            resolution: Number of cells per dimension
            
        Returns:
            Dictionary containing heatmap cells and metadata
        """
        from ..privacy.laplace import add_laplace_noise
        
        if bounds is None:
            bounds = self._get_default_bounds()
        
        points = await self.load_data(bounds=bounds)
        
        # Create grid
        lon_edges = np.linspace(bounds.min_lon, bounds.max_lon, resolution + 1)
        lat_edges = np.linspace(bounds.min_lat, bounds.max_lat, resolution + 1)
        
        # Count points in each cell
        if len(points) > 0:
            counts, _, _ = np.histogram2d(
                points[:, 0],
                points[:, 1],
                bins=[lon_edges, lat_edges]
            )
        else:
            counts = np.zeros((resolution, resolution))
        
        # Add Laplace noise to each cell
        # Sensitivity is 1 (adding/removing one point changes one cell by 1)
        cells = []
        max_count = 0
        
        for i in range(resolution):
            for j in range(resolution):
                true_count = counts[i, j]
                noisy_count = max(0, add_laplace_noise(true_count, 1.0, epsilon))
                max_count = max(max_count, noisy_count)
                
                cells.append({
                    "lon": (lon_edges[i] + lon_edges[i + 1]) / 2,
                    "lat": (lat_edges[j] + lat_edges[j + 1]) / 2,
                    "count": noisy_count,
                    "normalized": 0,  # Will be filled after max is known
                })
        
        # Normalize counts
        if max_count > 0:
            for cell in cells:
                cell["normalized"] = cell["count"] / max_count
        
        return {
            "cells": cells,
            "max_count": max_count,
            "resolution": resolution,
        }
    
    async def get_data_statistics(self) -> dict:
        """Get statistics about the loaded data."""
        # Query count and bounds from database
        result = await self.db.execute(text("""
            SELECT 
                COUNT(*) as total,
                MIN(longitude) as min_lon,
                MAX(longitude) as max_lon,
                MIN(latitude) as min_lat,
                MAX(latitude) as max_lat,
                AVG(longitude) as avg_lon,
                AVG(latitude) as avg_lat
            FROM taxi_pickups
        """))
        
        row = result.fetchone()
        
        if not row or row[0] == 0:
            # Return demo data stats
            return {
                "total_points": 50000,
                "min_longitude": settings.map_min_lon,
                "max_longitude": settings.map_max_lon,
                "min_latitude": settings.map_min_lat,
                "max_latitude": settings.map_max_lat,
                "center_longitude": (settings.map_min_lon + settings.map_max_lon) / 2,
                "center_latitude": (settings.map_min_lat + settings.map_max_lat) / 2,
                "source": "demo",
            }
        
        return {
            "total_points": row[0],
            "min_longitude": float(row[1]),
            "max_longitude": float(row[2]),
            "min_latitude": float(row[3]),
            "max_latitude": float(row[4]),
            "center_longitude": float(row[5]),
            "center_latitude": float(row[6]),
            "source": "database",
        }
