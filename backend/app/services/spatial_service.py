"""
Spatial query service using PrivTree for differential privacy.
"""

import numpy as np
from typing import Optional, Tuple
from pathlib import Path

from ..privacy.privtree import PrivTree, BoundingBox
from ..data_ingestion import load_points_as_array
from ..config import get_settings

settings = get_settings()


class SpatialService:
    """
    Handles spatial queries with differential privacy guarantees.
    
    Uses the PrivTree algorithm to create adaptive hierarchical
    decompositions that provide high resolution in dense areas
    while protecting sparse regions.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the spatial service.
        
        Args:
            data_path: Path to the taxi CSV data file
        """
        self.data_path = data_path
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
    
    def load_data(
        self,
        limit: Optional[int] = None,
        bounds: Optional[BoundingBox] = None
    ) -> np.ndarray:
        """
        Load taxi pickup data points.
        
        Uses caching to avoid reloading from disk on every query.
        """
        if self._cached_points is not None and limit is None:
            points = self._cached_points
        else:
            if not self.data_path or not Path(self.data_path).exists():
                # Return synthetic demo data if no real data available
                return self._generate_demo_data(bounds or self._get_default_bounds())
            
            bounds_tuple = None
            if bounds:
                bounds_tuple = (bounds.min_lon, bounds.max_lon, bounds.min_lat, bounds.max_lat)
            
            points = load_points_as_array(
                self.data_path,
                limit=limit,
                bounds=bounds_tuple
            )
            
            if limit is None:
                self._cached_points = points
        
        # Filter to bounds if specified
        if bounds and len(points) > 0:
            mask = (
                (points[:, 0] >= bounds.min_lon) &
                (points[:, 0] < bounds.max_lon) &
                (points[:, 1] >= bounds.min_lat) &
                (points[:, 1] < bounds.max_lat)
            )
            points = points[mask]
        
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
    
    def create_decomposition(
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
        
        # Load data points
        points = self.load_data(limit=max_points, bounds=bounds)
        
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
    
    def create_heatmap(
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
        
        points = self.load_data(bounds=bounds)
        
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
    
    def get_data_statistics(self) -> dict:
        """Get statistics about the loaded data."""
        points = self.load_data()
        
        if len(points) == 0:
            return {"error": "No data loaded"}
        
        return {
            "total_points": len(points),
            "min_longitude": float(np.min(points[:, 0])),
            "max_longitude": float(np.max(points[:, 0])),
            "min_latitude": float(np.min(points[:, 1])),
            "max_latitude": float(np.max(points[:, 1])),
            "center_longitude": float(np.mean(points[:, 0])),
            "center_latitude": float(np.mean(points[:, 1])),
        }

