"""
Spatial query service using PrivTree for differential privacy.
"""

import numpy as np
from typing import Optional, Tuple
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..privacy.privtree import PrivTree, BoundingBox
from ..privacy.adaptive_grid import AdaptiveGrid, GridCell, create_adaptive_grid_from_bounds
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
    
    async def create_adaptive_grid(
        self,
        epsilon: float,
        bounds: Optional[BoundingBox] = None,
        budget_split: float = 0.5,
        max_points: Optional[int] = None
    ) -> dict:
        """
        Create an Adaptive Grid decomposition of the spatial data.

        The Adaptive Grid (AG) algorithm by Qardaji et al. (2013) uses a
        two-level strategy:
        1. Coarse uniform grid to identify dense regions
        2. Fine subdivisions in dense cells for higher resolution

        Args:
            epsilon: Privacy budget for this query
            bounds: Optional bounding box (defaults to full dataset)
            budget_split: Fraction of epsilon for level 1 (default 0.5)
            max_points: Optional limit on points to process

        Returns:
            Dictionary containing GeoJSON and statistics
        """
        if bounds is None:
            bounds = self._get_default_bounds()

        # Load data points from PostgreSQL
        points = await self.load_data(limit=max_points, bounds=bounds)

        # Create Adaptive Grid using factory function
        ag = create_adaptive_grid_from_bounds(
            epsilon=epsilon,
            min_lon=bounds.min_lon,
            max_lon=bounds.max_lon,
            min_lat=bounds.min_lat,
            max_lat=bounds.max_lat,
            budget_split=budget_split,
        )

        # Build the grid
        ag.build(points)

        # Get results
        geojson = ag.to_geojson()
        statistics = ag.get_statistics()

        return {
            "geojson": geojson,
            "statistics": statistics,
            "point_count": len(points),
        }

    async def create_comparison(
        self,
        epsilon: float,
        bounds: Optional[BoundingBox] = None,
        max_points: Optional[int] = None
    ) -> dict:
        """
        Create both PrivTree and Adaptive Grid decompositions for comparison.

        Both algorithms receive the same epsilon budget independently,
        allowing for a fair side-by-side comparison.

        Args:
            epsilon: Privacy budget for EACH algorithm
            bounds: Optional bounding box (defaults to full dataset)
            max_points: Optional limit on points to process

        Returns:
            Dictionary containing both decomposition results
        """
        if bounds is None:
            bounds = self._get_default_bounds()

        # Load data once for both algorithms
        points = await self.load_data(limit=max_points, bounds=bounds)

        # ===== PrivTree Decomposition =====
        privtree = PrivTree(
            epsilon=epsilon,
            bounds=bounds,
            fanout=settings.privtree_fanout,
            theta=settings.privtree_theta,
        )
        privtree.build(points)
        privtree_geojson = privtree.to_geojson()
        privtree_stats = privtree.get_statistics()

        # ===== Adaptive Grid Decomposition =====
        ag = create_adaptive_grid_from_bounds(
            epsilon=epsilon,
            min_lon=bounds.min_lon,
            max_lon=bounds.max_lon,
            min_lat=bounds.min_lat,
            max_lat=bounds.max_lat,
        )
        ag.build(points)
        ag_geojson = ag.to_geojson()
        ag_stats = ag.get_statistics()

        return {
            "privtree": {
                "geojson": privtree_geojson,
                "statistics": privtree_stats,
            },
            "adaptive_grid": {
                "geojson": ag_geojson,
                "statistics": ag_stats,
            },
            "point_count": len(points),
            "epsilon_per_algorithm": epsilon,
        }

    async def calculate_mse(
        self,
        epsilon: float,
        bounds: Optional[BoundingBox] = None,
        num_trials: int = 10,
        max_points: Optional[int] = None
    ) -> dict:
        """
        Calculate Mean Squared Error for both algorithms.

        This method runs multiple trials of each algorithm and compares
        the noisy counts against the true counts to compute MSE.

        For fair comparison, we use a common uniform grid to evaluate
        both algorithms - computing the true count and the estimated
        count for each grid cell.

        Args:
            epsilon: Privacy budget for each algorithm
            bounds: Optional bounding box
            num_trials: Number of trials to average over
            max_points: Optional limit on points

        Returns:
            Dictionary containing MSE results for both algorithms
        """
        if bounds is None:
            bounds = self._get_default_bounds()

        # Load data once
        points = await self.load_data(limit=max_points, bounds=bounds)

        # Create evaluation grid (fixed resolution for fair comparison)
        eval_resolution = 20  # 20x20 = 400 cells
        lon_edges = np.linspace(bounds.min_lon, bounds.max_lon, eval_resolution + 1)
        lat_edges = np.linspace(bounds.min_lat, bounds.max_lat, eval_resolution + 1)

        # Compute true counts for the evaluation grid
        true_counts = np.zeros((eval_resolution, eval_resolution))
        for i in range(eval_resolution):
            for j in range(eval_resolution):
                cell_bounds = BoundingBox(
                    min_lon=lon_edges[i],
                    max_lon=lon_edges[i + 1],
                    min_lat=lat_edges[j],
                    max_lat=lat_edges[j + 1],
                )
                mask = (
                    (points[:, 0] >= cell_bounds.min_lon) &
                    (points[:, 0] < cell_bounds.max_lon) &
                    (points[:, 1] >= cell_bounds.min_lat) &
                    (points[:, 1] < cell_bounds.max_lat)
                )
                true_counts[i, j] = np.sum(mask)

        # Run multiple trials for each algorithm
        privtree_errors = []
        ag_errors = []

        for _ in range(num_trials):
            # ===== PrivTree Trial =====
            privtree = PrivTree(
                epsilon=epsilon,
                bounds=bounds,
                fanout=settings.privtree_fanout,
                theta=settings.privtree_theta,
            )
            privtree.build(points)
            pt_estimated = self._evaluate_on_grid(
                privtree.get_leaf_nodes(),
                lon_edges,
                lat_edges,
                eval_resolution,
            )
            privtree_errors.append(pt_estimated - true_counts)

            # ===== Adaptive Grid Trial =====
            ag = create_adaptive_grid_from_bounds(
                epsilon=epsilon,
                min_lon=bounds.min_lon,
                max_lon=bounds.max_lon,
                min_lat=bounds.min_lat,
                max_lat=bounds.max_lat,
            )
            ag.build(points)
            ag_estimated = self._evaluate_on_grid_ag(
                ag.result.cells if ag.result else [],
                lon_edges,
                lat_edges,
                eval_resolution,
            )
            ag_errors.append(ag_estimated - true_counts)

        # Compute MSE statistics
        privtree_all_errors = np.array(privtree_errors)
        ag_all_errors = np.array(ag_errors)

        privtree_mse = float(np.mean(privtree_all_errors ** 2))
        ag_mse = float(np.mean(ag_all_errors ** 2))

        return {
            "privtree": {
                "algorithm": "privtree",
                "mse": privtree_mse,
                "rmse": float(np.sqrt(privtree_mse)),
                "mean_error": float(np.mean(np.abs(privtree_all_errors))),
                "max_error": float(np.max(np.abs(privtree_all_errors))),
                "num_cells": eval_resolution * eval_resolution,
            },
            "adaptive_grid": {
                "algorithm": "adaptive_grid",
                "mse": ag_mse,
                "rmse": float(np.sqrt(ag_mse)),
                "mean_error": float(np.mean(np.abs(ag_all_errors))),
                "max_error": float(np.max(np.abs(ag_all_errors))),
                "num_cells": eval_resolution * eval_resolution,
            },
            "epsilon_used": epsilon,
            "num_trials": num_trials,
            "winner": "privtree" if privtree_mse < ag_mse else "adaptive_grid",
        }

    def _evaluate_on_grid(
        self,
        leaves: list,
        lon_edges: np.ndarray,
        lat_edges: np.ndarray,
        resolution: int,
    ) -> np.ndarray:
        """
        Evaluate PrivTree decomposition on a uniform grid.

        For each evaluation cell, find all overlapping leaves and
        estimate the count proportionally based on area overlap.
        """
        estimated = np.zeros((resolution, resolution))

        for i in range(resolution):
            for j in range(resolution):
                cell_min_lon = lon_edges[i]
                cell_max_lon = lon_edges[i + 1]
                cell_min_lat = lat_edges[j]
                cell_max_lat = lat_edges[j + 1]

                for leaf in leaves:
                    # Calculate intersection area
                    inter_min_lon = max(cell_min_lon, leaf.bounds.min_lon)
                    inter_max_lon = min(cell_max_lon, leaf.bounds.max_lon)
                    inter_min_lat = max(cell_min_lat, leaf.bounds.min_lat)
                    inter_max_lat = min(cell_max_lat, leaf.bounds.max_lat)

                    if inter_min_lon < inter_max_lon and inter_min_lat < inter_max_lat:
                        inter_area = (inter_max_lon - inter_min_lon) * (inter_max_lat - inter_min_lat)
                        leaf_area = leaf.bounds.area
                        if leaf_area > 0:
                            # Proportional count based on area overlap
                            proportion = inter_area / leaf_area
                            estimated[i, j] += (leaf.noisy_count or 0) * proportion

        return estimated

    def _evaluate_on_grid_ag(
        self,
        cells: list,
        lon_edges: np.ndarray,
        lat_edges: np.ndarray,
        resolution: int,
    ) -> np.ndarray:
        """
        Evaluate Adaptive Grid decomposition on a uniform grid.

        Similar to _evaluate_on_grid but for GridCell objects.
        """
        estimated = np.zeros((resolution, resolution))

        for i in range(resolution):
            for j in range(resolution):
                cell_min_lon = lon_edges[i]
                cell_max_lon = lon_edges[i + 1]
                cell_min_lat = lat_edges[j]
                cell_max_lat = lat_edges[j + 1]

                for ag_cell in cells:
                    # Calculate intersection area
                    inter_min_lon = max(cell_min_lon, ag_cell.min_lon)
                    inter_max_lon = min(cell_max_lon, ag_cell.max_lon)
                    inter_min_lat = max(cell_min_lat, ag_cell.min_lat)
                    inter_max_lat = min(cell_max_lat, ag_cell.max_lat)

                    if inter_min_lon < inter_max_lon and inter_min_lat < inter_max_lat:
                        inter_area = (inter_max_lon - inter_min_lon) * (inter_max_lat - inter_min_lat)
                        ag_cell_area = ag_cell.area
                        if ag_cell_area > 0:
                            proportion = inter_area / ag_cell_area
                            estimated[i, j] += ag_cell.noisy_count * proportion

        return estimated

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
