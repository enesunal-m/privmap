"""
Database-backed Adaptive Grid Algorithm Implementation.

This version uses database queries to count points instead of loading
all points into memory. This allows processing the full dataset (1.7M+ points)
without memory issues while maintaining accuracy.

OPTIMIZATION: Uses batch counting with a single SQL query to count all cells
in the grid at once, instead of individual queries per cell.

Based on the Adaptive Grid algorithm from Qardaji, Yang, and Li (ICDE 2013).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Awaitable, Dict, Tuple

from .laplace import generate_laplace_noise
from .adaptive_grid import GridCell, AdaptiveGridResult


# Type alias for the async count function (single cell)
CountFunction = Callable[[float, float, float, float], Awaitable[int]]

# Type alias for batch count function (returns grid of counts)
BatchCountFunction = Callable[
    [float, float, float, float, int, int],  # bounds + grid dimensions
    Awaitable[Dict[Tuple[int, int], int]]     # (i, j) -> count
]


class AdaptiveGridDB:
    """
    Database-backed Adaptive Grid implementation.

    Uses batch SQL queries to count all cells in a grid at once,
    avoiding the N*M individual queries that would cause timeouts.
    """

    def __init__(
        self,
        epsilon: float,
        min_lon: float,
        max_lon: float,
        min_lat: float,
        max_lat: float,
        count_fn: CountFunction,
        batch_count_fn: Optional[BatchCountFunction] = None,
        budget_split: float = 0.5,
        c1: float = 10.0,
        c2: float = 5.0,
        density_alpha: float = 1.5,
        min_m1: int = 4,
        max_m1: int = 50,
        min_m2: int = 2,
        max_m2: int = 10,
    ):
        """
        Initialize Adaptive Grid with database counting.

        Args:
            epsilon: Total privacy budget
            min_lon, max_lon, min_lat, max_lat: Bounding box coordinates
            count_fn: Async function that counts points given (min_lon, max_lon, min_lat, max_lat)
            batch_count_fn: Optional batch count function for efficient grid counting
            budget_split: Fraction of ε for level 1
            c1, c2: Constants for grid size formulas
            density_alpha: Threshold multiplier for dense cell detection
            min_m1/max_m1: Bounds on coarse grid dimension
            min_m2/max_m2: Bounds on fine grid dimension
        """
        self.epsilon = epsilon
        self.min_lon = min_lon
        self.max_lon = max_lon
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.count_fn = count_fn
        self.batch_count_fn = batch_count_fn
        self.budget_split = budget_split
        self.c1 = c1
        self.c2 = c2
        self.density_alpha = density_alpha
        self.min_m1 = min_m1
        self.max_m1 = max_m1
        self.min_m2 = min_m2
        self.max_m2 = max_m2

        # Split privacy budget
        self.epsilon1 = epsilon * budget_split
        self.epsilon2 = epsilon * (1 - budget_split)

        # Noise scales
        self.noise_scale1 = 1.0 / self.epsilon1
        self.noise_scale2 = 1.0 / self.epsilon2

        self._cell_counter = 0
        self.result: Optional[AdaptiveGridResult] = None
        self._total_count: int = 0

    def _generate_cell_id(self, prefix: str = "cell") -> str:
        self._cell_counter += 1
        return f"{prefix}_{self._cell_counter}"

    def _compute_m1(self, n_points: int) -> int:
        if n_points <= 0 or self.epsilon1 <= 0:
            return self.min_m1
        m1 = int(self.c1 * np.sqrt(n_points / self.epsilon1))
        return max(self.min_m1, min(self.max_m1, m1))

    def _compute_m2(self) -> int:
        if self.epsilon2 <= 0:
            return self.min_m2
        m2 = int(self.c2 * np.sqrt(self.epsilon2))
        return max(self.min_m2, min(self.max_m2, m2))

    async def build(self) -> AdaptiveGridResult:
        """
        Build the Adaptive Grid decomposition using database queries.

        Uses batch counting for efficiency when batch_count_fn is available.

        Returns:
            AdaptiveGridResult containing all cells with noisy counts
        """
        self._cell_counter = 0

        # Get total count from database
        self._total_count = await self.count_fn(
            self.min_lon, self.max_lon, self.min_lat, self.max_lat
        )

        # Compute grid dimensions
        m1 = self._compute_m1(self._total_count)
        m2 = self._compute_m2()

        # ===== PHASE 1: Coarse Grid =====
        cell_width = (self.max_lon - self.min_lon) / m1
        cell_height = (self.max_lat - self.min_lat) / m1

        # Use batch counting if available
        if self.batch_count_fn:
            coarse_counts = await self.batch_count_fn(
                self.min_lon, self.max_lon, self.min_lat, self.max_lat, m1, m1
            )
        else:
            coarse_counts = {}

        coarse_cells: List[GridCell] = []

        for i in range(m1):
            for j in range(m1):
                cell = GridCell(
                    min_lon=self.min_lon + i * cell_width,
                    max_lon=self.min_lon + (i + 1) * cell_width,
                    min_lat=self.min_lat + j * cell_height,
                    max_lat=self.min_lat + (j + 1) * cell_height,
                    level=1,
                    cell_id=self._generate_cell_id("L1"),
                )

                # Get true count from batch or individual query
                if self.batch_count_fn:
                    true_count = coarse_counts.get((i, j), 0)
                else:
                    true_count = await self.count_fn(
                        cell.min_lon, cell.max_lon, cell.min_lat, cell.max_lat
                    )

                # Add noise
                noise = generate_laplace_noise(self.noise_scale1)
                cell.noisy_count = max(0, true_count + noise)

                coarse_cells.append(cell)

        # Determine density threshold
        expected_avg = self._total_count / (m1 * m1)
        noise_threshold = self.noise_scale1 * 2
        density_threshold = max(noise_threshold, expected_avg * self.density_alpha)

        # ===== PHASE 2: Fine Grid for Dense Cells =====
        final_cells: List[GridCell] = []
        level2_count = 0

        # Identify dense cells that need subdivision
        dense_cells = [c for c in coarse_cells if c.noisy_count > density_threshold]

        # For each dense cell, do batch counting for its fine grid
        for coarse_cell in coarse_cells:
            if coarse_cell.noisy_count > density_threshold:
                # Dense cell: subdivide
                fine_cell_width = coarse_cell.width / m2
                fine_cell_height = coarse_cell.height / m2

                # Try batch counting for this coarse cell's fine grid
                if self.batch_count_fn:
                    fine_counts = await self.batch_count_fn(
                        coarse_cell.min_lon, coarse_cell.max_lon,
                        coarse_cell.min_lat, coarse_cell.max_lat,
                        m2, m2
                    )
                else:
                    fine_counts = {}

                for i in range(m2):
                    for j in range(m2):
                        fine_cell = GridCell(
                            min_lon=coarse_cell.min_lon + i * fine_cell_width,
                            max_lon=coarse_cell.min_lon + (i + 1) * fine_cell_width,
                            min_lat=coarse_cell.min_lat + j * fine_cell_height,
                            max_lat=coarse_cell.min_lat + (j + 1) * fine_cell_height,
                            level=2,
                            cell_id=self._generate_cell_id("L2"),
                            parent_id=coarse_cell.cell_id,
                        )

                        # Get true count
                        if self.batch_count_fn:
                            true_count = fine_counts.get((i, j), 0)
                        else:
                            true_count = await self.count_fn(
                                fine_cell.min_lon, fine_cell.max_lon,
                                fine_cell.min_lat, fine_cell.max_lat
                            )

                        noise = generate_laplace_noise(self.noise_scale2)
                        fine_cell.noisy_count = max(0, true_count + noise)

                        final_cells.append(fine_cell)
                        level2_count += 1
            else:
                # Sparse cell: keep as is
                final_cells.append(coarse_cell)

        self.result = AdaptiveGridResult(
            cells=final_cells,
            m1=m1,
            m2=m2,
            epsilon=self.epsilon,
            epsilon1=self.epsilon1,
            epsilon2=self.epsilon2,
            density_threshold=density_threshold,
            level1_cells=m1 * m1,
            level2_cells=level2_count,
        )

        return self.result

    def to_geojson(self) -> dict:
        """Convert the grid to GeoJSON FeatureCollection."""
        if self.result is None:
            return {"type": "FeatureCollection", "features": []}

        features = []
        for cell in self.result.cells:
            feature = {
                "type": "Feature",
                "geometry": cell.to_geojson(),
                "properties": {
                    "id": cell.cell_id,
                    "level": cell.level,
                    "depth": cell.level,
                    "count": cell.noisy_count,
                    "parent_id": cell.parent_id,
                }
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    def get_statistics(self) -> dict:
        """Get statistics about the built grid."""
        if self.result is None:
            return {}

        cells = self.result.cells
        counts = [cell.noisy_count for cell in cells]
        levels = [cell.level for cell in cells]

        level1_count = sum(1 for l in levels if l == 1)
        level2_count = sum(1 for l in levels if l == 2)

        return {
            "algorithm": "adaptive_grid",
            "total_cells": len(cells),
            "level1_cells": level1_count,
            "level2_cells": level2_count,
            "m1": self.result.m1,
            "m2": self.result.m2,
            "max_depth": 2 if level2_count > 0 else 1,
            "min_depth": 1,
            "avg_depth": sum(levels) / len(levels) if levels else 0,
            "total_noisy_count": sum(counts),
            "epsilon_used": self.epsilon,
            "epsilon1": self.epsilon1,
            "epsilon2": self.epsilon2,
            "noise_scale_l1": self.noise_scale1,
            "noise_scale_l2": self.noise_scale2,
            "density_threshold": self.result.density_threshold,
            "total_leaves": len(cells),
            "noise_scale": self.noise_scale2,
            "delta": 0.0,
            "total_true_count": self._total_count,
        }
