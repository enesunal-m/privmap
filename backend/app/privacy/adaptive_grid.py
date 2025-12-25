"""
Adaptive Grid (AG) Algorithm Implementation.

This module implements the Adaptive Grid algorithm for differentially private
spatial decomposition, based on the paper:
"Differentially Private Grids for Geospatial Data"
by Qardaji, Yang, and Li (ICDE 2013).

Key insight: Unlike PrivTree which uses recursive adaptive splitting, Adaptive Grid
uses a two-level strategy. The first level creates a coarse uniform grid to identify
dense regions, and the second level subdivides dense cells for finer granularity.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from .laplace import generate_laplace_noise, add_laplace_noise


@dataclass
class GridCell:
    """Represents a single cell in the adaptive grid."""
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    level: int  # 1 = coarse grid, 2 = fine grid
    noisy_count: float = 0.0
    cell_id: str = ""
    parent_id: Optional[str] = None  # For level 2 cells, reference to parent

    @property
    def width(self) -> float:
        return self.max_lon - self.min_lon

    @property
    def height(self) -> float:
        return self.max_lat - self.min_lat

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.min_lon + self.max_lon) / 2,
            (self.min_lat + self.max_lat) / 2
        )

    def contains_point(self, lon: float, lat: float) -> bool:
        """Check if a point is within this cell."""
        return (self.min_lon <= lon < self.max_lon and
                self.min_lat <= lat < self.max_lat)

    def to_geojson(self) -> dict:
        """Convert to GeoJSON polygon format."""
        return {
            "type": "Polygon",
            "coordinates": [[
                [self.min_lon, self.min_lat],
                [self.max_lon, self.min_lat],
                [self.max_lon, self.max_lat],
                [self.min_lon, self.max_lat],
                [self.min_lon, self.min_lat],
            ]]
        }

    def subdivide(self, m2: int) -> List['GridCell']:
        """
        Subdivide this cell into an m2 x m2 grid of smaller cells.

        Args:
            m2: Number of subdivisions per dimension

        Returns:
            List of m2^2 child cells
        """
        children = []
        cell_width = self.width / m2
        cell_height = self.height / m2

        for i in range(m2):
            for j in range(m2):
                child = GridCell(
                    min_lon=self.min_lon + i * cell_width,
                    max_lon=self.min_lon + (i + 1) * cell_width,
                    min_lat=self.min_lat + j * cell_height,
                    max_lat=self.min_lat + (j + 1) * cell_height,
                    level=2,
                    parent_id=self.cell_id,
                )
                children.append(child)

        return children


@dataclass
class AdaptiveGridResult:
    """Container for Adaptive Grid decomposition results."""
    cells: List[GridCell] = field(default_factory=list)
    m1: int = 0  # Coarse grid dimension
    m2: int = 0  # Fine grid dimension (for dense cells)
    epsilon: float = 0.0
    epsilon1: float = 0.0  # Budget for level 1
    epsilon2: float = 0.0  # Budget for level 2
    density_threshold: float = 0.0
    level1_cells: int = 0
    level2_cells: int = 0


class AdaptiveGrid:
    """
    Adaptive Grid: Two-level differentially private spatial decomposition.

    The algorithm works in two phases:

    Phase 1 (Coarse Grid):
    - Divide the domain into an m1 × m1 uniform grid
    - Query count for each cell and add Laplace noise using ε₁
    - Identify "dense" cells (noisy count above threshold)

    Phase 2 (Fine Grid):
    - For each dense cell, subdivide into m2 × m2 sub-cells
    - Query and add noise using ε₂
    - Non-dense cells remain as single cells

    Key parameters:
    - ε: Total privacy budget, split between phases (typically 50/50)
    - m1: Coarse grid dimension, computed as m1 = c₁ * √(N/ε₁)
    - m2: Fine grid dimension, computed as m2 = c₂ * √(ε₂)
    - Density threshold: Cells with noisy count > (N / m1²) * α are considered dense

    Reference: Qardaji et al., ICDE 2013
    """

    def __init__(
        self,
        epsilon: float,
        bounds: 'GridCell',  # Using GridCell as a simple bounds container
        budget_split: float = 0.5,  # Fraction of budget for level 1
        c1: float = 10.0,  # Constant for m1 calculation
        c2: float = 5.0,   # Constant for m2 calculation
        density_alpha: float = 1.5,  # Multiplier for density threshold
        min_m1: int = 4,   # Minimum coarse grid size
        max_m1: int = 50,  # Maximum coarse grid size
        min_m2: int = 2,   # Minimum fine grid size
        max_m2: int = 10,  # Maximum fine grid size
    ):
        """
        Initialize Adaptive Grid with privacy and spatial parameters.

        Args:
            epsilon: Total privacy budget
            bounds: Bounding box for the entire domain
            budget_split: Fraction of ε for level 1 (rest goes to level 2)
            c1: Constant for coarse grid size formula
            c2: Constant for fine grid size formula
            density_alpha: Threshold multiplier for dense cell detection
            min_m1/max_m1: Bounds on coarse grid dimension
            min_m2/max_m2: Bounds on fine grid dimension
        """
        self.epsilon = epsilon
        self.bounds = bounds
        self.budget_split = budget_split
        self.c1 = c1
        self.c2 = c2
        self.density_alpha = density_alpha
        self.min_m1 = min_m1
        self.max_m1 = max_m1
        self.min_m2 = min_m2
        self.max_m2 = max_m2

        # Split privacy budget between levels
        self.epsilon1 = epsilon * budget_split
        self.epsilon2 = epsilon * (1 - budget_split)

        # Noise scales for each level (sensitivity = 1 for count queries)
        self.noise_scale1 = 1.0 / self.epsilon1
        self.noise_scale2 = 1.0 / self.epsilon2

        self._cell_counter = 0
        self.result: Optional[AdaptiveGridResult] = None

    def _generate_cell_id(self, prefix: str = "cell") -> str:
        """Generate unique cell ID."""
        self._cell_counter += 1
        return f"{prefix}_{self._cell_counter}"

    def _compute_m1(self, n_points: int) -> int:
        """
        Compute coarse grid dimension m1.

        From the paper: m1 = c1 * sqrt(N / ε1)
        where N is the dataset size.

        Args:
            n_points: Number of data points

        Returns:
            Coarse grid dimension (m1 x m1 grid)
        """
        if n_points <= 0 or self.epsilon1 <= 0:
            return self.min_m1

        m1 = int(self.c1 * np.sqrt(n_points / self.epsilon1))
        return max(self.min_m1, min(self.max_m1, m1))

    def _compute_m2(self) -> int:
        """
        Compute fine grid dimension m2.

        From the paper: m2 = c2 * sqrt(ε2)

        Returns:
            Fine grid dimension (m2 x m2 subdivisions per dense cell)
        """
        if self.epsilon2 <= 0:
            return self.min_m2

        m2 = int(self.c2 * np.sqrt(self.epsilon2))
        return max(self.min_m2, min(self.max_m2, m2))

    def _count_points_in_cell(
        self,
        points: np.ndarray,
        cell: GridCell
    ) -> int:
        """Count number of points within a cell."""
        if len(points) == 0:
            return 0

        mask = (
            (points[:, 0] >= cell.min_lon) &
            (points[:, 0] < cell.max_lon) &
            (points[:, 1] >= cell.min_lat) &
            (points[:, 1] < cell.max_lat)
        )
        return int(np.sum(mask))

    def _create_coarse_grid(self, m1: int) -> List[GridCell]:
        """Create the level 1 coarse uniform grid."""
        cells = []
        cell_width = (self.bounds.max_lon - self.bounds.min_lon) / m1
        cell_height = (self.bounds.max_lat - self.bounds.min_lat) / m1

        for i in range(m1):
            for j in range(m1):
                cell = GridCell(
                    min_lon=self.bounds.min_lon + i * cell_width,
                    max_lon=self.bounds.min_lon + (i + 1) * cell_width,
                    min_lat=self.bounds.min_lat + j * cell_height,
                    max_lat=self.bounds.min_lat + (j + 1) * cell_height,
                    level=1,
                    cell_id=self._generate_cell_id("L1"),
                )
                cells.append(cell)

        return cells

    def build(self, points: np.ndarray) -> AdaptiveGridResult:
        """
        Build the Adaptive Grid decomposition from spatial data.

        This implements the two-phase algorithm from the paper:
        1. Create coarse grid and identify dense cells
        2. Subdivide dense cells into fine grid

        Args:
            points: Nx2 array of (lon, lat) coordinates

        Returns:
            AdaptiveGridResult containing all cells with noisy counts
        """
        self._cell_counter = 0
        n_points = len(points)

        # Compute grid dimensions
        m1 = self._compute_m1(n_points)
        m2 = self._compute_m2()

        # ===== PHASE 1: Coarse Grid =====
        coarse_cells = self._create_coarse_grid(m1)

        # Compute noisy counts for coarse cells
        for cell in coarse_cells:
            true_count = self._count_points_in_cell(points, cell)
            noise = generate_laplace_noise(self.noise_scale1)
            cell.noisy_count = max(0, true_count + noise)

        # Determine density threshold
        # A cell is "dense" if its noisy count exceeds the average density * alpha
        expected_avg = n_points / (m1 * m1)
        noise_threshold = self.noise_scale1 * 2  # 2x noise scale as minimum
        density_threshold = max(noise_threshold, expected_avg * self.density_alpha)

        # ===== PHASE 2: Fine Grid for Dense Cells =====
        final_cells: List[GridCell] = []
        level2_count = 0

        for coarse_cell in coarse_cells:
            if coarse_cell.noisy_count > density_threshold:
                # Dense cell: subdivide into m2 x m2 grid
                fine_cells = coarse_cell.subdivide(m2)

                # Assign IDs and compute noisy counts
                for fine_cell in fine_cells:
                    fine_cell.cell_id = self._generate_cell_id("L2")
                    true_count = self._count_points_in_cell(points, fine_cell)
                    noise = generate_laplace_noise(self.noise_scale2)
                    fine_cell.noisy_count = max(0, true_count + noise)

                final_cells.extend(fine_cells)
                level2_count += len(fine_cells)
            else:
                # Sparse cell: keep as is
                final_cells.append(coarse_cell)

        # Store and return result
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
        """
        Convert the grid to GeoJSON FeatureCollection.

        Each cell becomes a polygon feature with its noisy count
        and level as properties.
        """
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
                    "depth": cell.level,  # Alias for compatibility with PrivTree visualization
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
            # For compatibility with PrivTree statistics display
            "total_leaves": len(cells),
            "noise_scale": self.noise_scale2,  # Use level 2 noise scale for display
            "delta": 0.0,  # Not applicable to AG
        }


def create_adaptive_grid_from_bounds(
    epsilon: float,
    min_lon: float,
    max_lon: float,
    min_lat: float,
    max_lat: float,
    **kwargs
) -> AdaptiveGrid:
    """
    Factory function to create an AdaptiveGrid from coordinate bounds.

    Args:
        epsilon: Privacy budget
        min_lon, max_lon, min_lat, max_lat: Bounding box coordinates
        **kwargs: Additional parameters for AdaptiveGrid

    Returns:
        Configured AdaptiveGrid instance
    """
    bounds = GridCell(
        min_lon=min_lon,
        max_lon=max_lon,
        min_lat=min_lat,
        max_lat=max_lat,
        level=0,
        cell_id="bounds",
    )

    return AdaptiveGrid(epsilon=epsilon, bounds=bounds, **kwargs)
