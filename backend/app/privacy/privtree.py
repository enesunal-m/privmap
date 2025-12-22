"""
PrivTree Algorithm Implementation.

This module implements the PrivTree algorithm for differentially private
hierarchical spatial decomposition, based on the paper:
"PrivTree: A Differentially Private Algorithm for Hierarchical Decompositions"
by Zhang, Xiao, and Xie (SIGMOD 2016).

Key insight: Unlike traditional approaches that require noise proportional to
tree height h, PrivTree uses a constant amount of noise in split decisions
by introducing a carefully controlled bias term.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from collections import deque

from .laplace import generate_laplace_noise


@dataclass
class BoundingBox:
    """Represents a rectangular region in 2D space."""
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    
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
        """Check if a point is within this bounding box."""
        return (self.min_lon <= lon < self.max_lon and 
                self.min_lat <= lat < self.max_lat)
    
    def split_quadrants(self) -> List['BoundingBox']:
        """Split this box into 4 equal quadrants (for quadtree)."""
        mid_lon = (self.min_lon + self.max_lon) / 2
        mid_lat = (self.min_lat + self.max_lat) / 2
        
        return [
            # Bottom-left
            BoundingBox(self.min_lon, mid_lon, self.min_lat, mid_lat),
            # Bottom-right
            BoundingBox(mid_lon, self.max_lon, self.min_lat, mid_lat),
            # Top-left
            BoundingBox(self.min_lon, mid_lon, mid_lat, self.max_lat),
            # Top-right
            BoundingBox(mid_lon, self.max_lon, mid_lat, self.max_lat),
        ]
    
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


@dataclass
class PrivTreeNode:
    """
    A node in the PrivTree decomposition structure.
    
    Each node represents a spatial region and stores:
    - Its bounding box (the region it covers)
    - Its depth in the tree
    - Whether it's a leaf node
    - The noisy count of points in its region (for leaf nodes)
    """
    bounds: BoundingBox
    depth: int = 0
    is_leaf: bool = True
    children: List['PrivTreeNode'] = field(default_factory=list)
    noisy_count: Optional[float] = None
    node_id: str = ""
    
    def to_dict(self) -> dict:
        """Convert node to dictionary for serialization."""
        result = {
            "id": self.node_id,
            "bounds": {
                "min_lon": self.bounds.min_lon,
                "max_lon": self.bounds.max_lon,
                "min_lat": self.bounds.min_lat,
                "max_lat": self.bounds.max_lat,
            },
            "depth": self.depth,
            "is_leaf": self.is_leaf,
            "noisy_count": self.noisy_count,
        }
        if self.children:
            result["children"] = [c.to_dict() for c in self.children]
        return result


class PrivTree:
    """
    PrivTree: Parameter-free differentially private spatial decomposition.
    
    The algorithm adaptively partitions space based on data density while
    guaranteeing ε-differential privacy. Unlike traditional hierarchical
    methods, PrivTree doesn't require specifying a maximum tree height.
    
    Key parameters (from the paper):
    - ε (epsilon): Privacy budget
    - β (fanout): Number of children per node (4 for quadtree)
    - θ (theta): Threshold for splitting decision (typically 0)
    - δ (delta): Decay factor = λ * ln(β), controls bias reduction
    - λ (lambda): Noise scale = (2β - 1) / ((β - 1) * ε)
    
    The algorithm satisfies ε-DP when λ ≥ (2β - 1) / ((β - 1) * ε)
    """
    
    def __init__(
        self,
        epsilon: float,
        bounds: BoundingBox,
        fanout: int = 4,
        theta: float = 0.0,
        max_depth: int = 15,  # Safety limit to prevent infinite recursion
    ):
        """
        Initialize PrivTree with privacy and spatial parameters.
        
        Args:
            epsilon: Privacy budget (smaller = more private, more noise)
            bounds: The bounding box of the entire data domain
            fanout: Number of children per split (4 for 2D quadtree)
            theta: Threshold for split decision
            max_depth: Maximum tree depth (safety limit)
        """
        self.epsilon = epsilon
        self.bounds = bounds
        self.fanout = fanout
        self.theta = theta
        self.max_depth = max_depth
        
        # Calculate noise scale λ from Corollary 1 in the paper
        # For ε-DP: λ ≥ (2β - 1) / ((β - 1) * ε)
        # We use half epsilon for tree structure, half for leaf counts
        structure_epsilon = epsilon / 2
        self.noise_scale = (2 * fanout - 1) / ((fanout - 1) * structure_epsilon)
        
        # Calculate decay factor δ = λ * ln(β)
        self.delta = self.noise_scale * np.log(fanout)
        
        # For publishing noisy counts at leaves
        self.count_noise_scale = 2.0 / epsilon
        
        self.root: Optional[PrivTreeNode] = None
        self._node_counter = 0
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        self._node_counter += 1
        return f"node_{self._node_counter}"
    
    def _count_points_in_region(
        self,
        points: np.ndarray,
        bounds: BoundingBox
    ) -> int:
        """
        Count number of points within a bounding box.
        
        Args:
            points: Nx2 array of (lon, lat) coordinates
            bounds: The region to count points in
            
        Returns:
            Number of points in the region
        """
        if len(points) == 0:
            return 0
            
        mask = (
            (points[:, 0] >= bounds.min_lon) &
            (points[:, 0] < bounds.max_lon) &
            (points[:, 1] >= bounds.min_lat) &
            (points[:, 1] < bounds.max_lat)
        )
        return int(np.sum(mask))
    
    def _compute_biased_count(self, true_count: int, depth: int) -> float:
        """
        Compute the biased count for split decision.
        
        From the paper (Equation 5):
        b(v) = max{θ - δ, c(v) - depth(v) * δ}
        
        The bias term ensures that the privacy cost decreases exponentially
        with tree depth, allowing for a constant total privacy budget.
        """
        biased = true_count - depth * self.delta
        return max(self.theta - self.delta, biased)
    
    def build(self, points: np.ndarray) -> PrivTreeNode:
        """
        Build the PrivTree decomposition from spatial data.
        
        This implements Algorithm 2 from the paper. The key innovation is
        that the split decision uses a biased count that decays with depth,
        allowing constant noise regardless of tree height.
        
        Args:
            points: Nx2 array of (lon, lat) coordinates
            
        Returns:
            Root node of the constructed tree
        """
        self._node_counter = 0
        
        # Initialize root node
        self.root = PrivTreeNode(
            bounds=self.bounds,
            depth=0,
            node_id=self._generate_node_id()
        )
        
        # BFS queue: (node, points in this node's region)
        queue = deque([(self.root, points)])
        
        while queue:
            node, node_points = queue.popleft()
            
            # Count points in this region
            true_count = self._count_points_in_region(node_points, node.bounds)
            
            # Compute biased count (Equation 5)
            biased_count = self._compute_biased_count(true_count, node.depth)
            
            # Add Laplace noise for split decision
            noisy_biased_count = biased_count + generate_laplace_noise(self.noise_scale)
            
            # Decision: split if noisy biased count exceeds threshold
            should_split = (
                noisy_biased_count > self.theta and 
                node.depth < self.max_depth
            )
            
            if should_split:
                # Split into children
                node.is_leaf = False
                child_bounds = node.bounds.split_quadrants()
                
                for bounds in child_bounds:
                    child = PrivTreeNode(
                        bounds=bounds,
                        depth=node.depth + 1,
                        node_id=self._generate_node_id()
                    )
                    node.children.append(child)
                    
                    # Filter points for child region
                    child_points = self._filter_points(node_points, bounds)
                    queue.append((child, child_points))
            else:
                # Leaf node: store noisy count for query answering
                node.is_leaf = True
                noise = generate_laplace_noise(self.count_noise_scale)
                node.noisy_count = max(0, true_count + noise)
        
        return self.root
    
    def _filter_points(self, points: np.ndarray, bounds: BoundingBox) -> np.ndarray:
        """Filter points to those within bounds."""
        if len(points) == 0:
            return points
            
        mask = (
            (points[:, 0] >= bounds.min_lon) &
            (points[:, 0] < bounds.max_lon) &
            (points[:, 1] >= bounds.min_lat) &
            (points[:, 1] < bounds.max_lat)
        )
        return points[mask]
    
    def get_leaf_nodes(self) -> List[PrivTreeNode]:
        """Get all leaf nodes in the tree."""
        if self.root is None:
            return []
        
        leaves = []
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            if node.is_leaf:
                leaves.append(node)
            else:
                queue.extend(node.children)
        
        return leaves
    
    def to_geojson(self) -> dict:
        """
        Convert the tree to GeoJSON FeatureCollection.
        
        Each leaf node becomes a polygon feature with its noisy count
        as a property, suitable for visualization on a map.
        """
        features = []
        
        for leaf in self.get_leaf_nodes():
            feature = {
                "type": "Feature",
                "geometry": leaf.bounds.to_geojson(),
                "properties": {
                    "id": leaf.node_id,
                    "depth": leaf.depth,
                    "count": leaf.noisy_count,
                }
            }
            features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features,
        }
    
    def get_statistics(self) -> dict:
        """Get statistics about the built tree."""
        if self.root is None:
            return {}
        
        leaves = self.get_leaf_nodes()
        depths = [leaf.depth for leaf in leaves]
        counts = [leaf.noisy_count or 0 for leaf in leaves]
        
        return {
            "total_leaves": len(leaves),
            "max_depth": max(depths) if depths else 0,
            "min_depth": min(depths) if depths else 0,
            "avg_depth": sum(depths) / len(depths) if depths else 0,
            "total_noisy_count": sum(counts),
            "epsilon_used": self.epsilon,
            "noise_scale": self.noise_scale,
            "delta": self.delta,
        }

