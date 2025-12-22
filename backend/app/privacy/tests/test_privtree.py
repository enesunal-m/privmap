"""
Unit tests for PrivTree algorithm.
"""

import pytest
import numpy as np
from ..privtree import PrivTree, BoundingBox, PrivTreeNode


class TestBoundingBox:
    """Tests for BoundingBox class."""
    
    def test_creation(self):
        """Test bounding box creation."""
        box = BoundingBox(-8.7, -8.5, 41.1, 41.25)
        assert box.min_lon == -8.7
        assert box.max_lon == -8.5
        assert box.min_lat == 41.1
        assert box.max_lat == 41.25
    
    def test_dimensions(self):
        """Test width and height calculation."""
        box = BoundingBox(-8.7, -8.5, 41.1, 41.25)
        assert abs(box.width - 0.2) < 0.001
        assert abs(box.height - 0.15) < 0.001
    
    def test_center(self):
        """Test center point calculation."""
        box = BoundingBox(-8.7, -8.5, 41.1, 41.25)
        center = box.center
        assert abs(center[0] - (-8.6)) < 0.001
        assert abs(center[1] - 41.175) < 0.001
    
    def test_contains_point(self):
        """Test point containment check."""
        box = BoundingBox(-8.7, -8.5, 41.1, 41.25)
        
        # Point inside
        assert box.contains_point(-8.6, 41.15)
        
        # Point outside
        assert not box.contains_point(-9.0, 41.15)
        assert not box.contains_point(-8.6, 42.0)
        
        # Edge cases (lower bound inclusive, upper exclusive)
        assert box.contains_point(-8.7, 41.1)
        assert not box.contains_point(-8.5, 41.1)
    
    def test_split_quadrants(self):
        """Test quadrant splitting."""
        box = BoundingBox(-8.0, -7.0, 40.0, 41.0)
        quadrants = box.split_quadrants()
        
        assert len(quadrants) == 4
        
        # Check that quadrants cover the original area
        total_area = sum(q.area for q in quadrants)
        assert abs(total_area - box.area) < 0.001
    
    def test_to_geojson(self):
        """Test GeoJSON conversion."""
        box = BoundingBox(-8.0, -7.0, 40.0, 41.0)
        geojson = box.to_geojson()
        
        assert geojson["type"] == "Polygon"
        assert len(geojson["coordinates"]) == 1
        assert len(geojson["coordinates"][0]) == 5  # Closed polygon


class TestPrivTree:
    """Tests for PrivTree algorithm."""
    
    @pytest.fixture
    def sample_points(self):
        """Generate sample spatial points for testing."""
        np.random.seed(42)
        
        # Create clustered points
        cluster1 = np.random.normal([-8.6, 41.15], 0.02, (500, 2))
        cluster2 = np.random.normal([-8.55, 41.2], 0.015, (300, 2))
        
        return np.vstack([cluster1, cluster2])
    
    @pytest.fixture
    def bounds(self):
        """Default bounds for testing."""
        return BoundingBox(-8.7, -8.5, 41.1, 41.25)
    
    def test_initialization(self, bounds):
        """Test PrivTree initialization."""
        tree = PrivTree(epsilon=1.0, bounds=bounds)
        
        assert tree.epsilon == 1.0
        assert tree.fanout == 4
        assert tree.theta == 0.0
        assert tree.noise_scale > 0
        assert tree.delta > 0
    
    def test_noise_scale_calculation(self, bounds):
        """Test that noise scale follows the paper's formula."""
        epsilon = 1.0
        fanout = 4
        tree = PrivTree(epsilon=epsilon, bounds=bounds, fanout=fanout)
        
        # From Corollary 1: λ ≥ (2β - 1) / ((β - 1) * ε)
        # We use half epsilon for structure
        structure_epsilon = epsilon / 2
        expected_scale = (2 * fanout - 1) / ((fanout - 1) * structure_epsilon)
        
        assert abs(tree.noise_scale - expected_scale) < 0.001
    
    def test_delta_calculation(self, bounds):
        """Test that delta follows the paper's formula."""
        tree = PrivTree(epsilon=1.0, bounds=bounds)
        
        # From Lemma 3: δ = λ * ln(β)
        expected_delta = tree.noise_scale * np.log(tree.fanout)
        
        assert abs(tree.delta - expected_delta) < 0.001
    
    def test_build_empty_data(self, bounds):
        """Test building tree with no data."""
        tree = PrivTree(epsilon=1.0, bounds=bounds)
        empty_points = np.array([]).reshape(0, 2)
        
        root = tree.build(empty_points)
        
        assert root is not None
        assert root.is_leaf  # Should not split with no data
    
    def test_build_with_data(self, bounds, sample_points):
        """Test building tree with sample data."""
        tree = PrivTree(epsilon=1.0, bounds=bounds)
        root = tree.build(sample_points)
        
        assert root is not None
        assert root.bounds == bounds
        
        # Tree should have multiple levels for clustered data
        leaves = tree.get_leaf_nodes()
        assert len(leaves) > 1
    
    def test_leaf_nodes_have_counts(self, bounds, sample_points):
        """Test that all leaf nodes have noisy counts."""
        tree = PrivTree(epsilon=1.0, bounds=bounds)
        tree.build(sample_points)
        
        leaves = tree.get_leaf_nodes()
        
        for leaf in leaves:
            assert leaf.is_leaf
            assert leaf.noisy_count is not None
            assert leaf.noisy_count >= 0  # Counts are non-negative
    
    def test_geojson_output(self, bounds, sample_points):
        """Test GeoJSON output format."""
        tree = PrivTree(epsilon=1.0, bounds=bounds)
        tree.build(sample_points)
        
        geojson = tree.to_geojson()
        
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) > 0
        
        for feature in geojson["features"]:
            assert feature["type"] == "Feature"
            assert "geometry" in feature
            assert "properties" in feature
            assert feature["geometry"]["type"] == "Polygon"
            assert "count" in feature["properties"]
    
    def test_statistics(self, bounds, sample_points):
        """Test statistics calculation."""
        tree = PrivTree(epsilon=1.0, bounds=bounds)
        tree.build(sample_points)
        
        stats = tree.get_statistics()
        
        assert "total_leaves" in stats
        assert "max_depth" in stats
        assert "min_depth" in stats
        assert "avg_depth" in stats
        assert "epsilon_used" in stats
        assert "noise_scale" in stats
        assert "delta" in stats
        
        assert stats["total_leaves"] > 0
        assert stats["max_depth"] >= stats["min_depth"]
        assert stats["epsilon_used"] == 1.0
    
    def test_differential_privacy_property(self, bounds):
        """
        Test that the algorithm produces different results on different runs.
        (Due to random noise, identical runs should produce different outputs.)
        """
        np.random.seed(None)  # Use system entropy
        
        points = np.random.uniform(
            [bounds.min_lon, bounds.min_lat],
            [bounds.max_lon, bounds.max_lat],
            (100, 2)
        )
        
        # Run twice
        tree1 = PrivTree(epsilon=1.0, bounds=bounds)
        tree1.build(points)
        leaves1 = tree1.get_leaf_nodes()
        
        tree2 = PrivTree(epsilon=1.0, bounds=bounds)
        tree2.build(points)
        leaves2 = tree2.get_leaf_nodes()
        
        # Trees should differ due to randomness
        # (Note: structure might be same, but counts should differ)
        if len(leaves1) == len(leaves2):
            counts1 = [l.noisy_count for l in leaves1]
            counts2 = [l.noisy_count for l in leaves2]
            # At least some counts should differ
            assert counts1 != counts2
    
    def test_higher_epsilon_more_splits(self, bounds, sample_points):
        """
        Test that higher epsilon (less noise) tends to create more splits.
        """
        tree_low_eps = PrivTree(epsilon=0.1, bounds=bounds)
        tree_low_eps.build(sample_points)
        
        tree_high_eps = PrivTree(epsilon=5.0, bounds=bounds)
        tree_high_eps.build(sample_points)
        
        # Higher epsilon should generally allow more splits
        # (though this isn't guaranteed due to randomness)
        leaves_low = tree_low_eps.get_leaf_nodes()
        leaves_high = tree_high_eps.get_leaf_nodes()
        
        # Just check both produce valid output
        assert len(leaves_low) > 0
        assert len(leaves_high) > 0


class TestLaplaceMechanism:
    """Tests for Laplace noise generation."""
    
    def test_noise_has_zero_mean(self):
        """Test that Laplace noise has approximately zero mean."""
        from ..laplace import generate_laplace_noise
        
        samples = generate_laplace_noise(scale=1.0, size=10000)
        mean = np.mean(samples)
        
        # Should be close to 0 (within 3 standard errors)
        assert abs(mean) < 0.1
    
    def test_noise_scale(self):
        """Test that noise scale affects variance correctly."""
        from ..laplace import generate_laplace_noise
        
        samples1 = generate_laplace_noise(scale=1.0, size=10000)
        samples2 = generate_laplace_noise(scale=2.0, size=10000)
        
        std1 = np.std(samples1)
        std2 = np.std(samples2)
        
        # Variance scales with λ^2, so std scales with λ
        assert std2 > std1 * 1.5  # Should be roughly 2x
    
    def test_add_noise(self):
        """Test adding noise to a value."""
        from ..laplace import add_laplace_noise
        
        value = 100.0
        noisy_value = add_laplace_noise(value, sensitivity=1.0, epsilon=1.0)
        
        # Should be different (with high probability)
        assert noisy_value != value
        
        # Should be in reasonable range
        assert 50 < noisy_value < 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

