"""Tests for the area_tolerance parameter."""

from shapely.geometry import Polygon
from smoothify import smoothify


class TestAreaTolerance:
    """Test suite for area_tolerance parameter."""

    def test_area_tolerance_strict(self):
        """Test that strict tolerance produces accurate area."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        original_area = polygon.area  # 100 square units

        # Very strict tolerance: 0.001% of 100 = 0.001 absolute error
        smoothed = smoothify(
            polygon,
            segment_length=1.0,
            smooth_iterations=3,
            preserve_area=True,
            area_tolerance=0.001,
        )

        # Should be within strict tolerance (0.001% of 100 = 0.001 units)
        assert abs(smoothed.area - original_area) < original_area * 0.001 / 100

    def test_area_tolerance_relaxed(self):
        """Test that relaxed tolerance works."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        original_area = polygon.area  # 100 square units

        # More relaxed tolerance: 0.1% of 100 = 0.1 absolute error
        smoothed = smoothify(
            polygon,
            segment_length=1.0,
            smooth_iterations=3,
            preserve_area=True,
            area_tolerance=0.1,
        )

        # Should be within relaxed tolerance (0.1% of 100 = 0.1 units)
        assert abs(smoothed.area - original_area) < original_area * 0.1 / 100

    def test_area_tolerance_default(self):
        """Test that default tolerance works (0.01%)."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        original_area = polygon.area  # 100 square units

        # Use default tolerance (0.01% by not specifying area_tolerance)
        smoothed = smoothify(
            polygon,
            segment_length=1.0,
            smooth_iterations=3,
            preserve_area=True,
            # area_tolerance defaults to 0.01%
        )

        # Should preserve area with default tolerance (0.01% of 100 = 0.01 units)
        assert abs(smoothed.area - original_area) < original_area * 0.01 / 100

    def test_area_tolerance_comparison(self):
        """Test that stricter tolerance produces more accurate area."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        original_area = polygon.area  # 100 square units

        # Smooth with relaxed tolerance (0.1%)
        smoothed_relaxed = smoothify(
            polygon,
            segment_length=1.0,
            smooth_iterations=3,
            preserve_area=True,
            area_tolerance=0.1,
        )

        # Smooth with strict tolerance (0.0001%)
        smoothed_strict = smoothify(
            polygon,
            segment_length=1.0,
            smooth_iterations=3,
            preserve_area=True,
            area_tolerance=0.0001,
        )

        error_relaxed = abs(smoothed_relaxed.area - original_area)
        error_strict = abs(smoothed_strict.area - original_area)

        # Strict should be more accurate (or equal)
        # Small margin for numerical stability
        assert error_strict <= error_relaxed + original_area * 0.001 / 100
