"""Tests for core smoothify geometry functions."""

import pytest
from shapely.geometry import LineString, Polygon

from smoothify.smoothify_core import (
    _generate_starting_point_variants,
    _join_adjacent,
    _preserve_area_with_buffer,
    _rotate_polygon_start,
    _smoothify_geometry,
)


class TestRotatePolygonStart:
    """Test suite for polygon rotation functions."""

    def test_rotate_square(self):
        """Test rotating a square's starting point."""
        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        rotated = _rotate_polygon_start(square, shift=0.25)

        assert isinstance(rotated, Polygon)
        assert rotated.is_valid
        # Area should be identical
        assert abs(rotated.area - square.area) < 1e-10


class TestGenerateStartingPointVariants:
    """Test suite for generating polygon variants."""

    def test_polygon_variants(self):
        """Test generating variants for a polygon."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        variants = _generate_starting_point_variants(polygon, n_starting_points=4)

        assert len(variants) == 4
        assert all(isinstance(v, Polygon) for v in variants)
        assert all(v.is_valid for v in variants)

    def test_linestring_variants(self):
        """Test that linestrings return single variant."""
        line = LineString([(0, 0), (5, 5), (10, 0)])
        variants = _generate_starting_point_variants(line, n_starting_points=4)

        # LineStrings should return single item
        assert len(variants) == 1
        assert variants[0] == line


class TestPreserveAreaWithBuffer:
    """Test suite for area preservation function."""

    def test_preserve_area_exact(self):
        """Test preserving area when already exact."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        original_area = polygon.area

        preserved = _preserve_area_with_buffer(
            polygon, target_area=original_area, tolerance=1e-6
        )

        # Should return same polygon if area already matches
        assert abs(preserved.area - original_area) < 1e-6

    def test_preserve_area_smaller_polygon(self):
        """Test expanding a smaller polygon to target area."""
        small_polygon = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        target_area = 100.0  # Larger than 25

        preserved = _preserve_area_with_buffer(
            small_polygon, target_area=target_area, tolerance=1e-3
        )

        # Should be close to target area
        assert abs(preserved.area - target_area) < 1e-3
        assert preserved.area > small_polygon.area

    def test_preserve_area_larger_polygon(self):
        """Test shrinking a larger polygon to target area."""
        large_polygon = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        target_area = 100.0  # Smaller than 400

        preserved = _preserve_area_with_buffer(
            large_polygon, target_area=target_area, tolerance=1e-3
        )

        # Should be close to target area
        assert abs(preserved.area - target_area) < 1e-3
        assert preserved.area < large_polygon.area


class TestJoinAdjacent:
    """Test suite for joining adjacent geometries."""

    def test_join_two_squares(self):
        """Test joining two adjacent squares."""
        square1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        square2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])

        merged = _join_adjacent([square1, square2], segment_length=10.0)

        assert merged.is_valid
        # Should create a single merged geometry
        assert isinstance(merged, Polygon) or hasattr(merged, "geoms")

    def test_join_single_geometry(self):
        """Test joining a single geometry."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        merged = _join_adjacent(polygon, segment_length=10.0)

        assert merged.is_valid


class TestSmoothifyGeometry:
    """Test suite for the main smoothify_geometry function."""

    def test_smoothify_simple_polygon(self):
        """Test smoothing a simple polygon."""
        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        smoothed = _smoothify_geometry(
            square, segment_length=1.0, smooth_iterations=3, preserve_area=True
        )

        assert isinstance(smoothed, Polygon)
        assert smoothed.is_valid
        # Should have more vertices
        assert len(smoothed.exterior.coords) > len(square.exterior.coords)

    def test_smoothify_linestring(self):
        """Test smoothing a linestring."""
        line = LineString([(0, 0), (5, 5), (10, 0)])
        smoothed = _smoothify_geometry(
            line, segment_length=1.0, smooth_iterations=3, preserve_area=False
        )

        assert isinstance(smoothed, LineString)
        assert smoothed.is_valid
        # Endpoints should be preserved
        assert smoothed.coords[0] == line.coords[0]
        assert smoothed.coords[-1] == line.coords[-1]

    def test_smoothify_self_intersecting_linestring(self):
        """Test that a self-intersecting LineString is smoothed without error.

        Lines that cross themselves are geometrically valid. The smoothing
        pipeline must not split them into a MultiLineString via make_valid
        or unary_union, which node lines at self-intersection points.
        """
        # S-curve that crosses itself — triggers unary_union splitting
        line = LineString(
            [
                (0, 0),
                (2, 0),
                (3, 0),
                (4, 1),
                (3, 2),
                (2, 2),
                (1, 1),
                (2, 0.5),
                (3, 0.5),
                (4, 0),
                (5, 0),
                (7, 0),
            ]
        )
        assert not line.is_simple  # confirm it self-intersects

        smoothed = _smoothify_geometry(
            line, segment_length=0.5, smooth_iterations=3, preserve_area=False
        )

        assert isinstance(smoothed, LineString)

    def test_preserve_area_option(self):
        """Test that preserve_area option works."""
        square = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        original_area = square.area

        smoothed = _smoothify_geometry(
            square, segment_length=10.0, smooth_iterations=3, preserve_area=True
        )

        # Area should be close to original
        assert abs(smoothed.area - original_area) / original_area < 0.05  # Within 5%

    def test_no_preserve_area(self):
        """Test smoothing without area preservation."""
        square = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        original_area = square.area

        smoothed = _smoothify_geometry(
            square, segment_length=10.0, smooth_iterations=3, preserve_area=False
        )

        # Area will likely be different (smaller due to corner cutting)
        assert smoothed.is_valid
        assert smoothed.area < original_area

    def test_invalid_geometry_type(self):
        """Test that invalid geometry type raises error."""
        with pytest.raises((ValueError, AttributeError)):
            _smoothify_geometry(
                "not a geometry", segment_length=1.0, smooth_iterations=3
            )  # type: ignore
