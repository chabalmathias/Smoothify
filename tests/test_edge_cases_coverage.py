"""Tests for edge cases and error conditions to improve coverage."""

import pytest
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    Point,
    Polygon,
)

from smoothify import smoothify
from smoothify.geometry_ops import _smoothify_multilinestring
from smoothify.smoothify_core import (
    _generate_starting_point_variants,
    _preserve_area_with_buffer,
    _rotate_polygon_start,
    _smoothify_geometry,
)


class TestErrorConditions:
    """Test error conditions and edge cases for better coverage."""

    def test_smoothify_geometry_invalid_type(self):
        """Test that invalid geometry type raises ValueError."""
        point = Point(0, 0)
        with pytest.raises(ValueError, match="Input geometry must be"):
            _smoothify_geometry(point, segment_length=1.0)

    def test_smoothify_multilinestring_with_preserve_area(self):
        """Test MultiLineString smoothing (preserve_area is ignored for lines)."""
        line1 = LineString([(0, 0), (1, 1), (2, 0)])
        line2 = LineString([(3, 0), (4, 1), (5, 0)])
        multiline = MultiLineString([line1, line2])

        result = _smoothify_multilinestring(
            geom=multiline,
            segment_length=0.1,
            smooth_iterations=1,
            preserve_area=True,  # Should be ignored for linestrings
        )

        assert isinstance(result, MultiLineString)
        assert len(result.geoms) == 2

    def test_rotate_polygon_start_empty(self):
        """Test rotating an empty polygon."""
        empty_poly = Polygon()
        result = _rotate_polygon_start(empty_poly, shift=5)
        assert result.is_empty

    def test_generate_starting_point_variants_zero_n(self):
        """Test generating variants with n=0 returns empty list."""
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = _generate_starting_point_variants(square, n_starting_points=0)
        assert result == []

    def test_generate_starting_point_variants_negative_n(self):
        """Test generating variants with negative n returns empty list."""
        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        result = _generate_starting_point_variants(square, n_starting_points=-5)
        assert result == []

    def test_generate_starting_point_variants_n_exceeds_vertices(self):
        """Test when n_starting_points exceeds available vertices."""
        triangle = Polygon([(0, 0), (1, 0), (0.5, 1)])  # 3 vertices + closing
        result = _generate_starting_point_variants(triangle, n_starting_points=100)
        # Should be clamped to number of vertices - 1
        assert len(result) <= 3

    def test_smoothify_with_num_cores_zero(self):
        """Test smoothify with num_cores=0 uses all available cores."""
        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        collection = GeometryCollection([square])

        # This should use cpu_count() internally
        result = smoothify(collection, segment_length=1.0, num_cores=0)
        assert result is not None

    def test_smoothify_with_num_cores_negative(self):
        """Test smoothify with negative num_cores uses all available cores."""
        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        collection = GeometryCollection([square])

        result = smoothify(collection, segment_length=1.0, num_cores=-1)
        assert result is not None

    def test_smoothify_with_num_cores_one(self):
        """Test smoothify with num_cores=1 uses serial processing."""
        square1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        square2 = Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])
        collection = GeometryCollection([square1, square2])

        result = smoothify(collection, segment_length=1.0, num_cores=1)
        assert result is not None
        assert isinstance(result, GeometryCollection)

    def test_preserve_area_extreme_case(self):
        """Test area preservation with extreme area differences."""
        # Create a polygon and then a much smaller version
        original = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        # Severely shrink it
        shrunken = original.buffer(-40)  # Much smaller

        # Try to restore original area
        result = _preserve_area_with_buffer(
            shrunken, target_area=original.area, tolerance=0.1
        )
        assert isinstance(result, Polygon)
        # Should attempt to get close to target area
        assert result.area > shrunken.area

    def test_preserve_area_bracket_search_fallback(self):
        """Test the bracket search fallback when brentq fails."""
        # Create a very small polygon that might trigger edge cases
        tiny_poly = Polygon([(0, 0), (0.001, 0), (0.001, 0.001), (0, 0.001)])
        target_area = 10.0  # Much larger

        # This might trigger the ValueError fallback
        result = _preserve_area_with_buffer(
            tiny_poly, target_area=target_area, tolerance=0.01
        )
        assert isinstance(result, Polygon)

    def test_smoothify_geometry_produces_multipolygon_union(self):
        """Test case where union of variants produces MultiPolygon."""
        # Create a very thin polygon that might split when smoothed
        thin_polygon = Polygon(
            [
                (0, 0),
                (0.01, 0),
                (0.01, 10),
                (0.005, 5),
                (0, 10),
            ]  # Very thin with indent
        )

        # Use aggressive smoothing that might cause splitting
        result = _smoothify_geometry(
            thin_polygon, segment_length=0.001, smooth_iterations=5, preserve_area=False
        )
        # Should handle the MultiPolygon case and return largest
        assert isinstance(result, (Polygon, LineString))

    def test_smoothify_linear_ring_directly(self):
        """Test smoothing a LinearRing geometry."""
        ring = LinearRing([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = smoothify(ring, segment_length=1.0)
        # LinearRing is returned as LinearRing, not converted to Polygon
        assert isinstance(result, (Polygon, LinearRing))

    def test_smoothify_multilinestring_fallback_to_collection(self):
        """MultiLineString that can't keep its structure returns a Collection."""
        # Create lines that might not all process cleanly
        line1 = LineString([(0, 0), (1, 0)])  # Very short
        line2 = LineString([(10, 10), (11, 10)])
        multiline = MultiLineString([line1, line2])

        result = smoothify(multiline, segment_length=0.1)
        # Should return either MultiLineString or GeometryCollection
        assert isinstance(result, (MultiLineString, GeometryCollection))

    def test_auto_detect_segment_length_with_short_coords(self):
        """Test auto-detection with geometry having very few coordinates."""
        # Single segment linestring
        line = LineString([(0, 0), (10, 0)])
        result = smoothify(line, segment_length=None)
        assert result is not None

    def test_auto_detect_segment_length_all_zero_segments(self):
        """Test auto-detection with all zero-length segments raises error."""
        # Create degenerate geometry with duplicate points
        line = LineString([(0, 0), (0, 0), (0, 0)])
        # Should raise ValueError when can't auto-detect
        with pytest.raises(ValueError, match="Could not auto-detect segment_length"):
            smoothify(line, segment_length=None)

    def test_generate_starting_point_variants_invalid_type(self):
        """Test generating variants with invalid geometry type."""
        point = Point(0, 0)
        with pytest.raises(ValueError, match="Input geometry must be"):
            _generate_starting_point_variants(point, n_starting_points=5)

    def test_preserve_area_empty_polygon(self):
        """Test preserving area of empty polygon."""
        empty_poly = Polygon()
        result = _preserve_area_with_buffer(
            empty_poly, target_area=100.0, tolerance=0.1
        )
        assert result.is_empty

    def test_preserve_area_tolerance_fallback_refinement(self):
        """Test the refinement step when initial brentq doesn't meet tolerance."""
        # Create a scenario where we need refinement
        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        smoothed = square.buffer(-0.5)  # Slightly smaller

        # Use very tight tolerance to potentially trigger refinement
        result = _preserve_area_with_buffer(
            smoothed, target_area=square.area, tolerance=1e-8
        )
        assert isinstance(result, Polygon)
        # Should be very close to target area
        assert abs(result.area - square.area) < 0.01

    def test_preserve_area_bracket_expansion(self):
        """Test bracket expansion in area preservation algorithm."""
        # Create a very distorted polygon that needs large bracket expansion
        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        # Apply extreme buffer to create large area difference
        very_large = square.buffer(50)  # Much larger

        result = _preserve_area_with_buffer(
            very_large, target_area=square.area, tolerance=0.1
        )
        assert isinstance(result, Polygon)
        # Should shrink back toward target
        assert result.area < very_large.area

    def test_preserve_area_brentq_fallback(self):
        """Test ValueError fallback when brentq can't find a solution."""
        # This is difficult to trigger reliably, but we can test the fallback path
        # by using a polygon that might cause numerical issues
        tiny = Polygon([(0, 0), (1e-10, 0), (1e-10, 1e-10), (0, 1e-10)])
        target = 1000.0

        # Should use fallback and return closest result
        result = _preserve_area_with_buffer(tiny, target_area=target, tolerance=1e-8)
        assert isinstance(result, Polygon)
