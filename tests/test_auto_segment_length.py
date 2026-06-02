"""Tests for the auto pixel size detection function."""

import geopandas as gpd
import pytest
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)

from smoothify.geometry_ops import _auto_detect_segment_length


class TestAutoDetectPixelSize:
    """Test automatic pixel size detection from geometry segment lengths."""

    def test_simple_polygon(self):
        """Test detection with a simple square polygon."""
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        segment_length = _auto_detect_segment_length(polygon)
        assert segment_length == 1.0

    def test_polygon_with_simplified_edges(self):
        """Test detection handles simplified straight edges correctly."""
        # Polygon with 5-unit bottom edge (simplified), but 1-unit corners
        polygon = Polygon(
            [
                (0, 0),
                (5, 0),  # 5 units (simplified!)
                (5, 1),  # 1 unit (corner)
                (5, 5),  # 4 units
                (4, 5),
                (3, 5),
                (2, 5),
                (1, 5),
                (0, 5),  # 1 unit each
                (0, 4),
                (0, 3),
                (0, 2),
                (0, 1),  # 1 unit each
            ]
        )
        segment_length = _auto_detect_segment_length(polygon)
        # Should find minimum (1.0), not first segment (5.0)
        assert segment_length == 1.0

    def test_polygon_first_segment_not_representative(self):
        """Test when first segment is much larger than pixel size."""
        # First segment is 10 units (simplified edge)
        polygon = Polygon(
            [
                (0, 0),
                (10, 0),  # First: 10 units
                (10, 1),
                (11, 1),
                (11, 2),
                (10, 2),  # Min: 1 unit
                (10, 10),
                (0, 10),
            ]
        )
        segment_length = _auto_detect_segment_length(polygon)
        # Should detect 1.0, not 10.0
        assert segment_length == 1.0

    def test_polygon_with_hole(self):
        """Test detection checks interior rings (holes)."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Hole with smaller pixel size
        hole = [(2, 2), (2.5, 2), (2.5, 2.5), (2, 2.5)]
        polygon = Polygon(exterior, [hole])
        segment_length = _auto_detect_segment_length(polygon)
        # Should find minimum from hole (0.5), not exterior (10.0)
        assert segment_length == 0.5

    def test_polygon_with_multiple_holes(self):
        """Test detection with multiple holes."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole1 = [(2, 2), (4, 2), (4, 4), (2, 4)]
        hole2 = [(10, 10), (10.5, 10), (10.5, 10.5), (10, 10.5)]
        polygon = Polygon(exterior, [hole1, hole2])
        segment_length = _auto_detect_segment_length(polygon)
        # Should find minimum from hole2 (0.5)
        assert segment_length == 0.5

    def test_linestring(self):
        """Test detection with a LineString."""
        line = LineString([(0, 0), (2, 0), (2, 2), (4, 2)])
        segment_length = _auto_detect_segment_length(line)
        assert segment_length == 2.0

    def test_linestring_varying_segments(self):
        """Test LineString with varying segment lengths."""
        line = LineString(
            [
                (0, 0),
                (10, 0),  # 10 units
                (10, 1),
                (11, 1),
                (11, 2),  # 1 unit each
            ]
        )
        segment_length = _auto_detect_segment_length(line)
        # Should find minimum (1.0)
        assert segment_length == 1.0

    def test_linear_ring(self):
        """Test detection with a LinearRing."""
        ring = LinearRing([(0, 0), (3, 0), (3, 3), (0, 3)])
        segment_length = _auto_detect_segment_length(ring)
        assert segment_length == 3.0

    def test_multipolygon(self):
        """Test detection with MultiPolygon."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
        multi = MultiPolygon([poly1, poly2])
        segment_length = _auto_detect_segment_length(multi)
        # Should find minimum across all polygons (1.0 from poly2)
        assert segment_length == 1.0

    def test_multilinestring(self):
        """Test detection with MultiLineString."""
        line1 = LineString([(0, 0), (10, 0), (10, 10)])
        line2 = LineString([(20, 20), (21, 20), (21, 21)])
        multi = MultiLineString([line1, line2])
        segment_length = _auto_detect_segment_length(multi)
        # Should find minimum across all lines (1.0 from line2)
        assert segment_length == 1.0

    def test_geometry_collection(self):
        """Test detection with GeometryCollection."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        line = LineString([(20, 20), (20.5, 20), (20.5, 20.5)])
        collection = GeometryCollection([poly, line])
        segment_length = _auto_detect_segment_length(collection)
        # Should find minimum across all geometries (0.5 from line)
        assert segment_length == 0.5

    def test_geometry_collection_with_points(self):
        """Test that Point geometries are skipped."""
        poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        point = Point(10, 10)
        collection = GeometryCollection([point, poly])
        segment_length = _auto_detect_segment_length(collection)
        # Should skip point and use polygon
        assert segment_length == 5.0

    def test_list_of_geometries(self):
        """Test detection with a list of geometries."""
        geoms = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(20, 20), (21, 20), (21, 21), (20, 21)]),
            LineString([(30, 30), (30.5, 30), (30.5, 30.5)]),
        ]
        segment_length = _auto_detect_segment_length(geoms)
        # Should find minimum across all geometries (0.5)
        assert segment_length == 0.5

    def test_geodataframe(self):
        """Test detection with a GeoDataFrame."""
        geoms = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(20, 20), (21, 20), (21, 21), (20, 21)]),
            LineString([(30, 30), (30.2, 30), (30.2, 30.2)]),
        ]
        gdf = gpd.GeoDataFrame(geometry=geoms)
        segment_length = _auto_detect_segment_length(gdf)
        # Should find minimum (0.2)
        assert segment_length == pytest.approx(0.2, abs=1e-10)

    def test_geodataframe_samples_first_ten(self):
        """Test that GeoDataFrame only samples first 10 geometries."""
        # Create 15 geometries, last 5 have smaller pixel size
        geoms = [
            Polygon([(i, i), (i + 10, i), (i + 10, i + 10), (i, i + 10)])
            for i in range(10)
        ]
        geoms.extend(
            [
                Polygon([(i, i), (i + 0.1, i), (i + 0.1, i + 0.1), (i, i + 0.1)])
                for i in range(100, 105)
            ]
        )
        gdf = gpd.GeoDataFrame(geometry=geoms)
        segment_length = _auto_detect_segment_length(gdf)
        # Should only check first 10, so segment_length should be 10.0
        assert segment_length == 10.0

    def test_empty_polygon(self):
        """Test that empty geometries are handled gracefully."""
        empty = Polygon()
        with pytest.raises(ValueError, match="Could not auto-detect segment_length"):
            _auto_detect_segment_length(empty)

    def test_point_geometry(self):
        """Test that Point geometries raise an error."""
        point = Point(5, 5)
        with pytest.raises(ValueError, match="Could not auto-detect segment_length"):
            _auto_detect_segment_length(point)

    def test_collection_with_only_points(self):
        """Test GeometryCollection with only Points raises error."""
        collection = GeometryCollection([Point(0, 0), Point(1, 1), Point(2, 2)])
        with pytest.raises(ValueError, match="Could not auto-detect segment_length"):
            _auto_detect_segment_length(collection)

    def test_empty_list(self):
        """Test empty list raises error."""
        with pytest.raises(ValueError, match="Could not auto-detect segment_length"):
            _auto_detect_segment_length([])

    def test_list_with_only_empty_geometries(self):
        """Test list with only empty geometries raises error."""
        geoms = [Polygon(), LineString(), Polygon()]
        with pytest.raises(ValueError, match="Could not auto-detect segment_length"):
            _auto_detect_segment_length(geoms)

    def test_zero_length_segments_ignored(self):
        """Test that zero-length segments are ignored."""
        # Polygon with duplicate consecutive points (zero-length segment)
        polygon = Polygon([(0, 0), (0, 0), (5, 0), (5, 5), (0, 5)])
        segment_length = _auto_detect_segment_length(polygon)
        # Should skip zero-length and find 5.0
        assert segment_length == 5.0

    def test_large_geometry_sampling(self):
        """Test that large geometries are sampled efficiently."""
        # Create polygon with 1000 vertices, all 1-unit segments
        coords = [(i, 0) for i in range(500)]
        coords.extend([(500, i) for i in range(500)])
        coords.extend([(500 - i, 500) for i in range(500)])
        coords.extend([(0, 500 - i) for i in range(500)])
        polygon = Polygon(coords[:1000])  # Take first 1000
        segment_length = _auto_detect_segment_length(polygon)
        # Should still find 1.0 even with sampling
        assert segment_length == 1.0

    def test_fractional_segment_length(self):
        """Test detection with fractional pixel sizes."""
        polygon = Polygon(
            [
                (0, 0),
                (0.25, 0),
                (0.5, 0),
                (0.5, 0.25),
                (0.5, 0.5),
                (0.25, 0.5),
                (0, 0.5),
                (0, 0.25),
            ]
        )
        segment_length = _auto_detect_segment_length(polygon)
        assert segment_length == pytest.approx(0.25, abs=1e-10)

    def test_diagonal_segments(self):
        """Test with diagonal segments."""
        # Square rotated 45 degrees
        import math

        side = 1.0
        diag = side * math.sqrt(2)
        polygon = Polygon([(0, 0), (diag, 0), (diag, diag), (0, diag)])
        segment_length = _auto_detect_segment_length(polygon)
        # All segments should be same length
        assert segment_length == pytest.approx(diag, abs=1e-10)

    def test_mixed_precision(self):
        """Test with mixed precision coordinates."""
        polygon = Polygon(
            [
                (0.0, 0.0),
                (1.0000001, 0.0),
                (1.0000001, 0.5),
                (1.0, 0.5),
                (1.0, 1.0),
                (0.0, 1.0),
            ]
        )
        segment_length = _auto_detect_segment_length(polygon)
        # Should find very small segment from rounding
        assert segment_length < 0.5


class TestAutoDetectEdgeCases:
    """Test edge cases and robustness of auto-detection."""

    def test_sequence_samples_first_ten(self):
        """Test that sequences only sample first 10 items."""
        geoms = [
            Polygon([(i, i), (i + 10, i), (i + 10, i + 10), (i, i + 10)])
            for i in range(10)
        ]
        geoms.extend(
            [
                Polygon([(i, i), (i + 0.1, i), (i + 0.1, i + 0.1), (i, i + 0.1)])
                for i in range(100, 105)
            ]
        )
        segment_length = _auto_detect_segment_length(geoms)
        # Should only check first 10
        assert segment_length == 10.0

    def test_list_with_empty_geometries_mixed(self):
        """Test list with mix of empty and valid geometries."""
        geoms = [
            Polygon(),
            LineString(),
            Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
            Polygon(),
        ]
        segment_length = _auto_detect_segment_length(geoms)
        # Should skip empties and use the valid polygon
        assert segment_length == 3.0

    def test_multipolygon_with_holes(self):
        """Test MultiPolygon where one polygon has a hole with minimum segment."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        exterior2 = [(20, 20), (30, 20), (30, 30), (20, 30)]
        hole2 = [(22, 22), (22.1, 22), (22.1, 22.1), (22, 22.1)]
        poly2 = Polygon(exterior2, [hole2])
        multi = MultiPolygon([poly1, poly2])
        segment_length = _auto_detect_segment_length(multi)
        # Should find 0.1 from hole in poly2
        assert segment_length == pytest.approx(0.1, abs=1e-10)

    def test_very_small_segment_length(self):
        """Test detection with very small pixel sizes."""
        polygon = Polygon(
            [
                (0, 0),
                (0.001, 0),
                (0.002, 0),
                (0.002, 0.001),
                (0.002, 0.002),
                (0.001, 0.002),
                (0, 0.002),
                (0, 0.001),
            ]
        )
        segment_length = _auto_detect_segment_length(polygon)
        assert segment_length == pytest.approx(0.001, abs=1e-10)

    def test_very_large_segment_length(self):
        """Test detection with very large pixel sizes."""
        polygon = Polygon(
            [
                (0, 0),
                (1000, 0),
                (2000, 0),
                (2000, 1000),
                (2000, 2000),
                (1000, 2000),
                (0, 2000),
                (0, 1000),
            ]
        )
        segment_length = _auto_detect_segment_length(polygon)
        assert segment_length == 1000.0

    def test_irregular_polygon(self):
        """Test with irregular polygon (non-rectangular)."""
        # Triangle with varying segment lengths
        polygon = Polygon(
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (3, 0),  # 1-unit segments
                (2, 1),
                (1, 2),
                (0, 3),  # Longer diagonal segments
            ]
        )
        segment_length = _auto_detect_segment_length(polygon)
        # Should find 1.0 from bottom edge
        assert segment_length == 1.0

    def test_collection_with_nested_collections(self):
        """Test GeometryCollection containing other collections."""
        poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        line = LineString([(10, 10), (10.5, 10), (10.5, 10.5)])
        inner_collection = GeometryCollection([poly, line])
        # Note: Shapely flattens nested collections, but test anyway
        outer_collection = GeometryCollection([inner_collection])
        segment_length = _auto_detect_segment_length(outer_collection)
        assert segment_length == 0.5
