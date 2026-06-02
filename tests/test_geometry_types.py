"""Tests for correct handling of all geometry types and edge cases."""

import geopandas as gpd
import pytest
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from smoothify import smoothify


class TestGeometryTypeHandling:
    """Test that all geometry types are handled correctly."""

    def test_empty_polygon(self):
        """Test handling of empty polygon."""
        empty_poly = Polygon()
        result = smoothify(empty_poly, segment_length=1.0, smooth_iterations=3)

        # Empty geometries should be returned as-is
        assert result.is_empty
        assert result.is_valid
        assert isinstance(result, Polygon)

    def test_very_simple_polygon(self):
        """Test polygon with only 3 vertices (minimum)."""
        triangle = Polygon([(0, 0), (10, 0), (5, 10)])
        smoothed = smoothify(triangle, segment_length=1.0, smooth_iterations=3)

        assert isinstance(smoothed, Polygon)
        assert smoothed.is_valid
        # Should have more vertices after smoothing
        assert len(smoothed.exterior.coords) > 3

    def test_very_simple_linestring(self):
        """Test linestring with only 2 vertices (minimum)."""
        line = LineString([(0, 0), (10, 10)])
        smoothed = smoothify(line, segment_length=1.0, smooth_iterations=3)

        assert isinstance(smoothed, LineString)
        assert smoothed.is_valid
        # Endpoints should be preserved
        assert smoothed.coords[0] == line.coords[0]
        assert smoothed.coords[-1] == line.coords[-1]

    def test_multipolygon_many_polygons(self):
        """Test MultiPolygon with many small polygons."""
        polys = [
            Polygon(
                [
                    (i * 20, j * 20),
                    (i * 20 + 5, j * 20),
                    (i * 20 + 5, j * 20 + 5),
                    (i * 20, j * 20 + 5),
                ]
            )
            for i in range(5)
            for j in range(5)
        ]
        multi = MultiPolygon(polys)

        smoothed = smoothify(
            multi, segment_length=1.0, smooth_iterations=3, merge_multipolygons=False
        )

        assert smoothed.is_valid

    def test_geometry_collection_mixed_types(self):
        """Test GeometryCollection with mixed Polygon and LineString."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        line = LineString([(20, 0), (30, 10)])
        collection = GeometryCollection([poly, line])

        smoothed = smoothify(collection, segment_length=1.0, smooth_iterations=3)

        assert smoothed.is_valid
        assert isinstance(smoothed, GeometryCollection)

    def test_unsupported_point(self):
        """Test that Point raises appropriate error."""
        point = Point(5, 5)

        with pytest.raises(ValueError):
            smoothify(point, segment_length=1.0)

    def test_unsupported_multipoint(self):
        """Test that MultiPoint raises appropriate error."""
        multipoint = MultiPoint([(0, 0), (5, 5), (10, 10)])

        with pytest.raises(ValueError):
            smoothify(multipoint, segment_length=1.0)

    def test_polygon_with_multiple_holes(self):
        """Test polygon with multiple interior holes."""
        exterior = [(0, 0), (50, 0), (50, 50), (0, 50)]
        hole1 = [(10, 10), (20, 10), (20, 20), (10, 20)]
        hole2 = [(30, 30), (40, 30), (40, 40), (30, 40)]
        polygon = Polygon(exterior, [hole1, hole2])

        smoothed = smoothify(polygon, segment_length=1.0, smooth_iterations=3)

        assert isinstance(smoothed, Polygon)
        assert smoothed.is_valid
        # Should still have holes (though exact count may vary due to smoothing)
        assert len(smoothed.interiors) > 0

    def test_complex_multipolygon(self):
        """Test MultiPolygon with varying polygon sizes."""
        small = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        medium = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
        large = Polygon([(50, 50), (100, 50), (100, 100), (50, 100)])
        multi = MultiPolygon([small, medium, large])

        smoothed = smoothify(
            multi, segment_length=1.0, smooth_iterations=3, merge_multipolygons=False
        )

        assert smoothed.is_valid


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_linestring(self):
        """Test linestring with many vertices."""
        coords = [(i, i % 10) for i in range(100)]
        line = LineString(coords)

        smoothed = smoothify(line, segment_length=1.0, smooth_iterations=3)

        assert isinstance(smoothed, LineString)
        assert smoothed.is_valid
        # Endpoints preserved
        assert smoothed.coords[0] == line.coords[0]
        assert smoothed.coords[-1] == line.coords[-1]

    def test_polygon_with_many_vertices(self):
        """Test polygon with many vertices on exterior."""
        import math

        # Create circular polygon with many points
        n_points = 100
        coords = [
            (
                10 * math.cos(2 * math.pi * i / n_points),
                10 * math.sin(2 * math.pi * i / n_points),
            )
            for i in range(n_points)
        ]
        polygon = Polygon(coords)

        smoothed = smoothify(polygon, segment_length=0.5, smooth_iterations=2)

        assert isinstance(smoothed, Polygon)
        assert smoothed.is_valid

    def test_self_intersecting_after_smoothing(self):
        """Test that smoothing doesn't create invalid geometries."""
        # Create a polygon that might self-intersect if not careful
        polygon = Polygon([(0, 0), (10, 0), (5, 5), (10, 10), (0, 10)])

        smoothed = smoothify(polygon, segment_length=1.0, smooth_iterations=3)

        assert smoothed.is_valid  # Should not create self-intersections

    def test_adjacent_multipolygon_merge(self):
        """Test merging of adjacent polygons in MultiPolygon."""
        # Two adjacent squares
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])
        multi = MultiPolygon([poly1, poly2])

        # With merging
        smoothed_merged = smoothify(
            multi, segment_length=1.0, smooth_iterations=3, merge_multipolygons=True
        )

        assert smoothed_merged.is_valid

    def test_non_adjacent_multipolygon(self):
        """Test MultiPolygon with non-adjacent polygons."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(100, 100), (105, 100), (105, 105), (100, 105)])
        multi = MultiPolygon([poly1, poly2])

        smoothed = smoothify(
            multi, segment_length=1.0, smooth_iterations=3, merge_multipolygons=True
        )

        assert smoothed.is_valid


class TestGeoDataFrameTypes:
    """Test GeoDataFrame with different geometry types."""

    def test_geodataframe_all_polygons(self):
        """Test GeoDataFrame with only Polygon geometries."""
        polys = [
            Polygon([(i, i), (i + 5, i), (i + 5, i + 5), (i, i + 5)]) for i in range(10)
        ]
        gdf = gpd.GeoDataFrame(geometry=polys)

        smoothed = smoothify(
            gdf,
            segment_length=1.0,
            smooth_iterations=3,
            num_cores=1,
            merge_collection=False,
        )

        assert isinstance(smoothed, gpd.GeoDataFrame)
        assert len(smoothed) == len(gdf)
        assert all(geom.is_valid for geom in smoothed.geometry)

    def test_geodataframe_all_linestrings(self):
        """Test GeoDataFrame with only LineString geometries."""
        lines = [LineString([(i, i), (i + 5, i + 5)]) for i in range(10)]
        gdf = gpd.GeoDataFrame(geometry=lines)

        smoothed = smoothify(
            gdf,
            segment_length=1.0,
            smooth_iterations=3,
            num_cores=1,
            merge_collection=False,
        )

        assert isinstance(smoothed, gpd.GeoDataFrame)
        assert len(smoothed) == len(gdf)
        assert all(isinstance(geom, LineString) for geom in smoothed.geometry)

    def test_geodataframe_mixed_polygon_multipolygon(self):
        """Test GeoDataFrame with mix of Polygon and MultiPolygon."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
        multi = MultiPolygon([poly1, poly2])
        poly3 = Polygon([(20, 20), (25, 20), (25, 25), (20, 25)])

        gdf = gpd.GeoDataFrame(geometry=[poly1, multi, poly3])

        smoothed = smoothify(
            gdf,
            segment_length=1.0,
            smooth_iterations=3,
            num_cores=1,
            merge_collection=False,
        )

        assert isinstance(smoothed, gpd.GeoDataFrame)
        assert len(smoothed) == len(gdf)
        assert all(geom.is_valid for geom in smoothed.geometry)

    def test_geodataframe_preserves_attributes(self):
        """Test that GeoDataFrame attributes are preserved."""
        polys = [
            Polygon([(i, i), (i + 5, i), (i + 5, i + 5), (i, i + 5)]) for i in range(5)
        ]
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2, 3, 4, 5], "name": ["A", "B", "C", "D", "E"]}, geometry=polys
        )

        smoothed = smoothify(
            gdf,
            segment_length=1.0,
            smooth_iterations=3,
            num_cores=1,
            merge_collection=False,
        )

        assert "id" in smoothed.columns
        assert "name" in smoothed.columns
        assert list(smoothed["id"]) == [1, 2, 3, 4, 5]
        assert list(smoothed["name"]) == ["A", "B", "C", "D", "E"]

    def test_geodataframe_linestrings_not_converted_to_polygons(self):
        """LineStrings are not converted to Polygons when merge_collection=True.

        Regression test for bug where buffering operation in merge_collection
        was being applied to all geometries including LineStrings, converting
        them to Polygons.
        """
        # Create a GeoDataFrame with only LineStrings
        lines = [
            LineString([(i, i), (i + 5, i + 2), (i + 10, i + 5)]) for i in range(10)
        ]
        gdf = gpd.GeoDataFrame(geometry=lines)

        # Smooth with merge_collection=True (default)
        smoothed = smoothify(
            gdf,
            segment_length=1.0,
            smooth_iterations=3,
            num_cores=1,
            merge_collection=True,
        )

        # All geometries should still be LineStrings, not Polygons
        assert isinstance(smoothed, gpd.GeoDataFrame)
        assert len(smoothed) == len(gdf)
        assert all(isinstance(geom, LineString) for geom in smoothed.geometry)
        assert all(geom.is_valid for geom in smoothed.geometry)

    def test_geodataframe_mixed_types_merge_collection(self):
        """GeoDataFrame with mixed Polygon/LineString types and merge_collection=True.

        Ensures that merge_collection only applies buffering to Polygons,
        not to LineStrings or other geometry types.
        """
        # Create mixed geometry types
        polys = [
            Polygon([(i, i), (i + 5, i), (i + 5, i + 5), (i, i + 5)]) for i in range(5)
        ]
        lines = [LineString([(i + 20, i), (i + 25, i + 5)]) for i in range(5)]

        gdf = gpd.GeoDataFrame(geometry=polys + lines)

        # Smooth with merge_collection=True
        smoothed = smoothify(
            gdf,
            segment_length=1.0,
            smooth_iterations=3,
            num_cores=1,
            merge_collection=True,
        )

        assert isinstance(smoothed, gpd.GeoDataFrame)

        # Check that geometry types are preserved
        smoothed_types = smoothed.geometry.geom_type.value_counts()
        assert "Polygon" in smoothed_types or "MultiPolygon" in smoothed_types
        assert "LineString" in smoothed_types

        # Verify no LineStrings were converted to Polygons
        for i, geom in enumerate(smoothed.geometry):
            if i >= 5:  # These were originally LineStrings
                assert isinstance(geom, LineString), (
                    f"LineString at index {i} was converted to {type(geom)}"
                )

    def test_geodataframe_mixed_all_geometry_types_merge_collection(self):
        """Test GeoDataFrame with all supported mixed types with merge_collection=True.

        Tests Polygon, MultiPolygon, LineString, and MultiLineString together
        to ensure merge_collection correctly handles all combinations.
        Note: merge_collection can change row counts due to dissolve/explode operations.
        """
        # Create diverse mixed geometry types
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
        multi_poly = MultiPolygon(
            [
                Polygon([(20, 20), (25, 20), (25, 25), (20, 25)]),
                Polygon([(30, 30), (35, 30), (35, 35), (30, 35)]),
            ]
        )
        line1 = LineString([(40, 40), (45, 45), (50, 40)])
        line2 = LineString([(60, 60), (65, 65)])
        multi_line = MultiLineString(
            [LineString([(70, 70), (75, 75)]), LineString([(80, 80), (85, 85)])]
        )

        gdf = gpd.GeoDataFrame(
            geometry=[poly1, poly2, multi_poly, line1, line2, multi_line]
        )

        # Count original line-based geometries
        original_linestring_count = sum(
            1 for gt in gdf.geometry.geom_type if "LineString" in gt
        )

        # Smooth with merge_collection=True
        smoothed = smoothify(
            gdf,
            segment_length=1.0,
            smooth_iterations=3,
            num_cores=1,
            merge_collection=True,
        )

        assert isinstance(smoothed, gpd.GeoDataFrame)
        assert all(geom.is_valid for geom in smoothed.geometry)

        # Verify all (Multi)LineString geometries were preserved (not made Polygons)
        smoothed_types = smoothed.geometry.geom_type.value_counts()

        # Should have both (Multi)Polygon types and (Multi)LineString types
        has_polygon = "Polygon" in smoothed_types or "MultiPolygon" in smoothed_types
        has_linestring = (
            "LineString" in smoothed_types or "MultiLineString" in smoothed_types
        )

        assert has_polygon, "Expected to find Polygon or MultiPolygon geometries"
        assert has_linestring, (
            "Expected to find LineString or MultiLineString geometries"
        )

        # Count line-based geometries in result - should match original count
        # (LineStrings and MultiLineStrings should not be merged or exploded)
        smoothed_linestring_count = sum(
            1 for gt in smoothed.geometry.geom_type if "LineString" in gt
        )
        assert smoothed_linestring_count == original_linestring_count, (
            f"LineString count changed from {original_linestring_count} "
            f"to {smoothed_linestring_count}"
        )


class TestParameterRobustness:
    """Test robustness to different parameter values."""

    def test_zero_smooth_iterations(self):
        """Test with zero smoothing iterations returns original input unchanged."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        smoothed = smoothify(polygon, segment_length=1.0, smooth_iterations=0)

        assert smoothed is polygon

    def test_high_smooth_iterations(self):
        """Test with very high smoothing iterations."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        smoothed = smoothify(polygon, segment_length=1.0, smooth_iterations=10)

        assert isinstance(smoothed, Polygon)
        assert smoothed.is_valid
        # Should be very smooth with many vertices

    def test_very_small_segment_length(self):
        """Test with very small pixel size."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        smoothed = smoothify(polygon, segment_length=0.001, smooth_iterations=3)

        assert smoothed.is_valid

    def test_very_large_segment_length(self):
        """Test with pixel size larger than geometry."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        smoothed = smoothify(polygon, segment_length=100.0, smooth_iterations=3)

        assert smoothed.is_valid
