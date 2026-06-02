"""Tests for the main smoothify API."""

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

from smoothify import smoothify


class TestSmoothifyAPI:
    """Test suite for the main smoothify() function."""

    def test_smoothify_single_polygon(self):
        """Test smoothing a single polygon."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        smoothed = smoothify(geom=polygon, segment_length=1.0, smooth_iterations=3)

        assert isinstance(smoothed, Polygon)
        assert smoothed.is_valid
        assert len(smoothed.exterior.coords) > len(polygon.exterior.coords)

    def test_smoothify_single_linestring(self):
        """Test smoothing a single linestring."""
        line = LineString([(0, 0), (5, 5), (10, 0)])
        smoothed = smoothify(geom=line, segment_length=1.0, smooth_iterations=3)

        assert isinstance(smoothed, LineString)
        assert smoothed.is_valid
        # Endpoints preserved
        assert smoothed.coords[0] == line.coords[0]
        assert smoothed.coords[-1] == line.coords[-1]

    def test_smoothify_linear_ring(self):
        """Test smoothing a linear ring."""
        ring = LinearRing([(0, 0), (10, 0), (10, 10), (0, 10)])
        smoothed = smoothify(geom=ring, segment_length=1.0, smooth_iterations=3)

        # LinearRing should remain a LinearRing
        assert isinstance(smoothed, LinearRing)
        assert smoothed.is_valid
        # Should have more vertices after smoothing
        assert len(smoothed.coords) > len(ring.coords)

    def test_smoothify_multipolygon(self):
        """Test smoothing a multipolygon."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
        multi = MultiPolygon([poly1, poly2])

        smoothed = smoothify(geom=multi, segment_length=1.0, smooth_iterations=3)

        assert smoothed.is_valid
        # Should return MultiPolygon or GeometryCollection
        assert isinstance(smoothed, (MultiPolygon, GeometryCollection))

    def test_smoothify_multilinestring(self):
        """Test smoothing a multilinestring."""
        line1 = LineString([(0, 0), (5, 5)])
        line2 = LineString([(10, 10), (15, 15)])
        multi = MultiLineString([line1, line2])

        smoothed = smoothify(geom=multi, segment_length=1.0, smooth_iterations=3)

        assert smoothed.is_valid
        assert isinstance(smoothed, (MultiLineString, GeometryCollection))

    def test_smoothify_geometry_collection(self):
        """Test smoothing a geometry collection."""
        poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        line = LineString([(10, 10), (15, 15)])
        collection = GeometryCollection([poly, line])

        smoothed = smoothify(geom=collection, segment_length=1.0, smooth_iterations=3)

        assert smoothed.is_valid

    def test_smoothify_list_of_geometries(self):
        """Test smoothing a list of geometries."""
        poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        line = LineString([(10, 10), (15, 15)])
        geoms = [poly, line]

        smoothed = smoothify(geom=geoms, segment_length=1.0, smooth_iterations=3)

        assert smoothed.is_valid

    def test_mixed_geometry_list_preserves_all_types(self):
        """Test that list with polygon and linestring returns both geometries.

        Regression test for bug where linestrings were lost when smoothing
        a list containing both polygons and linestrings.
        """
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        linestring = LineString([(0, 0), (5, 5), (10, 0)])
        geoms = [polygon, linestring]

        smoothed = smoothify(
            geom=geoms, segment_length=1.0, smooth_iterations=3, merge_collection=False
        )

        assert smoothed.is_valid
        assert isinstance(smoothed, GeometryCollection)
        assert len(smoothed.geoms) == 2

        # Check that we have one polygon and one linestring
        geom_types = [type(g).__name__ for g in smoothed.geoms]
        assert "Polygon" in geom_types
        assert "LineString" in geom_types

    def test_mixed_geometry_collection_preserves_all_types(self):
        """Test that GeometryCollection with polygon and linestring returns both.

        Regression test for bug where linestrings were lost when smoothing
        a GeometryCollection containing both polygons and linestrings.
        """
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        linestring = LineString([(0, 0), (5, 5), (10, 0)])
        collection = GeometryCollection([polygon, linestring])

        smoothed = smoothify(
            geom=collection,
            segment_length=1.0,
            smooth_iterations=3,
            merge_collection=False,
        )

        assert smoothed.is_valid
        assert isinstance(smoothed, GeometryCollection)
        assert len(smoothed.geoms) == 2

        # Check that we have one polygon and one linestring
        geom_types = [type(g).__name__ for g in smoothed.geoms]
        assert "Polygon" in geom_types
        assert "LineString" in geom_types

    def test_smoothify_geodataframe(self):
        """Test smoothing a GeoDataFrame."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
        gdf = gpd.GeoDataFrame(
            {"name": ["poly1", "poly2"]}, geometry=[poly1, poly2], crs="EPSG:4326"
        )

        smoothed = smoothify(
            geom=gdf, segment_length=1.0, smooth_iterations=3, num_cores=1
        )

        assert isinstance(smoothed, gpd.GeoDataFrame)
        assert len(smoothed) == len(gdf)
        assert smoothed.crs == gdf.crs
        assert all(geom.is_valid for geom in smoothed.geometry)

    def test_smoothify_geodataframe_parallel(self):
        """Test smoothing a GeoDataFrame with parallel processing."""
        polys = [
            Polygon([(i, i), (i + 5, i), (i + 5, i + 5), (i, i + 5)])
            for i in range(0, 50, 10)
        ]
        gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326")

        smoothed = smoothify(
            geom=gdf, segment_length=1.0, smooth_iterations=3, num_cores=2
        )

        assert isinstance(smoothed, gpd.GeoDataFrame)
        assert len(smoothed) == len(gdf)
        assert all(geom.is_valid for geom in smoothed.geometry)

    def test_preserve_area_parameter(self):
        """Test preserve_area parameter."""
        polygon = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        original_area = polygon.area

        # With area preservation
        smoothed_preserved = smoothify(
            geom=polygon, segment_length=10.0, smooth_iterations=3, preserve_area=True
        )

        # Without area preservation
        smoothed_not_preserved = smoothify(
            geom=polygon, segment_length=10.0, smooth_iterations=3, preserve_area=False
        )

        # With preservation should be closer to original
        diff_preserved = abs(smoothed_preserved.area - original_area)
        diff_not_preserved = abs(smoothed_not_preserved.area - original_area)

        assert diff_preserved < diff_not_preserved

    def test_merge_collection_parameter(self):
        """Test merge_collection parameter."""
        # Two adjacent squares
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])
        collection = GeometryCollection([poly1, poly2])

        # With merging
        smoothed_merged = smoothify(
            geom=collection,
            segment_length=1.0,
            smooth_iterations=3,
            merge_collection=True,
        )

        # Without merging
        smoothed_not_merged = smoothify(
            geom=collection,
            segment_length=1.0,
            smooth_iterations=3,
            merge_collection=False,
        )

        assert smoothed_merged.is_valid
        assert smoothed_not_merged.is_valid

    def test_smooth_iterations_parameter(self):
        """Test that smooth_iterations affects smoothness."""
        polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

        smoothed_1 = smoothify(geom=polygon, segment_length=1.0, smooth_iterations=1)
        smoothed_3 = smoothify(geom=polygon, segment_length=1.0, smooth_iterations=3)
        smoothed_5 = smoothify(geom=polygon, segment_length=1.0, smooth_iterations=5)

        assert isinstance(smoothed_1, Polygon)
        assert isinstance(smoothed_3, Polygon)
        assert isinstance(smoothed_5, Polygon)
        # More iterations = more vertices
        assert (
            len(smoothed_1.exterior.coords)
            < len(smoothed_3.exterior.coords)
            < len(smoothed_5.exterior.coords)
        )

    def test_polygon_with_holes(self):
        """Test smoothing a polygon with holes."""
        exterior = [(0, 0), (20, 0), (20, 20), (0, 20)]
        hole = [(5, 5), (15, 5), (15, 15), (5, 15)]
        polygon = Polygon(exterior, [hole])

        smoothed = smoothify(geom=polygon, segment_length=1.0, smooth_iterations=3)

        assert isinstance(smoothed, Polygon)
        assert smoothed.is_valid
        # Should still have holes
        assert len(smoothed.interiors) > 0

    def test_invalid_input_type(self):
        """Test that invalid input raises error."""
        with pytest.raises(ValueError):
            smoothify(geom=Point(0, 0), segment_length=1.0)

        with pytest.raises(ValueError):
            smoothify(geom="not a geometry", segment_length=1.0)  # type: ignore

    def test_merge_field_with_geodataframe(self):
        """Test merge_field parameter with GeoDataFrame."""
        # Create adjacent polygons with different categories
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])  # Adjacent to poly1
        poly3 = Polygon([(0, 10), (10, 10), (10, 20), (0, 20)])  # Adjacent to poly1
        poly4 = Polygon([(30, 0), (40, 0), (40, 10), (30, 10)])  # Separate

        gdf = gpd.GeoDataFrame(
            {"category": ["A", "A", "B", "B"]},
            geometry=[poly1, poly2, poly3, poly4],
            crs="EPSG:4326",
        )

        # Smooth with merge_field - should dissolve by category
        smoothed = smoothify(
            geom=gdf,
            segment_length=1.0,
            smooth_iterations=3,
            merge_collection=True,
            merge_field="category",
            num_cores=1,
        )

        assert isinstance(smoothed, gpd.GeoDataFrame)
        assert all(geom.is_valid for geom in smoothed.geometry)
        # After dissolving by category and exploding, expect fewer/equal geometries
        assert len(smoothed) <= len(gdf)

    def test_merge_field_without_merge_collection_raises_error(self):
        """Test that merge_field requires merge_collection=True."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
        gdf = gpd.GeoDataFrame(
            {"category": ["A", "B"]}, geometry=[poly1, poly2], crs="EPSG:4326"
        )

        with pytest.raises(
            ValueError,
            match="merge_field is only supported when merge_collection is True",
        ):
            smoothify(
                geom=gdf,
                segment_length=1.0,
                merge_collection=False,
                merge_field="category",
                num_cores=1,
            )

    def test_merge_field_with_non_geodataframe_raises_error(self):
        """Test that merge_field only works with GeoDataFrame."""
        poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])

        with pytest.raises(
            ValueError, match="merge_field is only supported for GeoDataFrames"
        ):
            smoothify(
                geom=poly,
                segment_length=1.0,
                merge_field="category",  # type: ignore
            )

    def test_merge_field_with_invalid_column_raises_error(self):
        """Test that merge_field must be a valid column name."""
        poly1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        poly2 = Polygon([(10, 10), (15, 10), (15, 15), (10, 15)])
        gdf = gpd.GeoDataFrame(
            {"category": ["A", "B"]}, geometry=[poly1, poly2], crs="EPSG:4326"
        )

        with pytest.raises(
            ValueError, match="merge_field nonexistent not found in GeoDataFrame"
        ):
            smoothify(
                geom=gdf,
                segment_length=1.0,
                merge_collection=True,
                merge_field="nonexistent",
                num_cores=1,
            )

    def test_merge_field_none_with_geodataframe(self):
        """Test that merge_field=None works correctly (dissolves all geometries)."""
        poly1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        poly2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])  # Adjacent to poly1

        gdf = gpd.GeoDataFrame(
            {"category": ["A", "B"]}, geometry=[poly1, poly2], crs="EPSG:4326"
        )

        # Smooth with merge_collection=True but merge_field=None
        # This should dissolve all geometries together
        smoothed = smoothify(
            geom=gdf,
            segment_length=1.0,
            smooth_iterations=3,
            merge_collection=True,
            merge_field=None,
            num_cores=1,
        )

        assert isinstance(smoothed, gpd.GeoDataFrame)
        assert all(geom.is_valid for geom in smoothed.geometry)
        # After dissolving all and exploding, we should have 1 or more geometries
        assert len(smoothed) >= 1
