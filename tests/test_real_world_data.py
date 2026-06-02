"""Tests using real-world water body data from NAIP OWM."""

from pathlib import Path

import geopandas as gpd
import pytest

from smoothify import smoothify

TEST_DATA = Path(__file__).parent / "test_data" / "naip_owm_water_bodies.geojson"


@pytest.fixture
def water_gdf():
    return gpd.read_file(TEST_DATA)


@pytest.mark.slow
class TestRealWorldWaterBodies:
    """Smoke tests that smoothify works on real-world data with various settings.

    Marked ``slow`` (end-to-end smoothing over real water-body geometries, ~1 min
    total). Deselected by the pre-push hook for speed; CI still runs them."""

    def test_default_settings(self, water_gdf):
        result = smoothify(geom=water_gdf, smooth_iterations=3, num_cores=1)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(water_gdf)
        assert all(geom.is_valid for geom in result.geometry)

    def test_no_area_preservation(self, water_gdf):
        result = smoothify(geom=water_gdf, smooth_iterations=3, preserve_area=False)
        assert isinstance(result, gpd.GeoDataFrame)
        assert all(geom.is_valid for geom in result.geometry)

    def test_high_smooth_iterations(self, water_gdf):
        result = smoothify(geom=water_gdf, smooth_iterations=6)
        assert isinstance(result, gpd.GeoDataFrame)
        assert all(geom.is_valid for geom in result.geometry)

    def test_low_smooth_iterations(self, water_gdf):
        result = smoothify(geom=water_gdf, smooth_iterations=1)
        assert isinstance(result, gpd.GeoDataFrame)
        assert all(geom.is_valid for geom in result.geometry)

    def test_zero_smooth_iterations(self, water_gdf):
        result = smoothify(geom=water_gdf, smooth_iterations=0)
        assert result is water_gdf

    def test_explicit_segment_length(self, water_gdf):
        result = smoothify(geom=water_gdf, segment_length=5.0, smooth_iterations=3)
        assert isinstance(result, gpd.GeoDataFrame)
        assert all(geom.is_valid for geom in result.geometry)

    def test_large_segment_length(self, water_gdf):
        result = smoothify(geom=water_gdf, segment_length=20.0, smooth_iterations=3)
        assert isinstance(result, gpd.GeoDataFrame)
        assert all(geom.is_valid for geom in result.geometry)

    def test_strict_area_tolerance(self, water_gdf):
        result = smoothify(geom=water_gdf, smooth_iterations=3, area_tolerance=0.001)
        assert isinstance(result, gpd.GeoDataFrame)
        assert all(geom.is_valid for geom in result.geometry)

    def test_relaxed_area_tolerance(self, water_gdf):
        result = smoothify(geom=water_gdf, smooth_iterations=3, area_tolerance=1.0)
        assert isinstance(result, gpd.GeoDataFrame)
        assert all(geom.is_valid for geom in result.geometry)

    def test_merge_collection(self, water_gdf):
        result = smoothify(geom=water_gdf, smooth_iterations=3, merge_collection=True)
        assert isinstance(result, gpd.GeoDataFrame)
        assert all(geom.is_valid for geom in result.geometry)

    def test_no_merge_collection(self, water_gdf):
        result = smoothify(geom=water_gdf, smooth_iterations=3, merge_collection=False)
        assert isinstance(result, gpd.GeoDataFrame)
        assert all(geom.is_valid for geom in result.geometry)

    def test_parallel_processing(self, water_gdf):
        result = smoothify(geom=water_gdf, smooth_iterations=3, num_cores=2)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == len(water_gdf)
        assert all(geom.is_valid for geom in result.geometry)

    def test_merge_field(self, water_gdf):
        result = smoothify(
            geom=water_gdf,
            smooth_iterations=3,
            merge_collection=True,
            merge_field="class",
        )
        assert isinstance(result, gpd.GeoDataFrame)
        assert all(geom.is_valid for geom in result.geometry)
