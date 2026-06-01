"""Tests for how smoothify handles an *invalid* (self-intersecting) polygon.

Smoothify does not repair invalid input. Smoothing relies on
segmentize/simplify/union behaving predictably, which they do not for invalid
geometries (segmentize can even turn an invalid Polygon into a MultiPolygon).
So the contract is: **warn and return the geometry unchanged**, and tell the
user to run make_valid() first.

This must be consistent no matter how the geometry is passed in:

* as a single geometry,
* inside a list / GeometryCollection,
* inside a GeoDataFrame,

and regardless of merge_collection. These tests pin that behaviour down.
"""

import warnings

import geopandas as gpd
import pytest
from shapely import wkt
from shapely.geometry import GeometryCollection, Polygon

from smoothify import smoothify

# A deterministic self-intersecting polygon (is_valid == False). Its duplicated
# "1 -7" vertex and looping boundary make shapely report it invalid, and
# segmentize() turns it into a MultiPolygon -- the case that used to raise a
# cryptic "got MultiPolygon" error deep in the pipeline.
INVALID_WKT = (
    "POLYGON ((9 1, 9 0, 11 4, 7 5, 6 6, 4 6, 5 5, 4 5, 1 8, 0 10, 0 9, -2 9, "
    "-2 7, -3 5, -5 5, -6 4, -5 4, -9 4, -11 4, -10 2, -7 0, -8 -1, -5 -2, "
    "-5 -3, -6 -5, -6 -7, -6 -9, -5 -9, -3 -8, -1 -7, 1 -7, 0 -8, 1 -7, 4 -9, "
    "5 -9, 8 -8, 6 -4, 5 -3, 6 -1, 6 0, 9 1))"
)

# The classic "bowtie": two opposite-wound lobes meeting at a point, area 0.
BOWTIE_WKT = "POLYGON ((0 0, 10 10, 10 0, 0 10, 0 0))"


@pytest.fixture(params=[INVALID_WKT, BOWTIE_WKT], ids=["complex", "bowtie"])
def invalid_polygon(request) -> Polygon:
    poly = wkt.loads(request.param)
    assert not poly.is_valid, "fixture polygon is expected to be invalid"
    return poly


def _assert_warns_invalid(func):
    """Run func(), assert it warns about an invalid/self-intersecting geom,
    and return the result."""
    with pytest.warns(UserWarning, match="invalid|make_valid"):
        return func()


class TestInvalidPolygonSingle:
    """Invalid polygon passed directly as a single geometry."""

    def test_returns_unchanged_with_warning(self, invalid_polygon):
        result = _assert_warns_invalid(
            lambda: smoothify(invalid_polygon, segment_length=1.0, smooth_iterations=3)
        )
        # Returned as-is: same object, unchanged, never silently emptied.
        assert result is invalid_polygon
        assert not result.is_empty


class TestInvalidPolygonList:
    """Invalid polygon passed inside a list of geometries."""

    @pytest.mark.parametrize("merge_collection", [True, False])
    def test_returns_unchanged_with_warning(self, invalid_polygon, merge_collection):
        result = _assert_warns_invalid(
            lambda: smoothify(
                [invalid_polygon],
                segment_length=1.0,
                smooth_iterations=3,
                num_cores=1,
                merge_collection=merge_collection,
            )
        )
        assert isinstance(result, GeometryCollection)
        assert len(result.geoms) == 1
        out = result.geoms[0]
        # The single invalid geometry is passed through unchanged.
        assert out.equals(invalid_polygon)
        assert not out.is_empty


class TestInvalidPolygonGeoDataFrame:
    """Invalid polygon passed inside a GeoDataFrame."""

    @pytest.mark.parametrize("merge_collection", [True, False])
    def test_returns_unchanged_with_warning(self, invalid_polygon, merge_collection):
        gdf = gpd.GeoDataFrame(geometry=[invalid_polygon])
        result = _assert_warns_invalid(
            lambda: smoothify(
                gdf,
                segment_length=1.0,
                smooth_iterations=3,
                num_cores=1,
                merge_collection=merge_collection,
            )
        )
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
        out = result.geometry.iloc[0]
        assert out.equals(invalid_polygon)
        assert not out.is_empty


class TestMixedValidAndInvalid:
    """A collection holding both valid and invalid polygons: valid ones are
    smoothed, invalid ones pass through unchanged, and every input is
    represented in the output."""

    def _square(self, x):
        return Polygon([(x, 0), (x + 10, 0), (x + 10, 10), (x, 10)])

    def test_list_smooths_valid_passes_invalid(self):
        invalid = wkt.loads(INVALID_WKT)
        valid_a = self._square(0)
        valid_b = self._square(100)  # far away so merge won't fuse them
        result = _assert_warns_invalid(
            lambda: smoothify(
                [valid_a, invalid, valid_b],
                segment_length=1.0,
                smooth_iterations=3,
                num_cores=1,
                merge_collection=False,
            )
        )
        geoms = list(result.geoms)
        assert len(geoms) == 3
        # exactly one output equals the invalid input (passed through)
        passed_through = [g for g in geoms if g.equals(invalid)]
        assert len(passed_through) == 1
        # the other two are smoothed (changed) valid polygons
        smoothed = [g for g in geoms if not g.equals(invalid)]
        assert len(smoothed) == 2
        assert all(not g.equals(valid_a) and not g.equals(valid_b) for g in smoothed)

    def test_gdf_parallel_warns_from_main_process(self):
        # num_cores > 1 exercises the worker pool; the warning must still reach
        # the caller because invalid geoms are screened in the main process.
        invalid = wkt.loads(INVALID_WKT)
        gdf = gpd.GeoDataFrame(
            geometry=[self._square(0), invalid, self._square(100)]
        )
        result = _assert_warns_invalid(
            lambda: smoothify(
                gdf,
                segment_length=1.0,
                smooth_iterations=3,
                num_cores=2,
                merge_collection=False,
            )
        )
        assert len(result) == 3
        assert sum(g.equals(invalid) for g in result.geometry) == 1


class TestValidPolygonStillSmooths:
    """Guard must not affect valid input: a valid polygon is still smoothed
    (different from the input, and area is roughly preserved)."""

    def test_valid_polygon_is_smoothed_no_warning(self):
        square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning would fail the test
            result = smoothify(square, segment_length=1.0, smooth_iterations=3)
        assert isinstance(result, Polygon)
        assert result.is_valid
        assert not result.is_empty
        assert not result.equals(square)  # actually changed
        assert result.area == pytest.approx(square.area, rel=0.05)
