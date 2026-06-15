"""Regression test for a *valid* polygon whose smoothing intermediate was not.

Feature 5896 from a real raster dataset: a thin, ~staircase polygon only a few
segment-lengths across. It is perfectly valid on input, but when smoothed it
used to raise

    AssertionError: Resulting geometry must be Polygon or LineString.
    Got <class 'shapely.geometry.collection.GeometryCollection'>.

Cause: one rotated start-point variant self-intersects after simplify + Chaikin
where it rounds a neck only ~segment_length wide so the two sides cross. The
make_valid() repair of that variant returns a Polygon *plus* a zero-area line
filament at the pinch; unary_union keeps the 1-D debris, so the variant union
becomes a GeometryCollection that the rest of the pipeline could not handle.

The fix strips non-polygonal (zero-area) debris from each repaired variant
before the union, so the smoothed result is a clean, valid Polygon.
"""

import warnings

import pytest
from shapely import wkt
from shapely.geometry import Polygon

from smoothify import smoothify
from smoothify.smoothify_core import _polygonal_only, _smoothify_geometry

# Real feature 5896 (segment_length 10). Valid on input; its narrow tail is the
# spot where a smoothed variant used to self-intersect.
THIN_STAIRCASE_WKT = (
    "POLYGON ((-1567435.811070582 -3242808.72971921, "
    "-1567434.5427619545 -3242818.60906014, "
    "-1567444.252453771 -3242819.8555931244, "
    "-1567441.7129643494 -3242839.636522844, "
    "-1567432.003288262 -3242838.3899918776, "
    "-1567429.4652429952 -3242858.1597970645, "
    "-1567419.7651383937 -3242856.9145027213, "
    "-1567418.495409537 -3242866.8049685746, "
    "-1567399.0856563346 -3242864.313178158, "
    "-1567396.5476584304 -3242884.082987632, "
    "-1567386.847574424 -3242882.837719445, "
    "-1567394.4629497067 -3242823.517156064, "
    "-1567404.1630808434 -3242824.7624302953, "
    "-1567406.7010791986 -3242804.9926173063, "
    "-1567435.811070582 -3242808.72971921))"
)


def test_thin_staircase_smooths_to_valid_polygon():
    """The thin valid polygon smooths without raising and stays a valid Polygon."""
    poly = wkt.loads(THIN_STAIRCASE_WKT)
    assert poly.is_valid, "input fixture must be a valid polygon"

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # input is valid, so no warning is expected
        result = smoothify(poly, segment_length=10)

    assert isinstance(result, Polygon)
    assert result.is_valid
    assert not result.is_empty
    # Area is preserved (default preserve_area) to well within tolerance.
    assert result.area == pytest.approx(poly.area, rel=1e-3)


def test_core_intermediate_is_polygonal():
    """The core pipeline never yields non-polygonal debris for this geometry."""
    poly = wkt.loads(THIN_STAIRCASE_WKT)
    result = _smoothify_geometry(poly, segment_length=10, smooth_iterations=3)
    assert isinstance(result, Polygon)
    assert result.is_valid


def test_polygonal_only_drops_zero_area_debris():
    """The helper keeps area and discards stray lines/points from a collection."""
    from shapely.geometry import GeometryCollection, LineString

    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    debris = GeometryCollection([square, LineString([(0, 0), (5, -5)])])

    kept = _polygonal_only(debris)
    assert kept.geom_type in ("Polygon", "MultiPolygon")
    assert kept.area == square.area

    # Plain polygonal input is returned unchanged.
    assert _polygonal_only(square) is square
