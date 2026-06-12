"""Tests for congruent-geometry deduplication in batch smoothing.

Raster-derived datasets contain many translated copies of the same shape;
smoothify computes each distinct shape once and translates the result back.
That shortcut must be invisible: copies get exactly-translated results,
non-copies are unaffected, and geometries it cannot handle safely (3D) are
smoothed directly.
"""

import numpy as np
import geopandas as gpd
import shapely
from shapely.affinity import translate
from shapely.geometry import LineString, Polygon

from smoothify import smoothify


def pixel_blob() -> Polygon:
    """A small multi-pixel polygon (pixel size 1)."""
    return Polygon([(0, 0), (3, 0), (3, 1), (2, 1), (2, 2), (0, 2)])


class TestCongruenceDedup:
    def test_translated_copies_get_translated_results(self):
        base = pixel_blob()
        offsets = [(0, 0), (100, 0), (-50, 75), (1234.5, -987.25)]
        gdf = gpd.GeoDataFrame(geometry=[translate(base, dx, dy) for dx, dy in offsets])

        # merge_collection=False keeps output rows aligned 1:1 with input
        # (the dissolve+explode path does not preserve row order)
        out = smoothify(gdf, segment_length=1.0, num_cores=1, merge_collection=False)

        ref = out.geometry.iloc[0]
        for (dx, dy), got in zip(offsets[1:], out.geometry.iloc[1:], strict=True):
            expected = translate(ref, dx, dy)
            assert got.equals_exact(expected, tolerance=1e-9)

    def test_distinct_shapes_are_not_merged(self):
        a = pixel_blob()
        # same vertex count and bounding box, different shape
        b = Polygon([(0, 0), (3, 0), (3, 2), (1, 2), (1, 1), (0, 1)])
        gdf = gpd.GeoDataFrame(geometry=[a, translate(b, 50, 0)])

        out = smoothify(gdf, segment_length=1.0, num_cores=1, merge_collection=False)

        back = translate(out.geometry.iloc[1], -50, 0)
        assert not out.geometry.iloc[0].equals_exact(back, tolerance=1e-6)

    def test_3d_geometries_keep_their_z(self):
        """Regression: XY-congruent 3D lines must not swap or lose Z values.

        The dedup key hashes XY only and the translate-back drops Z, so 3D
        geometries are excluded from deduplication entirely.
        """
        a = LineString([(0, 0, 0), (10, 0, 5), (20, 10, 10)])
        b = LineString([(100, 0, 99), (110, 0, 42), (120, 10, 7)])
        gdf = gpd.GeoDataFrame(geometry=[a, b])

        out = smoothify(gdf, segment_length=2.0, num_cores=1)

        za = shapely.get_coordinates(out.geometry.iloc[0], include_z=True)[:, 2]
        zb = shapely.get_coordinates(out.geometry.iloc[1], include_z=True)[:, 2]
        assert not np.isnan(za).any()
        assert not np.isnan(zb).any()
        # each keeps its own Z range (endpoints are preserved by smoothing)
        assert za.min() == 0 and za.max() == 10
        assert zb.min() == 7 and zb.max() == 99
