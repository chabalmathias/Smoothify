"""Regression tests: smoothing a convex shape must not fold in concave cusps.

The polygon smoothing pipeline merges several start-point-rotated variants
with a union. With too few variants (e.g. 3 instead of 4), the union boundary
can fold into a concave cusp on small features where two variants' boundaries
cross without a third covering the junction — a convex 1x2-pixel rectangle
came out heart-shaped.
"""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.affinity import translate
from shapely.geometry import Polygon

from smoothify import smoothify

CUSP_REGRESSION_DATA = (
    Path(__file__).parent / "test_data" / "convex_pixel_rectangle.gpkg"
)


def concavity(geom: Polygon) -> float:
    """Fraction of the convex hull's area missing from the geometry."""
    return (geom.convex_hull.area - geom.area) / geom.convex_hull.area


class TestConvexInputsStayConvex:
    """Smoothed convex rectangles must stay (nearly) convex."""

    @pytest.mark.parametrize("width,height", [(9, 9), (9, 18), (9, 27), (18, 27)])
    def test_pixel_rectangle_has_no_cusp(self, width: float, height: float):
        """Small pixel-scale rectangles must smooth into convex blobs.

        The 3-variant bug produced concavity ~0.012 on the 9x18 case; correct
        output is convex to within numerical noise.
        """
        rect = Polygon([(0, 0), (width, 0), (width, height), (0, height)])

        smoothed = smoothify(rect, segment_length=9.0)

        assert concavity(smoothed) < 0.002

    def test_pixel_rectangle_at_utm_coordinates(self):
        """Same check away from the origin (real-world UTM coordinates)."""
        rect = translate(
            Polygon([(0, 0), (9, 0), (9, 18), (0, 18)]), 296838.0, 5791540.0
        )

        smoothed = smoothify(rect, segment_length=9.0)

        assert concavity(smoothed) < 0.002

    def test_reported_geometry_through_geodataframe_path(self):
        """The originally reported case: the exact GeoPackage feature, run
        through the GeoDataFrame pipeline (merge/dissolve, auto segment
        length) it was reported against rather than the single-geometry path.
        """
        gdf = gpd.read_file(CUSP_REGRESSION_DATA)

        smoothed = smoothify(gdf, num_cores=1)

        geom = smoothed.geometry.iloc[0]
        assert geom.is_valid
        assert concavity(geom) < 0.002
