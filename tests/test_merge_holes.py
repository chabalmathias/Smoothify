"""Tests for the merge_holes option: adjacent holes smooth as one opening.

Holes from diagonally adjacent raster cells touch at a corner. Smoothed
independently they become separate rounded shapes that overlap or leave a
fake land bridge; merge_holes joins them first so they smooth into a single
coherent opening (mirroring what merge_multipolygons does for shells).
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

from smoothify import smoothify


def max_concave_turn(geom: Polygon | MultiPolygon) -> float:
    """Sharpest concave turn (degrees) over all rings — a fold detector."""
    worst = 0.0
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            c = np.asarray(ring.coords)[:-1, :2]
            if len(c) < 4:
                continue
            v = np.diff(np.vstack([c, c[:2]]), axis=0)
            cross = v[:-1, 0] * v[1:, 1] - v[:-1, 1] * v[1:, 0]
            dot = (v[:-1] * v[1:]).sum(axis=1)
            orientation = np.sign(
                np.dot(c[:, 0], np.roll(c[:, 1], -1))
                - np.dot(c[:, 1], np.roll(c[:, 0], -1))
            )
            concave = cross * orientation < 0
            if concave.any():
                angles = np.degrees(np.arctan2(np.abs(cross[concave]), dot[concave]))
                worst = max(worst, float(angles.max()))
    return worst


def shell_with_corner_touching_holes() -> Polygon:
    """A large shell with two rectangular holes sharing the corner (50, 60)."""
    hole1 = [(20, 60), (50, 60), (50, 90), (20, 90)]
    hole2 = [(50, 10), (85, 10), (85, 60), (50, 60)]
    return Polygon([(-20, -30), (130, -30), (130, 130), (-20, 130)], [hole1, hole2])


def hole_count(geom: Polygon | MultiPolygon) -> int:
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    return sum(len(p.interiors) for p in polys)


class TestMergeHoles:
    def test_adjacent_holes_merge_into_one_opening(self):
        smoothed = smoothify(shell_with_corner_touching_holes(), segment_length=5.0)

        assert smoothed.is_valid
        assert hole_count(smoothed) == 1

    def test_merge_holes_false_keeps_holes_separate(self):
        smoothed = smoothify(
            shell_with_corner_touching_holes(), segment_length=5.0, merge_holes=False
        )

        assert smoothed.is_valid
        assert hole_count(smoothed) == 2

    def test_distant_holes_stay_separate(self):
        """Merging must only join holes that actually touch."""
        hole1 = [(10, 10), (30, 10), (30, 30), (10, 30)]
        hole2 = [(60, 60), (80, 60), (80, 80), (60, 80)]
        shell = Polygon(
            [(-20, -20), (110, -20), (110, 110), (-20, 110)], [hole1, hole2]
        )

        smoothed = smoothify(shell, segment_length=5.0)

        assert smoothed.is_valid
        assert hole_count(smoothed) == 2

    def test_merge_holes_through_geodataframe_path(self):
        gdf = gpd.GeoDataFrame(
            geometry=[shell_with_corner_touching_holes()], crs="EPSG:32750"
        )

        smoothed = smoothify(gdf, segment_length=5.0, num_cores=1)

        assert hole_count(smoothed.geometry.iloc[0]) == 1

    def test_composes_with_merge_collection_across_polygons(self):
        """Holes split across polygon boundaries merge after the dissolve.

        Holes of separate polygons can never touch directly (shell material
        always lies between them), but merge_collection can reunite a hole
        that polygonization split into edge-notches on two adjacent shells.
        merge_holes runs per polygon after that dissolve, so the reunited
        hole then merges with any pre-existing hole it touches.
        """
        # Left shell: a bay notch on its right edge plus an interior hole
        # corner-touching that notch at (40, 60).
        left = Polygon(
            [
                (0, 0),
                (60, 0),
                (60, 40),
                (40, 40),
                (40, 60),
                (60, 60),
                (60, 100),
                (0, 100),
            ],
            [[(20, 60), (40, 60), (40, 80), (20, 80)]],
        )
        # Right shell: the matching bay notch on its left edge.
        right = Polygon(
            [
                (60, 0),
                (120, 0),
                (120, 100),
                (60, 100),
                (60, 60),
                (80, 60),
                (80, 40),
                (60, 40),
            ]
        )
        gdf = gpd.GeoDataFrame(geometry=[left, right], crs="EPSG:32750")

        smoothed = smoothify(gdf, segment_length=5.0, num_cores=1)

        # Dissolve reunites the bay notches into one interior hole, and
        # merge_holes joins it with the corner-touching interior hole.
        assert len(smoothed) == 1
        assert hole_count(smoothed.geometry.iloc[0]) == 1

        separate = smoothify(gdf, segment_length=5.0, num_cores=1, merge_holes=False)
        assert hole_count(separate.geometry.iloc[0]) == 2

    def test_joined_pixel_holes_smooth_without_fold(self):
        """A joined hole with a one-pixel-wide arm must not grow a cusp.

        Joining a 1x2-pixel hole to a 2x3-pixel hole creates a bowtie whose
        narrow arm is exactly segment_length wide — the smoothing variants
        then disagree about it and their union carries a forked slit, which
        the area-preservation shrink sharpened into a ~150 degree fold before
        the fold-repair pass existed.
        """
        seg = 9.0
        h1 = [(0, 27), (9, 27), (9, 45), (0, 45)]  # 1x2 pixels
        h2 = [(9, 0), (27, 0), (27, 27), (9, 27)]  # 2x3 pixels
        shell = Polygon([(-60, -60), (90, -60), (90, 110), (-60, 110)], [h1, h2])

        smoothed = smoothify(shell, segment_length=seg)

        assert smoothed.is_valid
        assert hole_count(smoothed) == 1
        assert max_concave_turn(smoothed) < 60

    def test_fold_repair_preserves_area(self):
        """The fold-repair pass must still honour area preservation.

        The repair recomputes the result from the sealed variant union, so it
        must re-apply the area-preservation step — the bowtie that triggers
        the repair has to come back at its original area within tolerance.
        """
        from shapely.ops import unary_union

        seg = 9.0
        h1 = Polygon([(0, 27), (9, 27), (9, 45), (0, 45)])
        h2 = Polygon([(9, 0), (27, 0), (27, 27), (9, 27)])
        bowtie = unary_union(
            [h.buffer(seg / 250, join_style="mitre") for h in [h1, h2]]
        )

        smoothed = smoothify(bowtie, segment_length=seg)

        assert max_concave_turn(smoothed) < 60
        assert abs(smoothed.area - bowtie.area) / bowtie.area < 1e-4

    def test_lone_holes_keep_exact_area(self):
        """Holes that don't merge must not inherit the join buffer epsilon.

        merge_holes buffers holes to test adjacency; a hole with no neighbour
        must be smoothed from its original ring, not the inflated copy, so
        its area is preserved to the same tolerance as without merge_holes.
        """
        hole1 = [(20, 20), (60, 20), (60, 60), (20, 60)]
        hole2 = [(120, 120), (170, 120), (170, 170), (120, 170)]
        shell = Polygon(
            [(-30, -30), (220, -30), (220, 220), (-30, 220)], [hole1, hole2]
        )

        smoothed = smoothify(shell, segment_length=5.0)

        assert hole_count(smoothed) == 2
        assert abs(smoothed.area - shell.area) / shell.area < 1e-4

    def test_overlapping_donuts_keep_only_truly_open_regions(self):
        """Overlapping shells with overlapping holes resolve by union.

        With merge_collection=True the dissolve keeps a region open only
        where NEITHER feature has material: two overlapping square donuts
        leave three openings (a sliver of each original hole plus the
        hole-intersection), separated by real material, so merge_holes
        correctly leaves them distinct.
        """

        def donut(cx: float, outer: float = 40, inner: float = 20) -> Polygon:
            return Polygon(
                [
                    (cx - outer, -outer),
                    (cx + outer, -outer),
                    (cx + outer, outer),
                    (cx - outer, outer),
                ],
                [
                    [
                        (cx - inner, -inner),
                        (cx + inner, -inner),
                        (cx + inner, inner),
                        (cx - inner, inner),
                    ]
                ],
            )

        gdf = gpd.GeoDataFrame(geometry=[donut(0), donut(25)], crs="EPSG:32750")

        smoothed = smoothify(gdf, segment_length=5.0, num_cores=1)

        assert len(smoothed) == 1
        assert smoothed.geometry.iloc[0].is_valid
        assert hole_count(smoothed.geometry.iloc[0]) == 3
