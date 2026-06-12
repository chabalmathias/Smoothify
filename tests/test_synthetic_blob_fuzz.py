"""Seeded fuzz: random raster-style blobs must always smooth cleanly.

Generates random binary grids, polygonizes them into pixel-aligned blobs
(the library's target input), and screens every smoothed output with the
same defect checks as the real-world sweep: validity, no sharp concave
folds, and area preservation. Complements the Water.gpkg sweep with shapes
nobody hand-picked.
"""

import numpy as np
import pytest
from shapely import unary_union
from shapely.geometry import MultiPolygon, Polygon, box

from smoothify import smoothify

from .test_merge_holes import max_concave_turn

PIXEL = 9.0
MAX_CONCAVE_TURN = 100.0


def random_blobs(seed: int) -> list[Polygon]:
    """Polygonize a random binary grid into pixel-aligned blobs."""
    rng = np.random.default_rng(seed)
    grid = rng.random((12, 12)) < 0.45
    squares = [
        box(x * PIXEL, y * PIXEL, (x + 1) * PIXEL, (y + 1) * PIXEL)
        for y, x in zip(*np.nonzero(grid), strict=True)
    ]
    merged = unary_union(squares)
    parts = merged.geoms if isinstance(merged, MultiPolygon) else [merged]
    return [p for p in parts if isinstance(p, Polygon)]


@pytest.mark.parametrize("seed", range(12))
def test_random_blobs_smooth_cleanly(seed):
    blobs = random_blobs(seed)
    assert blobs, "degenerate seed produced no blobs"

    for blob in blobs:
        smoothed = smoothify(blob, segment_length=PIXEL)

        assert smoothed.is_valid
        assert not smoothed.is_empty
        assert max_concave_turn(smoothed) < MAX_CONCAVE_TURN, (
            f"fold on seed {seed}: {max_concave_turn(smoothed):.1f} degrees"
        )
        # preserve_area applies to the exterior and each hole separately;
        # net area still has to land close
        assert abs(smoothed.area - blob.area) / blob.area < 0.01
