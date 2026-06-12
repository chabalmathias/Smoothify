"""Regression tests: corner rounding must stay at segment_length scale.

Smoothing cuts corners at a fraction of each segment's length, so segments
must be capped (re-segmentized) after every simplify step. Without that cap,
simplify strips all collinear vertices from perfectly straight edges and
Chaikin then cuts at 1/4 of the full edge length — an 8 km square collapses
into a circle (hausdorff distance ~1,400 m) instead of keeping softly
rounded corners (~segment_length).
"""

from shapely.geometry import LineString, Polygon

from smoothify import smoothify


class TestCornerRoundingScale:
    """Straight-edged shapes must keep their shape, rounding only the corners."""

    def test_large_square_keeps_corners(self):
        """An 8 km square with a 10 m segment_length must stay a square."""
        square = Polygon([(0, 0), (8000, 0), (8000, 8000), (0, 8000)])
        segment_length = 10.0

        smoothed = smoothify(square, segment_length=segment_length)

        # Corner rounding deviates ~segment_length; the bug gave ~1,400 m.
        assert smoothed.hausdorff_distance(square) <= 4 * segment_length
        # Shape overlap: the bug moved ~15% of the area, fixed is ~0.02%.
        symdiff_pct = smoothed.symmetric_difference(square).area / square.area * 100
        assert symdiff_pct <= 0.5

    def test_l_shape_keeps_concave_corner(self):
        """Concave corners must round at segment_length scale too."""
        l_shape = Polygon([(0, 0), (100, 0), (100, 40), (60, 40), (60, 100), (0, 100)])
        segment_length = 5.0

        smoothed = smoothify(l_shape, segment_length=segment_length)

        assert smoothed.hausdorff_distance(l_shape) <= 4 * segment_length
        symdiff_pct = smoothed.symmetric_difference(l_shape).area / l_shape.area * 100
        assert symdiff_pct <= 5.0

    def test_long_linestring_keeps_bend(self):
        """A right-angle bend between long straight legs must stay in place."""
        line = LineString([(0, 0), (1000, 0), (1000, 1000)])
        segment_length = 5.0

        smoothed = smoothify(line, segment_length=segment_length)

        # The bug rounded the bend at 1/4 of the 1 km leg length.
        assert smoothed.hausdorff_distance(line) <= 4 * segment_length

    def test_staircase_still_smooths(self):
        """Capping segments must not stop real staircase noise being smoothed.

        Guards the other direction: a pixelated edge (raster artifact) should
        still be rounded into a curve, not preserved like a real straight edge.
        """
        pixel = 5.0
        points = [(0, 0)]
        x, y = 0.0, 0.0
        for _ in range(20):
            x += pixel
            points.append((x, y))
            y += pixel
            points.append((x, y))
        points += [(x, y + 100), (0, y + 100)]
        staircase = Polygon(points)

        smoothed = smoothify(staircase, segment_length=pixel)

        # Smoothing must remove the right-angle steps: the smoothed boundary
        # should be measurably shorter than the jagged original (a straight
        # diagonal is ~29% shorter; require at least a 5% reduction).
        assert smoothed.exterior.length < staircase.exterior.length * 0.95
