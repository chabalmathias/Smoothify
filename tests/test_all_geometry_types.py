"""Tests for all geometry types with simple and complex shapes.

Ensures smoothify handles every supported geometry type correctly,
from minimal examples through to complex real-world-like shapes.
"""

import math

import numpy as np
import pytest
from shapely import make_valid, wkt
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)

from smoothify import smoothify


# ---------------------------------------------------------------------------
# Polygons
# ---------------------------------------------------------------------------
class TestPolygonSimple:
    """Simple polygon inputs."""

    def test_triangle(self):
        poly = Polygon([(0, 0), (10, 0), (5, 8)])
        result = smoothify(poly, segment_length=1.0)
        assert isinstance(result, Polygon)
        assert result.is_valid
        assert len(result.exterior.coords) > 3

    def test_square(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = smoothify(poly, segment_length=1.0)
        assert isinstance(result, Polygon)
        assert result.is_valid

    def test_rectangle_thin(self):
        """Thin rectangle — smoothing shouldn't collapse it."""
        poly = Polygon([(0, 0), (100, 0), (100, 2), (0, 2)])
        result = smoothify(poly, segment_length=1.0)
        assert isinstance(result, Polygon)
        assert result.is_valid
        assert result.area > 0

    def test_regular_hexagon(self):
        coords = [
            (10 * math.cos(math.radians(60 * i)), 10 * math.sin(math.radians(60 * i)))
            for i in range(6)
        ]
        poly = Polygon(coords)
        result = smoothify(poly, segment_length=1.0)
        assert isinstance(result, Polygon)
        assert result.is_valid


class TestPolygonComplex:
    """Complex polygon inputs."""

    def test_star_shape(self):
        """Star polygon with concavities."""
        coords = []
        for i in range(10):
            angle = math.radians(36 * i)
            r = 10 if i % 2 == 0 else 4
            coords.append((r * math.cos(angle), r * math.sin(angle)))
        poly = Polygon(coords)
        result = smoothify(poly, segment_length=0.5)
        assert isinstance(result, Polygon)
        assert result.is_valid

    def test_polygon_with_one_hole(self):
        exterior = [(0, 0), (50, 0), (50, 50), (0, 50)]
        hole = [(15, 15), (35, 15), (35, 35), (15, 35)]
        poly = Polygon(exterior, [hole])
        result = smoothify(poly, segment_length=1.0)
        assert isinstance(result, (Polygon, MultiPolygon))
        assert result.is_valid

    def test_polygon_with_many_holes(self):
        exterior = [(0, 0), (100, 0), (100, 100), (0, 100)]
        holes = [
            [
                (10 + 25 * i, 10 + 25 * j),
                (20 + 25 * i, 10 + 25 * j),
                (20 + 25 * i, 20 + 25 * j),
                (10 + 25 * i, 20 + 25 * j),
            ]
            for i in range(3)
            for j in range(3)
        ]
        poly = Polygon(exterior, holes)
        result = smoothify(poly, segment_length=1.0)
        assert result.is_valid

    def test_circle_approximation(self):
        """High-vertex polygon approximating a circle."""
        n = 200
        coords = [
            (10 * math.cos(2 * math.pi * i / n), 10 * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
        poly = Polygon(coords)
        result = smoothify(poly, segment_length=0.5)
        assert isinstance(result, Polygon)
        assert result.is_valid

    def test_L_shaped_polygon(self):
        poly = Polygon(
            [
                (0, 0),
                (10, 0),
                (10, 5),
                (5, 5),
                (5, 10),
                (0, 10),
            ]
        )
        result = smoothify(poly, segment_length=0.5)
        assert isinstance(result, Polygon)
        assert result.is_valid

    def test_pixelated_polygon(self):
        """Staircase polygon typical of raster-to-vector conversion."""
        coords = []
        for i in range(10):
            coords.append((i, i))
            coords.append((i + 1, i))
        for i in range(10, 0, -1):
            coords.append((i, i + 5))
            coords.append((i - 1, i + 5))
        poly = Polygon(coords)
        result = smoothify(poly, segment_length=1.0)
        assert isinstance(result, Polygon)
        assert result.is_valid


# ---------------------------------------------------------------------------
# LineStrings
# ---------------------------------------------------------------------------
class TestLineStringSimple:
    """Simple LineString inputs."""

    def test_two_point_line(self):
        line = LineString([(0, 0), (10, 10)])
        result = smoothify(line, segment_length=1.0)
        assert isinstance(result, LineString)
        assert result.coords[0] == (0.0, 0.0)
        assert result.coords[-1] == (10.0, 10.0)

    def test_three_point_angle(self):
        line = LineString([(0, 0), (5, 5), (10, 0)])
        result = smoothify(line, segment_length=1.0)
        assert isinstance(result, LineString)

    def test_straight_line(self):
        """Perfectly straight line should remain straight-ish."""
        line = LineString([(0, 0), (5, 0), (10, 0)])
        result = smoothify(line, segment_length=1.0)
        assert isinstance(result, LineString)

    def test_right_angle(self):
        line = LineString([(0, 0), (10, 0), (10, 10)])
        result = smoothify(line, segment_length=1.0)
        assert isinstance(result, LineString)


class TestLineStringComplex:
    """Complex LineString inputs."""

    def test_zigzag(self):
        coords = [(i, (i % 2) * 5) for i in range(20)]
        line = LineString(coords)
        result = smoothify(line, segment_length=0.5)
        assert isinstance(result, LineString)
        assert result.coords[0] == line.coords[0]
        assert result.coords[-1] == line.coords[-1]

    def test_spiral(self):
        t = np.linspace(0, 4 * np.pi, 200)
        x = t * np.cos(t)
        y = t * np.sin(t)
        line = LineString(np.column_stack([x, y]))
        result = smoothify(line, segment_length=0.5)
        assert isinstance(result, LineString)

    def test_self_intersecting_line(self):
        """Self-intersecting line must stay a LineString, not be split."""
        line = LineString(
            [
                (0, 0),
                (2, 0),
                (3, 0),
                (4, 1),
                (3, 2),
                (2, 2),
                (1, 1),
                (2, 0.5),
                (3, 0.5),
                (4, 0),
                (5, 0),
                (7, 0),
            ]
        )
        assert not line.is_simple
        result = smoothify(line, segment_length=0.5)
        assert isinstance(result, LineString)

    def test_random_walk(self):
        """Long random-walk line (river-like)."""
        rng = np.random.default_rng(42)
        n = 300
        x = np.cumsum(rng.standard_normal(n) * 0.2)
        y = np.cumsum(rng.standard_normal(n) * 0.2)
        line = LineString(np.column_stack([x, y]))
        result = smoothify(line, segment_length=0.5)
        assert isinstance(result, LineString)

    def test_many_vertices(self):
        coords = [(i * 0.1, math.sin(i * 0.1) * 5) for i in range(500)]
        line = LineString(coords)
        result = smoothify(line, segment_length=0.5)
        assert isinstance(result, LineString)

    def test_hairpin_turn(self):
        """Tight U-turn."""
        line = LineString([(0, 0), (10, 0), (10.5, 0.5), (10, 1), (0, 1)])
        result = smoothify(line, segment_length=0.5)
        assert isinstance(result, LineString)


# ---------------------------------------------------------------------------
# LinearRings
# ---------------------------------------------------------------------------
class TestLinearRing:
    """LinearRing inputs."""

    def test_simple_ring(self):
        ring = LinearRing([(0, 0), (10, 0), (10, 10), (0, 10)])
        result = smoothify(ring, segment_length=1.0)
        assert isinstance(result, (LinearRing, Polygon))
        assert result.is_valid

    def test_triangular_ring(self):
        ring = LinearRing([(0, 0), (10, 0), (5, 8)])
        result = smoothify(ring, segment_length=1.0)
        assert isinstance(result, (LinearRing, Polygon))
        assert result.is_valid

    def test_complex_ring(self):
        """Ring with many vertices."""
        n = 50
        coords = [
            (10 * math.cos(2 * math.pi * i / n), 10 * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
        ring = LinearRing(coords)
        result = smoothify(ring, segment_length=0.5)
        assert isinstance(result, (LinearRing, Polygon))
        assert result.is_valid


# ---------------------------------------------------------------------------
# MultiPolygons
# ---------------------------------------------------------------------------
class TestMultiPolygonSimple:
    """Simple MultiPolygon inputs."""

    def test_two_squares(self):
        p1 = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        p2 = Polygon([(20, 20), (25, 20), (25, 25), (20, 25)])
        mp = MultiPolygon([p1, p2])
        result = smoothify(mp, segment_length=1.0, merge_multipolygons=False)
        assert result.is_valid

    def test_single_polygon_multi(self):
        """MultiPolygon with one polygon."""
        p = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        mp = MultiPolygon([p])
        result = smoothify(mp, segment_length=1.0, merge_multipolygons=False)
        assert result.is_valid


class TestMultiPolygonComplex:
    """Complex MultiPolygon inputs."""

    def test_adjacent_squares_merged(self):
        """Adjacent polygons that should merge."""
        p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        p2 = Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])
        mp = MultiPolygon([p1, p2])
        result = smoothify(mp, segment_length=1.0, merge_multipolygons=True)
        assert result.is_valid

    def test_many_small_polygons(self):
        """Grid of small polygons."""
        polys = [
            Polygon(
                [
                    (i * 15, j * 15),
                    (i * 15 + 5, j * 15),
                    (i * 15 + 5, j * 15 + 5),
                    (i * 15, j * 15 + 5),
                ]
            )
            for i in range(4)
            for j in range(4)
        ]
        mp = MultiPolygon(polys)
        result = smoothify(mp, segment_length=1.0, merge_multipolygons=False)
        assert result.is_valid

    def test_mixed_size_polygons(self):
        tiny = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        large = Polygon([(50, 50), (150, 50), (150, 150), (50, 150)])
        mp = MultiPolygon([tiny, large])
        result = smoothify(mp, segment_length=1.0, merge_multipolygons=False)
        assert result.is_valid

    def test_polygon_with_holes_in_multi(self):
        exterior1 = [(0, 0), (30, 0), (30, 30), (0, 30)]
        hole1 = [(5, 5), (15, 5), (15, 15), (5, 15)]
        p1 = Polygon(exterior1, [hole1])
        p2 = Polygon([(50, 50), (60, 50), (60, 60), (50, 60)])
        mp = MultiPolygon([p1, p2])
        result = smoothify(mp, segment_length=1.0, merge_multipolygons=False)
        assert result.is_valid


# ---------------------------------------------------------------------------
# MultiLineStrings
# ---------------------------------------------------------------------------
class TestMultiLineStringSimple:
    """Simple MultiLineString inputs."""

    def test_two_lines(self):
        ml = MultiLineString(
            [
                [(0, 0), (5, 5)],
                [(10, 0), (15, 5)],
            ]
        )
        result = smoothify(ml, segment_length=1.0)
        assert result.is_valid

    def test_single_line_multi(self):
        ml = MultiLineString([[(0, 0), (5, 5), (10, 0)]])
        result = smoothify(ml, segment_length=1.0)
        assert result.is_valid


class TestMultiLineStringComplex:
    """Complex MultiLineString inputs."""

    def test_many_lines(self):
        lines = [[(i * 10, 0), (i * 10 + 5, 5), (i * 10 + 10, 0)] for i in range(10)]
        ml = MultiLineString(lines)
        result = smoothify(ml, segment_length=1.0)
        assert result.is_valid

    def test_zigzag_lines(self):
        lines = []
        for j in range(5):
            coords = [(i, (i % 2) * 3 + j * 10) for i in range(15)]
            lines.append(coords)
        ml = MultiLineString(lines)
        result = smoothify(ml, segment_length=0.5)
        assert result.is_valid

    def test_long_and_short_lines(self):
        short = LineString([(0, 0), (1, 1)])
        long_line = LineString(
            [(10, 10)] + [(10 + i, 10 + math.sin(i)) for i in range(100)]
        )
        ml = MultiLineString([short, long_line])
        result = smoothify(ml, segment_length=1.0)
        assert result.is_valid

    def test_self_intersecting_multilinestring(self):
        """MultiLineString containing a self-intersecting line."""
        si_line = LineString(
            [
                (0, 0),
                (2, 0),
                (3, 0),
                (4, 1),
                (3, 2),
                (2, 2),
                (1, 1),
                (2, 0.5),
                (3, 0.5),
                (4, 0),
                (5, 0),
                (7, 0),
            ]
        )
        normal_line = LineString([(10, 0), (15, 5), (20, 0)])
        ml = MultiLineString([si_line, normal_line])
        result = smoothify(ml, segment_length=0.5)
        assert result.is_valid


# ---------------------------------------------------------------------------
# GeometryCollections
# ---------------------------------------------------------------------------
class TestGeometryCollectionSimple:
    """Simple GeometryCollection inputs."""

    def test_polygon_and_line(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        line = LineString([(20, 0), (30, 10)])
        gc = GeometryCollection([poly, line])
        result = smoothify(gc, segment_length=1.0)
        assert isinstance(result, GeometryCollection)
        assert result.is_valid

    def test_single_polygon_collection(self):
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        gc = GeometryCollection([poly])
        result = smoothify(gc, segment_length=1.0)
        assert result.is_valid


class TestGeometryCollectionComplex:
    """Complex GeometryCollection inputs."""

    def test_all_supported_types(self):
        """Collection with every supported geometry type."""
        poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        line = LineString([(20, 0), (25, 5), (30, 0)])
        mp = MultiPolygon(
            [
                Polygon([(40, 0), (45, 0), (45, 5), (40, 5)]),
                Polygon([(50, 0), (55, 0), (55, 5), (50, 5)]),
            ]
        )
        ml = MultiLineString(
            [
                [(60, 0), (65, 5)],
                [(70, 0), (75, 5)],
            ]
        )
        gc = GeometryCollection([poly, line, mp, ml])
        result = smoothify(gc, segment_length=1.0)
        assert result.is_valid

    def test_many_mixed_geometries(self):
        geoms = []
        for i in range(5):
            geoms.append(
                Polygon(
                    [
                        (i * 20, 0),
                        (i * 20 + 8, 0),
                        (i * 20 + 8, 8),
                        (i * 20, 8),
                    ]
                )
            )
        for i in range(5):
            geoms.append(
                LineString(
                    [
                        (i * 20, 20),
                        (i * 20 + 5, 25),
                        (i * 20 + 10, 20),
                    ]
                )
            )
        gc = GeometryCollection(geoms)
        result = smoothify(gc, segment_length=1.0)
        assert result.is_valid


# ---------------------------------------------------------------------------
# List inputs
# ---------------------------------------------------------------------------
class TestListInput:
    """List of geometries inputs."""

    def test_list_of_polygons(self):
        polys = [
            Polygon([(i * 20, 0), (i * 20 + 10, 0), (i * 20 + 10, 10), (i * 20, 10)])
            for i in range(3)
        ]
        result = smoothify(polys, segment_length=1.0)
        assert result.is_valid

    def test_list_of_linestrings(self):
        lines = [
            LineString([(i * 20, 0), (i * 20 + 5, 5), (i * 20 + 10, 0)])
            for i in range(3)
        ]
        result = smoothify(lines, segment_length=1.0)
        assert result.is_valid

    def test_mixed_list(self):
        geoms = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            LineString([(20, 0), (30, 10)]),
        ]
        result = smoothify(geoms, segment_length=1.0)
        assert result.is_valid


# ---------------------------------------------------------------------------
# Smoothing parameters across types
# ---------------------------------------------------------------------------
class TestSmoothingParamsAllTypes:
    """Test different smooth_iterations and segment_length across types."""

    @pytest.fixture(params=[1, 3, 5])
    def iterations(self, request):
        return request.param

    @pytest.fixture(params=[0.5, 1.0, 5.0])
    def seg_len(self, request):
        return request.param

    def test_polygon_iterations(self, iterations, seg_len):
        poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        result = smoothify(poly, segment_length=seg_len, smooth_iterations=iterations)
        assert isinstance(result, Polygon)
        assert result.is_valid

    def test_linestring_iterations(self, iterations, seg_len):
        line = LineString([(0, 0), (5, 5), (10, 0), (15, 5), (20, 0)])
        result = smoothify(line, segment_length=seg_len, smooth_iterations=iterations)
        assert isinstance(result, LineString)

    def test_multipolygon_iterations(self, iterations, seg_len):
        mp = MultiPolygon(
            [
                Polygon([(0, 0), (8, 0), (8, 8), (0, 8)]),
                Polygon([(20, 0), (28, 0), (28, 8), (20, 8)]),
            ]
        )
        result = smoothify(
            mp,
            segment_length=seg_len,
            smooth_iterations=iterations,
            merge_multipolygons=False,
        )
        assert result.is_valid

    def test_multilinestring_iterations(self, iterations, seg_len):
        ml = MultiLineString(
            [
                [(0, 0), (5, 5), (10, 0)],
                [(20, 0), (25, 5), (30, 0)],
            ]
        )
        result = smoothify(ml, segment_length=seg_len, smooth_iterations=iterations)
        assert result.is_valid


# ---------------------------------------------------------------------------
# Area preservation across polygon types
# ---------------------------------------------------------------------------
class TestAreaPreservation:
    """Area preservation for different polygon shapes."""

    @pytest.mark.parametrize("preserve", [True, False])
    def test_square_area(self, preserve):
        poly = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
        result = smoothify(poly, segment_length=5.0, preserve_area=preserve)
        assert isinstance(result, Polygon)
        if preserve:
            assert abs(result.area - poly.area) / poly.area < 0.05

    @pytest.mark.parametrize("preserve", [True, False])
    def test_star_area(self, preserve):
        coords = []
        for i in range(10):
            angle = math.radians(36 * i)
            r = 20 if i % 2 == 0 else 8
            coords.append((r * math.cos(angle), r * math.sin(angle)))
        poly = Polygon(coords)
        result = smoothify(poly, segment_length=1.0, preserve_area=preserve)
        assert isinstance(result, Polygon)
        assert result.is_valid

    @pytest.mark.parametrize("preserve", [True, False])
    def test_polygon_with_hole_area(self, preserve):
        exterior = [(0, 0), (50, 0), (50, 50), (0, 50)]
        hole = [(15, 15), (35, 15), (35, 35), (15, 35)]
        poly = Polygon(exterior, [hole])
        result = smoothify(poly, segment_length=1.0, preserve_area=preserve)
        assert result.is_valid


# ---------------------------------------------------------------------------
# Degenerate / tricky geometries
# ---------------------------------------------------------------------------
class TestDegenerateGeometries:
    """Degenerate, invalid, or tricky geometries that stress edge cases."""

    def test_bowtie_polygon(self):
        """Self-intersecting (bowtie) polygon — must be made valid first."""
        geom = wkt.loads("POLYGON((0 0, 2 2, 2 0, 0 2, 0 0))")
        assert not geom.is_valid
        geom = make_valid(geom)
        result = smoothify(geom, segment_length=0.5)
        assert result.is_valid

    def test_polygon_hole_touching_exterior(self):
        """Hole that shares a vertex with the exterior ring."""
        geom = wkt.loads("POLYGON((0 0,10 0,10 10,0 10,0 0),(5 0,8 3,5 6,2 3,5 0))")
        assert geom.is_valid
        result = smoothify(geom, segment_length=1.0)
        assert result.is_valid

    def test_spike_polygon(self):
        """Polygon with a degenerate zero-width spike."""
        geom = wkt.loads("POLYGON((0 0,10 0,10 10,5 10,5 0,0 0))")
        # Shapely treats this as invalid; make_valid turns it into a collection
        geom = make_valid(geom)
        result = smoothify(geom, segment_length=1.0)
        assert result.is_valid

    def test_multipolygon_touching_at_edge(self):
        """Two polygons sharing an entire edge — invalid MultiPolygon."""
        geom = wkt.loads(
            "MULTIPOLYGON(((0 0,5 0,5 5,0 5,0 0)),((5 0,10 0,10 5,5 5,5 0)))"
        )
        geom = make_valid(geom)
        result = smoothify(geom, segment_length=1.0)
        assert result.is_valid

    def test_empty_polygon(self):
        geom = wkt.loads("POLYGON EMPTY")
        result = smoothify(geom, segment_length=1.0)
        assert result.is_empty

    def test_empty_linestring(self):
        geom = LineString()
        result = smoothify(geom, segment_length=1.0)
        assert result.is_empty

    def test_empty_multipolygon(self):
        geom = MultiPolygon()
        result = smoothify(geom, segment_length=1.0)
        assert result.is_valid

    def test_empty_multilinestring(self):
        geom = MultiLineString()
        result = smoothify(geom, segment_length=1.0)
        assert result.is_valid

    def test_duplicate_points_linestring(self):
        """LineString with consecutive duplicate points."""
        geom = wkt.loads("LINESTRING(0 0,0 0,1 1,2 2,2 2)")
        result = smoothify(geom, segment_length=0.5)
        assert isinstance(result, LineString)
        assert result.is_valid

    def test_near_zero_area_polygon(self):
        """Extremely thin polygon that is nearly degenerate."""
        poly = Polygon([(0, 0), (100, 0), (100, 0.001), (0, 0.001)])
        result = smoothify(poly, segment_length=0.01)
        assert result.is_valid

    def test_polygon_with_collinear_points(self):
        """Polygon where several consecutive vertices are collinear."""
        poly = Polygon(
            [
                (0, 0),
                (2, 0),
                (5, 0),
                (8, 0),
                (10, 0),
                (10, 10),
                (0, 10),
            ]
        )
        result = smoothify(poly, segment_length=1.0)
        assert isinstance(result, Polygon)
        assert result.is_valid

    def test_linestring_with_collinear_points(self):
        """LineString where all points are collinear."""
        line = LineString([(0, 0), (1, 0), (2, 0), (5, 0), (10, 0)])
        result = smoothify(line, segment_length=1.0)
        assert isinstance(result, LineString)

    def test_almost_closed_linestring(self):
        """LineString whose start and end are nearly identical."""
        line = LineString([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0.001)])
        result = smoothify(line, segment_length=1.0)
        assert isinstance(result, LineString)

    def test_polygon_hole_nearly_fills_exterior(self):
        """Hole that is almost as large as the exterior."""
        exterior = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(0.5, 0.5), (9.5, 0.5), (9.5, 9.5), (0.5, 9.5)]
        poly = Polygon(exterior, [hole])
        result = smoothify(poly, segment_length=0.5)
        assert result.is_valid

    def test_reversed_ring_orientation(self):
        """Polygon with clockwise exterior (non-standard orientation)."""
        # Shapely normalises this, but real files may have it
        poly = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])  # CW
        result = smoothify(poly, segment_length=1.0)
        assert isinstance(result, Polygon)
        assert result.is_valid

    def test_zero_area_ring(self):
        """Polygon that degenerates to a line (zero area)."""
        poly = Polygon([(0, 0), (10, 0), (5, 0)])
        # Shapely makes this empty/invalid
        if poly.is_empty or not poly.is_valid:
            poly = make_valid(poly)
        result = smoothify(poly, segment_length=1.0)
        assert result.is_valid

    def test_nearly_coincident_vertices(self):
        """Polygon with vertices separated by < 1e-10."""
        poly = Polygon(
            [
                (0, 0),
                (10, 0),
                (10, 1e-11),
                (10, 10),
                (0, 10),
            ]
        )
        result = smoothify(poly, segment_length=1.0)
        assert result.is_valid

    def test_overlapping_multipolygon(self):
        """MultiPolygon with overlapping members."""
        p1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        p2 = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
        geom = make_valid(MultiPolygon([p1, p2]))
        result = smoothify(geom, segment_length=1.0)
        assert result.is_valid


# ---------------------------------------------------------------------------
# Z coordinates
# ---------------------------------------------------------------------------
class TestZCoordinates:
    """Geometries with Z dimension (common in real shapefiles)."""

    def test_polygon_z(self):
        poly = Polygon([(0, 0, 5), (10, 0, 5), (10, 10, 5), (0, 10, 5)])
        result = smoothify(poly, segment_length=1.0)
        assert isinstance(result, Polygon)
        assert result.is_valid

    def test_linestring_z(self):
        line = LineString([(0, 0, 0), (5, 5, 10), (10, 0, 20)])
        result = smoothify(line, segment_length=1.0)
        assert isinstance(result, LineString)

    def test_multipolygon_z(self):
        p1 = Polygon([(0, 0, 1), (5, 0, 1), (5, 5, 1), (0, 5, 1)])
        p2 = Polygon([(20, 20, 2), (25, 20, 2), (25, 25, 2), (20, 25, 2)])
        mp = MultiPolygon([p1, p2])
        result = smoothify(mp, segment_length=1.0, merge_multipolygons=False)
        assert result.is_valid

    def test_multilinestring_z(self):
        ml = MultiLineString(
            [
                [(0, 0, 0), (5, 5, 10)],
                [(10, 0, 0), (15, 5, 10)],
            ]
        )
        result = smoothify(ml, segment_length=1.0)
        assert result.is_valid


# ---------------------------------------------------------------------------
# Numeric edge cases
# ---------------------------------------------------------------------------
class TestNumericEdgeCases:
    """Extreme coordinate values and numeric stability."""

    def test_very_large_coordinates(self):
        """Coordinates typical of projected CRS (e.g. UTM easting/northing)."""
        poly = Polygon(
            [
                (500000, 6000000),
                (500100, 6000000),
                (500100, 6000100),
                (500000, 6000100),
            ]
        )
        result = smoothify(poly, segment_length=10.0)
        assert isinstance(result, Polygon)
        assert result.is_valid

    def test_very_small_coordinates(self):
        """Tiny fractional coordinates."""
        poly = Polygon(
            [
                (0, 0),
                (1e-6, 0),
                (1e-6, 1e-6),
                (0, 1e-6),
            ]
        )
        result = smoothify(poly, segment_length=1e-7)
        assert isinstance(result, Polygon)
        assert result.is_valid

    def test_large_coordinate_linestring(self):
        line = LineString(
            [
                (1e7, 1e7),
                (1e7 + 50, 1e7 + 50),
                (1e7 + 100, 1e7),
            ]
        )
        result = smoothify(line, segment_length=5.0)
        assert isinstance(result, LineString)

    def test_negative_coordinates(self):
        poly = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])
        result = smoothify(poly, segment_length=1.0)
        assert isinstance(result, Polygon)
        assert result.is_valid

    def test_mixed_sign_coordinates(self):
        """Geometry spanning the origin."""
        line = LineString([(-10, -5), (0, 5), (10, -5)])
        result = smoothify(line, segment_length=1.0)
        assert isinstance(result, LineString)


# ---------------------------------------------------------------------------
# Collapse / degenerate output cases
# ---------------------------------------------------------------------------
class TestCollapseOutputs:
    """Inputs that may collapse or degenerate after smoothing."""

    def test_tiny_polygon_survives(self):
        """Very small polygon should not vanish."""
        poly = Polygon([(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)])
        result = smoothify(poly, segment_length=0.01)
        assert result.is_valid
        # Should still have some area
        if isinstance(result, Polygon):
            assert result.area > 0

    def test_tiny_triangle(self):
        poly = Polygon([(0, 0), (0.01, 0), (0.005, 0.01)])
        result = smoothify(poly, segment_length=0.001)
        assert result.is_valid

    def test_very_short_linestring(self):
        """Two-point line shorter than segment_length."""
        line = LineString([(0, 0), (0.01, 0.01)])
        result = smoothify(line, segment_length=1.0)
        assert isinstance(result, LineString)

    def test_large_segment_length_polygon(self):
        """segment_length much larger than the geometry — aggressive simplification."""
        poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        result = smoothify(poly, segment_length=100.0)
        assert result.is_valid

    def test_large_segment_length_linestring(self):
        line = LineString([(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)])
        result = smoothify(line, segment_length=100.0)
        assert isinstance(result, LineString)


# ---------------------------------------------------------------------------
# Invariant / property tests
# ---------------------------------------------------------------------------
class TestInvariants:
    """Property-based invariants that should hold for all smoothing operations."""

    def test_idempotence_polygon(self):
        """Smoothing twice should not drift much after the first pass."""
        poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        once = smoothify(poly, segment_length=2.0)
        twice = smoothify(once, segment_length=2.0)
        assert isinstance(once, Polygon)
        assert isinstance(twice, Polygon)
        # Second pass should change area by < 5%
        assert abs(twice.area - once.area) / once.area < 0.05

    def test_idempotence_linestring(self):
        line = LineString([(0, 0), (5, 5), (10, 0), (15, 5), (20, 0)])
        once = smoothify(line, segment_length=1.0)
        twice = smoothify(once, segment_length=1.0)
        assert isinstance(once, LineString)
        assert isinstance(twice, LineString)
        # Length should not drift wildly
        assert abs(twice.length - once.length) / once.length < 0.15

    def test_bounded_area_distortion_polygon(self):
        """Area change should be bounded (within 10% without preserve_area)."""
        poly = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
        result = smoothify(poly, segment_length=5.0, preserve_area=False)
        assert isinstance(result, Polygon)
        ratio = result.area / poly.area
        assert 0.8 < ratio < 1.2, f"Area ratio {ratio:.3f} outside bounds"

    def test_area_preservation_tight(self):
        """With preserve_area=True, area should be within tolerance."""
        poly = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
        result = smoothify(
            poly, segment_length=5.0, preserve_area=True, area_tolerance=0.01
        )
        assert isinstance(result, Polygon)
        pct_error = abs(result.area - poly.area) / poly.area * 100
        assert pct_error < 1.0, f"Area error {pct_error:.4f}% exceeds 1%"

    def test_hausdorff_distance_bounded_polygon(self):
        """Smoothed polygon should stay close to original."""
        poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        result = smoothify(poly, segment_length=2.0)
        assert isinstance(result, Polygon)
        dist = poly.hausdorff_distance(result)
        # Should not drift more than a few segment_lengths
        assert dist < 10.0, f"Hausdorff distance {dist:.2f} too large"

    def test_hausdorff_distance_bounded_linestring(self):
        """Smoothed line should stay close to original."""
        line = LineString([(0, 0), (5, 5), (10, 0), (15, 5), (20, 0)])
        result = smoothify(line, segment_length=1.0)
        assert isinstance(result, LineString)
        dist = line.hausdorff_distance(result)
        assert dist < 5.0, f"Hausdorff distance {dist:.2f} too large"

    def test_envelope_containment_polygon(self):
        """Smoothed polygon envelope should be similar to original."""
        poly = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
        result = smoothify(poly, segment_length=5.0)
        assert isinstance(result, Polygon)
        # Result bounding box should not be wildly different
        orig_bounds = poly.bounds
        res_bounds = result.bounds
        for i in range(4):
            assert abs(orig_bounds[i] - res_bounds[i]) < 10.0

    def test_vertex_count_increases(self):
        """Smoothing should generally add vertices."""
        poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        result = smoothify(poly, segment_length=2.0)
        assert isinstance(result, Polygon)
        assert len(result.exterior.coords) > len(poly.exterior.coords)

    def test_output_always_valid(self):
        """Diverse set of inputs should always produce valid output."""
        inputs = [
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            LineString([(0, 0), (5, 5), (10, 0)]),
            MultiPolygon(
                [
                    Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
                    Polygon([(20, 20), (25, 20), (25, 25), (20, 25)]),
                ]
            ),
            MultiLineString([[(0, 0), (5, 5)], [(10, 0), (15, 5)]]),
            GeometryCollection(
                [
                    Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                    LineString([(20, 0), (30, 10)]),
                ]
            ),
        ]
        for geom in inputs:
            result = smoothify(geom, segment_length=1.0)
            assert result.is_valid, f"Invalid output for {geom.geom_type}"

    def test_linestring_endpoints_preserved(self):
        """Endpoints must be preserved across various LineStrings."""
        lines = [
            LineString([(0, 0), (10, 10)]),
            LineString([(0, 0), (5, 5), (10, 0)]),
            LineString([(i, math.sin(i)) for i in range(20)]),
        ]
        for line in lines:
            result = smoothify(line, segment_length=1.0)
            assert isinstance(result, LineString)
            assert result.coords[0] == pytest.approx(line.coords[0], abs=1e-6)
            assert result.coords[-1] == pytest.approx(line.coords[-1], abs=1e-6)
