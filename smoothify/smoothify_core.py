from typing import cast

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from shapely import make_valid
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiPolygon,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# Chaikin's corner cut moves a right-angle vertex inward by ~1/4 of the
# adjacent segment length, so smoothing segments of FACTOR * segment_length
# deviates at most ~segment_length at a right angle — the same deviation
# budget the simplify(segment_length) step already allows — while letting
# shallow facet chains (e.g. simplified curves) blend into smooth curves.
_CHAIKIN_SEGMENT_FACTOR = 4


def _chaikin_corner_cutting(
    geom: Polygon | LineString, num_iterations: int = 1, reverse: bool = False
) -> Polygon | LineString:
    """Apply Chaikin's corner cutting algorithm to smooth a geometry.

    Chaikin's algorithm iteratively replaces each line segment with two new segments
    by cutting corners at 1/4 and 3/4 positions, creating a smooth curve. This
    implementation uses vectorized NumPy operations for performance and handles
    both closed (Polygon) and open (LineString) geometries."""

    is_closed = isinstance(geom, Polygon)
    points = np.array(
        geom.exterior.coords if is_closed else geom.coords, dtype=np.float64
    )

    if is_closed:
        # Remove duplicate endpoint for closed rings
        points = points[:-1]
        endpoints = None
    else:
        # Store endpoints for open linestrings
        endpoints = (points[0], points[-1])

    if reverse:
        points = points[::-1]

    for _ in range(num_iterations):
        # Get point pairs for corner cutting
        if is_closed:
            p0 = points
            p1 = np.roll(points, -1, axis=0)
        else:
            p0 = points[:-1]
            p1 = points[1:]

        # Vectorized smoothing at 1/4 and 3/4 positions
        # Pre-allocate result array for better performance
        n_new_points = len(p0) * 2
        points = np.empty((n_new_points, points.shape[1]), dtype=np.float64)
        points[0::2] = 0.75 * p0 + 0.25 * p1  # q points
        points[1::2] = 0.25 * p0 + 0.75 * p1  # r points

        # Restore endpoints for open linestrings
        if endpoints is not None:
            points = np.vstack([endpoints[0], points, endpoints[1]])

    # Add closing point for polygons
    if is_closed:
        points = np.vstack([points, points[0:1]])

    if reverse:
        points = points[::-1]

    return LineString(points) if isinstance(geom, LineString) else Polygon(points)


def _rotate_ring_coords(ring: LinearRing, shift: float) -> "npt.NDArray[np.float64]":
    """Rotate a linear ring coordinate sequence by a fractional shift.

    Used to create multiple starting point variants of a polygon for smoothing,
    which are later merged to avoid artifacts from a fixed starting vertex."""

    coords = np.array(ring.coords)
    n = len(coords) - 1  # Exclude duplicate closing point
    shift_idx = int(round(shift * n)) % n
    # Use numpy array slicing and concatenation
    rotated = np.vstack(
        [
            coords[shift_idx:],
            coords[1 : shift_idx + 1],
            coords[shift_idx : shift_idx + 1],
        ]
    )
    return rotated


def _rotate_polygon_start(polygon: Polygon, shift: float) -> Polygon:
    """Rotate a polygon's exterior start vertex by a fractional shift.

    Creates a topologically identical polygon with a different starting vertex.
    Used in conjunction with smoothing to avoid artifacts at the start/end point.
    """

    if polygon.is_empty:
        return polygon

    exterior = LinearRing(_rotate_ring_coords(polygon.exterior, shift))
    return Polygon(exterior)


def _generate_starting_point_variants(
    geom: BaseGeometry, n_starting_points: int
) -> list[Polygon | LineString]:
    """Generate evenly rotated variants of a Polygon to avoid smoothing artifacts.

    For Polygons, creates multiple versions with different starting vertices. When
    smoothed and merged, these variants eliminate artifacts that can occur at the
    arbitrary start/end point of a polygon's coordinate ring. LineStrings are
    returned as-is since they have fixed endpoints."""

    if n_starting_points <= 0:
        return []
    if isinstance(geom, LineString):
        return [geom]

    elif isinstance(geom, Polygon):
        if n_starting_points > len(geom.exterior.coords) - 1:
            n_starting_points = len(geom.exterior.coords) - 1

        variants = []

        for i in range(n_starting_points):
            shift = i / n_starting_points
            if shift == 0:
                shifted_geoms = geom
            else:
                shifted_geoms = _rotate_polygon_start(geom, shift)

            variants.append(shifted_geoms)

        return variants
    elif isinstance(geom, MultiPolygon):
        # Invalid input is screened out before smoothing, so a valid Polygon
        # should never become multi-part here. If it does, segmentize/simplify
        # split it unexpectedly — surface that rather than a bare type error.
        raise ValueError(
            "Preprocessing produced a MultiPolygon from a single Polygon "
            f"({len(geom.geoms)} parts); expected it to stay single-part. "
            "This usually means the input geometry was invalid "
            "(self-intersecting). Repair it with shapely's make_valid() first."
        )
    else:
        raise ValueError(
            f"Input geometry must be a Polygon or LineString, got {type(geom)}."
        )


def _preserve_area_with_buffer(
    polygon: Polygon,
    target_area: float,
    tolerance: float = 1e-6,
) -> Polygon:
    """Restore original polygon area after smoothing via iterative buffering.

    Smoothing operations can slightly change polygon area. This function uses
    Brent's method (root-finding algorithm) to find the optimal buffer distance
    that restores the original area within the specified tolerance."""

    if polygon.is_empty:
        return polygon

    current_area = polygon.area
    if abs(current_area - target_area) <= tolerance:
        return polygon

    # Approximate buffer distance needed (assuming circular shape)
    perimeter = polygon.length
    initial_guess = (target_area - current_area) / perimeter if perimeter > 0 else 0

    def area_delta(distance: float) -> float:
        buffered_polygon = polygon.buffer(distance)
        return float(buffered_polygon.area - target_area)

    scale = (
        abs(initial_guess) * 2 if initial_guess != 0 else (polygon.area / 3.1416) ** 0.5
    )
    max_distance = max(0.1, scale)

    lo, hi = -max_distance, max_distance
    f_lo, f_hi = area_delta(lo), area_delta(hi)

    for _ in range(20):
        if f_lo * f_hi < 0:
            break
        lo *= 1.5
        hi *= 1.5
        f_lo, f_hi = area_delta(lo), area_delta(hi)

    try:
        # Use brentq to find optimal buffer distance
        # xtol controls precision of buffer distance finding, not area precision
        # Use a reasonable xtol relative to the search space
        xtol = min(tolerance * 0.01, (hi - lo) * 0.001)
        optimal_distance = cast(float, brentq(area_delta, lo, hi, xtol=xtol))
        result = polygon.buffer(optimal_distance)

        # Verify the result meets tolerance, otherwise try to refine
        if abs(result.area - target_area) > tolerance:
            # Try a tighter search if we didn't meet tolerance
            xtol = min(tolerance * 0.001, (hi - lo) * 0.0001)
            optimal_distance = cast(float, brentq(area_delta, lo, hi, xtol=xtol))
            result = polygon.buffer(optimal_distance)

        return result
    except ValueError:
        # Fallback: just return the closest one
        buffered_lo = polygon.buffer(lo)
        buffered_hi = polygon.buffer(hi)
        return min(
            [buffered_lo, buffered_hi],
            key=lambda candidate: abs(candidate.area - target_area),
        )


def _join_adjacent(
    geom: list[BaseGeometry] | BaseGeometry, segment_length: float
) -> BaseGeometry:
    """Merge adjacent geometries using a small buffer-dissolve operation.

    Applies a tiny buffer (segment_length / 1000) to join geometries that are nearly
    touching or share edges, then unions them together into a single geometry or
    collection. This is useful for merging polygons derived from adjacent raster
    cells with the same classification.

    Note: Only applies to Polygon and MultiPolygon geometries. Other geometry types
    (LineString, LinearRing, etc.) are returned unchanged."""

    if isinstance(geom, list):
        geom_combined = unary_union(geom)
    else:
        geom_combined = geom
    poly_types = []
    other_types = []
    if isinstance(geom_combined, GeometryCollection):
        for geom in geom_combined.geoms:
            if isinstance(geom, Polygon | MultiPolygon):
                poly_types.append(geom)
            else:
                other_types.append(geom)
        geom_combined = GeometryCollection(poly_types)
    elif not isinstance(geom_combined, Polygon | MultiPolygon):
        # If the geometry is not a polygon type, return it unchanged
        return geom_combined

    merged_geom = geom_combined.buffer(segment_length / 1000)
    merged_geom = unary_union(merged_geom)

    if other_types:
        return GeometryCollection([merged_geom] + other_types)

    return merged_geom


def _smoothify_geometry(
    geom: Polygon | LineString,
    segment_length: float,
    smooth_iterations: int = 3,
    preserve_area: bool = True,
    area_tolerance: float = 0.01,
) -> Polygon | LineString:
    """Core smoothing pipeline using Chaikin corner cutting with area preservation.

    This is the main smoothing algorithm that:
    1. Adds intermediate vertices along line segments (segmentize)
    2. Generates multiple rotated variants (for Polygons) to avoid artifacts
    3. Simplifies each variant to remove noise, then re-segmentizes so no
       segment exceeds segment_length * _CHAIKIN_SEGMENT_FACTOR (simplify
       strips collinear vertices, and Chaikin's corner cuts scale with
       segment length — unbounded segments would over-round sharp corners)
    4. Applies Chaikin corner cutting to smooth
    5. Merges all variants via union to eliminate start-point artifacts
    6. Simplifies the merged result and re-segmentizes again, then applies
       a final smoothing pass
    7. Optionally restores original area via buffering (for Polygons)"""

    if geom.geom_type == "Polygon":
        original_area = geom.area
    elif geom.geom_type == "LineString":
        original_area = None
    else:
        raise ValueError("Input geometry must be a Polygon or LineString.")

    geom_segmented = geom.segmentize(segment_length / 2)

    geom_iterations = []

    if isinstance(geom, Polygon):
        starting_point_geoms = _generate_starting_point_variants(
            geom_segmented, n_starting_points=4
        )
        for moved_start in starting_point_geoms:
            # Simplify strips noise below segment_length, but on straight edges
            # it also strips every densified vertex, leaving arbitrarily long
            # segments. Chaikin cuts corners at 1/4 of each segment's length,
            # so re-segmentize afterwards to cap the rounding at segment_length
            # scale.
            moved_start = moved_start.simplify(
                tolerance=segment_length,
                preserve_topology=True,
            ).segmentize(segment_length * _CHAIKIN_SEGMENT_FACTOR)
            moved_start = cast(Polygon, moved_start)
            for reverse in [False, True]:
                smoothed = _chaikin_corner_cutting(
                    geom=moved_start,
                    num_iterations=smooth_iterations,
                    reverse=reverse,
                )
                geom_iterations.append(smoothed)

    else:
        moved_start = geom_segmented.simplify(
            tolerance=segment_length,
            preserve_topology=True,
        ).segmentize(segment_length * _CHAIKIN_SEGMENT_FACTOR)
        moved_start = cast(LineString, moved_start)
        smoothed = _chaikin_corner_cutting(
            geom=moved_start,
            num_iterations=smooth_iterations,
        )
        geom_iterations.append(smoothed)

    if isinstance(geom, Polygon):
        geom_iterations = [make_valid(g) for g in geom_iterations]

        dissolved_poly = make_valid(unary_union(geom_iterations)).simplify(
            tolerance=segment_length / 5,
            preserve_topology=True,
        )

        # If the union is a MultiPolygon, take the largest geometry
        if isinstance(dissolved_poly, MultiPolygon):
            largest_geom = max(dissolved_poly.geoms, key=lambda x: x.area)
            dissolved_poly = largest_geom
    else:
        # LineStrings: skip make_valid/unary_union — self-intersecting lines
        # are geometrically valid, and unary_union would split them at
        # crossing points into a MultiLineString
        dissolved_poly = geom_iterations[0].simplify(
            tolerance=segment_length / 5,
            preserve_topology=True,
        )

    assert isinstance(dissolved_poly, (Polygon, LineString)), (
        f"Resulting geometry must be Polygon or LineString. Got {type(dissolved_poly)}."
    )

    # The simplify above can again leave long segments on straight stretches
    # (a Chaikin-rounded corner deviates < segment_length / 5 from the sharp
    # original, so simplify removes it entirely); re-segmentize so the final
    # Chaikin pass cannot re-round at a larger scale.
    dissolved_poly = dissolved_poly.segmentize(segment_length * _CHAIKIN_SEGMENT_FACTOR)

    smoothed_geom = _chaikin_corner_cutting(
        geom=dissolved_poly,
        num_iterations=smooth_iterations,
    )
    if (
        original_area is not None
        and isinstance(smoothed_geom, Polygon)
        and preserve_area
    ):
        # Convert percentage to absolute tolerance based on original area
        # area_tolerance is percentage (e.g., 0.01 = 0.01% error = 99.99% preservation)
        absolute_tolerance = original_area * (area_tolerance / 100.0)

        smoothed_geom = _preserve_area_with_buffer(
            polygon=smoothed_geom,
            target_area=original_area,
            tolerance=absolute_tolerance,
        )

    return smoothed_geom
