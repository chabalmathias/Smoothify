import hashlib
import warnings
from functools import partial
from multiprocessing import get_context
from typing import Callable, Optional, Sequence, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from joblib import Parallel, delayed
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from shapely import make_valid

from .smoothify_core import (
    _join_adjacent,
    _max_concave_turn_degrees,
    _smoothify_geometry,
)

_INVALID_GEOM_HINT = (
    "Repair it first with shapely's make_valid() "
    "(or geopandas GeoSeries.make_valid()) and smooth again."
)


def _warn_invalid(count: int, geom_type: Optional[str] = None) -> None:
    """Emit a single warning that ``count`` invalid geometries were skipped.

    Called in the main process (not inside parallel workers) so the warning
    reliably reaches the user."""
    if count == 1 and geom_type is not None:
        msg = (
            f"Skipping invalid {geom_type} (e.g. self-intersecting); returning "
            f"it unchanged. {_INVALID_GEOM_HINT}"
        )
    else:
        plural = "geometry" if count == 1 else "geometries"
        msg = (
            f"Skipping {count} invalid {plural} (e.g. self-intersecting); "
            f"returning them unchanged. {_INVALID_GEOM_HINT}"
        )
    warnings.warn(msg, stacklevel=2)


def _partition_valid(
    geoms: Sequence[BaseGeometry],
) -> tuple[list[BaseGeometry], list[BaseGeometry]]:
    """Split geometries into (smoothable, invalid).

    Empty geometries count as smoothable (they pass straight through the
    smoothing functions unchanged). Invalid geometries are pulled aside so the
    merge/dissolve step can't silently repair them and so they are never sent
    to worker processes."""
    valid: list[BaseGeometry] = []
    invalid: list[BaseGeometry] = []
    for g in geoms:
        if g.is_empty or g.is_valid:
            valid.append(g)
        else:
            invalid.append(g)
    return valid, invalid


def _smooth_deduplicated(
    geoms: Sequence[BaseGeometry],
    run_batch: Callable[[list[BaseGeometry]], list[BaseGeometry]],
) -> list[BaseGeometry]:
    """Smooth geometries, computing each congruent shape only once.

    Raster-derived datasets typically contain many translated copies of the
    same shape (e.g. single-pixel polygons). Smoothing commutes with
    translation, so congruent geometries are grouped by their origin-
    normalized coordinates, one representative per group is smoothed (via
    run_batch, which may be serial or parallel), and the result is translated
    back to each occurrence."""

    rep_indices: list[int] = []
    key_to_rep: dict[tuple[str, int, bytes], int] = {}
    # per input geometry: (representative position, translation offset)
    plan: list[tuple[int, "np.ndarray"]] = []

    for i, geom in enumerate(geoms):
        coords = shapely.get_coordinates(geom)
        if len(coords) == 0 or shapely.has_z(geom):
            # Empty geometry: nothing to normalize. 3D geometry: the key
            # hashes XY only and the translate-back drops Z, so XY-congruent
            # geometries with different Z would silently swap or lose their
            # Z values — smooth these directly instead.
            rep_indices.append(i)
            plan.append((len(rep_indices) - 1, np.zeros(2)))
            continue
        mins = coords.min(axis=0)
        # Ring/part boundaries are not encoded in the flat coordinate dump,
        # so include the interior ring count to keep keys unambiguous.
        n_rings = (
            shapely.get_num_interior_rings(geom) if isinstance(geom, Polygon) else 0
        )
        # Key on a digest rather than the raw coordinate bytes: the keys live
        # for the whole batch, and raw keys would hold ~1.6x the dataset's
        # coordinate memory on large GeoDataFrames of distinct shapes. A
        # 16-byte BLAKE2b digest makes key memory independent of geometry
        # size (collision odds ~1e-24).
        digest = hashlib.blake2b(
            (coords - mins).tobytes(), digest_size=16
        ).digest()
        key = (geom.geom_type, int(n_rings), digest)
        rep_pos = key_to_rep.get(key)
        if rep_pos is None:
            rep_indices.append(i)
            rep_pos = len(rep_indices) - 1
            key_to_rep[key] = rep_pos
            plan.append((rep_pos, np.zeros(2)))
        else:
            rep_mins = shapely.get_coordinates(geoms[rep_indices[rep_pos]]).min(axis=0)
            plan.append((rep_pos, mins - rep_mins))

    smoothed_reps = run_batch([geoms[i] for i in rep_indices])

    results: list[BaseGeometry] = []
    for rep_pos, offset in plan:
        rep_result = smoothed_reps[rep_pos]
        if offset[0] == 0 and offset[1] == 0:
            results.append(rep_result)
        else:
            results.append(shapely.transform(rep_result, lambda c, o=offset: c + o))
    return results


def _smoothify_multipolygon(
    geom: BaseGeometry,
    segment_length: float,
    smooth_iterations: int,
    merge_multipolygons: bool,
    preserve_area: bool,
    area_tolerance: float = 0.01,
) -> MultiPolygon:
    """Smooth a MultiPolygon, optionally merging adjacent polygons first.

    If merge_multipolygons is True, applies a small buffer-dissolve operation
    to join adjacent polygons before smoothing. This is useful for MultiPolygons
    derived from adjacent raster cells with the same classification."""

    if merge_multipolygons:
        geom = _join_adjacent(geom=geom, segment_length=segment_length)

    if isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)

    elif isinstance(geom, Polygon):
        polygons = [geom]

    else:
        raise ValueError(
            f"Expected output of unary_union to be MultiPolygon or Polygon, got {type(geom)}"  # noqa: E501
        )
    for polygon in polygons:
        assert isinstance(polygon, Polygon)

    smoothed = [
        _smoothify_polygon(
            geom=polygon,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            preserve_area=preserve_area,
            area_tolerance=area_tolerance,
        )
        for polygon in polygons
    ]

    # Flatten any MultiPolygons from hole subtraction
    flattened = []
    for geom_result in smoothed:
        if isinstance(geom_result, MultiPolygon):
            flattened.extend(geom_result.geoms)
        else:
            flattened.append(geom_result)

    return MultiPolygon(flattened)


def _smoothify_linearing(
    geom: LinearRing,
    segment_length: float,
    smooth_iterations: int = 3,
    preserve_area: bool = False,
) -> LinearRing:
    """Smooth a LinearRing by converting to Polygon, smoothing, then extracting the ring.

    LinearRings are closed loops that can be smoothed using polygon smoothing
    techniques, then converted back to a LinearRing. Uses default area_tolerance
    of 0.01% for area preservation when preserve_area is True."""  # noqa: E501

    smoothed_polygon = _smoothify_geometry(
        geom=Polygon(geom),
        segment_length=segment_length,
        smooth_iterations=smooth_iterations,
        preserve_area=preserve_area,
    )
    # Cast to Polygon since we passed in a Polygon
    smoothed_polygon = cast(Polygon, smoothed_polygon)
    return LinearRing(smoothed_polygon.exterior.coords)


def _extract_and_fill_holes(geom: Polygon) -> tuple[list[Polygon], Polygon]:
    """Extract interior holes from a polygon as separate polygons.

    Separates a polygon's interior rings (holes) from its exterior shell so they
    can be smoothed independently. This enables individual smoothing of holes and
    maintains hole area."""  # noqa: E501

    holes = []
    for interior in geom.interiors:
        holes.append(Polygon(interior))
    filled_polygon = Polygon(geom.exterior)
    return holes, filled_polygon


def _smoothify_polygon(
    geom: Polygon,
    segment_length: float,
    smooth_iterations: int = 3,
    preserve_area: bool = True,
    area_tolerance: float = 0.01,
) -> Polygon | MultiPolygon:
    """Smooth a Polygon while preserving interior holes.

    Smooths the exterior shell and each interior hole independently, then
    recombines them. This approach prevents artifacts at hole boundaries and
    maintains proper polygon topology. May return a MultiPolygon if hole
    subtraction splits the polygon."""

    holes, filled_polygon = _extract_and_fill_holes(geom)

    smooth_polygon = _smoothify_geometry(
        geom=filled_polygon,
        segment_length=segment_length,
        smooth_iterations=smooth_iterations,
        preserve_area=preserve_area,
        area_tolerance=area_tolerance,
    )

    # Smooth all holes and subtract them from the smoothed exterior
    if holes and isinstance(smooth_polygon, Polygon):
        smoothed_hole_polygons = []
        for hole in holes:
            smooth_hole = _smoothify_geometry(
                geom=hole,
                segment_length=segment_length,
                smooth_iterations=smooth_iterations,
                preserve_area=preserve_area,
                area_tolerance=area_tolerance,
            )
            if isinstance(smooth_hole, Polygon):
                smoothed_hole_polygons.append(smooth_hole)

        # Subtract all holes in one difference call; parts of a hole lying
        # outside the smoothed exterior are ignored by difference, so they
        # cannot add area
        if smoothed_hole_polygons:
            smooth_polygon = smooth_polygon.difference(
                unary_union(smoothed_hole_polygons)
            )
            # A smoothed hole can cross the independently smoothed exterior;
            # the difference then clips it, leaving sharp concave cusps at
            # the (often tangential) crossing points. Repair with a small
            # opening (erode-dilate: removes hair-thin material needles
            # between hole rim and boundary) followed by a closing
            # (dilate-erode: seals thin slits) — both deviate at most their
            # radius and only at sub-smoothing-scale features.
            if _max_concave_turn_degrees(smooth_polygon) > 60:
                radius = segment_length / 4
                repaired = (
                    smooth_polygon.buffer(-radius)
                    .buffer(radius)
                    .buffer(radius)
                    .buffer(-radius)
                )
                if not repaired.is_empty:
                    smooth_polygon = make_valid(repaired)

    return cast("Polygon | MultiPolygon", smooth_polygon)


def _smoothify_multilinestring(
    geom: MultiLineString,
    segment_length: float,
    smooth_iterations: int = 3,
    preserve_area: bool = False,
) -> MultiLineString:
    """Smooth every LineString within a MultiLineString independently.

    Applies smoothing to each line segment in the collection while maintaining
    the MultiLineString structure. Area preservation is not applicable to
    LineStrings and is ignored."""

    lines = [
        _smoothify_linestring(
            geom=line,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            preserve_area=preserve_area,
        )
        for line in geom.geoms
    ]
    return MultiLineString(lines)


def _smoothify_linestring(
    geom: LineString,
    segment_length: float,
    smooth_iterations: int = 3,
    preserve_area: bool = False,
) -> LineString:
    """Smooth a LineString using Chaikin's corner-cutting algorithm.

    Transforms jagged line features (e.g., road networks from classified imagery)
    into smooth curves while preserving endpoints."""

    smooth_line = _smoothify_geometry(
        geom=geom,
        segment_length=segment_length,
        smooth_iterations=smooth_iterations,
        preserve_area=preserve_area,
    )
    if not isinstance(smooth_line, LineString):
        raise ValueError(
            f"Expected output of smoothify_geometry to be LineString, got {type(smooth_line)}"  # noqa: E501
        )
    return smooth_line


def _smoothify_geodataframe(
    gdf: gpd.GeoDataFrame,
    segment_length: float,
    num_cores: int,
    smooth_iterations: int,
    merge_collection: bool,
    merge_multipolygons: bool,
    preserve_area: bool,
    area_tolerance: float = 0.01,
    merge_field: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Smooth all geometries in a GeoDataFrame with optional parallel processing.

    Processes each geometry in a GeoDataFrame using Chaikin corner cutting.
    Optionally merges adjacent features before smoothing and supports parallel
    execution for large datasets."""  # noqa: E501

    modified_gdf = gdf.copy()

    # Screen out invalid geometries up front. Smoothing can't process them and
    # the merge/dissolve step below would otherwise silently repair them --
    # both inconsistent with the single-geometry path. They are set aside and
    # appended back unchanged so the result still covers every input feature.
    valid_mask = modified_gdf.geometry.is_valid | modified_gdf.geometry.is_empty
    invalid_gdf = modified_gdf[~valid_mask].copy()
    if len(invalid_gdf) > 0:
        _warn_invalid(len(invalid_gdf))
        modified_gdf = modified_gdf[valid_mask].copy()

    if merge_collection:
        # Only merge polygons, not linestrings
        # Separate polygons from other geometry types
        polygon_mask = modified_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])

        if polygon_mask.any():
            # Process polygons
            polygon_gdf = modified_gdf[polygon_mask].copy()
            # Mitre join keeps corners as single vertices (round joins add ~8
            # vertices per corner, inflating the dissolve union for no visible
            # gain at this tiny buffer distance)
            polygon_gdf.geometry = polygon_gdf.buffer(
                segment_length / 1000, join_style="mitre"
            )
            polygon_gdf = polygon_gdf.dissolve(by=merge_field)
            if merge_field is not None:
                polygon_gdf = polygon_gdf.reset_index()
            polygon_gdf = polygon_gdf.explode(index_parts=False, ignore_index=True)

            # Combine with non-polygon geometries
            if not polygon_mask.all():
                other_gdf = modified_gdf[~polygon_mask].copy()
                modified_gdf = gpd.GeoDataFrame(
                    pd.concat([polygon_gdf, other_gdf], ignore_index=True), crs=gdf.crs
                )
            else:
                modified_gdf = polygon_gdf
        # If no polygons, leave modified_gdf as is

    smoothify_partial = partial(
        _smoothify_single,
        segment_length=segment_length,
        smooth_iterations=smooth_iterations,
        merge_multipolygons=merge_multipolygons,
        preserve_area=preserve_area,
        area_tolerance=area_tolerance,
    )

    def run_batch(geoms: list[BaseGeometry]) -> list[BaseGeometry]:
        if num_cores == 1:
            return [smoothify_partial(g) for g in geoms]
        return cast(
            "list[BaseGeometry]",
            Parallel(n_jobs=num_cores)(delayed(smoothify_partial)(g) for g in geoms),
        )

    modified_gdf.geometry = _smooth_deduplicated(
        list(modified_gdf.geometry), run_batch
    )

    # Re-attach any invalid geometries that were set aside, unchanged, so the
    # output still has a row for every input feature.
    if len(invalid_gdf) > 0:
        modified_gdf = gpd.GeoDataFrame(
            pd.concat([modified_gdf, invalid_gdf], ignore_index=True),
            crs=gdf.crs,
        )

    return modified_gdf


def _smoothify_single(
    geom: BaseGeometry,
    segment_length: float,
    smooth_iterations: int,
    merge_multipolygons: bool,
    preserve_area: bool,
    area_tolerance: float = 0.01,
) -> BaseGeometry:
    """Smooth a single geometry by dispatching to the appropriate type-specific function.

    Central dispatcher that routes geometries to specialized smoothing functions
    based on their type. Handles all supported geometry types including simple
    and multi-part geometries.

    Invalid geometries (e.g. self-intersecting polygons) are not smoothed:
    smoothing relies on segmentize/simplify/union behaving predictably, which
    they do not for invalid input. Such geometries are returned unchanged with
    a warning so batch jobs keep running and output stays aligned 1:1 with
    input."""  # noqa: E501
    if geom.is_empty:
        return geom

    if not geom.is_valid:
        warnings.warn(
            f"Skipping invalid {geom.geom_type} (e.g. self-intersecting); "
            "returning it unchanged. Repair it first with shapely's "
            "make_valid() (or geopandas GeoSeries.make_valid()) and smooth "
            "again.",
            stacklevel=2,
        )
        return geom

    if isinstance(geom, Polygon):
        return _smoothify_polygon(
            geom=geom,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            preserve_area=preserve_area,
            area_tolerance=area_tolerance,
        )
    elif isinstance(geom, LinearRing):
        return _smoothify_linearing(
            geom=geom,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            preserve_area=False,
        )
    elif isinstance(geom, LineString):
        return _smoothify_linestring(
            geom=geom,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            preserve_area=False,
        )
    elif isinstance(geom, MultiPolygon):
        return _smoothify_multipolygon(
            geom=geom,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            merge_multipolygons=merge_multipolygons,
            preserve_area=preserve_area,
            area_tolerance=area_tolerance,
        )
    elif isinstance(geom, MultiLineString):
        return _smoothify_multilinestring(
            geom=geom,
            segment_length=segment_length,
            smooth_iterations=smooth_iterations,
            preserve_area=False,
        )
    return geom


def _smoothify_bulk(
    geom: GeometryCollection | MultiPolygon | MultiLineString,
    segment_length: float,
    num_cores: int,
    smooth_iterations: int,
    merge_collection: bool,
    merge_multipolygons: bool,
    preserve_area: bool,
    area_tolerance: float = 0.01,
) -> GeometryCollection | MultiPolygon | MultiLineString:
    """Smooth a collection of geometries using parallel processing.

    Processes GeometryCollection, MultiPolygon, or MultiLineString by distributing
    component geometries across multiple worker processes for efficient parallel
    execution on large datasets.

    Optionally merges adjacent geometries before smoothing, which is useful for
    collections of polygons derived from adjacent raster cells with the same
    classification."""

    input_type = type(geom)

    # Screen out invalid geometries up front: smoothing can't process them, and
    # the merge step below would otherwise silently repair them (inconsistent
    # with the single-geometry path). Set aside and appended back unchanged.
    valid_geoms, invalid_geoms = _partition_valid(list(geom.geoms))
    if invalid_geoms:
        _warn_invalid(len(invalid_geoms))
        geom = GeometryCollection(valid_geoms)

    # join adjacent polygons found in a geometry collection
    if merge_collection and isinstance(geom, GeometryCollection):
        geom_joined = _join_adjacent(geom=geom, segment_length=segment_length)
        geom = GeometryCollection(geom_joined)

    # join adjacent polygons found within a multipolygon
    if merge_multipolygons and isinstance(geom, MultiPolygon):
        geom_joined = _join_adjacent(geom=geom, segment_length=segment_length)
        geom = GeometryCollection(geom_joined)

    if isinstance(geom, MultiLineString):
        geom = GeometryCollection(geom.geoms)

    smoothify_partial = partial(
        _smoothify_single,
        segment_length=segment_length,
        smooth_iterations=smooth_iterations,
        merge_multipolygons=merge_multipolygons,
        preserve_area=preserve_area,
        area_tolerance=area_tolerance,
    )

    def run_batch(geoms: list[BaseGeometry]) -> list[BaseGeometry]:
        if num_cores == 1:
            return [smoothify_partial(g) for g in geoms]
        # Use spawn to avoid fork warnings in multi-threaded environments
        # (Python 3.13+)
        ctx = get_context("spawn")
        with ctx.Pool(num_cores) as pool:
            return pool.map(smoothify_partial, geoms)

    geom_smoothed = _smooth_deduplicated(list(geom.geoms), run_batch)

    # Re-attach untouched invalid geometries so every input is represented.
    geom_smoothed = list(geom_smoothed) + invalid_geoms

    if input_type == MultiPolygon:
        if geom_smoothed and all(isinstance(g, Polygon) for g in geom_smoothed):
            return MultiPolygon([g for g in geom_smoothed if isinstance(g, Polygon)])
    elif input_type == MultiLineString:
        if geom_smoothed and all(isinstance(g, LineString) for g in geom_smoothed):
            return MultiLineString(
                [g for g in geom_smoothed if isinstance(g, LineString)]
            )

    return GeometryCollection(geom_smoothed)


def _auto_detect_segment_length(
    geom: BaseGeometry | Sequence[BaseGeometry] | gpd.GeoDataFrame,
) -> float:
    """Auto-detect segment length from the minimum segment length of the geometry.

    For pixelated/rasterized geometries, finds the shortest segment which typically
    represents the true pixel size. Polygonization often skips vertices along straight
    edges, but corners retain the minimum segment length equal to the pixel size.

    Samples up to 100 segments per geometry for efficiency on large datasets.

    Args:
        geom: Geometry, Sequence of geometries, GeoDataFrame, or collection to analyze.

    Returns:
        float: Estimated segment length based on minimum segment length.

    Raises:
        ValueError: If no suitable geometry found for auto-detection.
    """

    def get_min_segment_length(
        geom: BaseGeometry, max_samples: int = 100
    ) -> Optional[float]:
        """Get the minimum segment length from a geometry.

        Samples up to max_samples segments to find the shortest one.
        For pixelated geometries, the minimum segment represents the pixel size.
        """
        if geom.is_empty:
            return None

        if isinstance(geom, Point):
            return None

        min_length = None

        def compute_min_segment_from_coords(
            coords_list: list[tuple[float, ...]],
        ) -> Optional[float]:
            """Vectorized computation of minimum segment length from coordinates."""
            coords_array = np.array(coords_list)
            if len(coords_array) < 2:
                return None

            # Vectorized distance calculation
            diffs = coords_array[1:] - coords_array[:-1]
            distances = np.linalg.norm(diffs, axis=1)

            # Filter out zero-length segments
            valid_distances = distances[distances > 0]
            if len(valid_distances) == 0:
                return None

            # Sample if too many segments
            if len(valid_distances) > max_samples:
                step = len(valid_distances) // max_samples
                valid_distances = valid_distances[::step]

            return float(np.min(valid_distances))

        if isinstance(geom, Polygon):
            # Check exterior ring
            coords = list(geom.exterior.coords)
            length = compute_min_segment_from_coords(coords)
            if length is not None:
                min_length = length

            # Check interior rings (holes)
            for interior in geom.interiors:
                coords = list(interior.coords)
                length = compute_min_segment_from_coords(coords)
                if length is not None:
                    if min_length is None or length < min_length:
                        min_length = length

        elif isinstance(geom, (LineString, LinearRing)):
            coords = list(geom.coords)
            min_length = compute_min_segment_from_coords(coords)

        elif isinstance(geom, (MultiPolygon, MultiLineString)):
            for sub_geom in geom.geoms:
                length = get_min_segment_length(sub_geom, max_samples)
                if length is not None:
                    if min_length is None or length < min_length:
                        min_length = length

        elif isinstance(geom, GeometryCollection):
            for sub_geom in geom.geoms:
                length = get_min_segment_length(sub_geom, max_samples)
                if length is not None:
                    if min_length is None or length < min_length:
                        min_length = length

        return min_length

    # Handle GeoDataFrame
    min_segment_length = None
    if isinstance(geom, gpd.GeoDataFrame):
        # Sample first few geometries to find minimum
        for geometry in geom.geometry.head(10):
            length = get_min_segment_length(geometry)
            if length is not None:
                if min_segment_length is None or length < min_segment_length:
                    min_segment_length = length
    elif isinstance(geom, Sequence) and not isinstance(geom, (str, bytes)):
        # Handle sequence of geometries
        for i, geometry in enumerate(geom):
            if i >= 10:  # Sample first 10 geometries
                break
            if isinstance(geometry, BaseGeometry):
                length = get_min_segment_length(geometry)
                if length is not None:
                    if min_segment_length is None or length < min_segment_length:
                        min_segment_length = length
    else:
        # Handle single geometry or collection
        min_segment_length = get_min_segment_length(cast(BaseGeometry, geom))

    if min_segment_length is not None:
        return min_segment_length

    raise ValueError(
        "Could not auto-detect segment_length: no suitable geometry with coordinates found. "  # noqa: E501
        "Please provide segment_length explicitly."
    )
