# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- Single-core smoothing is ~3.5x faster on typical raster-derived data (benchmarked on `examples/Water.gpkg`: 6.8s → 1.9s, including the cost of the new fold-repair check below). Output differences are sub-pixel (bounded by the algorithm's own start-point noise floor); area preservation accuracy is unchanged or slightly better. The main changes:
  - Removed the reversed-direction smoothing variants for polygons: Chaikin corner cutting is direction-invariant on closed rings, so they duplicated the forward variants bit-for-bit and only inflated the variant union (output unchanged).
  - Holes are now subtracted in a single `difference` call against their union instead of one `intersection` + `difference` pair per hole (output unchanged).
  - The tiny merge/dissolve buffer now uses mitre joins, which keep corners as single vertices instead of adding ~8 arc vertices per corner (boundary differences at the millimetre scale of the buffer itself).
  - Pre-union Chaikin smoothing of the start-point variants is capped at 2 iterations; detail beyond that was erased by the post-union simplify anyway, while doubling the vertex count entering the expensive union. The final smoothing pass still runs the full `smooth_iterations`.
  - The area-preservation root finder brackets the root from one side using its linear estimate, caches evaluations, and uses a step tolerance derived from the area tolerance via the perimeter, roughly halving the number of buffer operations.
  - Congruent geometries (translated copies of the same shape, common in raster-derived data) are now smoothed once and the result translated to each occurrence; this also reduces work dispatched to parallel workers.

### Added
- New `merge_holes` option (default `True`): holes that touch or nearly touch (e.g. diagonally adjacent raster cells) are joined before smoothing, so they smooth into one coherent opening instead of separate overlapping shapes leaving a fake land bridge. Pass `merge_holes=False` for the previous per-hole behaviour. Mirrors what `merge_multipolygons` does for shells.
- `examples/merge_holes_examples.ipynb`: worked examples of `merge_holes` and its interplay with `merge_collection` (touching holes, overlapping donuts, holes split across features).
- `benchmarks/bench_water.py`: single-core timing/profiling benchmark with baseline output comparison.

### Fixed
- Fixed sharp concave folds in smoothed output on shapes with features about one `segment_length` wide (e.g. a hole with a one-pixel-wide arm, as produced by `merge_holes` joining a small hole to a larger one). The start-point smoothing variants can disagree about such features, leaving a forked slit in their union that the area-preservation shrink sharpens into a cusp. The final result is now checked for sharp concave turns and, when one is found, recomputed from the variant union sealed with a small closing (dilate-erode at `segment_length / 4`).
- Fixed sharp cusps where a smoothed hole crosses the independently smoothed exterior and gets clipped by the hole subtraction (previously up to a 180-degree fold at the tangential crossing points). When detected, the result is repaired with a small opening + closing (at `segment_length / 4`), which removes hair-thin material needles and seals thin slits without visibly moving the boundary.
- Fixed shapes with long straight edges being massively over-rounded (e.g. a large square with a small `segment_length` collapsed into a circle). The simplify steps strip all collinear vertices from straight edges, and Chaikin's corner cuts scale with segment length, so corner rounding grew with edge length instead of `segment_length`. Geometries are now re-segmentized after each simplify step, capping corner rounding at roughly `segment_length` while still smoothing raster staircase artifacts into curves. Applies to all geometry types. Note: outputs are somewhat denser (more vertices) and smoothing is ~20% slower on typical raster-derived data.

## [0.2.3] - 2026-06-02

### Added
- The package now ships inline type hints and a `py.typed` marker, so type checkers (mypy, Pyright/Pylance) pick up `smoothify()`'s signatures and overloads in downstream code.

### Changed
- Invalid geometries (e.g. self-intersecting polygons) are now returned unchanged with a warning instead of crashing with a cryptic error or silently collapsing to an empty geometry. Behaviour is consistent across single geometries, lists/collections, and GeoDataFrames (and regardless of `merge_collection`). Repair them with shapely's `make_valid()` first if you want them smoothed.

## [0.2.2] - 2026-03-24

### Fixed
- Fixed LineString smoothing: self-intersecting lines are no longer run through the polygon-only `make_valid`/`unary_union` step, which could split them into a MultiLineString at crossing points
- Preserve coordinate dimensionality during Chaikin corner cutting instead of forcing 2D output, so smoothed geometries keep their original dimensions

## [0.2.1] - 2026-02-25

### Fixed
- Fixed `TopologyException` crash on thin/elongated polygons by validating smoothed variants before union
- Fixed crash when hole subtraction splits a polygon into a MultiPolygon (e.g. tiny holes relative to segment length)

## [0.2.0] - 2026-02-25

### Fixed
- `smooth_iterations=0` now returns the original input unchanged instead of running the geometry through segmentize/simplify pipeline without smoothing

## [0.1.0] - 2025-11-25

### Added
- Initial public release
- Core smoothing functionality using Chaikin's corner-cutting algorithm
- Support for all Shapely geometry types (Polygon, LineString, MultiPolygon, etc.)
- Automatic segment length detection
- Parallel processing support
- Area preservation for polygons