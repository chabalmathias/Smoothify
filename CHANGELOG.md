# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
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