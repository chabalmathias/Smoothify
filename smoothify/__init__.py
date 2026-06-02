"""
Smoothify - Geometry Smoothing Package

A Python package for smoothing and refining geometries derived from raster data
classifications. Transforms jagged polygons and lines resulting from raster-to-vector
conversion into smooth, visually appealing features using an optimized implementation
of Chaikin's corner-cutting algorithm.

Supports:
    - Polygons (including those with holes)
    - LineStrings
    - MultiPolygons
    - MultiLineStrings
    - GeometryCollections
    - GeoDataFrames

Main function:
    smoothify() - Apply Chaikin corner-cutting smoothing to geometries
"""

try:
    from ._version import __version__
except ImportError:
    # Source checkout without a build step (e.g. running tests directly from
    # the repo). setuptools-scm writes _version.py at build time.
    __version__ = "0.0.0+unknown"

from .coordinator import smoothify

# Package-wide exports
__all__ = [
    "smoothify",
    "__version__",
]
