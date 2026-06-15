"""Raster-noise geometry generators.

Each generator turns a seeded random raster into pixel-aligned polygons the way
real raster-to-vector conversion does: build a noise field, threshold it into
classes, and dissolve the boxes of each class into polygons. Holes appear for
free wherever one class fully encloses cells of another, so no special-casing
is needed to exercise the hole-handling paths.

A generator returns ``(polygons, params)`` where ``params`` is a JSON-friendly
dict recording exactly how the case was built, so any failure is reproducible
from its seed alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.ndimage import gaussian_filter
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.ops import unary_union

# Map-unit size of one raster cell. Fixed per case but recorded in params; the
# pipeline's behaviour is scale-relative so a single value exercises the logic.
PIXEL = 9.0


@dataclass
class FuzzCase:
    """One generated batch of polygons plus everything needed to rebuild it."""

    seed: int
    generator: str
    pixel: float
    params: dict
    polygons: list[Polygon] = field(default_factory=list)


def _polygonize_mask(mask: "np.ndarray", pixel: float) -> list[Polygon]:
    """Dissolve the true cells of a boolean grid into pixel-aligned polygons.

    Cells of other classes enclosed by this mask become interior rings (holes)
    automatically, since they are simply absent from the union."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    boxes = [
        box(x * pixel, y * pixel, (x + 1) * pixel, (y + 1) * pixel)
        for y, x in zip(xs, ys, strict=True)
    ]
    merged = unary_union(boxes)
    parts = merged.geoms if isinstance(merged, MultiPolygon) else [merged]
    return [p for p in parts if isinstance(p, Polygon) and not p.is_empty]


def _normalized_field(rng: "np.random.Generator", shape, sigma: float) -> "np.ndarray":
    """Spatially-correlated noise in [0, 1] (Gaussian-blurred white noise).

    sigma ~ 0 gives speckle; larger sigma gives organic, coastline-like blobs —
    the spectrum of shapes real landcover rasters produce."""
    field = gaussian_filter(rng.random(shape), sigma=sigma, mode="reflect")
    lo, hi = float(field.min()), float(field.max())
    if hi - lo < 1e-12:
        return np.zeros(shape)
    return (field - lo) / (hi - lo)


# --- generators ------------------------------------------------------------
# Each takes (rng) and returns (polygons, params).


def white_noise_blobs(rng: "np.random.Generator"):
    """Uncorrelated white noise -> binary threshold. Lots of tiny speckle and
    diagonally-touching cells: the pathological, jagged extreme."""
    size = int(rng.integers(8, 22))
    density = float(rng.uniform(0.3, 0.6))
    grid = rng.random((size, size)) < density
    polys = _polygonize_mask(grid, PIXEL)
    return polys, {"size": size, "density": round(density, 4)}


def blurred_noise_classes(rng: "np.random.Generator"):
    """Correlated noise thresholded into N classes by quantile. Produces
    adjacent multi-class polygons sharing edges, plus enclosed-class holes."""
    size = int(rng.integers(16, 44))
    sigma = float(rng.uniform(1.0, 5.0))
    n_classes = int(rng.integers(2, 6))
    f = _normalized_field(rng, (size, size), sigma)
    edges = np.quantile(f, np.linspace(0.0, 1.0, n_classes + 1))
    labels = np.digitize(f, edges[1:-1])
    polys: list[Polygon] = []
    for k in range(n_classes):
        polys.extend(_polygonize_mask(labels == k, PIXEL))
    return polys, {
        "size": size,
        "sigma": round(sigma, 4),
        "n_classes": n_classes,
    }


def thin_channels(rng: "np.random.Generator"):
    """A narrow quantile band of correlated noise -> long thin arms one or two
    cells wide. These are what trigger the simplify-near-tolerance forks and
    the concave-seal branch in the core."""
    size = int(rng.integers(20, 48))
    sigma = float(rng.uniform(1.5, 4.0))
    centre = float(rng.uniform(0.35, 0.65))
    half_width = float(rng.uniform(0.04, 0.1))
    f = _normalized_field(rng, (size, size), sigma)
    mask = np.abs(f - centre) < half_width
    polys = _polygonize_mask(mask, PIXEL)
    return polys, {
        "size": size,
        "sigma": round(sigma, 4),
        "centre": round(centre, 4),
        "half_width": round(half_width, 4),
    }


def ring_features(rng: "np.random.Generator"):
    """Concentric-ish bands from |field - level| -> shells with large holes,
    stressing area preservation on interiors and hole survival."""
    size = int(rng.integers(20, 48))
    sigma = float(rng.uniform(2.0, 5.0))
    level = float(rng.uniform(0.4, 0.6))
    band = float(rng.uniform(0.08, 0.18))
    f = _normalized_field(rng, (size, size), sigma)
    mask = np.abs(f - level) > band  # everything except a thin ring -> holes
    polys = _polygonize_mask(mask, PIXEL)
    return polys, {
        "size": size,
        "sigma": round(sigma, 4),
        "level": round(level, 4),
        "band": round(band, 4),
    }


def dense_many_classes(rng: "np.random.Generator"):
    """Lightly-blurred noise cut into many (6-12) quantile classes on a large
    grid. Maximises shared multi-class edges and small enclosed features —
    stresses adjacency merging and produces lots of tiny holed polygons."""
    size = int(rng.integers(36, 64))
    sigma = float(rng.uniform(0.6, 2.0))
    n_classes = int(rng.integers(6, 13))
    f = _normalized_field(rng, (size, size), sigma)
    edges = np.quantile(f, np.linspace(0.0, 1.0, n_classes + 1))
    labels = np.digitize(f, edges[1:-1])
    polys: list[Polygon] = []
    for k in range(n_classes):
        polys.extend(_polygonize_mask(labels == k, PIXEL))
    return polys, {"size": size, "sigma": round(sigma, 4), "n_classes": n_classes}


def swiss_cheese(rng: "np.random.Generator"):
    """A mostly-solid region peppered with single-cell holes (white speckle
    punched out of a filled block). Many interior rings of pixel scale — the
    hole-extraction, hole-merging, and per-hole area-preservation paths, and
    where large segment lengths make holes collapse."""
    size = int(rng.integers(20, 40))
    hole_density = float(rng.uniform(0.1, 0.35))
    grid = rng.random((size, size)) >= hole_density  # True = solid
    # Keep a solid border so the holes stay interior rings, not edge notches.
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
    polys = _polygonize_mask(grid, PIXEL)
    return polys, {"size": size, "hole_density": round(hole_density, 4)}


GENERATORS: list[Callable] = [
    white_noise_blobs,
    blurred_noise_classes,
    thin_channels,
    ring_features,
    dense_many_classes,
    swiss_cheese,
]


def generate_case(seed: int) -> FuzzCase:
    """Build a deterministic case from a seed.

    The generator is chosen by the seed so coverage rotates across all styles
    as the seed range grows, and every case is fully reproducible."""
    rng = np.random.default_rng(seed)
    gen = GENERATORS[seed % len(GENERATORS)]
    polys, params = gen(rng)
    return FuzzCase(
        seed=seed,
        generator=gen.__name__,
        pixel=PIXEL,
        params=params,
        polygons=polys,
    )
