#!/usr/bin/env python
"""Render before/after plots for selected fuzz failures.

Reads cases by (seed, index), re-runs the pipeline, and draws the original
pixel-aligned polygon next to the smoothed output, annotating the broken
invariant. Saved to a PNG for triage.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from shapely.geometry import MultiPolygon, Polygon  # noqa: E402

from fuzz import check_polygon  # noqa: E402
from fuzz.runner import _valid_polygons  # noqa: E402
from fuzz.generators import generate_case  # noqa: E402
from smoothify import smoothify  # noqa: E402
from smoothify.smoothify_core import _max_concave_turn_degrees  # noqa: E402

# (seed, index) -> the worst folds and all hole_lost cases from the 4000-seed run
CASES = [
    (1421, 6),
    (3985, 9),
    (1421, 8),
    (3138, 1),
    (1105, 16),
    (3669, 78),
]


def _draw(ax, geom, title, color):
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for p in polys:
        if not isinstance(p, Polygon) or p.is_empty:
            continue
        xs, ys = p.exterior.xy
        ax.fill(xs, ys, facecolor=color, edgecolor="black", lw=0.8, alpha=0.5)
        for ring in p.interiors:
            hx, hy = ring.xy
            ax.fill(hx, hy, facecolor="white", edgecolor="black", lw=0.8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)


def main() -> int:
    n = len(CASES)
    fig, axes = plt.subplots(n, 2, figsize=(11, 5 * n))
    if n == 1:
        axes = [axes]

    for row, (seed, index) in enumerate(CASES):
        case = generate_case(seed)
        original = _valid_polygons(case.polygons)[index]
        smoothed = smoothify(original, segment_length=case.pixel)
        defects = check_polygon(original, smoothed, min_feature_area=case.pixel**2)
        turn = _max_concave_turn_degrees(smoothed)
        flags = ", ".join(d.invariant for d in defects)

        _draw(
            axes[row][0],
            original,
            f"seed {seed} #{index} input\n{case.generator}",
            "tab:blue",
        )
        _draw(
            axes[row][1],
            smoothed,
            f"smoothed -- {flags}\nmax concave turn {turn:.1f}deg",
            "tab:red",
        )

    fig.tight_layout()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/fuzz_worst.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
