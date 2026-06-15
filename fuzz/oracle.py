"""Invariants every smoothed polygon must satisfy.

The oracle is the part that actually finds bugs: a generator only matters in so
far as it produces shapes that make an invariant fail. Each check returns zero
or more :class:`Defect`s; an empty list means the output is acceptable.

These mirror the contract the real-world sweep and the existing blob fuzz test
enforce, so a defect here is a defect there too.
"""

from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import explain_validity

from smoothify.smoothify_core import _max_concave_turn_degrees

# A correctly smoothed boundary is gently curved on the concave side; a sharp
# inward turn means a fold/slit. Matches the threshold the blob fuzz test uses.
MAX_CONCAVE_TURN_DEG = 100.0
# Fractional area change allowed (preserve_area targets ~0.01%, so 1% is a very
# loose ceiling that only trips on real failures).
AREA_TOLERANCE = 0.01


@dataclass
class Defect:
    """A single violated invariant."""

    invariant: str
    detail: str


def check_polygon(
    original: Polygon,
    smoothed,
    *,
    min_feature_area: float = 0.0,
) -> list[Defect]:
    """Screen one smoothed polygon against all invariants.

    ``original`` must be a valid single Polygon (invalid input is returned
    unchanged by the pipeline, so feeding it here would test nothing).
    ``min_feature_area`` is the area below which a vanished hole is considered
    legitimate rounding rather than a dropped feature (use ~pixel**2)."""
    defects: list[Defect] = []

    if smoothed.is_empty:
        defects.append(Defect("empty", "smoothed output is empty"))
        return defects

    if not smoothed.is_valid:
        defects.append(Defect("invalid", explain_validity(smoothed)))

    # A single input polygon should never fragment into multiple parts.
    if isinstance(smoothed, MultiPolygon):
        defects.append(
            Defect(
                "split",
                f"single polygon became MultiPolygon ({len(smoothed.geoms)} parts)",
            )
        )

    turn = _max_concave_turn_degrees(smoothed)
    if turn > MAX_CONCAVE_TURN_DEG:
        defects.append(Defect("fold", f"{turn:.1f}deg concave turn"))

    if original.area > 0:
        err = abs(smoothed.area - original.area) / original.area
        if err > AREA_TOLERANCE:
            defects.append(Defect("area", f"{err * 100:.3f}% area change"))

    # Hole survival: large holes in the input must not vanish entirely. Small
    # holes (below min_feature_area) may legitimately round away.
    big_holes_in = sum(
        1 for r in original.interiors if Polygon(r).area > max(min_feature_area, 0)
    )
    if big_holes_in:
        holes_out = (
            len(smoothed.interiors)
            if isinstance(smoothed, Polygon)
            else sum(len(p.interiors) for p in smoothed.geoms)
        )
        if holes_out == 0:
            defects.append(
                Defect(
                    "hole_lost", f"{big_holes_in} significant hole(s) -> 0 holes out"
                )
            )

    return defects
