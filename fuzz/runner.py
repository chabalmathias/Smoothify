"""Drive generated cases through the public pipeline and collect failures.

Two kinds of failure are collected, never raised (collect-and-report):

- ``error``  -- ``smoothify()`` raised an exception. This is the primary target:
  inputs that crash the pipeline. The exception type, message, and traceback
  are captured for triage.
- ``defect`` -- it returned, but the output violated an invariant (fold, lost
  hole, area drift). Secondary; can be switched off with ``check_invariants``.

Every input is run through a *sweep* of segment lengths, since which value is
passed strongly changes the code path (and the failures, as the seg=9 vs seg=11
runs showed).
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field

from shapely import make_valid
from shapely.geometry import MultiPolygon, Polygon

from smoothify import smoothify

from .generators import generate_case
from .oracle import Defect, check_polygon

# Segment lengths to try per input, as multipliers of the case pixel size, plus
# ``None`` (auto-detect). Spans from far below the pixel size (many vertices, no
# simplification) to far above it (whole features collapse below tolerance) —
# the extremes are where degenerate intermediate geometry is most likely.
DEFAULT_SEGMENT_MULTIPLIERS = (0.05, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 10.0, 50.0)


@dataclass
class Failure:
    """One (input polygon, segment_length) pair that errored or misbehaved."""

    seed: int
    generator: str
    pixel: float
    params: dict
    index: int  # which polygon within the case
    segment_length: float | None
    kind: str  # "error" or "defect"
    error: str = ""  # "ValueError: ..." for kind == "error"
    traceback: str = ""
    defects: list[Defect] = field(default_factory=list)
    original_wkt: str = ""

    def summary(self) -> str:
        seg = "auto" if self.segment_length is None else f"{self.segment_length:g}"
        head = f"seed={self.seed} {self.generator} #{self.index} seg={seg}"
        if self.kind == "error":
            return f"{head}: ERROR {self.error}"
        flags = ", ".join(f"{d.invariant}({d.detail})" for d in self.defects)
        return f"{head}: {flags}"


def _valid_polygons(polys: list[Polygon]) -> list[Polygon]:
    """Repair inputs to valid single polygons.

    Box unions can self-touch at a corner (diagonal adjacency), producing an
    invalid bow-tie that the pipeline would return unchanged. make_valid splits
    it at the pinch; keep the resulting polygon parts so we always feed the
    smoother valid, smoothable geometry."""
    out: list[Polygon] = []
    for p in polys:
        if p.is_empty:
            continue
        g = p if p.is_valid else make_valid(p)
        if isinstance(g, Polygon) and not g.is_empty:
            out.append(g)
        elif isinstance(g, MultiPolygon):
            out.extend(part for part in g.geoms if not part.is_empty)
    return out


def segment_sweep(pixel: float) -> list[float | None]:
    """Default per-case segment-length sweep: auto-detect plus multiples."""
    return [None, *(round(m * pixel, 6) for m in DEFAULT_SEGMENT_MULTIPLIERS)]


def run_seed(
    seed: int,
    *,
    segment_lengths: list[float | None] | None = None,
    check_invariants: bool = True,
    include_collection: bool = True,
) -> list[Failure]:
    """Generate the case for ``seed``, smooth every polygon at every segment
    length in the sweep, and return all errors (and optionally defects).

    ``segment_lengths`` defaults to :func:`segment_sweep` of the case pixel
    size; pass an explicit list (``None`` entries mean auto-detect) to override.

    With ``include_collection`` the whole case is also smoothed as one list
    (``num_cores=1`` to keep multiprocessing out of the fuzzer), exercising the
    merge/bulk path — only errors are collected there, since merging changes
    area and topology so the per-polygon oracle no longer applies.
    """
    case = generate_case(seed)
    segs = segment_lengths if segment_lengths is not None else segment_sweep(case.pixel)
    polys = _valid_polygons(case.polygons)
    failures: list[Failure] = []

    def base(i: int, seg: float | None, poly: Polygon | None) -> dict:
        return {
            "seed": seed,
            "generator": case.generator,
            "pixel": case.pixel,
            "params": case.params,
            "index": i,
            "segment_length": seg,
            "original_wkt": poly.wkt if poly is not None else "",
        }

    def record_error(i, seg, poly, exc):
        failures.append(
            Failure(
                **base(i, seg, poly),
                kind="error",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
        )

    for i, poly in enumerate(polys):
        for seg in segs:
            try:
                smoothed = smoothify(poly, segment_length=seg)
            except Exception as exc:  # noqa: BLE001 -- any crash is a finding
                record_error(i, seg, poly, exc)
                continue

            if check_invariants:
                defects = check_polygon(poly, smoothed, min_feature_area=case.pixel**2)
                if defects:
                    failures.append(
                        Failure(**base(i, seg, poly), kind="defect", defects=defects)
                    )

    # Collection/merge path: smooth all polygons of the case at once.
    if include_collection and polys:
        for seg in segs:
            try:
                smoothify(polys, segment_length=seg, num_cores=1)
            except Exception as exc:  # noqa: BLE001
                record_error(-1, seg, None, exc)
    return failures


def run_seeds(
    seeds,
    *,
    segment_lengths: list[float | None] | None = None,
    check_invariants: bool = True,
    include_collection: bool = True,
    n_jobs: int = 1,
) -> list[Failure]:
    """Run a range of seeds, optionally in parallel via joblib.

    Returns the flattened list of all failures across all seeds."""
    seeds = list(seeds)
    kw = dict(
        segment_lengths=segment_lengths,
        check_invariants=check_invariants,
        include_collection=include_collection,
    )
    if n_jobs == 1:
        results = [run_seed(s, **kw) for s in seeds]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(delayed(run_seed)(s, **kw) for s in seeds)
    return [f for batch in results for f in batch]
