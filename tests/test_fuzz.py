"""Bounded, seeded fuzz over raster-noise geometries (CI gate).

The primary gate is: no generated polygon may make ``smoothify()`` raise. A
separate, looser check screens invariants (folds, lost holes, area drift) at
the matched segment length only.

This is deliberately lean so it stays fast in CI — a small seed range and a
representative segment subset (including one extreme). The exhaustive sweep
(every generator, the full segment range, the collection path, tens of
thousands of seeds) lives in ``scripts/fuzz_run.py``.
"""

import pytest

from fuzz import generate_case, run_seed
from fuzz.generators import PIXEL

# Rotates through all generator styles a few times each; kept small for speed.
SEEDS = range(48)

# Representative subset of the full sweep: auto-detect, below/at/above the pixel
# size, and one extreme. The script covers the rest.
CI_SEGMENTS = [None, 0.5 * PIXEL, PIXEL, 2.0 * PIXEL, 50.0 * PIXEL]


@pytest.mark.parametrize("seed", SEEDS)
def test_no_input_raises_across_segment_sweep(seed):
    """No generated geometry should crash the pipeline at any segment length,
    whether smoothed individually or as a merged collection."""
    errors = [
        f
        for f in run_seed(seed, segment_lengths=CI_SEGMENTS, check_invariants=False)
        if f.kind == "error"
    ]
    assert not errors, "\n".join(f"{f.summary()}\n{f.traceback}" for f in errors)


@pytest.mark.parametrize("seed", SEEDS)
def test_invariants_hold_at_matched_segment_length(seed):
    """Softer check: at segment_length == pixel size, outputs stay well-formed.

    Kept separate from the error gate because folds on thin features are a known
    open issue surfaced by this harness; this asserts the matched-resolution
    case that the real-world sweep also relies on."""
    failures = run_seed(seed, segment_lengths=[PIXEL], include_collection=False)
    defects = [f for f in failures if f.kind == "defect"]
    assert not defects, "\n".join(f.summary() for f in defects)


def test_generators_produce_geometry():
    """Guard against a degenerate generator silently emitting nothing, which
    would make the fuzz pass vacuously."""
    produced = sum(len(generate_case(s).polygons) for s in SEEDS)
    assert produced > 0
