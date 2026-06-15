#!/usr/bin/env python
"""Heavy, unbounded fuzz run over raster-noise geometries.

Primary target: inputs that make ``smoothify()`` raise. Every generated polygon
is run through a sweep of segment lengths; any exception is captured (type,
message, traceback) and written to disk with its reproducing seed. Invariant
violations (folds, lost holes, area drift) are reported too unless
``--errors-only`` is set.

Examples:
    # Hunt for crashes across the default segment sweep
    python scripts/fuzz_run.py --iters 20000 --jobs 8 --errors-only

    # Custom segment lengths (absolute units; "auto" = auto-detect)
    python scripts/fuzz_run.py --iters 5000 --segments auto,2,4.5,9,18,36
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Make the repo root importable so ``import fuzz`` works when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fuzz import run_seeds  # noqa: E402


def _parse_segments(spec: str | None):
    """Parse "auto,2,4.5,9" -> [None, 2.0, 4.5, 9.0]; None -> use defaults."""
    if not spec:
        return None
    out: list[float | None] = []
    for tok in spec.split(","):
        tok = tok.strip()
        out.append(None if tok.lower() == "auto" else float(tok))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=1000, help="number of seeds to run")
    ap.add_argument("--start", type=int, default=0, help="first seed")
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers (joblib)")
    ap.add_argument(
        "--segments",
        type=str,
        default=None,
        help='comma list of segment lengths, e.g. "auto,2,9,18" (default: sweep)',
    )
    ap.add_argument(
        "--errors-only",
        action="store_true",
        help="only report exceptions, skip invariant checks (faster)",
    )
    ap.add_argument(
        "--no-collection",
        action="store_true",
        help="skip smoothing each case as a merged collection (per-polygon only)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("fuzz_failures"),
        help="directory for failure artifacts",
    )
    args = ap.parse_args()

    seeds = range(args.start, args.start + args.iters)
    segments = _parse_segments(args.segments)
    print(
        f"Fuzzing seeds {args.start}..{args.start + args.iters - 1} "
        f"(jobs={args.jobs}, errors_only={args.errors_only}, "
        f"segments={'default sweep' if segments is None else segments})"
    )

    failures = run_seeds(
        seeds,
        segment_lengths=segments,
        check_invariants=not args.errors_only,
        include_collection=not args.no_collection,
        n_jobs=args.jobs,
    )

    errors = [f for f in failures if f.kind == "error"]
    defects = [f for f in failures if f.kind == "defect"]
    print(f"\n{len(errors)} error(s), {len(defects)} invariant defect(s).")

    if not failures:
        print("Clean run.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for n, f in enumerate(failures):
        seg = "auto" if f.segment_length is None else f"{f.segment_length:g}"
        stem = f"{f.kind}_seed{f.seed}_{f.generator}_{f.index}_seg{seg}_{n}"
        (args.out / f"{stem}.wkt").write_text(f.original_wkt)
        if f.kind == "error":
            (args.out / f"{stem}.traceback.txt").write_text(f.traceback)
        record = {
            "kind": f.kind,
            "seed": f.seed,
            "generator": f.generator,
            "pixel": f.pixel,
            "params": f.params,
            "index": f.index,
            "segment_length": f.segment_length,
            "error": f.error,
            "defects": [
                {"invariant": d.invariant, "detail": d.detail} for d in f.defects
            ],
        }
        (args.out / f"{stem}.json").write_text(json.dumps(record, indent=2))
        index.append(record)
    (args.out / "summary.json").write_text(json.dumps(index, indent=2))
    print(f"Artifacts written to {args.out}/")

    if errors:
        # Group crashes by exception type + message so distinct bugs are obvious.
        by_error: Counter = Counter(f.error for f in errors)
        print("\nERRORS by signature:")
        for sig, count in by_error.most_common():
            print(f"  [{count}] {sig}")
        print("\nFirst few errors (reproduce with the seed + segment):")
        for f in errors[:15]:
            print(f"  {f.summary()}")

    if defects:
        by_inv: Counter = Counter(d.invariant for f in defects for d in f.defects)
        print("\nDEFECTS by invariant:")
        for name, count in by_inv.most_common():
            print(f"  {name:12s} {count}")

    # Exit non-zero only when a crash was found; defects are advisory.
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
