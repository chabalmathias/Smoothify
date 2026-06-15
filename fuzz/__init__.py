"""Geometry fuzzing harness for the smoothify pipeline.

Generates large volumes of raster-derived polygons (noise -> threshold into
N classes -> vectorise into pixel-aligned polygons) and screens every smoothed
output against a set of invariants (the oracle). Two entry points share this
core:

- ``tests/test_fuzz.py``  -- bounded, seeded, CI-safe regression gate
- ``scripts/fuzz_run.py`` -- unbounded heavy run that logs failures to disk

The generators and oracle live here so both entry points stay in lockstep.
"""

from .generators import GENERATORS, FuzzCase, generate_case
from .oracle import Defect, check_polygon
from .runner import Failure, run_seed, run_seeds

__all__ = [
    "GENERATORS",
    "FuzzCase",
    "generate_case",
    "Defect",
    "check_polygon",
    "Failure",
    "run_seed",
    "run_seeds",
]
