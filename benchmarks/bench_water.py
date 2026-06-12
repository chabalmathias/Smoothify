"""Benchmark smoothify on examples/Water.gpkg (single core).

Usage:
    python benchmarks/bench_water.py                  # time it
    python benchmarks/bench_water.py --profile        # cProfile, print top hotspots
    python benchmarks/bench_water.py --save-baseline  # save output for comparison
    python benchmarks/bench_water.py --compare        # compare output to baseline
"""

import argparse
import cProfile
import pstats
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely

from smoothify import smoothify

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "examples" / "Water.gpkg"
BASELINE = ROOT / "benchmarks" / "baseline_output.gpkg"


def run(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return smoothify(gdf, num_cores=1)


def quality_report(result: gpd.GeoDataFrame, baseline: gpd.GeoDataFrame) -> None:
    """Compare result to baseline: per-feature symmetric difference / hausdorff."""
    if len(result) != len(baseline):
        print(f"FEATURE COUNT DIFFERS: {len(result)} vs baseline {len(baseline)}")
        return
    sym_ratios = []
    hausdorffs = []
    for g_new, g_old in zip(result.geometry, baseline.geometry, strict=True):
        if g_old.is_empty and g_new.is_empty:
            continue
        denom = g_old.area if g_old.area > 0 else 1.0
        sym_ratios.append(g_new.symmetric_difference(g_old).area / denom)
        hausdorffs.append(g_new.hausdorff_distance(g_old))
    sym = np.array(sym_ratios)
    hd = np.array(hausdorffs)
    print(f"identical features:        {(sym == 0).sum()}/{len(sym)}")
    print(f"sym-diff area ratio  mean: {sym.mean():.2e}  max: {sym.max():.2e}")
    print(f"hausdorff dist (m)   mean: {hd.mean():.3f}  max: {hd.max():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="only first N features")
    args = parser.parse_args()

    gdf = gpd.read_file(DATA)
    if args.limit:
        gdf = gdf.head(args.limit).copy()
    nverts = shapely.get_num_coordinates(gdf.geometry.values).sum()
    print(f"{len(gdf)} features, {nverts} vertices")

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        result = run(gdf)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative").print_stats(30)
        stats.sort_stats("tottime").print_stats(30)
    else:
        times = []
        result = None
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            result = run(gdf)
            times.append(time.perf_counter() - t0)
        print(f"time: best {min(times):.2f}s  " + " ".join(f"{t:.2f}" for t in times))

    assert result is not None
    if args.save_baseline:
        result.to_file(BASELINE, driver="GPKG")
        print(f"baseline saved to {BASELINE}")
    elif args.compare:
        baseline = gpd.read_file(BASELINE)
        quality_report(result, baseline)


if __name__ == "__main__":
    main()
