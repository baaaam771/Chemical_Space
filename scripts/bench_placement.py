"""Measure how placement scales with compound count -- on CPU, and on GPU if
RAPIDS cuML is installed.

Every figure past 100k in the planning notes is extrapolated from one machine.
Run this on the box you mean to deploy on and the extrapolation becomes a
measurement.

    python scripts/bench_placement.py                   # CPU path (what ships today)
    python scripts/bench_placement.py --gpu             # add the cuML paths
    python scripts/bench_placement.py --gpu --skip-cpu --n 100000 500000 1000000

This file is deliberately standalone: with --skip-cpu it imports nothing from
chemspace, so it can be copied to a GPU box on its own and run with just
numpy, scipy and cuml. The synthetic matrices are shaped like the real
domains -- physchem is 13 dense standardised columns under euclidean, while
disease/target/moiety are wide sparse binary matrices under jaccard.

Two GPU modes are timed, because they are a real choice and not a detail:

  fast   whatever is fastest for that metric, with no random_state --
         build_algo="nn_descent" for euclidean, the default build for
         jaccard (cuML 26.10 raises NotImplementedError for nn_descent on
         sparse jaccard input, so there is no choice to make there)
  repro  random_state=0 -- reproducible, documented as slower and more
         memory-hungry

The CPU path today is the "repro" one (random_state=0, hence a single thread).
Record reproducibility does not actually depend on either: it comes from
session.save(). The comparison is here so the choice gets made on numbers.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time

import numpy as np

DENSE_D = 13          # PHYSCHEM_FEATURES
SPARSE_D = 400        # a mid-sized disease vocabulary
SPARSE_RATE = 0.0075  # ~3 labels per compound
PROJECT_TO = (500_000, 2_897_819)   # option C, and all of ChEMBL


def make(n: int, kind: str, rng) -> np.ndarray:
    if kind == "dense":
        return rng.standard_normal((n, DENSE_D)).astype(np.float32)
    return (rng.random((n, SPARSE_D)) < SPARSE_RATE).astype(np.float32)


def min_cluster_size(n: int) -> int:
    """The rule placement.ClusterConfig uses."""
    return int(min(max(max(5, round(math.sqrt(n) / 2)), 2), max(n // 2, 2)))


def bench_cpu(X: np.ndarray, metric: str) -> float:
    from chemspace.domains.base import DomainFeatures
    from chemspace.placement import place

    n = X.shape[0]
    features = DomainFeatures(
        X=X,
        compound_ids=tuple(f"C{i}" for i in range(n)),
        covered=np.ones(n, bool),
        feature_names=tuple(f"f{i}" for i in range(X.shape[1])),
        metric=metric,
        kind="continuous" if metric == "euclidean" else "binary",
    )
    start = time.perf_counter()
    place(features)
    return time.perf_counter() - start


def bench_gpu(X: np.ndarray, metric: str, mode: str) -> tuple[float, float]:
    """cuML's equivalent of place(): UMAP to 2-D, then HDBSCAN on the coords.

    Two documented cuML constraints shape this, and neither costs us anything:
      * jaccard is supported for *sparse* inputs only -- and these matrices are
        sparse binary already, so CSR is where they belong anyway;
      * HDBSCAN is euclidean-only -- and we cluster on the 2-D coordinates
        rather than the feature space, which is euclidean by construction.

    Returns (seconds, peak GPU MB).
    """
    import cupy
    from cuml.cluster import HDBSCAN
    from cuml.manifold import UMAP

    n = X.shape[0]
    data = X
    if metric == "jaccard":
        from scipy.sparse import csr_matrix

        data = csr_matrix(X)

    kwargs: dict = {}
    if mode == "repro":
        kwargs["random_state"] = 0
    elif metric == "euclidean":
        kwargs["build_algo"] = "nn_descent"
    # else: fastest available for this metric is the default build with no
    # random_state. Measured on cuML 26.10: nn_descent raises
    # NotImplementedError for jaccard on sparse input, so the three Jaccard
    # domains have no nn_descent path to choose.

    cupy.get_default_memory_pool().free_all_blocks()
    free_before, _total = cupy.cuda.runtime.memGetInfo()

    # Sample free memory while the work runs. Comparing before and after does
    # not work: cuML allocates through RMM, which hands memory back to its own
    # pool when the estimator is done, so the transient peak -- the number that
    # decides whether this fits beside another job -- is gone by the time the
    # call returns. Measured the wrong way, a run that had just demanded 16 GiB
    # reported 17 MB.
    low_water = free_before
    stop = threading.Event()

    def watch() -> None:
        nonlocal low_water
        while not stop.is_set():
            free, _ = cupy.cuda.runtime.memGetInfo()
            low_water = min(low_water, free)
            stop.wait(0.02)

    sampler = threading.Thread(target=watch, daemon=True)
    sampler.start()
    try:
        start = time.perf_counter()
        emb = UMAP(
            n_components=2,
            n_neighbors=min(15, max(2, n - 1)),
            min_dist=0.1,
            metric=metric,
            **kwargs,
        ).fit_transform(data)
        HDBSCAN(min_cluster_size=min_cluster_size(n)).fit_predict(emb)
        elapsed = time.perf_counter() - start
    finally:
        stop.set()
        sampler.join(timeout=1.0)

    return elapsed, max(free_before - low_water, 0) / 1e6


def human(sec: float) -> str:
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 5400:
        return f"{sec / 60:.0f}분"
    return f"{sec / 3600:.1f}시간"


def project(points: list[tuple[int, float]], label: str) -> None:
    """Fit seconds ~ n**k over the measured points and read off the big sizes.

    The first measurement of each shape carries JIT and compile warm-up, so it
    is dropped -- leaving it in makes the curve look flatter than it is.
    """
    usable = [(n, s) for n, s in points if s > 0][1:]
    if len(usable) < 2:
        return
    xs = np.log([n for n, _ in usable])
    ys = np.log([s for _, s in usable])
    slope, intercept = np.polyfit(xs, ys, 1)
    bits = "  ".join(
        f"{t:,} -> {human(math.exp(intercept) * t ** slope)}" for t in PROJECT_TO
    )
    print(f"    {label}: seconds ~ n^{slope:.2f}   {bits}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="*",
                        default=[12_804, 25_000, 50_000, 100_000])
    parser.add_argument("--gpu", action="store_true", help="also time the cuML paths")
    parser.add_argument("--skip-cpu", action="store_true",
                        help="GPU only; also drops the chemspace import entirely")
    parser.add_argument("--modes", nargs="*", default=["fast", "repro"],
                        choices=["fast", "repro"])
    args = parser.parse_args()

    # A Korean Windows console is cp949 and cannot encode this script's output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    modes: list[str] = []
    if args.gpu:
        try:
            import cuml
            import cupy

            dev = cupy.cuda.runtime.getDeviceProperties(0)
            _free, total = cupy.cuda.runtime.memGetInfo()
            print(
                f"GPU 0: {dev['name'].decode()} · VRAM {total / 1e9:.0f} GB "
                f"· compute capability {dev['major']}.{dev['minor']} "
                f"· cuML {cuml.__version__}"
            )
            print(f"GPUs visible: {cupy.cuda.runtime.getDeviceCount()}\n")
            modes = list(args.modes)
        except Exception as exc:  # noqa: BLE001 - saying why is the whole point
            print(f"cuML unavailable ({exc}); CPU only.\n")
    if args.skip_cpu and not modes:
        print("Nothing to measure: --skip-cpu with no working GPU path.")
        return 2

    cols = ([] if args.skip_cpu else ["CPU"]) + [f"GPU/{m}" for m in modes]
    head = f"{'shape':<22}{'n':>9}" + "".join(f"{c:>13}" for c in cols)
    print(head)
    print("-" * len(head))

    rng = np.random.default_rng(0)
    for kind, metric in (("dense", "euclidean"), ("sparse", "jaccard")):
        label = f"{metric}/{DENSE_D if kind == 'dense' else SPARSE_D} {kind}"
        series: dict[str, list[tuple[int, float]]] = {c: [] for c in cols}
        peaks: dict[str, float] = {}
        for n in args.n:
            X = make(n, kind, rng)
            cells = []
            if not args.skip_cpu:
                try:
                    sec = bench_cpu(X, metric)
                    series["CPU"].append((n, sec))
                    cells.append(f"{sec:.1f}s")
                except Exception as exc:  # noqa: BLE001
                    cells.append("실패")
                    print(f"  CPU {label} n={n}: {type(exc).__name__}: {exc}")
            for mode in modes:
                key = f"GPU/{mode}"
                try:
                    sec, peak_mb = bench_gpu(X, metric, mode)
                    series[key].append((n, sec))
                    peaks[key] = max(peaks.get(key, 0.0), peak_mb)
                    cells.append(f"{sec:.1f}s")
                except Exception as exc:  # noqa: BLE001
                    cells.append("실패")
                    print(f"  {key} {label} n={n}: {type(exc).__name__}: {exc}")
            print(f"{label:<22}{n:>9,}" + "".join(f"{c:>13}" for c in cells),
                  flush=True)
        for col in cols:
            project(series[col], f"{label} · {col}")
        for col, peak_mb in peaks.items():
            print(f"    {label} · {col}: peak GPU {peak_mb:,.0f} MB")
        print()

    print("First row of each shape carries JIT/compile warm-up and is left out "
          "of the fit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
