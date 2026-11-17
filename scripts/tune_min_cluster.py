"""Is the degradation at 73,864 the data, or the cluster-size rule?

``min_cluster_size`` defaults to ``round(sqrt(n) / 2)``, which was chosen at
12,804 compounds where it gives 57. At 73,864 the same rule asks for 136, and
HDBSCAN answers by calling a great deal of the map noise -- the target domain
went from 6.2% noise and 58 clusters to 44.1% and 132, and the moiety domain
collapsed from 34 clusters to 14.

That could be two different things:

  * the expansion genuinely blurred the structure, or
  * the rule outgrew its tuning and is now demanding clusters larger than the
    real groups.

They are told apart by holding the data fixed and varying only the rule. This
re-clusters the *already embedded* coordinates, so each row costs an HDBSCAN
fit rather than another UMAP.

    uv run python scripts/tune_min_cluster.py
    uv run python scripts/tune_min_cluster.py --domains target moiety
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd

import chemspace
from chemspace.domains import default_registry
from chemspace.placement import NOISE, UNCOVERED, ClusterConfig, _cluster, place
from chemspace.quality import _external_agreement


def atc_labels(ds, compound_ids) -> np.ndarray:
    """WHO ATC main group, for compounds carrying exactly one.

    Same rule as ``scripts/cluster_quality.py``; repeated rather than imported
    so this script does not depend on how that one is run.
    """
    col = ds.compounds["atc_level1"].reindex(list(compound_ids))
    out = []
    for v in col:
        v = "" if v is None or v != v else str(v)
        out.append("" if (not v or ";" in v) else v)
    return np.array(out, dtype=object)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument(
        "--snapshot",
        default=None,
        help="measure a specific snapshot dir instead of the newest; the rule "
        "has to hold at both sizes, so it is worth running on each",
    )
    ap.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        default=[20, 40, 60, 90, 136],
        help="min_cluster_size values to try (default spans sqrt(n)/10 .. sqrt(n)/2)",
    )
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s",
                        stream=sys.stdout)

    ds = chemspace.load(snapshot=args.snapshot) if args.snapshot else chemspace.load()
    registry = default_registry(ds)
    names = args.domains or list(registry.names)
    ids = tuple(ds.compounds.index)
    external = atc_labels(ds, ids)
    auto = max(5, round(np.sqrt(len(ids)) / 2))
    print(f"{ds} · 현재 규칙 min_cluster_size = {auto}\n")

    rows = []
    for name in names:
        domain = registry[name]
        feats = domain.featurize(ids)
        # Cached from the coherence run, so this is a disk read, not a UMAP.
        base = place(feats)
        fit = base.labels != UNCOVERED
        emb = base.coords[fit]
        X = feats.X[fit]
        ext = np.asarray(external)[fit]

        for mcs in args.sizes:
            t0 = time.perf_counter()
            lab, method = _cluster(
                emb, X, feats.metric, ClusterConfig(min_cluster_size=mcs)
            )
            scored = lab != NOISE
            ami, purity, labelled = _external_agreement(lab[scored], ext[scored])
            rows.append(
                {
                    "domain": domain.label,
                    "min_size": mcs,
                    "clusters": int(len({int(v) for v in lab} - {NOISE})),
                    "noise": f"{float((lab == NOISE).mean()):.1%}",
                    "ami_atc": None if ami is None else round(ami, 3),
                    "purity": None if purity is None else f"{purity:.1%}",
                    "labelled": labelled,
                    "sec": round(time.perf_counter() - t0, 1),
                    "auto": "←현재" if mcs == auto else "",
                }
            )
            print(f"  {domain.label:6} {method:28} {rows[-1]['clusters']:>4}군집 "
                  f"noise {rows[-1]['noise']:>6} AMI {rows[-1]['ami_atc']}", flush=True)

    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "\n군집 수가 늘고 noise가 줄며 AMI가 함께 오르면 규칙이 과했던 것이고,"
        "\nAMI가 어느 값에서도 오르지 않으면 확장이 실제로 구조를 흐린 것입니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
