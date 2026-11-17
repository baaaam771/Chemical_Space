"""Are the clusters meaningful, or patterns drawn in noise?

`scripts/smoke_test.py` measures that the domains *disagree* (ARI near zero).
That is necessary and not sufficient: two random partitions also score ARI 0.
This script measures the other half -- whether each domain's clusters hold
together on their own -- and prints every score beside a null obtained by
running the same pipeline on the same matrix with each feature column
shuffled.

    uv run python scripts/cluster_quality.py
    uv run python scripts/cluster_quality.py --domains target disease
    uv run python scripts/cluster_quality.py --subset 3000   # faster

Read it as a comparison, never as an absolute. A silhouette of 0.2 means
nothing by itself; a silhouette of 0.2 where the null scores 0.02 means the
structure is real, and one where the null scores 0.19 means it is not.
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
from chemspace.placement import place
from chemspace.quality import coherence, null_coherence

#: ATC feeds the disease features, so agreement there is partly circular.
ATC_IS_CIRCULAR = {"disease"}


def without_atc(ds):
    """Disease terms with the ATC entries removed.

    The disease domain mixes MeSH indications with ATC groups, and ATC is also
    the external yardstick -- so its score is partly a reflection. Measured
    both ways on ChEMBL_37, the circularity is worth AMI 0.713 vs 0.508: real
    but not the whole story, since MeSH indications genuinely track
    therapeutic area. Reporting both is the only honest option.
    """
    stripped = {
        c: tuple(t for t in ts if not t.startswith("ATC:"))
        for c, ts in ds.disease_terms.items()
    }
    return {c: ts for c, ts in stripped.items() if ts}


def atc_labels(ds, compound_ids) -> np.ndarray:
    """WHO ATC anatomical main group, for compounds that have exactly one.

    An external yardstick only works if it is external: ATC is assigned by
    people who never saw this pipeline. Compounds in several groups are left
    unlabelled rather than forced into one.
    """
    col = ds.compounds["atc_level1"].reindex(list(compound_ids))
    out = []
    for v in col:
        v = "" if v is None or v != v else str(v)
        out.append("" if (not v or ";" in v) else v)
    return np.array(out, dtype=object)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", nargs="*", default=None, help="default: all")
    ap.add_argument("--subset", type=int, default=0,
                    help="score a random subset of this size instead of everything")
    ap.add_argument("--seed", type=int, default=0)
    # Two snapshots now coexist (12,804 and 73,864) and ``load()`` takes the
    # newest. Comparing them means asking for the older one by name, otherwise
    # the only way to score it is to move directories around.
    ap.add_argument(
        "--snapshot",
        default=None,
        help="score a specific snapshot dir instead of the newest",
    )
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s",
                        stream=sys.stdout)
    pd.set_option("display.width", 200)

    ds = chemspace.load(snapshot=args.snapshot) if args.snapshot else chemspace.load()
    registry = default_registry(ds)
    names = args.domains or list(registry.names)

    ids = tuple(ds.compounds.index)
    if args.subset and args.subset < len(ids):
        rng = np.random.default_rng(args.seed)
        ids = tuple(np.array(ids)[rng.choice(len(ids), args.subset, replace=False)])
    external = atc_labels(ds, ids)
    print(f"{ds} · scoring {len(ids):,} compounds · "
          f"ATC label on {(external != '').sum():,}\n")

    rows = []
    for name in names:
        domain = registry[name]
        t0 = time.perf_counter()
        feats = domain.featurize(ids)
        placement = place(feats)
        # Both runs get re-cut at the real run's cluster count so the
        # silhouettes are comparable; left free, the null wins simply by
        # finding fewer and larger clusters.
        k = max(len(placement.cluster_ids), 2)
        real = coherence(
            feats, placement, domain=domain.label,
            external=external, circular=name in ATC_IS_CIRCULAR,
            matched_k=k, seed=args.seed,
        )
        null = null_coherence(
            feats, domain=domain.label, external=external,
            matched_k=k, seed=args.seed,
        )
        rows += [real.row(), null.row()]

        if name == "disease":
            # Same domain with the circular part taken out of the features.
            keep = ds.disease_terms
            try:
                ds.disease_terms = without_atc(ds)
                f2 = domain.featurize(ids)
                p2 = place(f2)
                rows.append(
                    coherence(
                        f2, p2, domain=f"{domain.label} (ATC 제외)", external=external,
                        matched_k=max(len(p2.cluster_ids), 2), seed=args.seed,
                    ).row()
                )
            finally:
                ds.disease_terms = keep

        print(f"  {domain.label:6} ({name:9}) {time.perf_counter() - t0:6.1f}s", flush=True)

    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        "\nsil_coords  : separation on the drawn map -- near-circular, a floor only"
        "\nsil_features: the same clusters scored in the original feature space (k differs!)"
        "\nsil_at_k    : feature-space silhouette, both runs cut at the same k"
        "\nami_atc     : agreement with WHO's therapeutic classification"
        "\n(null)      : identical pipeline, feature columns shuffled independently"
        "\n순환        : ATC also feeds this domain's features"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
