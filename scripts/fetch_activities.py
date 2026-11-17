"""Top up an existing ChEMBL snapshot with measured activities.

The curated ``mechanism`` table is a dead end for target annotation: it covers
5,954 compounds in the whole of ChEMBL, so collecting more compounds never
widens it. ``activity`` holds ~24.5M measured potencies and is the only route
to target coverage at scale.

This script adds that layer to a snapshot that already exists, instead of
re-downloading molecules, indications and mechanisms -- which matters because
changing the potency threshold is something you will want to try more than
once.

    uv run python scripts/fetch_activities.py                    # pchembl >= 7
    uv run python scripts/fetch_activities.py --min-pchembl 6
    uv run python scripts/fetch_activities.py --force            # re-download

It rewrites ``activities.parquet`` and ``targets.parquet`` and updates the
manifest. Nothing else in the snapshot is touched.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from chemspace.data.chembl import (
    ChemblClient,
    _collect_targets,
    fetch_activities,
    resolve_data_path,
    snapshot_dir,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--snapshot", default=None, help="default: the newest one")
    ap.add_argument(
        "--min-pchembl",
        type=float,
        default=7.0,
        help="potency cutoff; 7 = 100 nM, 6 = 1 uM (default: 7)",
    )
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--force", action="store_true", help="refetch even if present")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    snap = Path(args.snapshot) if args.snapshot else snapshot_dir(args.cache_dir)
    if snap is None:
        print(f"no snapshot under {resolve_data_path(args.cache_dir)}; "
              f"run scripts/fetch_chembl.py first")
        return 2

    manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
    mols = pd.read_parquet(snap / "molecules.parquet")
    mechs = pd.read_parquet(snap / "mechanisms.parquet")
    ids = sorted(mols["chembl_id"].dropna().unique())
    keep = set(ids)
    print(f"{snap.name}: {len(ids):,} compounds · pchembl >= {args.min_pchembl}\n")

    out = snap / "activities.parquet"
    if out.exists() and not args.force:
        acts = pd.read_parquet(out)
        print(f"reusing {out.name} ({len(acts):,} rows); --force to refetch")
    else:
        client = ChemblClient()
        t0 = time.perf_counter()
        acts = fetch_activities(
            client, ids, min_pchembl=args.min_pchembl, batch_size=args.batch_size
        )
        acts = acts[acts["chembl_id"].isin(keep)]
        acts.to_parquet(out, index=False)
        print(f"\nactivities: {len(acts):,} rows in {time.perf_counter() - t0:.0f}s")

    act_compounds = set(acts["chembl_id"]) if len(acts) else set()
    mech_compounds = set(mechs["chembl_id"]) & keep

    # Targets have to be re-collected for the union: an activity-derived target
    # with no protein-class hierarchy could only ever cluster by its own name.
    target_ids = sorted(
        set(mechs["target_chembl_id"].dropna())
        | set(acts["target_chembl_id"].dropna() if len(acts) else [])
    )
    print(f"collecting metadata for {len(target_ids):,} targets "
          f"(mechanism + activity)")
    targets = _collect_targets(ChemblClient(), target_ids)
    targets.to_parquet(snap / "targets.parquet", index=False)

    manifest["min_pchembl"] = args.min_pchembl
    manifest.setdefault("counts", {}).update(
        activities=int(len(acts)),
        activity_compounds=int(len(act_compounds)),
        mechanism_compounds=int(len(mech_compounds)),
        target_compounds=int(len(mech_compounds | act_compounds)),
        targets=int(targets["target_chembl_id"].nunique()) if len(targets) else 0,
    )
    (snap / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    n = len(ids)
    both = mech_compounds | act_compounds
    print("\n표적 주석 커버리지")
    print(f"  mechanism (큐레이션)  {len(mech_compounds):>7,} / {n:,}"
          f"  ({len(mech_compounds)/n:.1%})")
    print(f"  activity  (측정값)    {len(act_compounds):>7,} / {n:,}"
          f"  ({len(act_compounds)/n:.1%})")
    print(f"  합집합                {len(both):>7,} / {n:,}"
          f"  ({len(both)/n:.1%})")
    print(f"\n-> {snap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
