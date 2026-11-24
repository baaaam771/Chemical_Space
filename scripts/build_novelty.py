"""Compute the 신규성 (chemical-novelty) overlay cache for a snapshot.

Novelty = Tanimoto (ECFP4) similarity to the nearest **approved drug**
(``max_phase>=4``): far from every marketed drug = novel space. ChEMBL has no
such field, so it is computed from SMILES and cached per snapshot
(``novelty.parquet``), merged at load as a per-compound overlay. The pass is a
fingerprint build plus one bulk-similarity sweep per compound against the
approved reference, so it is cached rather than recomputed each load.

    uv run python scripts/build_novelty.py                 # newest snapshot
    uv run python scripts/build_novelty.py --snapshot chembl_37_t50

Tiers (nearest-approved Tanimoto): 기존 ≥0.7 · 중간 0.4–0.7 · 신규 <0.4 --
see ``chemspace.data.novelty``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from chemspace.data.chembl import resolve_data_path, snapshot_dir
from chemspace.data.novelty import (
    APPROVED_MIN_PHASE,
    morgan_fps,
    nearest_similarity,
    novelty_tier,
)

OUT_NAME = "novelty.parquet"


def build_one(snap: Path) -> Path:
    mols = pd.read_parquet(
        snap / "molecules.parquet",
        columns=["chembl_id", "canonical_smiles", "max_phase"],
    )
    n = len(mols)
    phase = pd.to_numeric(mols["max_phase"], errors="coerce")
    approved = phase >= APPROVED_MIN_PHASE
    print(f"{snap.name}: {n:,} molecules · 승인약 참조 {int(approved.sum()):,}", flush=True)

    t0 = time.perf_counter()
    fps = morgan_fps(mols["canonical_smiles"])
    print(f"  fingerprints built ({time.perf_counter() - t0:.0f}s)", flush=True)
    ref_fps = [f for f, a in zip(fps, approved) if a]
    nn = nearest_similarity(fps, ref_fps)
    print(f"  nearest-approved sweep done ({time.perf_counter() - t0:.0f}s)", flush=True)

    tiers = [novelty_tier(float(s) if not np.isnan(s) else np.nan) for s in nn]
    df = pd.DataFrame({"chembl_id": mols["chembl_id"], "nn_sim": nn, "novelty": tiers})
    out = snap / OUT_NAME
    df.to_parquet(out, index=False)

    vc = df["novelty"].value_counts(dropna=False)
    valid = nn[~np.isnan(nn)]
    print(f"  wrote {out}  ({time.perf_counter() - t0:.0f}s total)", flush=True)
    if len(valid):
        print(
            f"  nn_sim  min {valid.min():.2f} · 중앙 {np.median(valid):.2f} · "
            f"평균 {valid.mean():.2f} · max {valid.max():.2f}",
            flush=True,
        )
    print("  티어: " + " · ".join(f"{k or '없음'} {v:,}" for k, v in vc.items()), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--snapshot", default=None, help="snapshot dir name (default: newest)")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    if args.snapshot:
        snap = resolve_data_path(args.cache_dir) / args.snapshot
    else:
        snap = snapshot_dir(args.cache_dir)
    if snap is None or not (snap / "molecules.parquet").exists():
        print("no snapshot found")
        return 2
    build_one(snap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
