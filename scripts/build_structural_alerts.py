"""Compute the 구조알림 (structural-alert) overlay cache for a snapshot.

구조알림 is the one xlsx-taxonomy axis that collection cannot fill -- ChEMBL
holds no PAINS/Brenk flag -- so it is computed from SMILES with RDKit's
published FilterCatalogs (PAINS, NIH, Brenk). The result is a pure function of
structure but the pass is ~5 ms/mol (minutes over a whole snapshot), so it is
cached per snapshot as ``structural_alerts.parquet`` and merged at load time.
Gitignored like the other derived caches; this script regenerates it.

    uv run python scripts/build_structural_alerts.py              # newest snapshot
    uv run python scripts/build_structural_alerts.py --snapshot chembl_37_t50
    uv run python scripts/build_structural_alerts.py --all        # every snapshot

The label per compound is the single most-severe family it matches (PAINS >
NIH > Brenk), blank when it matches none -- see ``chemspace.data.alerts``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from chemspace.data.alerts import alert_flags, label_from
from chemspace.data.chembl import resolve_data_path, snapshot_dir

OUT_NAME = "structural_alerts.parquet"


def build_one(snap: Path, log_every: int = 10_000) -> Path:
    mols = pd.read_parquet(
        snap / "molecules.parquet", columns=["chembl_id", "canonical_smiles"]
    )
    n = len(mols)
    print(f"{snap.name}: {n:,} molecules", flush=True)
    t0 = time.perf_counter()
    rows = []
    for i, (cid, smi) in enumerate(
        zip(mols["chembl_id"], mols["canonical_smiles"]), start=1
    ):
        flags = alert_flags(smi if isinstance(smi, str) else "")
        rows.append(
            {
                "chembl_id": cid,
                "pains": flags["PAINS"],
                "nih": flags["NIH"],
                "brenk": flags["BRENK"],
                "struct_alert": label_from(flags),
            }
        )
        if i % log_every == 0:
            dt = time.perf_counter() - t0
            print(f"  {i:,}/{n:,}  ({dt:.0f}s, {dt / i * 1000:.1f} ms/mol)", flush=True)

    df = pd.DataFrame(rows)
    out = snap / OUT_NAME
    df.to_parquet(out, index=False)

    hits = df["struct_alert"].value_counts()
    clean = int((df["struct_alert"] == "").sum())
    dt = time.perf_counter() - t0
    print(f"  wrote {out}  ({dt:.0f}s total)", flush=True)
    print(
        f"  PAINS {int(df['pains'].sum()):,} · NIH {int(df['nih'].sum()):,} · "
        f"Brenk {int(df['brenk'].sum()):,} · 알림없음 {clean:,} "
        f"({clean / n:.1%})",
        flush=True,
    )
    print(
        "  라벨: "
        + " · ".join(f"{k or '없음'} {v:,}" for k, v in hits.items()),
        flush=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--snapshot", default=None, help="snapshot dir name (default: newest)")
    ap.add_argument("--all", action="store_true", help="build for every cached snapshot")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    base = resolve_data_path(args.cache_dir)
    if args.all:
        snaps = sorted(d for d in base.iterdir() if (d / "molecules.parquet").exists())
    elif args.snapshot:
        snaps = [base / args.snapshot]
    else:
        s = snapshot_dir(args.cache_dir)
        if s is None:
            print("no snapshot found")
            return 2
        snaps = [s]

    for snap in snaps:
        if not (snap / "molecules.parquet").exists():
            print(f"skip {snap} (no molecules.parquet)")
            continue
        build_one(snap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
