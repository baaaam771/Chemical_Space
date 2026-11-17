"""The B rule came out at 548,589 -- bigger than "all human targets". Why?

Because restricting to the drug layer's own targets removes almost nothing:
those 2,898 proteins *are* the well-studied druggable proteome, and ChEMBL's
measured potencies are concentrated on exactly them. An anchor that keeps
everything is not an anchor.

So this measures rules that actually bound the result, and prints them beside
each other. Two families:

``potency``
    Raise the cutoff. Cheap, but it does not spread: the compounds that reach
    1 nM are the ones on the most heavily optimised targets, so the map gets
    deeper on kinases rather than wider across biology.

``per-target cap``
    Keep only the strongest N compounds for each target. Bounded by
    construction -- at most ``targets x N`` -- and it spends the budget evenly
    across targets instead of letting the most-screened ones take it all.

    uv run python scripts/count_variants.py --db <chembl_37.db>
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time

import pandas as pd

from chemspace.data.chembl import snapshot_dir
from chemspace.data.dataset import PROTEIN_TARGET_TYPES

PAIRS = """
    SELECT md.chembl_id AS cid, t.chembl_id AS tid, MAX(act.pchembl_value) AS p
      FROM activities act
      JOIN assays a               ON a.assay_id = act.assay_id
      JOIN target_dictionary t    ON t.tid = a.tid
      JOIN {anchor} an            ON an.chembl_id = t.chembl_id
      JOIN molecule_dictionary md ON md.molregno = act.molregno
      JOIN compound_structures cs ON cs.molregno = md.molregno
     WHERE act.pchembl_value >= ?
       AND cs.canonical_smiles IS NOT NULL
     GROUP BY md.chembl_id, t.chembl_id
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--cache-dir", default="data/cache")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    snap = snapshot_dir(args.cache_dir)
    tg = pd.read_parquet(snap / "targets.parquet")
    prot = tg[tg["target_type"].isin(PROTEIN_TARGET_TYPES)]
    anchor_ids = set(prot["target_chembl_id"].dropna().astype(str))
    mech = pd.read_parquet(snap / "mechanisms.parquet")
    mech_ids = set(mech["target_chembl_id"].dropna().astype(str)) & anchor_ids
    current = set(pd.read_parquet(snap / "molecules.parquet")["chembl_id"].astype(str))

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.execute("PRAGMA temp_store = MEMORY")
    con.execute("PRAGMA cache_size = -2000000")
    for name, ids in (("anchor", anchor_ids), ("mechtgt", mech_ids)):
        con.execute(f"CREATE TEMP TABLE {name} (chembl_id TEXT PRIMARY KEY)")
        con.executemany(f"INSERT OR IGNORE INTO {name} VALUES (?)", [(i,) for i in ids])

    print(f"기준 표적: 활성·기전 전체 단백질 {len(anchor_ids):,} · "
          f"기전(큐레이션)만 {len(mech_ids):,}\n")
    print(f"{'규칙':44} {'화합물':>10} {'합집합':>10}  {'시간':>7}")
    print("-" * 78)

    rows = []

    def run(label: str, sql: str, params: tuple) -> None:
        t0 = time.perf_counter()
        got = {r[0] for r in con.execute(sql, params)}
        dt = time.perf_counter() - t0
        union = len(got | current)
        print(f"{label:44} {len(got):>10,} {union:>10,}  {dt:>6.0f}s", flush=True)
        rows.append((label, len(got), union))

    for anchor, tag in (("anchor", "전체 단백질 표적"), ("mechtgt", "기전 표적만")):
        for cut in (7.0, 8.0, 9.0):
            run(
                f"{tag} · pchembl>={cut:.0f}",
                f"SELECT DISTINCT cid FROM ({PAIRS.format(anchor=anchor)})",
                (cut,),
            )

    for anchor, tag in (("anchor", "전체 단백질 표적"), ("mechtgt", "기전 표적만")):
        for cap in (25, 50, 100):
            run(
                f"{tag} · pchembl>=7 · 표적당 상위 {cap}",
                f"""
                WITH pairs AS ({PAIRS.format(anchor=anchor)}),
                     ranked AS (
                         SELECT cid, ROW_NUMBER() OVER (
                                    PARTITION BY tid ORDER BY p DESC, cid
                                ) AS rn
                           FROM pairs
                     )
                SELECT DISTINCT cid FROM ranked WHERE rn <= {cap}
                """,
                (7.0,),
            )

    print(f"\n현재 {len(current):,}개 · plotly 실용 한계 약 100,000점")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
