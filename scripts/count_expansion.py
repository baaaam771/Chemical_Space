"""How many compounds would each expansion rule bring in?

Counting this over the REST API is not possible -- distinct-molecule counts
need the rows themselves, and there are millions. Against the SQLite release
it is a few seconds, which is the whole reason for downloading it.

    uv run python scripts/count_expansion.py --db data/chembl_dump/chembl_37/chembl_37.db

Nothing is written. This only reports, so the download can be made once and
the rule argued about afterwards.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

from chemspace.data.chembl import snapshot_dir
from chemspace.data.dataset import PROTEIN_TARGET_TYPES


def anchor_targets(snapshot: Path) -> set[str]:
    """The protein targets our current drug layer already hits.

    Cell lines are left out deliberately: they made up 35.7% of the measured
    (compound, target) pairs and would anchor the expansion on "tested against
    HeLa" rather than on biology.
    """
    tgts = pd.read_parquet(snapshot / "targets.parquet")
    keep = tgts[tgts["target_type"].isin(PROTEIN_TARGET_TYPES)]
    return set(keep["target_chembl_id"].dropna().astype(str))


def q(con: sqlite3.Connection, sql: str, params: tuple = ()) -> int:
    t0 = time.perf_counter()
    n = con.execute(sql, params).fetchone()[0]
    print(f"    {time.perf_counter() - t0:5.1f}s", flush=True)
    return int(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="path to chembl_NN.db")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--min-pchembl", type=float, default=7.0)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    snap = snapshot_dir(args.cache_dir)
    if snap is None:
        print("no snapshot yet; run scripts/fetch_chembl.py first")
        return 2
    anchors = anchor_targets(snap)
    current = set(
        pd.read_parquet(snap / "molecules.parquet")["chembl_id"].dropna().astype(str)
    )
    print(f"현재 스냅샷 {snap.name}: 화합물 {len(current):,} · 단백질 표적 {len(anchors):,}\n")

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.execute("PRAGMA temp_store = MEMORY")

    # The anchor set goes into a temp table: 4k ids in an IN clause is slower
    # than a join and hits SQLite's parameter limit.
    con.execute("CREATE TEMP TABLE anchor (chembl_id TEXT PRIMARY KEY)")
    con.executemany("INSERT OR IGNORE INTO anchor VALUES (?)", [(a,) for a in anchors])

    print(f"pchembl >= {args.min_pchembl} 기준\n")

    print("  B · 기존 표적에 붙는 화합물")
    b = q(
        con,
        """
        SELECT COUNT(DISTINCT md.chembl_id)
          FROM activities act
          JOIN assays a            ON a.assay_id = act.assay_id
          JOIN target_dictionary t ON t.tid = a.tid
          JOIN anchor an           ON an.chembl_id = t.chembl_id
          JOIN molecule_dictionary md ON md.molregno = act.molregno
          JOIN compound_structures cs ON cs.molregno = md.molregno
         WHERE act.pchembl_value >= ?
        """,
        (args.min_pchembl,),
    )

    print("  C · 사람 단백질 표적 전체")
    c = q(
        con,
        """
        SELECT COUNT(DISTINCT md.chembl_id)
          FROM activities act
          JOIN assays a            ON a.assay_id = act.assay_id
          JOIN target_dictionary t ON t.tid = a.tid
          JOIN molecule_dictionary md ON md.molregno = act.molregno
          JOIN compound_structures cs ON cs.molregno = md.molregno
         WHERE act.pchembl_value >= ?
           AND t.organism = 'Homo sapiens'
        """,
        (args.min_pchembl,),
    )

    print("  D · 구조가 있는 ChEMBL 전체")
    d = q(con, "SELECT COUNT(*) FROM compound_structures WHERE canonical_smiles IS NOT NULL")

    print(f"\n{'규칙':28} {'화합물':>12} {'현재 대비':>10}")
    for name, n in (("B 기존 표적 중심", b), ("C 사람 표적 전체", c), ("D ChEMBL 전체", d)):
        print(f"{name:28} {n:>12,} {n / len(current):>9.0f}배")

    # B is the one we are about to build, so report the union it really makes.
    rows = con.execute(
        """
        SELECT DISTINCT md.chembl_id
          FROM activities act
          JOIN assays a            ON a.assay_id = act.assay_id
          JOIN target_dictionary t ON t.tid = a.tid
          JOIN anchor an           ON an.chembl_id = t.chembl_id
          JOIN molecule_dictionary md ON md.molregno = act.molregno
          JOIN compound_structures cs ON cs.molregno = md.molregno
         WHERE act.pchembl_value >= ?
        """,
        (args.min_pchembl,),
    ).fetchall()
    found = {r[0] for r in rows}
    union = found | current
    print(
        f"\nB를 지금 집합과 합치면: {len(union):,}개"
        f"  (새로 들어오는 것 {len(union - current):,} · 이미 있던 것 {len(found & current):,})"
    )
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
