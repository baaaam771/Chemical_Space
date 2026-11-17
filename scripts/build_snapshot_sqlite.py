"""Build a snapshot from the ChEMBL SQLite release instead of the REST API.

The REST path (``scripts/fetch_chembl.py``) is right for the drug layer: 12,804
compounds, a few thousand requests, done. It does not survive an order of
magnitude more -- the ``activity`` endpoint already answers 500 for ``__in``
batches wider than about 20, so a pull of a hundred thousand compounds' worth
of potencies is hours of requests with no way to count the result in advance.

This builds the same six tables from the local SQLite release, with the same
column names, so everything downstream (``Dataset.load``, the domains, the
cache) is unchanged.

    uv run python scripts/build_snapshot_sqlite.py --db <chembl_37.db> --count-only
    uv run python scripts/build_snapshot_sqlite.py --db <chembl_37.db> --out chembl_37_b

The compound set is the union of

  * everything already in the current snapshot (the drug layer), and
  * every compound with a measured potency at or above ``--min-pchembl``
    against a *protein* target that the drug layer already hits.

The second half is the point: it widens the map along the biology the drug
layer is already annotated for, rather than along all of ChEMBL. Cell-line
"targets" are excluded -- they were 35.7% of the measured pairs and would
anchor the expansion on what a compound was screened against rather than on
what it binds.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from chemspace.data.chembl import (
    PROPERTY_FIELDS,
    _MOL_FLAGS,
    resolve_data_path,
    snapshot_dir,
)
from chemspace.data.dataset import PROTEIN_TARGET_TYPES

log = logging.getLogger("build")

#: Tables and the columns this script reads, checked before any work starts so
#: a schema difference between releases fails with a name rather than a
#: half-written snapshot.
REQUIRED: dict[str, tuple[str, ...]] = {
    # Every flag and property the REST path carries is listed, so a release
    # that renamed one says which rather than writing a column of nulls.
    "molecule_dictionary": ("molregno", "chembl_id", "pref_name", *_MOL_FLAGS),
    "compound_structures": ("molregno", "canonical_smiles", "standard_inchi_key"),
    "compound_properties": ("molregno", *PROPERTY_FIELDS),
    "molecule_hierarchy": ("molregno", "parent_molregno"),
    "molecule_atc_classification": ("molregno", "level5"),
    "drug_indication": ("molregno", "mesh_id", "mesh_heading", "max_phase_for_ind"),
    "drug_mechanism": ("molregno", "tid", "action_type", "mechanism_of_action"),
    "activities": ("molregno", "assay_id", "pchembl_value", "standard_type"),
    "assays": ("assay_id", "tid", "assay_type"),
    "target_dictionary": ("tid", "chembl_id", "pref_name", "target_type", "organism"),
    "target_components": ("tid", "component_id"),
    "component_sequences": ("component_id", "accession"),
    "component_class": ("component_id", "protein_class_id"),
    "protein_classification": (
        "protein_class_id", "parent_id", "pref_name", "short_name", "class_level",
    ),
}


def check_schema(con: sqlite3.Connection) -> None:
    have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    missing = []
    for table, cols in REQUIRED.items():
        if table not in have:
            missing.append(table)
            continue
        present = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        for c in cols:
            if c not in present:
                missing.append(f"{table}.{c}")
    if missing:
        raise SystemExit("schema mismatch: " + ", ".join(missing))


def anchors_and_current(snap: Path) -> tuple[set[str], set[str]]:
    """The drug layer's protein targets, and its compounds."""
    tgts = pd.read_parquet(snap / "targets.parquet")
    keep = tgts[tgts["target_type"].isin(PROTEIN_TARGET_TYPES)]
    anchors = set(keep["target_chembl_id"].dropna().astype(str))
    mols = pd.read_parquet(snap / "molecules.parquet")
    return anchors, set(mols["chembl_id"].dropna().astype(str))


def select_compounds(
    con: sqlite3.Connection,
    anchors: set[str],
    current: set[str],
    min_pchembl: float,
    per_target: int = 0,
) -> pd.DataFrame:
    """One row per kept compound: molregno and chembl_id.

    ``per_target`` keeps only the strongest N compounds for each target.
    Without it the rule does not bound anything: measured potency in ChEMBL is
    concentrated on exactly the proteins the drug layer already hits, so
    "restrict to our targets" returned 548,589 compounds -- more than *all*
    human targets (539,505). The cap is what makes the set a size we choose,
    and it spends that budget evenly over targets rather than letting the
    most-screened handful take it all.
    """
    con.execute("CREATE TEMP TABLE anchor (chembl_id TEXT PRIMARY KEY)")
    con.executemany("INSERT OR IGNORE INTO anchor VALUES (?)", [(a,) for a in anchors])
    con.execute("CREATE TEMP TABLE current (chembl_id TEXT PRIMARY KEY)")
    con.executemany("INSERT OR IGNORE INTO current VALUES (?)", [(c,) for c in current])

    pairs = """
        SELECT md.molregno AS molregno, md.chembl_id AS chembl_id,
               t.chembl_id AS tid, MAX(act.pchembl_value) AS p
          FROM activities act
          JOIN assays a               ON a.assay_id = act.assay_id
          JOIN target_dictionary t    ON t.tid = a.tid
          JOIN anchor an              ON an.chembl_id = t.chembl_id
          JOIN molecule_dictionary md ON md.molregno = act.molregno
          JOIN compound_structures cs ON cs.molregno = md.molregno
         WHERE act.pchembl_value >= ?
           AND cs.canonical_smiles IS NOT NULL
         GROUP BY md.molregno, t.chembl_id
    """
    if per_target > 0:
        sql = f"""
            CREATE TEMP TABLE keep AS
            WITH pairs AS ({pairs}),
                 ranked AS (
                     SELECT molregno, chembl_id, ROW_NUMBER() OVER (
                                PARTITION BY tid ORDER BY p DESC, chembl_id
                            ) AS rn
                       FROM pairs
                 )
            SELECT DISTINCT molregno, chembl_id FROM ranked WHERE rn <= {per_target}
        """
    else:
        sql = f"""
            CREATE TEMP TABLE keep AS
            SELECT DISTINCT molregno, chembl_id FROM ({pairs})
        """
    con.execute(sql, (min_pchembl,))
    # The drug layer joins in whole, including the compounds with no measured
    # potency at all -- losing approved drugs to widen the map would be a
    # strange trade.
    con.execute(
        """
        INSERT INTO keep (molregno, chembl_id)
        SELECT md.molregno, md.chembl_id
          FROM molecule_dictionary md
          JOIN current c ON c.chembl_id = md.chembl_id
         WHERE md.molregno NOT IN (SELECT molregno FROM keep)
        """
    )
    con.execute("CREATE INDEX temp.keep_mol ON keep(molregno)")
    return pd.read_sql_query("SELECT molregno, chembl_id FROM keep", con)


# ------------------------------------------------------------------ tables


def molecules(con: sqlite3.Connection) -> pd.DataFrame:
    flags = ", ".join(f"md.{f}" for f in _MOL_FLAGS)
    props = ", ".join(f"cp.{f}" for f in PROPERTY_FIELDS)
    df = pd.read_sql_query(
        f"""
        SELECT md.chembl_id              AS chembl_id,
               md.pref_name              AS pref_name,
               pmd.chembl_id             AS parent_chembl_id,
               cs.canonical_smiles       AS canonical_smiles,
               cs.standard_inchi_key     AS standard_inchi_key,
               {flags},
               {props}
          FROM keep k
          JOIN molecule_dictionary md   ON md.molregno = k.molregno
          LEFT JOIN compound_structures cs ON cs.molregno = md.molregno
          LEFT JOIN compound_properties cp ON cp.molregno = md.molregno
          LEFT JOIN molecule_hierarchy mh  ON mh.molregno = md.molregno
          LEFT JOIN molecule_dictionary pmd ON pmd.molregno = mh.parent_molregno
        """,
        con,
    )

    atc = pd.read_sql_query(
        """
        SELECT md.chembl_id AS chembl_id, ac.level5 AS level5
          FROM keep k
          JOIN molecule_dictionary md ON md.molregno = k.molregno
          JOIN molecule_atc_classification ac ON ac.molregno = md.molregno
        """,
        con,
    )
    if len(atc):
        grouped = atc.groupby("chembl_id")["level5"].apply(list)
        codes = grouped.apply(lambda v: ";".join(sorted(set(map(str, v)))))
        level1 = grouped.apply(lambda v: ";".join(sorted({str(c)[0] for c in v if c})))
        df["atc_codes"] = df["chembl_id"].map(codes).fillna("")
        df["atc_level1"] = df["chembl_id"].map(level1).fillna("")
    else:
        df["atc_codes"] = ""
        df["atc_level1"] = ""
    return df


def indications(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT md.chembl_id  AS chembl_id,
               pmd.chembl_id AS parent_chembl_id,
               di.mesh_id, di.mesh_heading, di.efo_id, di.efo_term,
               di.max_phase_for_ind
          FROM keep k
          JOIN molecule_dictionary md ON md.molregno = k.molregno
          JOIN drug_indication di     ON di.molregno = md.molregno
          LEFT JOIN molecule_hierarchy mh   ON mh.molregno = md.molregno
          LEFT JOIN molecule_dictionary pmd ON pmd.molregno = mh.parent_molregno
        """,
        con,
    ).drop_duplicates()


def mechanisms(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT md.chembl_id  AS chembl_id,
               pmd.chembl_id AS parent_chembl_id,
               t.chembl_id   AS target_chembl_id,
               dm.action_type, dm.mechanism_of_action,
               dm.direct_interaction, dm.disease_efficacy
          FROM keep k
          JOIN molecule_dictionary md ON md.molregno = k.molregno
          JOIN drug_mechanism dm      ON dm.molregno = md.molregno
          LEFT JOIN target_dictionary t     ON t.tid = dm.tid
          LEFT JOIN molecule_hierarchy mh   ON mh.molregno = md.molregno
          LEFT JOIN molecule_dictionary pmd ON pmd.molregno = mh.parent_molregno
        """,
        con,
    ).drop_duplicates()


def activities(con: sqlite3.Connection, min_pchembl: float) -> pd.DataFrame:
    """Measured potencies for the kept compounds, at or above the cutoff.

    Only one row survives per (compound, target): the strongest. ChEMBL holds
    many assays against the same protein and the domain only ever uses the
    best one, so collapsing here keeps the file a tenth of the size.
    """
    df = pd.read_sql_query(
        """
        SELECT md.chembl_id AS chembl_id,
               t.chembl_id  AS target_chembl_id,
               MAX(act.pchembl_value) AS pchembl_value,
               act.standard_type, a.assay_type,
               t.organism   AS target_organism
          FROM keep k
          JOIN molecule_dictionary md ON md.molregno = k.molregno
          JOIN activities act         ON act.molregno = md.molregno
          JOIN assays a               ON a.assay_id = act.assay_id
          JOIN target_dictionary t    ON t.tid = a.tid
         WHERE act.pchembl_value >= ?
         GROUP BY md.chembl_id, t.chembl_id
        """,
        con,
        params=(min_pchembl,),
    )
    return df


def targets(con: sqlite3.Connection) -> pd.DataFrame:
    """Targets reachable from the kept compounds, with their class links."""
    return pd.read_sql_query(
        """
        WITH tids AS (
            SELECT DISTINCT a.tid AS tid
              FROM keep k
              JOIN activities act ON act.molregno = k.molregno
              JOIN assays a       ON a.assay_id = act.assay_id
            UNION
            SELECT DISTINCT dm.tid
              FROM keep k
              JOIN drug_mechanism dm ON dm.molregno = k.molregno
        )
        SELECT t.chembl_id   AS target_chembl_id,
               t.pref_name   AS target_pref_name,
               t.target_type AS target_type,
               t.organism    AS organism,
               tc.component_id        AS component_id,
               csq.accession          AS accession,
               cc.protein_class_id    AS protein_class_id
          FROM tids
          JOIN target_dictionary t ON t.tid = tids.tid
          LEFT JOIN target_components tc   ON tc.tid = t.tid
          LEFT JOIN component_sequences csq ON csq.component_id = tc.component_id
          LEFT JOIN component_class cc      ON cc.component_id = tc.component_id
        """,
        con,
    ).drop_duplicates()


def protein_classes(con: sqlite3.Connection) -> pd.DataFrame:
    """One row per class with its l1..l6 path, as the REST path produces."""
    raw = pd.read_sql_query(
        "SELECT protein_class_id, parent_id, pref_name, short_name, class_level "
        "FROM protein_classification",
        con,
    )
    by_id = {int(r.protein_class_id): r for r in raw.itertuples()}

    def path(cid: int) -> list[str]:
        out: list[str] = []
        seen: set[int] = set()
        while cid is not None and cid in by_id and cid not in seen:
            seen.add(cid)
            rec = by_id[cid]
            parent = rec.parent_id
            # The synthetic root has no parent and is not a real class.
            if parent is not None and not pd.isna(parent) and rec.pref_name:
                out.append(str(rec.pref_name))
            cid = None if parent is None or pd.isna(parent) else int(parent)
        return list(reversed(out))

    rows = []
    for cid, rec in by_id.items():
        p = path(cid)
        row = {
            "protein_class_id": cid,
            "class_level": rec.class_level,
            "pref_name": rec.pref_name,
            "short_name": rec.short_name,
            "class_path": " > ".join(p),
        }
        for i in range(1, 7):
            row[f"l{i}"] = p[i - 1] if len(p) >= i else None
        rows.append(row)
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="path to chembl_NN.db")
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--out", default=None, help="snapshot dir name (default: <version>_b)")
    ap.add_argument("--min-pchembl", type=float, default=7.0)
    ap.add_argument(
        "--per-target",
        type=int,
        default=50,
        help="keep only the strongest N compounds per target; 0 = no cap "
        "(measured: 25 -> 47,580 compounds, 50 -> 73,864, 100 -> 114,932, "
        "none -> 557,550)",
    )
    ap.add_argument(
        "--count-only",
        action="store_true",
        help="report how big the set would be and write nothing",
    )
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout,
    )

    base = snapshot_dir(args.cache_dir)
    if base is None:
        print("no existing snapshot to take the anchor targets from")
        return 2
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    version = manifest["chembl_version"]
    anchors, current = anchors_and_current(base)
    log.info("anchor: %s · 화합물 %d · 단백질 표적 %d", base.name, len(current), len(anchors))

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.execute("PRAGMA temp_store = MEMORY")
    con.execute("PRAGMA cache_size = -2000000")  # 2 GB page cache
    check_schema(con)

    t0 = time.perf_counter()
    keep = select_compounds(
        con, anchors, current, args.min_pchembl, per_target=args.per_target
    )
    log.info(
        "화합물 %d개 선택 (%.0fs) — 새로 %d, 기존 %d",
        len(keep), time.perf_counter() - t0,
        len(set(keep["chembl_id"]) - current), len(set(keep["chembl_id"]) & current),
    )
    if args.count_only:
        return 0

    out = resolve_data_path(args.cache_dir) / (args.out or f"{version.lower()}_b")
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for name, fn in (
        ("molecules", lambda: molecules(con)),
        ("indications", lambda: indications(con)),
        ("mechanisms", lambda: mechanisms(con)),
        ("activities", lambda: activities(con, args.min_pchembl)),
        ("targets", lambda: targets(con)),
        ("protein_classes", lambda: protein_classes(con)),
    ):
        t = time.perf_counter()
        df = fn()
        df.to_parquet(out / f"{name}.parquet", index=False)
        counts[name] = len(df)
        log.info("  %-15s %9d행  %5.0fs", name, len(df), time.perf_counter() - t)

    mols = pd.read_parquet(out / "molecules.parquet")
    acts = pd.read_parquet(out / "activities.parquet")
    mechs = pd.read_parquet(out / "mechanisms.parquet")
    inds = pd.read_parquet(out / "indications.parquet")
    tgts = pd.read_parquet(out / "targets.parquet")
    mech_c = set(mechs["chembl_id"].dropna())
    act_c = set(acts["chembl_id"].dropna())
    new_manifest = {
        "chembl_version": version,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "source": "sqlite",
        "built_from": str(args.db),
        "rule": (
            "drug layer + pchembl>=%.1f on its protein targets"
            % args.min_pchembl
            + (f", strongest {args.per_target} per target" if args.per_target else "")
        ),
        "per_target": args.per_target,
        "anchor_snapshot": base.name,
        "min_phase": manifest.get("min_phase"),
        "min_pchembl": args.min_pchembl,
        "counts": {
            "molecules": int(len(mols)),
            "indications": int(len(inds)),
            "indication_compounds": int(inds["chembl_id"].nunique()),
            "mechanisms": int(len(mechs)),
            "mechanism_compounds": len(mech_c),
            "activities": int(len(acts)),
            "activity_compounds": len(act_c),
            "target_compounds": len(mech_c | act_c),
            "targets": int(tgts["target_chembl_id"].nunique()),
            "protein_classes": counts["protein_classes"],
        },
    }
    (out / "manifest.json").write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")

    n = len(mols)
    print(f"\n{out}")
    print(json.dumps(new_manifest["counts"], indent=2, ensure_ascii=False))
    print(f"\n표적 주석 커버리지  {len(mech_c | act_c):,} / {n:,}  "
          f"({len(mech_c | act_c) / n:.1%})")
    print(f"적응증 커버리지      {inds['chembl_id'].nunique():,} / {n:,}  "
          f"({inds['chembl_id'].nunique() / n:.1%})")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
