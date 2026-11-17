"""Fetch the ChEMBL approved/clinical drug layer through the public REST API.

Only compounds with a development phase assigned (``max_phase >= 0.5``) are
pulled.  That is deliberate: the disease domain depends on ``drug_indication``
rows and the target domain on ``mechanism`` rows, and both exist almost only for
drugs that reached the clinic.  A larger activity-based slice would leave those
two domains with no signal.

Verified against ChEMBL_37 (16,809 molecules at ``max_phase >= 0.5``).

Table chain for the target domain -- note the classification hangs off
``target_component``, *not* ``target``:

    mechanism.target_chembl_id
      -> target.target_components[].component_id
      -> target_component.protein_classifications[].protein_classification_id
      -> protein_classification (a parent_id hierarchy, walked into l1..l6)

Everything is cached as parquet under ``cache_dir``, keyed by the release that
``/status`` reports, so a run is reproducible against a named snapshot.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd
import requests

log = logging.getLogger(__name__)

API_ROOT = "https://www.ebi.ac.uk/chembl/api/data"
EBI_HOST = "https://www.ebi.ac.uk"
USER_AGENT = "chemspace/0.1 (chemical space exploration tool)"

_PAGE_SIZE = 1000
_SLEEP_BETWEEN_PAGES = 0.1
_MAX_RETRIES = 4
_IN_BATCH = 40  # keeps the `__in` query string well inside URL length limits

TABLES = (
    "molecules",
    "indications",
    "mechanisms",
    "activities",
    "targets",
    "protein_classes",
)


class ChemblError(RuntimeError):
    pass


@dataclass(frozen=True)
class Snapshot:
    version: str  # e.g. "ChEMBL_37"
    fetched_at: str

    @property
    def slug(self) -> str:
        return self.version.lower().replace(" ", "_")


class ChemblClient:
    def __init__(self, session: requests.Session | None = None, timeout: float = 90.0):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        self.timeout = timeout

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict:
        last: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    raise ChemblError(f"HTTP {r.status_code} from {r.url}")
                r.raise_for_status()
                return r.json()
            except Exception as exc:  # noqa: BLE001 - retried
                last = exc
                wait = 2.0 * (attempt + 1)
                log.warning("ChEMBL request failed (%s); retry in %.0fs", exc, wait)
                time.sleep(wait)
        raise ChemblError(f"giving up on {url}: {last}")

    def status(self) -> Snapshot:
        payload = self._get(f"{API_ROOT}/status.json")
        version = str(payload.get("chembl_db_version") or "unknown")
        return Snapshot(version=version, fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"))

    def paged(self, endpoint: str, params: dict[str, Any] | None = None) -> Iterator[dict]:
        """Walk a collection endpoint, following ``page_meta.next``."""
        params = dict(params or {})
        params.setdefault("limit", _PAGE_SIZE)
        url = f"{API_ROOT}/{endpoint}.json"
        first, seen = True, 0
        while url:
            payload = self._get(url, params=params if first else None)
            meta = payload.get("page_meta", {})
            rows = payload[_collection_key(payload)]
            yield from rows
            seen += len(rows)
            if seen and meta.get("total_count"):
                log.info("  %s: %d/%s", endpoint, seen, meta["total_count"])
            nxt = meta.get("next")
            url = f"{EBI_HOST}{nxt}" if nxt else None
            first = False
            if url:
                time.sleep(_SLEEP_BETWEEN_PAGES)

    def by_ids(
        self,
        endpoint: str,
        field: str,
        ids: Iterable[Any],
        only: str | None = None,
        batch_size: int = _IN_BATCH,
        **filters: Any,
    ) -> Iterator[dict]:
        """Fetch records in ``__in`` batches. ``only`` trims the payload a lot.

        ``filters`` are passed through as extra query parameters, which is how
        the activity pull restricts itself to measured potencies at or above a
        threshold instead of dragging down every assay result ever recorded.
        """
        ids = [str(i) for i in ids]
        extra: dict[str, Any] = {"only": only} if only else {}
        extra.update(filters)
        for batch in _batched(ids, batch_size):
            params = {f"{field}__in": ",".join(batch), **extra}
            yield from self.paged(endpoint, params)


def _collection_key(payload: dict) -> str:
    for k, v in payload.items():
        if k != "page_meta" and isinstance(v, list):
            return k
    raise ChemblError(f"no collection in payload keys={list(payload)}")


def _batched(seq: list[Any], n: int) -> Iterator[list[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# --------------------------------------------------------------------------- #
# Flatteners -- field lists verified against the live ChEMBL_37 API.
# --------------------------------------------------------------------------- #

# molecule_properties actually returns exactly these; cx_logp / hba_lipinski /
# molecular_species are NOT present on this endpoint.
#
# Four of these (hba, hbd, aromatic_rings, heavy_atoms) collide with the RDKit
# columns in `derive`. The loader prefixes this whole block with `chembl_` so
# the unprefixed name always means "recomputed by RDKit"; see
# `Dataset._prefix_chembl_properties`.
PROPERTY_FIELDS = (
    "full_mwt",
    "mw_freebase",
    "alogp",
    "psa",
    "hba",
    "hbd",
    "rtb",
    "aromatic_rings",
    "heavy_atoms",
    "qed_weighted",
    "num_ro5_violations",
    "ro3_pass",
    "np_likeness_score",
    "full_molformula",
)

_MOL_FLAGS = (
    "max_phase",
    "first_approval",
    "molecule_type",
    "structure_type",
    "withdrawn_flag",
    "therapeutic_flag",
    "black_box_warning",
    "first_in_class",
    "orphan",
    "chirality",
    "oral",
    "parenteral",
    "topical",
    "natural_product",
    "prodrug",
    "polymer_flag",
    "inorganic_flag",
    "usan_stem_definition",
)

_NUMERIC = (
    "max_phase",
    "first_approval",
    *(f for f in PROPERTY_FIELDS if f != "full_molformula"),
)


def _flatten_molecule(rec: dict) -> dict:
    structures = rec.get("molecule_structures") or {}
    props = rec.get("molecule_properties") or {}
    hierarchy = rec.get("molecule_hierarchy") or {}
    atc = rec.get("atc_classifications") or []
    row: dict[str, Any] = {
        "chembl_id": rec.get("molecule_chembl_id"),
        "pref_name": rec.get("pref_name"),
        "parent_chembl_id": hierarchy.get("parent_chembl_id"),
        # ATC codes ("L01EB01"): the first letter is the anatomical main group,
        # a therapeutic-area signal with better coverage than drug_indication.
        "atc_codes": ";".join(atc),
        "atc_level1": ";".join(sorted({c[0] for c in atc if c})),
        "canonical_smiles": structures.get("canonical_smiles"),
        "standard_inchi_key": structures.get("standard_inchi_key"),
    }
    for f in _MOL_FLAGS:
        row[f] = rec.get(f)
    for f in PROPERTY_FIELDS:
        row[f] = props.get(f)
    return row


def _flatten_indication(rec: dict) -> dict:
    return {
        "chembl_id": rec.get("molecule_chembl_id"),
        "parent_chembl_id": rec.get("parent_molecule_chembl_id"),
        "mesh_id": rec.get("mesh_id"),
        "mesh_heading": rec.get("mesh_heading"),
        "efo_id": rec.get("efo_id"),
        "efo_term": rec.get("efo_term"),
        "max_phase_for_ind": rec.get("max_phase_for_ind"),
    }


def _flatten_mechanism(rec: dict) -> dict:
    return {
        "chembl_id": rec.get("molecule_chembl_id"),
        "parent_chembl_id": rec.get("parent_molecule_chembl_id"),
        "target_chembl_id": rec.get("target_chembl_id"),
        "action_type": rec.get("action_type"),
        "mechanism_of_action": rec.get("mechanism_of_action"),
        "direct_interaction": rec.get("direct_interaction"),
        "disease_efficacy": rec.get("disease_efficacy"),
    }


ACTIVITY_ONLY = (
    "molecule_chembl_id,target_chembl_id,pchembl_value,standard_type,"
    "assay_type,target_organism"
)


def _flatten_activity(rec: dict) -> dict:
    return {
        "chembl_id": rec.get("molecule_chembl_id"),
        "target_chembl_id": rec.get("target_chembl_id"),
        "pchembl_value": rec.get("pchembl_value"),
        "standard_type": rec.get("standard_type"),
        "assay_type": rec.get("assay_type"),
        "target_organism": rec.get("target_organism"),
    }


def fetch_activities(
    client: "ChemblClient",
    compound_ids: Iterable[str],
    min_pchembl: float = 7.0,
    batch_size: int = 20,
) -> pd.DataFrame:
    """Measured potencies at or above ``min_pchembl`` for the given compounds.

    This is the only route to target annotation at scale. The curated
    ``mechanism`` table covers 5,954 compounds in the whole of ChEMBL -- a hard
    ceiling no amount of extra compounds moves -- while ``activity`` holds
    ~24.5M measurements. The threshold is what keeps that from becoming noise:
    pChEMBL 7 is 100 nM, potent enough that the compound is doing something
    real at the target.

    Batches are small on purpose. The endpoint answers 500 for wide ``__in``
    queries against this table, which is a server-side cost limit rather than
    a bad request, so the usual batch of 40 has to come down.
    """
    rows = [
        _flatten_activity(r)
        for r in client.by_ids(
            "activity",
            "molecule_chembl_id",
            compound_ids,
            only=ACTIVITY_ONLY,
            batch_size=batch_size,
            pchembl_value__gte=str(min_pchembl),
        )
    ]
    if not rows:
        return pd.DataFrame(columns=[f for f in _flatten_activity({})])
    df = pd.DataFrame(rows).dropna(subset=["chembl_id", "target_chembl_id"])
    df = _coerce(df, ["pchembl_value"])
    return df.drop_duplicates()


def _collect_targets(client: "ChemblClient", target_ids: Sequence[str]) -> pd.DataFrame:
    """Target rows plus their protein-class links, for the given target ids.

    Shared by the mechanism and activity paths so both kinds of target get the
    same class hierarchy -- otherwise an activity-derived target would land on
    the map with no L1/L2/L3 and could only ever cluster by its own name.
    """
    tgt_rows: list[dict] = []
    comp_to_target: dict[Any, list[str]] = {}
    for rec in client.by_ids(
        "target",
        "target_chembl_id",
        target_ids,
        only="target_chembl_id,pref_name,target_type,organism,target_components",
    ):
        tid = rec.get("target_chembl_id")
        comps = rec.get("target_components") or []
        if not comps:
            tgt_rows.append(
                {
                    "target_chembl_id": tid,
                    "target_pref_name": rec.get("pref_name"),
                    "target_type": rec.get("target_type"),
                    "organism": rec.get("organism"),
                    "component_id": None,
                    "accession": None,
                }
            )
        for c in comps:
            cid = c.get("component_id")
            comp_to_target.setdefault(cid, []).append(tid)
            tgt_rows.append(
                {
                    "target_chembl_id": tid,
                    "target_pref_name": rec.get("pref_name"),
                    "target_type": rec.get("target_type"),
                    "organism": rec.get("organism"),
                    "component_id": cid,
                    "accession": c.get("accession"),
                }
            )
    targets = pd.DataFrame(tgt_rows)

    # protein_classifications live on target_component, not target. `only`
    # keeps the multi-kB sequence field out of the response.
    comp_ids = sorted({c for c in comp_to_target if c is not None})
    comp_class: list[dict] = []
    for rec in client.by_ids(
        "target_component",
        "component_id",
        comp_ids,
        only="component_id,accession,protein_classifications",
    ):
        for cls in rec.get("protein_classifications") or []:
            comp_class.append(
                {
                    "component_id": rec.get("component_id"),
                    "protein_class_id": cls.get("protein_classification_id"),
                }
            )
    cc = pd.DataFrame(comp_class).drop_duplicates()
    log.info(
        "      %d targets, %d components, %d class links",
        targets["target_chembl_id"].nunique() if len(targets) else 0,
        len(comp_ids),
        len(cc),
    )
    if len(cc):
        return targets.merge(cc, on="component_id", how="left")
    return targets.assign(protein_class_id=pd.NA)


def _protein_class_table(records: list[dict]) -> pd.DataFrame:
    """Turn the flat parent_id hierarchy into one row per class with l1..l6."""
    by_id = {r["protein_class_id"]: r for r in records}

    def path(cid: Any) -> list[str]:
        out: list[str] = []
        seen: set[Any] = set()
        while cid is not None and cid in by_id and cid not in seen:
            seen.add(cid)
            rec = by_id[cid]
            name = rec.get("pref_name")
            if rec.get("parent_id") is not None and name:  # skip the synthetic root
                out.append(str(name))
            cid = rec.get("parent_id")
        return list(reversed(out))

    rows = []
    for cid, rec in by_id.items():
        p = path(cid)
        row: dict[str, Any] = {
            "protein_class_id": cid,
            "class_level": rec.get("class_level"),
            "pref_name": rec.get("pref_name"),
            "short_name": rec.get("short_name"),
            "class_path": " > ".join(p),
        }
        for i in range(1, 7):
            row[f"l{i}"] = p[i - 1] if len(p) >= i else None
        rows.append(row)
    return pd.DataFrame(rows)


def _coerce(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """ChEMBL returns numbers as strings ('max_phase': '4.0')."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def project_root(start: str | Path | None = None) -> Path | None:
    """Nearest ancestor of ``start`` (default: cwd) holding ``pyproject.toml``.

    ``None`` when there is no such directory, which is the normal case for an
    installed package being used from somewhere else entirely.
    """
    here = Path(start or Path.cwd()).resolve()
    for base in (here, *here.parents):
        if (base / "pyproject.toml").exists():
            return base
    return None


def resolve_data_path(relative: str | Path) -> Path:
    """Turn a path like ``data/cache`` into one that works from any directory.

    A notebook opened in ``notebooks/`` has that as its working directory, so a
    bare ``data/cache`` points at ``notebooks/data/cache`` and the snapshot
    looks missing.  Resolution order: an absolute path as given, then the path
    under the current directory if it already exists, then under the project
    root.  New data is created under the project root when there is one, so
    fetching from a notebook still fills the project's own cache.
    """
    rel = Path(relative)
    if rel.is_absolute():
        return rel
    if (Path.cwd() / rel).exists():
        return Path.cwd() / rel
    root = project_root()
    if root is not None:
        return root / rel
    return Path.cwd() / rel


def snapshot_dir(cache_dir: str | Path = "data/cache", version: str | None = None) -> Path | None:
    """Return the newest cached snapshot dir, or the one matching ``version``."""
    cache_dir = resolve_data_path(cache_dir)
    if not cache_dir.exists():
        return None
    cands = [d for d in cache_dir.iterdir() if (d / "manifest.json").exists()]
    if version:
        cands = [d for d in cands if d.name == version.lower().replace(" ", "_")]
    if not cands:
        return None
    return max(cands, key=lambda d: json.loads((d / "manifest.json").read_text())["fetched_at"])


def fetch_snapshot(
    cache_dir: str | Path = "data/cache",
    min_phase: float = 0.5,
    min_pchembl: float = 7.0,
    force: bool = False,
    client: ChemblClient | None = None,
) -> Path:
    """Download the drug layer and write parquet tables. Returns the snapshot dir.

    Roughly 3-6 minutes against ChEMBL_37, plus the activity pull, which is
    the slow part because it is batched 20 compounds at a time.
    Re-running is free unless ``force``.
    """
    client = client or ChemblClient()
    cache_dir = resolve_data_path(cache_dir)
    snap = client.status()
    out = cache_dir / snap.slug

    if (out / "manifest.json").exists() and not force:
        log.info("reusing cached snapshot %s", out)
        return out

    out.mkdir(parents=True, exist_ok=True)
    log.info("fetching %s -> %s", snap.version, out)

    log.info("[1/6] molecules (max_phase >= %s)", min_phase)
    mols = pd.DataFrame(
        _flatten_molecule(r) for r in client.paged("molecule", {"max_phase__gte": min_phase})
    )
    mols = mols.dropna(subset=["chembl_id", "canonical_smiles"]).drop_duplicates("chembl_id")
    mols = _coerce(mols, _NUMERIC)
    mols.to_parquet(out / "molecules.parquet", index=False)
    keep = set(mols["chembl_id"])
    log.info("      %d molecules", len(mols))

    # drug_indication / mechanism are small enough to pull whole and filter
    # locally; that also keeps the cache usable for a wider molecule set later.
    log.info("[2/6] drug indications")
    inds = pd.DataFrame(_flatten_indication(r) for r in client.paged("drug_indication"))
    inds = inds[inds["chembl_id"].isin(keep)].drop_duplicates()
    inds = _coerce(inds, ["max_phase_for_ind"])
    inds.to_parquet(out / "indications.parquet", index=False)
    log.info("      %d rows covering %d compounds", len(inds), inds["chembl_id"].nunique())

    log.info("[3/6] mechanisms")
    mechs = pd.DataFrame(_flatten_mechanism(r) for r in client.paged("mechanism"))
    mechs = mechs[mechs["chembl_id"].isin(keep)].drop_duplicates()
    mechs.to_parquet(out / "mechanisms.parquet", index=False)
    log.info("      %d rows covering %d compounds", len(mechs), mechs["chembl_id"].nunique())

    log.info("[4/6] measured activities (pchembl >= %s)", min_pchembl)
    acts = fetch_activities(client, sorted(keep), min_pchembl=min_pchembl)
    acts = acts[acts["chembl_id"].isin(keep)]
    acts.to_parquet(out / "activities.parquet", index=False)
    log.info(
        "      %d rows covering %d compounds and %d targets",
        len(acts),
        acts["chembl_id"].nunique() if len(acts) else 0,
        acts["target_chembl_id"].nunique() if len(acts) else 0,
    )

    log.info("[5/6] targets and their components")
    # The union, not just the mechanism targets: an activity-derived target
    # with no class hierarchy could only ever cluster by its own name.
    target_ids = sorted(
        set(mechs["target_chembl_id"].dropna())
        | set(acts["target_chembl_id"].dropna() if len(acts) else [])
    )
    targets = _collect_targets(client, target_ids)
    targets.to_parquet(out / "targets.parquet", index=False)

    log.info("[6/6] protein classification hierarchy")
    pc_records = list(client.paged("protein_classification"))
    pcs = _protein_class_table(pc_records)
    pcs.to_parquet(out / "protein_classes.parquet", index=False)
    log.info("      %d classes", len(pcs))

    mech_compounds = set(mechs["chembl_id"])
    act_compounds = set(acts["chembl_id"]) if len(acts) else set()
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "chembl_version": snap.version,
                "fetched_at": snap.fetched_at,
                "min_phase": min_phase,
                "min_pchembl": min_pchembl,
                "counts": {
                    "molecules": int(len(mols)),
                    "indications": int(len(inds)),
                    "indication_compounds": int(inds["chembl_id"].nunique()),
                    "mechanisms": int(len(mechs)),
                    "mechanism_compounds": int(len(mech_compounds)),
                    "activities": int(len(acts)),
                    "activity_compounds": int(len(act_compounds)),
                    "target_compounds": int(len(mech_compounds | act_compounds)),
                    "targets": int(targets["target_chembl_id"].nunique()),
                    "protein_classes": int(len(pcs)),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("done -> %s", out)
    return out
