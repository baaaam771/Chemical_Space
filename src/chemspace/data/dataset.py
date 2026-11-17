"""Load a ChEMBL snapshot into the tables and lookups the domains need."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from chemspace.data import derive
from chemspace.data.chembl import PROPERTY_FIELDS, resolve_data_path, snapshot_dir

log = logging.getLogger(__name__)

PROTEIN_TARGET_TYPES = frozenset(
    {
        "SINGLE PROTEIN",
        "PROTEIN COMPLEX",
        "PROTEIN COMPLEX GROUP",
        "PROTEIN FAMILY",
        "CHIMERIC PROTEIN",
        "PROTEIN-PROTEIN INTERACTION",
        "SELECTIVITY GROUP",
        "NUCLEIC-ACID",
    }
)
"""Target types that name a molecule a compound can bind.

ChEMBL's ``activity`` table calls anything an assay was run against a
"target", and 36% of the pairs we pulled point at a CELL-LINE -- plus ORGANISM,
TISSUE and ADMET readouts. Those are measurements of an effect, not of a
binding partner: docetaxel came back with 544 cell lines and two proteins.
Left in, the target map would group cytotoxic chemotherapies by which
screening panel they happened to be run through and present it as shared
pharmacology.

Applied to the activity side only. The curated ``mechanism`` table is small
and deliberate -- where it names an organism for an antibacterial, that is a
statement about how the drug works, not an assay artifact.
"""

MAX_ACTIVITY_TARGETS = 25
"""Strongest measured targets kept per compound.

See Dataset._cap_per_compound: ChEMBL's activity counts track study effort, so
an uncapped target vector separates well-examined compounds from the rest.
Twenty-five is wide enough for a real polypharmacology profile and narrow
enough that a kinase-panel screen does not swamp the feature space.
"""

#: ATC anatomical main groups. The first letter of an ATC code is a coarse
#: therapeutic-area label: 2,818 compounds carry one (57% of the approved
#: subset).  Measured against ChEMBL_37 it adds only 259 compounds that have no
#: MeSH indication, so it is *not* a coverage fix -- its value is being a
#: 14-way axis that complements the ~5,000 fine-grained MeSH terms, and giving
#: the disease domain something meaningful to colour points by.
ATC_LEVEL1 = {
    "A": "소화계·대사",
    "B": "혈액·조혈",
    "C": "심혈관",
    "D": "피부",
    "G": "비뇨생식·성호르몬",
    "H": "전신호르몬",
    "J": "전신 항감염",
    "L": "항암·면역조절",
    "M": "근골격계",
    "N": "신경계",
    "P": "항기생충",
    "R": "호흡기",
    "S": "감각기관",
    "V": "기타",
}


@dataclass
class Dataset:
    """One ChEMBL snapshot, ready to be looked at through any domain."""

    version: str
    compounds: pd.DataFrame
    """One row per compound, indexed by ``chembl_id``: ChEMBL fields plus the
    RDKit descriptors, scaffolds and modality label from :mod:`.derive`."""

    indications: pd.DataFrame
    targets: pd.DataFrame
    """Long: one row per (compound, target, protein class) with ``l1``..``l6``."""

    fps: np.ndarray
    """(n, 2048) uint8 ECFP4 of the whole molecule, rows aligned to ``compounds``."""

    scaffold_fps: np.ndarray
    """(n, 2048) uint8 ECFP4 of the Murcko scaffold."""

    disease_terms: dict[str, tuple[str, ...]]
    target_terms: dict[str, tuple[str, ...]]
    manifest: dict

    _row_of: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._row_of = {cid: i for i, cid in enumerate(self.compounds.index)}

    # ---------------------------------------------------------------- loading

    @classmethod
    def load(
        cls,
        snapshot: str | Path | None = None,
        cache_dir: str | Path = "data/cache",
        rebuild_derived: bool = False,
    ) -> Dataset:
        snap = Path(snapshot) if snapshot else snapshot_dir(cache_dir)
        if snap is None:
            looked = resolve_data_path(cache_dir)
            raise FileNotFoundError(
                f"no ChEMBL snapshot under {looked} (asked for {cache_dir!r}, "
                f"cwd {Path.cwd()}). Run: python scripts/fetch_chembl.py"
            )
        manifest = json.loads((snap / "manifest.json").read_text(encoding="utf-8"))
        log.info("loading %s (%s)", snap, manifest["chembl_version"])

        mols = pd.read_parquet(snap / "molecules.parquet")
        inds = pd.read_parquet(snap / "indications.parquet")
        mechs = pd.read_parquet(snap / "mechanisms.parquet")
        # Optional: snapshots taken before the activity pull existed have no
        # such file, and should still load.
        act_path = snap / "activities.parquet"
        acts = pd.read_parquet(act_path) if act_path.exists() else None
        tgts = pd.read_parquet(snap / "targets.parquet")
        pcs = pd.read_parquet(snap / "protein_classes.parquet")

        derived, fps, sfps = cls._derived(snap, mols, rebuild=rebuild_derived)

        compounds = (
            cls._prefix_chembl_properties(mols)
            .merge(derived, on="chembl_id", how="inner")
            .set_index("chembl_id")
        )
        # Keep array rows aligned with the (possibly reduced) compound index.
        order = {cid: i for i, cid in enumerate(derived["chembl_id"])}
        rows = np.array([order[c] for c in compounds.index], dtype=int)
        fps, sfps = fps[rows], sfps[rows]

        targets = cls._target_table(
            mechs, acts, tgts, pcs, keep=set(compounds.index)
        )
        inds = inds[inds["chembl_id"].isin(compounds.index)]

        return cls(
            version=manifest["chembl_version"],
            compounds=compounds,
            indications=inds,
            targets=targets,
            fps=fps,
            scaffold_fps=sfps,
            disease_terms=cls._disease_terms(compounds, inds),
            target_terms=cls._target_terms(targets),
            manifest=manifest,
        )

    @staticmethod
    def _prefix_chembl_properties(mols: pd.DataFrame) -> pd.DataFrame:
        """Move ChEMBL's own descriptors under ``chembl_*``.

        Four of them (``hba``, ``hbd``, ``aromatic_rings``, ``heavy_atoms``)
        have the same names as the RDKit columns.  Left alone, the merge
        silently renames both sides to ``_x``/``_y`` and every later reference
        becomes a coin flip.  Prefixing makes the rule explicit: a bare
        descriptor name is always the RDKit value, which is the one with no
        gaps, and ``chembl_*`` is there for comparison.
        """
        rename = {f: f"chembl_{f}" for f in PROPERTY_FIELDS if f in mols.columns}
        return mols.rename(columns=rename)

    @staticmethod
    def _derived(
        snap: Path, mols: pd.DataFrame, rebuild: bool
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        out = snap / "derived"
        table, arrays = out / "compounds.parquet", out / "fingerprints.npz"
        if table.exists() and arrays.exists() and not rebuild:
            npz = np.load(arrays)
            return pd.read_parquet(table), npz["fps"], npz["scaffold_fps"]

        log.info("deriving descriptors/scaffolds/modality for %d molecules", len(mols))
        out.mkdir(parents=True, exist_ok=True)
        derived, fps, sfps = derive.derive_table(mols)
        derived.to_parquet(table, index=False)
        np.savez_compressed(arrays, fps=fps, scaffold_fps=sfps)
        return derived, fps, sfps

    @staticmethod
    def _target_table(
        mechs: pd.DataFrame,
        acts: pd.DataFrame | None,
        tgts: pd.DataFrame,
        pcs: pd.DataFrame,
        keep: set[str],
        max_activity_targets: int = MAX_ACTIVITY_TARGETS,
    ) -> pd.DataFrame:
        """One row per (compound, target, protein class), from both sources.

        ``mechanism`` is curated intent -- what the drug was made to hit -- and
        covers 5,954 compounds in the whole of ChEMBL. ``activity`` is measured
        potency and covers hundreds of thousands. They answer different
        questions, so a ``source`` column keeps them apart for reporting, but
        they share one term namespace: ``TGT:EGFR`` means the same target
        whichever table it came from, and splitting it would stop two compounds
        that hit the same protein from ever sharing a feature.
        """
        cols = ["target_chembl_id", "target_pref_name", "target_type", "organism", "protein_class_id"]
        t = tgts[[c for c in cols if c in tgts.columns]].drop_duplicates()
        pc_cols = ["protein_class_id", "class_path", "l1", "l2", "l3", "l4", "l5", "l6"]

        frames = [
            mechs[mechs["chembl_id"].isin(keep)].assign(
                source="mechanism", pchembl_value=np.nan
            )
        ]
        if acts is not None and len(acts):
            a = acts[acts["chembl_id"].isin(keep)]
            # Drop assay readouts that are not binding partners *before*
            # capping, so a compound's 25 slots go to real targets instead of
            # being filled by the cell-line panel it was screened against.
            # one row per target: t still carries a row per protein class
            types = t.drop_duplicates("target_chembl_id").set_index(
                "target_chembl_id"
            )["target_type"]
            kind = a["target_chembl_id"].map(types)
            a = a[kind.isin(PROTEIN_TARGET_TYPES)]
            a = Dataset._cap_per_compound(a, max_activity_targets)
            frames.append(
                a[["chembl_id", "target_chembl_id", "pchembl_value"]].assign(
                    source="activity"
                )
            )

        df = pd.concat(frames, ignore_index=True, sort=False)
        df = (
            df.merge(t, on="target_chembl_id", how="left")
            .merge(pcs[pc_cols], on="protein_class_id", how="left")
        )
        return df.drop_duplicates()

    @staticmethod
    def _cap_per_compound(acts: pd.DataFrame, cap: int) -> pd.DataFrame:
        """Keep each compound's strongest ``cap`` targets.

        ChEMBL's activity volume tracks how much a compound has been studied,
        not how promiscuous it is: a drug profiled against a kinase panel
        carries hundreds of measured targets while an equally promiscuous but
        less-examined one carries three. Left uncapped, the target map would
        separate famous compounds from obscure ones and call it biology.
        """
        if cap <= 0 or acts.empty:
            return acts
        best = (
            acts.sort_values("pchembl_value", ascending=False)
            .drop_duplicates(["chembl_id", "target_chembl_id"])
        )
        return best.groupby("chembl_id", group_keys=False, sort=False).head(cap)

    @staticmethod
    def _disease_terms(
        compounds: pd.DataFrame, inds: pd.DataFrame
    ) -> dict[str, tuple[str, ...]]:
        """MeSH indication headings plus ATC anatomical main groups.

        Terms are prefixed by source (``MeSH:``, ``EFO:``, ``ATC:``) so a
        cluster profile can say where a term came from, and so the coarse and
        fine signals can be told apart when reading a map.
        """
        terms: dict[str, set[str]] = {}
        have = inds.dropna(subset=["mesh_heading"])
        for cid, heading in zip(have["chembl_id"], have["mesh_heading"]):
            terms.setdefault(cid, set()).add(f"MeSH:{heading}")
        efo = inds[inds["mesh_heading"].isna()].dropna(subset=["efo_term"])
        for cid, term in zip(efo["chembl_id"], efo["efo_term"]):
            terms.setdefault(cid, set()).add(f"EFO:{term}")
        atc = compounds["atc_level1"].dropna()
        for cid, letters in atc.items():
            for letter in str(letters).split(";"):
                if letter:
                    terms.setdefault(cid, set()).add(
                        f"ATC:{ATC_LEVEL1.get(letter, letter)}"
                    )
        return {k: tuple(sorted(v)) for k, v in terms.items()}

    @staticmethod
    def _target_terms(targets: pd.DataFrame) -> dict[str, tuple[str, ...]]:
        """Protein-class levels 1-3 plus the specific target.

        Including several levels lets a subset separate either coarsely
        (Enzyme vs Membrane receptor) or finely (one kinase family), depending
        on what the selected compounds actually share.
        """
        terms: dict[str, set[str]] = {}
        for rec in targets.itertuples(index=False):
            s = terms.setdefault(rec.chembl_id, set())
            for lvl in ("l1", "l2", "l3"):
                v = getattr(rec, lvl, None)
                if isinstance(v, str) and v:
                    s.add(f"{lvl.upper()}:{v}")
            name = getattr(rec, "target_pref_name", None)
            if isinstance(name, str) and name:
                s.add(f"TGT:{name}")
        return {k: tuple(sorted(v)) for k, v in terms.items()}

    # ----------------------------------------------------------------- access

    @property
    def all_ids(self) -> tuple[str, ...]:
        return tuple(self.compounds.index)

    def rows(self, compound_ids: Sequence[str]) -> np.ndarray:
        """Row indices into ``fps`` / ``scaffold_fps`` / ``compounds``."""
        return np.array([self._row_of[c] for c in compound_ids], dtype=int)

    def frame(self, compound_ids: Sequence[str]) -> pd.DataFrame:
        return self.compounds.loc[list(compound_ids)]

    @property
    def fingerprint(self) -> str:
        """Short hash of the snapshot manifest.

        The version string alone is not enough to key a cache on: adding the
        activity layer changes every target feature vector while ChEMBL_37
        stays ChEMBL_37, so a cached target placement from before the pull
        would be served as if it were current. The manifest carries the row
        counts and the thresholds, so it moves whenever the content does.
        """
        blob = json.dumps(self.manifest, sort_keys=True, default=str)
        return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()

    def coverage(self) -> pd.DataFrame:
        """Per-domain data coverage -- the first thing to check on a new snapshot."""
        n = len(self.compounds)
        have_target = set(self.target_terms)
        have_disease = set(self.disease_terms)
        have_mesh = {
            c for c, ts in self.disease_terms.items() if any(t.startswith("MeSH:") for t in ts)
        }
        src = self.targets.get("source")
        if src is None:  # a snapshot from before the activity pull
            have_mech, have_act_only = have_target, set()
        else:
            have_mech = set(self.targets.loc[src == "mechanism", "chembl_id"])
            have_act_only = set(self.targets.loc[src == "activity", "chembl_id"]) - have_mech
        rows = [
            ("물성 physchem", n, "RDKit descriptors"),
            ("모이어티 moiety", int((self.compounds["murcko_smiles"] != "").sum()), "Murcko scaffold"),
            ("모달리티 modality", n, "rules + ChEMBL molecule_type"),
            ("질환 disease", len(have_disease), "MeSH indication or ATC group"),
            ("  └ MeSH만", len(have_mesh), "drug_indication rows only"),
            ("표적 target", len(have_target), "mechanism + activity -> protein class"),
            ("  └ mechanism만", len(have_mech), "curated mechanism of action"),
            ("  └ activity만", len(have_act_only), "measured potency, no curated MoA"),
        ]
        df = pd.DataFrame(rows, columns=["domain", "compounds", "source"])
        df["coverage"] = (df["compounds"] / n).map(lambda v: f"{v:.1%}")
        return df

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Dataset {self.version}: {len(self.compounds)} compounds>"
