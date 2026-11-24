"""Load a ChEMBL snapshot into the tables and lookups the domains need."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
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
class TargetESM:
    """Per-accession ESM-2 sequence embeddings (``target_esm.npz``).

    Shared across snapshots and keyed by UniProt accession; a compound's
    sequence vector is the mean over the accessions of the targets it hits.
    """

    dim: int
    index: dict[str, int]
    matrix: np.ndarray  # (n_accessions, dim) float32

    def mean_vector(self, accessions: Sequence[str]) -> np.ndarray | None:
        rows = [self.index[a] for a in accessions if a in self.index]
        if not rows:
            return None
        return self.matrix[rows].mean(axis=0)

    @classmethod
    def load(cls, path: Path) -> "TargetESM | None":
        if not path.exists():
            return None
        npz = np.load(path, allow_pickle=True)
        accs = [str(a) for a in npz["accessions"]]
        mat = np.ascontiguousarray(npz["emb"], dtype=np.float32)
        return cls(dim=int(mat.shape[1]), index={a: i for i, a in enumerate(accs)}, matrix=mat)


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

    target_accessions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """Compound -> its targets' UniProt accessions, for the sequence domain."""
    target_embeddings: "TargetESM | None" = None
    """Precomputed per-accession ESM-2 sequence embeddings, if the snapshot
    ships ``target_esm.npz``. ``None`` means the sequence domain is unavailable."""

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

        # Display overlays (labels, not domains): colour/filter any map by them.
        for col, series in cls._overlay_columns(compounds, mechs).items():
            compounds[col] = series
        # Computed overlays read from their per-snapshot caches (blank if absent).
        compounds["struct_alert"] = cls._computed_label(
            snap, "structural_alerts.parquet", "struct_alert", compounds.index
        )
        compounds["novelty"] = cls._computed_label(
            snap, "novelty.parquet", "novelty", compounds.index
        )

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
            target_accessions=cls._target_accessions(targets),
            target_embeddings=TargetESM.load(snap / "target_esm.npz"),
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
        cols = ["target_chembl_id", "target_pref_name", "target_type", "organism", "protein_class_id", "accession"]
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
    def _overlay_columns(
        compounds: pd.DataFrame, mechs: pd.DataFrame
    ) -> dict[str, object]:
        """Per-compound display overlays: MoA, development stage, safety flag.

        These are *labels*, not feature spaces, so they are deliberately not
        domains -- embedding a k-value label makes a map of k identical points
        (the lesson the modality domain is built around). They ride along as
        fields the UI can colour or filter any existing map by. All three are
        read from columns the snapshot already ships, so nothing is
        re-extracted; the sparse ones (MoA, stage) are simply blank where ChEMBL
        has no annotation, which the UI shows as 미분류.
        """
        idx = compounds.index

        def num(col: str) -> pd.Series:
            if col not in compounds.columns:
                return pd.Series(np.nan, index=idx)
            return pd.to_numeric(compounds[col], errors="coerce")

        # Development stage from ChEMBL max_phase. Two levels: a coarse L1
        # (approved/clinical/pre) and the finer L2 phase, so a map can be
        # coloured at either granularity.
        phase = num("max_phase")
        dev_stage = np.select(
            [phase >= 4, phase > 0], ["승인", "임상"], default="전임상/미상"
        )
        dev_stage_l2 = np.select(
            [phase >= 4, phase == 3, phase == 2, phase == 1, phase > 0],
            ["Phase 4 (시판)", "Phase 3", "Phase 2", "Phase 1", "Early Phase 1"],
            default="전임상/미상",
        )

        # Safety flags; a withdrawal is the stronger statement, so it wins.
        wd = num("withdrawn_flag").fillna(0)
        bb = num("black_box_warning").fillna(0)
        safety = np.select([wd == 1, bb == 1], ["시판중단", "블랙박스경고"], default="")

        # MoA. L1 = the compound's first curated action_type (INHIBITOR -> Inhibitor).
        # L2 refines only INHIBITOR into reversible/irreversible via the covalent
        # warhead flag (this snapshot ships no reversibility field); the other
        # action types (PARTIAL AGONIST, PAM, NAM, …) are already L2-specific.
        if "action_type" in mechs.columns:
            at = (
                mechs.dropna(subset=["action_type"])
                .drop_duplicates("chembl_id")
                .set_index("chembl_id")["action_type"]
                .astype(str)
                .reindex(idx)
            )
            moa = at.str.title().fillna("").to_numpy()
            cov = (
                (compounds["covalent_tier"] == "strong").to_numpy()
                if "covalent_tier" in compounds.columns
                else np.zeros(len(idx), dtype=bool)
            )
            is_inh = (at == "INHIBITOR").fillna(False).to_numpy()
            moa_l2 = np.where(
                at.isna().to_numpy(),
                "",
                np.where(
                    is_inh,
                    np.where(cov, "Irreversible (Covalent) Inhibitor", "Reversible Inhibitor"),
                    at.str.title().fillna("").to_numpy(),
                ),
            )
        else:
            moa = np.full(len(idx), "", dtype=object)
            moa_l2 = np.full(len(idx), "", dtype=object)

        # Modality L2 from molecule_type + size (RDKit mw). Blank molecule_type
        # (27% of the snapshot) stays 미분류; the structural modality domain still
        # places those, this only labels them.
        mtype = (
            compounds["molecule_type"].fillna("").astype(str).to_numpy()
            if "molecule_type" in compounds.columns
            else np.full(len(idx), "", dtype=object)
        )
        w = num("mw").to_numpy()
        sm = mtype == "Small molecule"
        modality_l2 = np.select(
            [
                sm & (w >= 500) & (w <= 1000),
                sm & (w < 500),
                mtype == "Antibody",
                np.isin(mtype, ["Protein", "Peptide"]) & (w > 1000),
            ],
            [
                "Macrocycle-like (MW 500-1000)",
                "Synthetic organic (MW<500)",
                "mAb/ADC/Bispecific",
                "Large Biologics",
            ],
            default="",
        )

        # Physicochemical L2: simplified BCS / bRo5 subclass (RDKit mw/clogp/tpsa),
        # following the xlsx taxonomy. Blank where no descriptors exist.
        mw = num("mw").to_numpy()
        lp = num("clogp").to_numpy()
        psa = num("tpsa").to_numpy()
        physchem_l2 = np.select(
            [
                (mw <= 500) & (lp >= 1) & (lp <= 3) & (psa <= 90),
                (mw <= 500) & (lp > 3) & (psa <= 90),
                (mw <= 500) & (lp <= 1) & (psa > 90),
                (mw > 500) & (lp > 5),
                (mw > 500),
                (lp > 5),
            ],
            [
                "BCS1 (고용해·고투과)",
                "BCS2 (저용해·고투과)",
                "BCS3 (고용해·저투과)",
                "고MW·고지용성",
                "고MW (>500)",
                "고지용성 (LogP>5)",
            ],
            default="기타(소분자)",
        )
        physchem_l2 = np.where(np.isnan(mw), "", physchem_l2)

        return {
            "moa": moa,
            "moa_l2": moa_l2,
            "dev_stage": dev_stage,
            "dev_stage_l2": dev_stage_l2,
            "modality_l2": modality_l2,
            "physchem_l2": physchem_l2,
            "safety": safety,
        }

    @staticmethod
    def _computed_label(snap: Path, filename: str, column: str, idx: pd.Index) -> pd.Series:
        """A per-compound computed-overlay label from its per-snapshot cache.

        구조알림 and 신규성 are *computed* (not in ChEMBL), so each is built once
        by its ``scripts/build_*.py`` into a parquet keyed by ``chembl_id`` and
        merged here. A snapshot built before a cache existed simply gets a blank
        column -- it still loads, the overlay is just empty until the script is
        run. Compounds absent from the cache also stay blank.
        """
        path = snap / filename
        if not path.exists():
            return pd.Series("", index=idx, dtype=object)
        df = pd.read_parquet(path, columns=["chembl_id", column])
        s = df.set_index("chembl_id")[column].astype(str)
        return s.reindex(idx).fillna("")

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

    @staticmethod
    def _target_accessions(targets: pd.DataFrame) -> dict[str, tuple[str, ...]]:
        """Compound -> UniProt accessions of the targets it hits.

        Parallel to :meth:`_target_terms` but keeps the accession, which is the
        key into the ESM-2 sequence embeddings.
        """
        if "accession" not in targets.columns:
            return {}
        acc: dict[str, set[str]] = {}
        for rec in targets.itertuples(index=False):
            a = getattr(rec, "accession", None)
            if isinstance(a, str) and a:
                acc.setdefault(rec.chembl_id, set()).add(a)
        return {k: tuple(sorted(v)) for k, v in acc.items()}

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
