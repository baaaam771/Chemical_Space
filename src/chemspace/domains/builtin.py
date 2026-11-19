"""The five built-in domains.

Each one answers "what makes these compounds similar?" differently, so the same
compound set lands in a different arrangement under each.

A note on the modality domain, because it is the one that is easy to get wrong:
modality is a *label*, and one-hot encoding a label produces a map with k
identical points and no structure.  So the modality domain embeds the
**structural properties that define modality** -- size, flexibility, ring-system
count, amide backbone, largest ring, electrophilic warheads -- and uses the
label only to colour the result.  Peptides, macrocycles and covalent compounds
then occupy regions you can actually select.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd

from chemspace.data.dataset import Dataset
from chemspace.data.derive import COVALENT_WARHEADS, PHYSCHEM_FEATURES
from chemspace.domains.base import (
    Domain,
    DomainFeatures,
    multi_hot,
    prune_unsupported,
    standardize_within_subset,
)

__all__ = [
    "PhysChemDomain",
    "DiseaseDomain",
    "TargetDomain",
    "MoietyDomain",
    "ModalityDomain",
]


def _top(counter: pd.Series, k: int = 5) -> list[tuple[str, int]]:
    return [(str(i), int(v)) for i, v in counter.head(k).items()]


def _strip(term: str) -> str:
    """Drop the source prefix: ``MeSH:Breast Neoplasms`` -> ``Breast Neoplasms``."""
    return term.split(":", 1)[1] if ":" in term else term


def _section(title: str, series: pd.Series, top: int) -> dict:
    """One block of the 구성 종류 view: the distinct-kind count and the first
    ``top`` kinds with their frequencies."""
    return {
        "title": title,
        "total": int(len(series)),
        "items": [{"name": _strip(str(i)), "count": int(v)} for i, v in series.head(top).items()],
    }


def _lead_summary(
    specific: pd.Series, context: pd.Series, empty: str, k: int = 3, tag_min: float = 0.5
) -> str:
    """A cluster label that leads with the *specific* things it contains and
    tags the broad category in parentheses -- but only when that category is
    actually representative.

    The broad category alone (14 ATC groups, or L1 protein classes) is too
    coarse to tell two clusters apart -- every oncology cluster reads
    "항암·면역조절". Leading with the actual indications or targets is what makes
    a cluster's meaning legible; the parenthesised context keeps the altitude.

    The tag is the single most common context term, which is only a fair label
    when the set is reasonably homogeneous. On a mixed selection the leads are
    cancers while the top ATC area is CNS, and "Neoplasms … (신경계)" is simply
    wrong -- so the tag is dropped unless the top context term holds at least
    ``tag_min`` of the context annotations. Falls back to the context list when
    nothing specific is annotated.
    """
    leads = [_strip(str(i)) for i in specific.head(k).index]
    if leads:
        tag = ""
        total = float(context.sum()) if len(context) else 0.0
        if total and float(context.iloc[0]) / total >= tag_min:
            tag = f" ({_strip(str(context.index[0]))})"
        return " · ".join(leads) + tag
    if len(context):
        return " · ".join(f"{_strip(str(i))} ({v})" for i, v in context.head(k).items())
    return empty


class _DatasetDomain(Domain):
    def __init__(self, dataset: Dataset):
        self.ds = dataset

    def color_series(self, compound_ids: Sequence[str]) -> pd.Series | None:
        """Categorical column worth colouring points by, if any."""
        return None


# --------------------------------------------------------------------------- #


class PhysChemDomain(_DatasetDomain):
    name = "physchem"
    label = "물성"
    description = "분자량·지용성·극성 등 물리화학 기술자 (RDKit 계산, 부분집합 내 z-score)"

    def featurize(self, compound_ids: Sequence[str]) -> DomainFeatures:
        ids = tuple(compound_ids)
        raw = self.ds.frame(ids)[list(PHYSCHEM_FEATURES)].to_numpy(dtype=np.float64)
        X, notes = standardize_within_subset(raw)
        names = tuple(
            n for n, keep in zip(PHYSCHEM_FEATURES, _kept_mask(raw)) if keep
        )
        return DomainFeatures(
            X=X,
            compound_ids=ids,
            covered=np.ones(len(ids), dtype=bool),
            feature_names=names,
            metric="euclidean",
            kind="continuous",
            notes=notes,
        )

    def profile(self, compound_ids: Sequence[str]) -> dict:
        df = self.ds.frame(compound_ids)
        med = df[list(PHYSCHEM_FEATURES)].median(numeric_only=True)
        return {
            "n": len(df),
            "summary": (
                f"MW {med['mw']:.0f} · cLogP {med['clogp']:.1f} · "
                f"TPSA {med['tpsa']:.0f} · QED {med['qed']:.2f}"
            ),
            "medians": {k: round(float(v), 2) for k, v in med.items()},
            "region": _physchem_region(med),
        }

    def composition(self, compound_ids: Sequence[str], top: int = 25) -> list[dict]:
        # Continuous, so there are no "kinds" to count -- show each descriptor's
        # median instead, which is what characterises a physchem region.
        med = self.ds.frame(compound_ids)[list(PHYSCHEM_FEATURES)].median(numeric_only=True)
        items = [{"name": str(k), "count": round(float(v), 2)} for k, v in med.items()]
        return [{"title": "물리화학 기술자 (중앙값)", "total": len(items), "items": items}]


def _kept_mask(raw: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.nanstd(np.asarray(raw, dtype=np.float64), axis=0) > eps


def _physchem_region(med: pd.Series) -> str:
    mw, clogp = float(med["mw"]), float(med["clogp"])
    if mw < 300 and clogp < 3:
        return "fragment/lead-like"
    if mw <= 500 and clogp <= 5:
        return "drug-like (Ro5)"
    return "bRo5"


# --------------------------------------------------------------------------- #


class DiseaseDomain(_DatasetDomain):
    name = "disease"
    label = "질환"
    description = "MeSH 적응증 + ATC 해부학적 주군 (다중 라벨, Jaccard)"

    def __init__(self, dataset: Dataset, min_support: int | None = None):
        super().__init__(dataset)
        self.min_support = min_support

    def featurize(self, compound_ids: Sequence[str]) -> DomainFeatures:
        ids = tuple(compound_ids)
        X, names, notes = multi_hot(ids, self.ds.disease_terms, self.min_support)
        covered = np.array([c in self.ds.disease_terms for c in ids], dtype=bool)
        return DomainFeatures(
            X=X,
            compound_ids=ids,
            covered=covered,
            feature_names=names,
            metric="jaccard",
            kind="binary",
            notes=notes,
        )

    def profile(self, compound_ids: Sequence[str]) -> dict:
        terms = [t for c in compound_ids for t in self.ds.disease_terms.get(c, ())]
        counts = pd.Series(terms, dtype="object").value_counts()
        mesh = counts[counts.index.str.startswith("MeSH:")]
        efo = counts[counts.index.str.startswith("EFO:")]
        atc = counts[counts.index.str.startswith("ATC:")]
        # The specific indications (MeSH + EFO) lead; the ATC area tags the
        # altitude. Both are already computed -- the old summary just threw the
        # specific half away and showed only the 14 coarse ATC groups.
        specific = pd.concat([mesh, efo]).sort_values(ascending=False)
        covered = sum(1 for c in compound_ids if c in self.ds.disease_terms)
        return {
            "n": len(compound_ids),
            "covered": covered,
            "summary": _lead_summary(specific, atc, "질환 정보 없음"),
            "top_indications": _top(mesh),
            "top_areas": _top(atc),
        }

    def color_series(self, compound_ids: Sequence[str]) -> pd.Series | None:
        def primary(cid: str) -> str:
            atc = [t for t in self.ds.disease_terms.get(cid, ()) if t.startswith("ATC:")]
            return atc[0].split(":", 1)[1] if atc else "미분류"

        return pd.Series([primary(c) for c in compound_ids], index=list(compound_ids))

    def composition(self, compound_ids: Sequence[str], top: int = 25) -> list[dict]:
        terms = [t for c in compound_ids for t in self.ds.disease_terms.get(c, ())]
        counts = pd.Series(terms, dtype="object").value_counts()
        mesh = counts[counts.index.str.startswith("MeSH:")]
        efo = counts[counts.index.str.startswith("EFO:")]
        atc = counts[counts.index.str.startswith("ATC:")]
        specific = pd.concat([mesh, efo]).sort_values(ascending=False)
        out = []
        if len(specific):
            out.append(_section("질병 (MeSH/EFO)", specific, top))
        if len(atc):
            out.append(_section("치료영역 (ATC)", atc, top))
        return out

    def term_counts(self, compound_ids: Sequence[str]) -> dict:
        return Counter(t for c in compound_ids for t in self.ds.disease_terms.get(c, ()))


# --------------------------------------------------------------------------- #


class TargetDomain(_DatasetDomain):
    name = "target"
    label = "표적"
    description = "표적의 단백질 계열 L1~L3 + 개별 표적 — 큐레이션 MoA + 측정 활성 (Jaccard)"

    def __init__(self, dataset: Dataset, min_support: int | None = None):
        super().__init__(dataset)
        self.min_support = min_support

    def featurize(self, compound_ids: Sequence[str]) -> DomainFeatures:
        ids = tuple(compound_ids)
        X, names, notes = multi_hot(ids, self.ds.target_terms, self.min_support)
        covered = np.array([c in self.ds.target_terms for c in ids], dtype=bool)
        return DomainFeatures(
            X=X,
            compound_ids=ids,
            covered=covered,
            feature_names=names,
            metric="jaccard",
            kind="binary",
            notes=notes,
        )

    def profile(self, compound_ids: Sequence[str]) -> dict:
        terms = [t for c in compound_ids for t in self.ds.target_terms.get(c, ())]
        counts = pd.Series(terms, dtype="object").value_counts()
        fam = counts[counts.index.str.startswith(("L1:", "L2:"))]
        l1 = counts[counts.index.str.startswith("L1:")]
        tgt = counts[counts.index.str.startswith("TGT:")]
        covered = sum(1 for c in compound_ids if c in self.ds.target_terms)
        # Lead with the specific proteins, tag the family -- same reasoning as
        # the disease summary: "Membrane receptor" does not tell two target
        # clusters apart, "Alpha-1A adrenergic receptor" does. The tag is built
        # from L1 alone (one per compound) so its share is a clean fraction;
        # fam mixes L1 and L2 and would double-count.
        return {
            "n": len(compound_ids),
            "covered": covered,
            "summary": _lead_summary(tgt, l1, "표적 정보 없음"),
            "top_families": _top(fam),
            "top_targets": _top(tgt),
        }

    def color_series(self, compound_ids: Sequence[str]) -> pd.Series | None:
        def primary(cid: str) -> str:
            l1 = [t for t in self.ds.target_terms.get(cid, ()) if t.startswith("L1:")]
            return l1[0].split(":", 1)[1] if l1 else "미분류"

        return pd.Series([primary(c) for c in compound_ids], index=list(compound_ids))

    def composition(self, compound_ids: Sequence[str], top: int = 25) -> list[dict]:
        terms = [t for c in compound_ids for t in self.ds.target_terms.get(c, ())]
        counts = pd.Series(terms, dtype="object").value_counts()
        tgt = counts[counts.index.str.startswith("TGT:")]
        l1 = counts[counts.index.str.startswith("L1:")]
        l2 = counts[counts.index.str.startswith("L2:")]
        out = []
        if len(tgt):
            out.append(_section("단백질 표적 (TGT)", tgt, top))
        if len(l1):
            out.append(_section("단백질 계열 L1", l1, top))
        if len(l2):
            out.append(_section("단백질 계열 L2", l2, top))
        return out

    def term_counts(self, compound_ids: Sequence[str]) -> dict:
        return Counter(t for c in compound_ids for t in self.ds.target_terms.get(c, ()))


# --------------------------------------------------------------------------- #


class MoietyDomain(_DatasetDomain):
    name = "moiety"
    label = "모이어티"
    description = "Murcko 스캐폴드의 ECFP4 비트 (부분집합 내 지지도 없는 비트 제거, Jaccard)"

    def __init__(self, dataset: Dataset, min_support: int = 2):
        super().__init__(dataset)
        self.min_support = min_support

    def featurize(self, compound_ids: Sequence[str]) -> DomainFeatures:
        ids = tuple(compound_ids)
        rows = self.ds.rows(ids)
        raw = self.ds.scaffold_fps[rows].astype(np.float32)
        names = tuple(f"bit{i}" for i in range(raw.shape[1]))
        X, names, notes = prune_unsupported(raw, names, self.min_support)
        covered = (self.ds.frame(ids)["murcko_smiles"] != "").to_numpy(dtype=bool)
        return DomainFeatures(
            X=X,
            compound_ids=ids,
            covered=covered,
            feature_names=names,
            metric="jaccard",
            kind="binary",
            notes={**notes, "vocab_full": raw.shape[1]},
        )

    def profile(self, compound_ids: Sequence[str]) -> dict:
        df = self.ds.frame(compound_ids)
        scaf = df["murcko_smiles"].replace("", np.nan).dropna().value_counts()
        generic = df["murcko_generic"].replace("", np.nan).dropna().value_counts()
        return {
            "n": len(df),
            "covered": int((df["murcko_smiles"] != "").sum()),
            "summary": (
                f"{len(scaf)}개 스캐폴드 / 최빈 {scaf.iloc[0]}개"
                if len(scaf)
                else "스캐폴드 없음"
            ),
            "n_scaffolds": int(len(scaf)),
            "top_scaffolds": _top(scaf),
            "top_frameworks": _top(generic, 3),
        }

    def composition(self, compound_ids: Sequence[str], top: int = 25) -> list[dict]:
        df = self.ds.frame(compound_ids)
        scaf = df["murcko_smiles"].replace("", np.nan).dropna().value_counts()
        generic = df["murcko_generic"].replace("", np.nan).dropna().value_counts()
        out = []
        if len(scaf):
            out.append(_section("스캐폴드 (Murcko)", scaf, top))
        if len(generic):
            out.append(_section("골격 (generic)", generic, top))
        return out


# --------------------------------------------------------------------------- #

#: Structural properties that distinguish one modality from another.
_MODALITY_CONTINUOUS = (
    "mw",
    "rotb",
    "max_ring_size",
    "n_amide_bonds",
    "n_ring_systems",
    "fsp3",
    "tpsa",
    "hbd",
    "hba",
    "aromatic_rings",
    "heavy_atoms",
    "ring_count",
)


class ModalityDomain(_DatasetDomain):
    name = "modality"
    label = "모달리티"
    description = (
        "모달리티를 규정하는 구조 특성(크기·유연성·고리계·아마이드 골격·warhead)으로 배치하고 "
        "규칙 기반 라벨은 색으로 표시"
    )

    def featurize(self, compound_ids: Sequence[str]) -> DomainFeatures:
        ids = tuple(compound_ids)
        df = self.ds.frame(ids)
        cont = df[list(_MODALITY_CONTINUOUS)].to_numpy(dtype=np.float64)

        names = [n for n, _, _ in COVALENT_WARHEADS]
        wh = np.zeros((len(ids), len(names) + 2), dtype=np.float64)
        for j, name in enumerate(names):
            wh[:, j] = (df["covalent_warhead"] == name).to_numpy(dtype=float)
        # The strong-tier flag, not "any electrophile": a generic enone says
        # little about modality, so it should not pull a compound toward the
        # covalent region of the map.
        wh[:, -2] = (df["covalent_tier"] == "strong").to_numpy(dtype=float)
        wh[:, -1] = (df["e3_ligand"] != "").to_numpy(dtype=float)

        raw = np.hstack([cont, wh])
        all_names = (
            *_MODALITY_CONTINUOUS,
            *(f"warhead:{n}" for n in names),
            "has_strong_warhead",
            "has_e3_ligand",
        )
        X, notes = standardize_within_subset(raw)
        names = tuple(n for n, keep in zip(all_names, _kept_mask(raw)) if keep)
        return DomainFeatures(
            X=X,
            compound_ids=ids,
            covered=np.ones(len(ids), dtype=bool),
            feature_names=names,
            metric="euclidean",
            kind="continuous",
            notes=notes,
        )

    def profile(self, compound_ids: Sequence[str]) -> dict:
        df = self.ds.frame(compound_ids)
        counts = df["modality"].value_counts()
        strong = df.loc[df["covalent_tier"] == "strong", "covalent_warhead"].value_counts()
        weak = df.loc[df["covalent_tier"] == "weak", "covalent_warhead"].value_counts()
        return {
            "n": len(df),
            "summary": " · ".join(f"{i} ({v})" for i, v in counts.head(3).items()),
            "modalities": _top(counts),
            "warheads": _top(strong, 3),
            "weak_electrophiles": _top(weak, 3),
            "n_e3_ligand": int((df["e3_ligand"] != "").sum()),
        }

    def color_series(self, compound_ids: Sequence[str]) -> pd.Series | None:
        return self.ds.frame(compound_ids)["modality"]

    def composition(self, compound_ids: Sequence[str], top: int = 25) -> list[dict]:
        counts = self.ds.frame(compound_ids)["modality"].value_counts()
        return [_section("모달리티", counts, top)] if len(counts) else []

    def term_counts(self, compound_ids: Sequence[str]) -> dict:
        return Counter(f"MOD:{m}" for m in self.ds.frame(compound_ids)["modality"])
