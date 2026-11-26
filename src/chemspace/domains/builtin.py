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
    auto_min_support,
    multi_hot,
    prune_unsupported,
    standardize_within_subset,
)

__all__ = [
    "PhysChemDomain",
    "DiseaseDomain",
    "TargetDomain",
    "TargetFamilyDomain",
    "TargetFamilyMapDomain",
    "TargetActivityDomain",
    "TargetSeqDomain",
    "TargetKnowledgeDomain",
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


class TargetFamilyDomain(TargetDomain):
    """The target view re-laid-out so the curated protein families *group*.

    The plain 표적 domain weights every token equally, so the individual target
    (``TGT:``) tokens -- the most numerous -- dominate the Jaccard distance and
    the map fragments into many small per-target clusters. The curated family is
    present but visually scattered.

    This domain up-weights the curated family: under a set-based Jaccard metric,
    repeating a token is how you weight it, so ``FAM:Kinase`` enters the feature
    set several times and ``SUB:Kinase▸TK`` a couple of times, while each
    ``TGT:`` stays at weight one. Same-family compounds then share many bits and
    pull into one region; the individual targets still differ, so the region
    keeps internal spread instead of collapsing to a single stacked point (the
    modality lesson). Narrowing into one family re-featurizes that subset, where
    the family tokens are constant and the subfamily/target tokens separate it.

    This is a *label-guided* layout -- the geometry is deliberately organised by
    the family we curated, not discovered from structure alone. That is exactly
    the ask: a map whose big shapes are the pharmacological families, coloured by
    the same curation (표적 계열) so shape and colour agree. The weights are the
    one knob between "crisp blobs" and "natural connections".
    """

    name = "target_family"
    label = "표적(계열)"
    description = (
        "큐레이션 단백질 계열(L1/L2)을 가중한 표적 토큰 — 같은 계열이 한 영역으로 뭉치고 "
        "개별 표적이 내부를 퍼뜨림 (Jaccard). 색은 표적 계열과 일치."
    )

    #: Token repeats = weight under Jaccard. Tuned for "blobs, not dots": strong
    #: enough that families separate, light enough that targets still spread.
    FAM_WEIGHT = 4
    SUB_WEIGHT = 2

    def _family_memberships(self, ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
        df = self.ds.frame(ids)
        l1_map = df["target_l1"].to_dict() if "target_l1" in df.columns else {}
        l2_map = df["target_l2"].to_dict() if "target_l2" in df.columns else {}
        terms: dict[str, tuple[str, ...]] = {}
        for c in ids:
            toks: list[str] = []
            l1 = str(l1_map.get(c) or "")
            l2 = str(l2_map.get(c) or "")
            if l1:
                toks += [f"FAM:{l1}#{k}" for k in range(self.FAM_WEIGHT)]
                if l2:
                    toks += [f"SUB:{l1}>{l2}#{k}" for k in range(self.SUB_WEIGHT)]
            # Individual targets give within-family spread (weight 1).
            toks += [t for t in self.ds.target_terms.get(c, ()) if t.startswith("TGT:")]
            if toks:
                terms[c] = tuple(toks)
        return terms

    def featurize(self, compound_ids: Sequence[str]) -> DomainFeatures:
        ids = tuple(compound_ids)
        memberships = self._family_memberships(ids)
        X, names, notes = multi_hot(ids, memberships, self.min_support)
        covered = np.array([bool(memberships.get(c)) for c in ids], dtype=bool)
        return DomainFeatures(
            X=X,
            compound_ids=ids,
            covered=covered,
            feature_names=names,
            metric="jaccard",
            kind="binary",
            notes={**notes, "fam_weight": self.FAM_WEIGHT, "sub_weight": self.SUB_WEIGHT},
        )

    def color_series(self, compound_ids: Sequence[str]) -> pd.Series | None:
        df = self.ds.frame(compound_ids)
        if "target_l1" not in df.columns:
            return super().color_series(compound_ids)
        return df["target_l1"].astype(str).replace("", "미분류")


def _label_codes(values: np.ndarray) -> np.ndarray:
    """Map strings to int codes for supervision; '' -> -1 (unknown)."""
    uniq = {v: i for i, v in enumerate(sorted({v for v in values if v}))}
    return np.array([uniq.get(v, -1) for v in values], dtype=np.int64)


class TargetFamilyMapDomain(TargetDomain):
    """Semi-supervised family map: the curated family *guides* the layout.

    Method A (``TargetFamilyDomain``) up-weighted discrete family tokens under
    Jaccard, which over-fragments -- near-duplicate token sets scatter into many
    small islands instead of forming family continents. This domain fixes that by
    separating the two jobs:

    * **Geometry comes from chemical structure** (the compound's own ECFP bits),
      which is smooth and continuous -- no token-set duplication, no shattering.
    * **The macro layout comes from the curated family as a supervision label**
      handed to UMAP (``y``). Same-family compounds are drawn into one region;
      different families separate. The result is the thing the plain overlay
      could not give: coherent family continents, each an internal chemical
      landscape.

    The supervision is *adaptive*, which is what makes drill-down work. When the
    subset spans several families, it supervises by curated L1 -> eight
    continents. Narrow into one family and every point shares that L1, so it
    supervises by L2 instead -> the subfamilies (TK, CMGC, …) separate. Coverage
    is the compounds with a curated family; the rest (no protein target, or a
    target outside the eight families) are placed in the uncovered band.

    The clustering is label-driven too: each point's cluster is its curated family
    (the same adaptive code), so the clusters ARE the continents -- selecting a
    cluster selects a whole family -- instead of HDBSCAN carving each continent
    into many density blobs that disagree with the colouring.
    """

    name = "target_family_map"
    label = "표적(계열지도)"
    description = (
        "화학구조(ECFP)로 배치하되 큐레이션 계열을 UMAP 지도 신호(준지도)로 줘 "
        "같은 계열이 한 대륙으로 모임 — 좁히면 하위계열로 자동 전환. 색은 표적 계열과 일치."
    )

    #: 0 ignores the family, 1 lets it dominate. Mid-high gives separated
    #: continents that still keep their internal chemical structure.
    TARGET_WEIGHT = 0.6

    def _family_labels(self, ids: tuple[str, ...]) -> np.ndarray:
        df = self.ds.frame(ids)
        l1 = (
            df["target_l1"].astype(str).to_numpy()
            if "target_l1" in df.columns
            else np.full(len(ids), "", dtype=object)
        )
        l2 = (
            df["target_l2"].astype(str).to_numpy()
            if "target_l2" in df.columns
            else np.full(len(ids), "", dtype=object)
        )
        covered = l1 != ""
        # One family in view -> separate its subfamilies; many -> separate families.
        if len({v for v in l1[covered]}) > 1:
            label_vals = l1
        else:
            label_vals = np.where(l2 != "", l2, l1)
        return np.where(covered, label_vals, "")

    def featurize(self, compound_ids: Sequence[str]) -> DomainFeatures:
        ids = tuple(compound_ids)
        rows = self.ds.rows(ids)
        raw = self.ds.fps[rows].astype(np.float32)
        names = tuple(f"bit{i}" for i in range(raw.shape[1]))
        X, names, notes = prune_unsupported(raw, names, self.min_support)

        labels = self._family_labels(ids)
        covered = labels != ""
        sup = _label_codes(labels)
        n_fam = len({v for v in labels if v})
        return DomainFeatures(
            X=X,
            compound_ids=ids,
            covered=covered,
            feature_names=names,
            metric="jaccard",
            kind="binary",
            notes={**notes, "supervised_families": n_fam},
            supervision=sup,
            supervision_weight=self.TARGET_WEIGHT,
            # The clusters ARE the curated families (adaptive: L1 across families,
            # L2 within one), so the grouping matches the colouring exactly and the
            # continents are the clusters -- no HDBSCAN over-segmentation.
            cluster_labels=sup,
        )

    def color_series(self, compound_ids: Sequence[str]) -> pd.Series | None:
        df = self.ds.frame(compound_ids)
        if "target_l1" not in df.columns:
            return super().color_series(compound_ids)
        return df["target_l1"].astype(str).replace("", "미분류")


def _standardize_covered(X: np.ndarray, covered: np.ndarray, eps: float = 1e-9):
    """Z-score columns using the covered rows' own mean/spread, drop constants.

    Unlike :func:`standardize_within_subset`, the statistics come from the
    covered rows only (uncovered rows are all-zero placeholders that
    ``place`` never embeds), so they do not pull the mean toward zero.
    """
    X = np.asarray(X, dtype=np.float64)
    cov = X[covered]
    if cov.shape[0] < 2:
        return np.zeros((X.shape[0], 0), np.float32), {"kept": 0}
    mu, sd = cov.mean(0), cov.std(0)
    keep = sd > eps
    Z = np.zeros((X.shape[0], int(keep.sum())), np.float32)
    Z[covered] = ((cov[:, keep] - mu[keep]) / sd[keep]).astype(np.float32)
    return Z, {"kept": int(keep.sum()), "dropped_constant": int((~keep).sum())}


class TargetSeqDomain(TargetDomain):
    """The target domain re-laid-out by protein *sequence* similarity.

    Same compounds, same target tokens for every text tab (so a cluster still
    reads as its enriched ``TGT:``/family terms), but the map and clusters are
    built from ESM-2 sequence embeddings instead of Jaccard over target tokens.
    Homologous targets land near each other even when they share no token, and
    the distances actually spread -- see the measurement in the README. Only
    compounds with a UniProt-mapped protein target are covered.
    """

    name = "target_seq"
    label = "표적(서열)"
    description = "표적 단백질 서열 임베딩 (ESM-2 650M, 잔기 평균 → 부분집합 z-score) — 같은 표적 토큰으로 설명"

    def featurize(self, compound_ids: Sequence[str]) -> DomainFeatures:
        ids = tuple(compound_ids)
        esm = self.ds.target_embeddings
        acc = self.ds.target_accessions
        dim = esm.dim if esm else 0
        X = np.zeros((len(ids), dim), np.float64)
        covered = np.zeros(len(ids), dtype=bool)
        if esm is not None:
            for i, c in enumerate(ids):
                v = esm.mean_vector(acc.get(c, ()))
                if v is not None:
                    X[i] = v
                    covered[i] = True
        Z, notes = _standardize_covered(X, covered)
        return DomainFeatures(
            X=Z,
            compound_ids=ids,
            covered=covered,
            feature_names=tuple(f"esm{i}" for i in range(Z.shape[1])),
            metric="euclidean",
            kind="continuous",
            notes=notes,
        )


class TargetKnowledgeDomain(TargetDomain):
    """The target view re-laid-out by LLM *knowledge* of the proteins.

    Same compounds and the same target tokens for every text tab (so a cluster
    still reads as its enriched ``TGT:``/family terms), but the map is built from
    a language model's text-embedding of each target's description instead of
    sequence (ESM) or shared-ligand (Jaccard) similarity. Targets a model treats
    as biologically related -- same pathway, same disease, same mechanism -- land
    near each other even when they share no ligand and little sequence. A
    compound's vector is the mean over the knowledge vectors of its targets.

    This is a *knowledge*-derived layout: it reflects what the model has read,
    which need not line up with measured bioactivity -- see the README note. Only
    compounds whose targets are in ``target_knowledge.npz`` are covered; the
    domain is registered only when that cache exists.
    """

    name = "target_knowledge"
    label = "표적(지식)"
    description = (
        "표적 설명(이름·계열·질환)의 LLM 텍스트 임베딩으로 배치 — 기능적으로 "
        "가까운 표적이 모임 (부분집합 z-score). 같은 표적 토큰으로 설명. "
        "측정 활성이 아닌 '지식' 기반이라 생물활성과 다를 수 있음."
    )

    def featurize(self, compound_ids: Sequence[str]) -> DomainFeatures:
        ids = tuple(compound_ids)
        emb = self.ds.target_knowledge
        tids = self.ds.target_ids
        dim = emb.dim if emb else 0
        X = np.zeros((len(ids), dim), np.float64)
        covered = np.zeros(len(ids), dtype=bool)
        if emb is not None:
            for i, c in enumerate(ids):
                v = emb.mean_vector(tids.get(c, ()))
                if v is not None:
                    X[i] = v
                    covered[i] = True
        Z, notes = _standardize_covered(X, covered)
        return DomainFeatures(
            X=Z,
            compound_ids=ids,
            covered=covered,
            feature_names=tuple(f"k{i}" for i in range(Z.shape[1])),
            metric="euclidean",
            kind="continuous",
            notes=notes,
        )


class TargetActivityDomain(TargetDomain):
    """The target view laid out by the *measured bioactivity profile*.

    Each compound is a vector over targets weighted by how strongly it hits each
    (pchembl), and the map places compounds by the *shape* of that profile
    (cosine): compounds with the same polypharmacology -- the same targets hit at
    the same relative potencies -- land together, whatever their structure.

    This is the activity-grounded counterpart to the knowledge/sequence views:
    it is built from measured data, not from what a model knows or from sequence
    similarity, so it does not inherit the "knowledge space ≠ bioactivity space"
    gap. Needs no external model -- the activity + mechanism layers already ship.
    Only compounds with at least one supported target (in this subset) are placed.
    """

    name = "target_activity"
    label = "표적(활성)"
    description = (
        "측정 활성 프로파일 — 표적별 pchembl 벡터의 방향(cosine)으로 배치. 같은 표적을 "
        "비슷한 세기로 치는(폴리파마콜로지가 닮은) 화합물이 모임. 외부 모델 불필요."
    )

    def featurize(self, compound_ids: Sequence[str]) -> DomainFeatures:
        ids = tuple(compound_ids)
        prof = self.ds.target_activity
        counts = Counter(t for c in ids for (t, _) in prof.get(c, ()))
        threshold = (
            auto_min_support(len(ids)) if self.min_support is None else int(self.min_support)
        )
        vocab = sorted(t for t, n in counts.items() if n >= threshold)
        if not vocab and counts:  # threshold too aggressive for this subset
            vocab = sorted(counts)
        index = {t: j for j, t in enumerate(vocab)}
        X = np.zeros((len(ids), len(vocab)), dtype=np.float32)
        for i, c in enumerate(ids):
            for t, v in prof.get(c, ()):
                j = index.get(t)
                if j is not None:
                    X[i, j] = v
        # Covered = rows that survive subset pruning with a non-zero profile, so
        # cosine never sees a zero-norm vector.
        covered = (X != 0).any(axis=1)
        return DomainFeatures(
            X=X,
            compound_ids=ids,
            covered=covered,
            feature_names=tuple(f"TGT:{t}" for t in vocab),
            metric="cosine",
            kind="continuous",
            notes={"vocab_full": len(counts), "vocab_kept": len(vocab), "min_support": threshold},
        )

    def color_series(self, compound_ids: Sequence[str]) -> pd.Series | None:
        df = self.ds.frame(compound_ids)
        if "target_l1" not in df.columns:
            return super().color_series(compound_ids)
        return df["target_l1"].astype(str).replace("", "미분류")


def _family_colour(ds, compound_ids):
    """Colour any target-derived domain by the curated L1 family, blanks -> 미분류."""
    df = ds.frame(compound_ids)
    if "target_l1" not in df.columns:
        return None
    return df["target_l1"].astype(str).replace("", "미분류")


class TargetFamilyActivityDomain(TargetDomain):
    """A — dense per-family activity profile.

    Each compound becomes a short vector over the curated subfamilies, weighted
    by its strongest pchembl on each. The sparse thousand-target token space
    (which scatters) collapses to a dense low-dim one that clusters as cleanly as
    the continuous physchem descriptors.
    """

    name = "target_fam_activity"
    label = "표적(계열활성)"
    description = (
        "큐레이션 계열별 최강 pchembl의 밀집 벡터 — 희소 표적을 저차원으로 압축해 "
        "물성처럼 또렷한 군집 (부분집합 z-score, euclidean)."
    )

    def featurize(self, compound_ids: Sequence[str]) -> DomainFeatures:
        from chemspace.data.dataset import DEFAULT_ACTIVITY_PCHEMBL
        from chemspace.data.target_class import curate

        ids = tuple(compound_ids)
        tg = self.ds.targets
        cols = [c for c in ("chembl_id", "l1", "l2", "l3", "l4", "pchembl_value") if c in tg.columns]
        sub = tg[tg["chembl_id"].isin(set(ids))][cols].copy()
        for c in ("l1", "l2", "l3", "l4"):
            if c not in sub.columns:
                sub[c] = ""
        if not sub.empty:
            combos = sub[["l1", "l2", "l3", "l4"]].drop_duplicates()
            fam = combos.apply(
                lambda r: " > ".join(x for x in curate(r["l1"], r["l2"], r["l3"], r["l4"]) if x),
                axis=1,
            )
            combos = combos.assign(fam=fam.values)
            sub = sub.merge(combos, on=["l1", "l2", "l3", "l4"], how="left")
            sub = sub[sub["fam"].astype(str) != ""]
        if sub.empty:
            return DomainFeatures(
                X=np.zeros((len(ids), 0), np.float32),
                compound_ids=ids,
                covered=np.zeros(len(ids), bool),
                feature_names=(),
                metric="euclidean",
                kind="continuous",
                notes={"families": 0},
            )
        sub["pch"] = pd.to_numeric(sub["pchembl_value"], errors="coerce").fillna(
            DEFAULT_ACTIVITY_PCHEMBL
        ) if "pchembl_value" in sub.columns else float(DEFAULT_ACTIVITY_PCHEMBL)
        piv = sub.pivot_table(index="chembl_id", columns="fam", values="pch", aggfunc="max")
        fams = sorted(piv.columns)
        raw = piv.reindex(index=list(ids), columns=fams).fillna(0.0).to_numpy(np.float64)
        covered = (raw != 0).any(axis=1)
        Z, notes = _standardize_covered(raw, covered)
        return DomainFeatures(
            X=Z,
            compound_ids=ids,
            covered=covered,
            feature_names=tuple(f"fam{i}" for i in range(Z.shape[1])),
            metric="euclidean",
            kind="continuous",
            notes={**notes, "families": len(fams)},
        )

    def color_series(self, compound_ids):
        s = _family_colour(self.ds, compound_ids)
        return s if s is not None else super().color_series(compound_ids)


class TargetSvdDomain(TargetDomain):
    """B — latent-factor compression of the sparse target token matrix.

    TruncatedSVD squeezes the high-cardinality binary target vectors into a dense
    ~50-dim space before UMAP, so near-orthogonal sparse rows gain shared axes and
    the map is smoother than the raw Jaccard one.
    """

    name = "target_svd"
    label = "표적(잠재)"
    description = "표적 토큰 행렬을 잠재요인(SVD ~50차원)으로 압축 후 배치 — 희소성을 줄여 더 매끈한 군집 (euclidean)."
    N_COMPONENTS = 50

    def featurize(self, compound_ids: Sequence[str]) -> DomainFeatures:
        from sklearn.decomposition import TruncatedSVD

        ids = tuple(compound_ids)
        Xb, _names, notes = multi_hot(ids, self.ds.target_terms, self.min_support)
        covered = np.array([c in self.ds.target_terms for c in ids], dtype=bool)
        if Xb.shape[1] >= 3:
            k = int(min(self.N_COMPONENTS, Xb.shape[1] - 1))
            Z = TruncatedSVD(n_components=k, random_state=0).fit_transform(
                Xb.astype(np.float64)
            ).astype(np.float32)
        else:
            Z = Xb.astype(np.float32)
        return DomainFeatures(
            X=Z,
            compound_ids=ids,
            covered=covered,
            feature_names=tuple(f"svd{i}" for i in range(Z.shape[1])),
            metric="euclidean",
            kind="continuous",
            notes={**notes, "svd_dim": int(Z.shape[1])},
        )

    def color_series(self, compound_ids):
        s = _family_colour(self.ds, compound_ids)
        return s if s is not None else super().color_series(compound_ids)


class TargetCooccurDomain(TargetDomain):
    """C — target2vec: embed targets by co-occurrence, a compound is their mean.

    Targets hit together by the same compounds get nearby latent vectors (an SVD
    of the log co-occurrence matrix, the classic word-embedding recipe), so a
    compound placed at the mean of its targets' vectors carries target-relatedness
    the raw one-hot tokens miss.
    """

    name = "target2vec"
    label = "표적(동시출현)"
    description = (
        "같은 화합물이 함께 치는 표적으로 표적을 임베딩(동시출현 SVD), 화합물은 그 평균 — "
        "표적 간 관계로 구조화 (euclidean)."
    )
    DIM = 32

    def featurize(self, compound_ids: Sequence[str]) -> DomainFeatures:
        from sklearn.decomposition import TruncatedSVD

        ids = tuple(compound_ids)
        Xb, _names, notes = multi_hot(ids, self.ds.target_terms, self.min_support)
        covered = np.array([c in self.ds.target_terms for c in ids], dtype=bool)
        v = Xb.shape[1]
        if v >= 3 and Xb.shape[0] >= 3:
            xb = Xb.astype(np.float64)
            co = xb.T @ xb  # (v, v) co-occurrence counts
            k = int(min(self.DIM, v - 1))
            tv = TruncatedSVD(n_components=k, random_state=0).fit_transform(np.log1p(co))
            rs = xb.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1.0
            Z = ((xb @ tv) / rs).astype(np.float32)
        else:
            Z = Xb.astype(np.float32)
        return DomainFeatures(
            X=Z,
            compound_ids=ids,
            covered=covered,
            feature_names=tuple(f"t2v{i}" for i in range(Z.shape[1])),
            metric="euclidean",
            kind="continuous",
            notes={**notes, "dim": int(Z.shape[1])},
        )

    def color_series(self, compound_ids):
        s = _family_colour(self.ds, compound_ids)
        return s if s is not None else super().color_series(compound_ids)


class TargetTerritoryDomain(TargetFamilyMapDomain):
    """E — near-territory layout: B with a much stronger family weight.

    Pushing UMAP's target_weight close to 1 makes the curated family dominate the
    layout, so families form near-disjoint territories (approximating the '분류로
    영역을 먼저 나눈다' method) at the cost of within-family distances meaning less.
    """

    name = "target_territory"
    label = "표적(영역)"
    description = "B의 계열 가중치를 크게 높여(≈0.9) 가족이 거의 겹치지 않는 영역으로 — 영역 우선 배치 근사."
    TARGET_WEIGHT = 0.9


def _minmax01(a: np.ndarray) -> np.ndarray:
    """Scale each column to [0, 1] independently; a constant column -> 0.5."""
    a = np.asarray(a, dtype=np.float64)
    out = np.full_like(a, 0.5)
    for j in range(a.shape[1]):
        col = a[:, j]
        lo, hi = float(col.min()), float(col.max())
        if hi > lo:
            out[:, j] = (col - lo) / (hi - lo)
    return out


class TargetAtlasDomain(TargetFamilyMapDomain):
    """지도첩 — 큐레이션 계열을 캔버스의 고정 칸에 강제 분할 배치(겹침 0).

    E(영역)는 UMAP ``target_weight``를 높여 가족이 '거의' 안 겹치게 한 **준지도**
    근사였다. 이 도메인은 인코딩에 가족을 발견시키려 하지 않고, 아예 캔버스를 보이는
    계열 수만큼의 칸으로 나눠 **각 계열을 자기 칸에만** 둔다. 칸이 서로 떨어져 있으니
    계열 간 겹침은 구조적으로 0이고, 칸 안에서는 화학구조(ECFP)의 국소 분산 두 축
    (TruncatedSVD)으로 정렬해 세부 구조도 읽힌다. UMAP의 무작위성이 없어 같은
    부분집합은 언제나 같은 지도를 그린다 — '분류 값이 이미 있으니 수작업으로라도 강제로
    모은다'를 결정론적으로 구현한 것. 군집은 계열 라벨 그대로라 **칸 = 군집 = 계열**.

    부분집합이 여러 계열에 걸치면 L1(8대륙)으로, 한 계열로 좁히면 L2(하위계열)로 칸을
    나눈다 — ``TargetFamilyMapDomain``의 적응형 라벨을 그대로 따른다.
    """

    name = "target_atlas"
    label = "표적(지도첩)"
    description = (
        "큐레이션 계열별로 캔버스를 칸으로 나눠 강제 배치(겹침 0) — 칸 안은 ECFP "
        "국소 분산축(SVD)으로 정렬. 결정론적, 칸=군집=계열."
    )

    #: Padding inside each cell (keeps points off the gutter) and the gutter
    #: width between cells, both in cell-width units.
    PAD = 0.08
    GAP = 0.35

    def featurize(self, compound_ids: Sequence[str]) -> DomainFeatures:
        from dataclasses import replace

        base = super().featurize(compound_ids)
        labels = self._family_labels(tuple(compound_ids))
        coords = self._atlas_coords(base.X, labels, base.covered)
        return replace(
            base,
            coords=coords,
            notes={**base.notes, "layout": "atlas-partition"},
        )

    def _atlas_coords(
        self, X: np.ndarray, labels: np.ndarray, covered: np.ndarray
    ) -> np.ndarray:
        import math

        from sklearn.decomposition import TruncatedSVD

        n = X.shape[0]
        coords = np.zeros((n, 2), dtype=np.float32)
        fams = sorted({str(labels[i]) for i in range(n) if covered[i]})
        k = len(fams)
        if k == 0:
            return coords
        # A slightly wide grid reads better than a square one (8 -> 4x2).
        cols = max(1, int(round(math.sqrt(k * 1.6))))
        step = 1.0 + self.GAP
        lo, hi = self.PAD, 1.0 - self.PAD
        for idx, fam in enumerate(fams):
            col, row = idx % cols, idx // cols
            ox, oy = col * step, -row * step
            mask = covered & (labels == fam)
            Xf = X[mask]
            local = self._local_layout(Xf, TruncatedSVD)
            local = _minmax01(local) * (hi - lo) + lo
            local[:, 0] += ox
            local[:, 1] += oy
            coords[mask] = local.astype(np.float32)
        return coords

    @staticmethod
    def _local_layout(Xf: np.ndarray, TruncatedSVD) -> np.ndarray:
        """Within-cell coordinates for one family: its ECFP local variance axes,
        or a deterministic ring when it is too small / degenerate for SVD."""
        m = Xf.shape[0]
        if m >= 3 and Xf.shape[1] >= 2:
            try:
                svd = TruncatedSVD(n_components=2, random_state=0)
                emb = svd.fit_transform(Xf.astype(np.float32))
                if np.isfinite(emb).all() and np.ptp(emb) > 0:
                    return emb
            except Exception:  # pragma: no cover - SVD fallback
                pass
        ang = np.linspace(0.0, 2.0 * np.pi, max(m, 1), endpoint=False)
        return np.column_stack([0.5 + 0.2 * np.cos(ang), 0.5 + 0.2 * np.sin(ang)])


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
