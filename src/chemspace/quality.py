"""Is a clustering meaningful, or is it drawing patterns in noise?

The project already measures that re-placing a subset under a new domain
*regroups* it (adjusted Rand index near zero between domains). That is a
necessary result and an insufficient one: ARI near zero is equally consistent
with

  * both clusterings being meaningful and seeing different things -- what the
    tool is for, and
  * one of them being noise.

Two random partitions also score ARI 0. This module measures the other half:
whether each domain's clusters hold together on their own.

Three readings, because no single one is safe alone:

``silhouette_coords``
    How separated the clusters are on the map the user sees. Close to
    circular -- HDBSCAN found these clusters in those very coordinates -- so
    it is a floor, not evidence.

``silhouette_features``
    The same clusters scored in the *original* feature space under the
    domain's own metric. This is the honest one for the embedding step: it
    asks whether UMAP's layout reflects structure that was already in the
    features, or whether the separation is something the projection invented.

``ami`` against an external label
    Agreement with WHO's ATC therapeutic classification, which is assigned by
    people who never saw this pipeline. Only meaningful where ATC does not
    feed the features -- see ``Coherence.circular``.

Every reading is reported beside a **null**: the same pipeline run on the same
matrix with each feature column shuffled independently. That keeps each
feature's prevalence and destroys the co-occurrence between them, so the null
answers "what score would this domain get if there were nothing to find?".
A real number close to its null means the clusters are an artifact, whatever
the absolute value looks like.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import numpy as np

from chemspace.domains.base import DomainFeatures
from chemspace.placement import NOISE, UNCOVERED, ClusterConfig, EmbedConfig, Placement, place

log = logging.getLogger(__name__)

__all__ = [
    "Coherence",
    "coherence",
    "shuffled_features",
    "null_coherence",
    "silhouette_at_k",
]

#: Silhouette is O(n^2) in pairwise distances; score a sample above this.
SILHOUETTE_SAMPLE = 4000


@dataclass(frozen=True)
class Coherence:
    """How well one domain's clusters hold together."""

    domain: str
    n: int
    n_scored: int
    """Compounds the scores are computed on: covered, and not noise."""

    n_clusters: int
    noise_frac: float
    silhouette_coords: float
    silhouette_features: float
    silhouette_matched: float = float("nan")
    """Feature-space silhouette with the cluster count held fixed.

    Silhouette rewards few large clusters, so the plain reading cannot be
    compared between a real run and its null when the two found different
    numbers of clusters -- measured on the modality domain, the null scored
    *higher* on 6 clusters than the real run did on 17. Re-cutting both at the
    same k with k-means on the coordinates makes the comparison mean
    something.
    """

    ami: float | None = None
    purity: float | None = None
    n_labelled: int = 0
    circular: bool = False
    """True when the external label also feeds this domain's features."""

    def row(self) -> dict:
        return {
            "domain": self.domain,
            "n": self.n,
            "clusters": self.n_clusters,
            "noise": f"{self.noise_frac:.1%}",
            "sil_coords": round(self.silhouette_coords, 3),
            "sil_features": round(self.silhouette_features, 3),
            "sil_at_k": round(self.silhouette_matched, 3),
            "ami_atc": None if self.ami is None else round(self.ami, 3),
            "purity": None if self.purity is None else f"{self.purity:.1%}",
            "labelled": self.n_labelled,
            "note": "순환" if self.circular else "",
        }


def _scored_mask(placement: Placement) -> np.ndarray:
    """Points a cluster score can be computed on.

    Noise and uncovered are not clusters: scoring them as if they were would
    reward a run that dumped everything into one of those two buckets.
    """
    return (placement.labels != NOISE) & (placement.labels != UNCOVERED)


def _silhouette(X: np.ndarray, labels: np.ndarray, metric: str, seed: int) -> float:
    from sklearn.metrics import silhouette_score

    if len(np.unique(labels)) < 2 or len(labels) < 3:
        return float("nan")
    data = X.astype(bool) if metric == "jaccard" else np.ascontiguousarray(X, dtype=np.float64)
    kwargs = {"metric": metric, "random_state": seed}
    if len(labels) > SILHOUETTE_SAMPLE:
        kwargs["sample_size"] = SILHOUETTE_SAMPLE
    try:
        return float(silhouette_score(data, labels, **kwargs))
    except Exception as exc:  # noqa: BLE001 - degenerate input is a result
        log.warning("silhouette failed (%s)", exc)
        return float("nan")


def silhouette_at_k(
    X: np.ndarray, coords: np.ndarray, metric: str, k: int, seed: int = 0
) -> float:
    """Feature-space silhouette after re-cutting the map into exactly ``k``.

    Reuses the embedding that has already been computed, so this costs a
    k-means fit rather than a second UMAP.
    """
    from sklearn.cluster import KMeans

    n = coords.shape[0]
    k = int(min(max(k, 2), max(n - 1, 2)))
    if n <= k or not X.shape[1]:
        return float("nan")
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(coords)
    return _silhouette(X, labels, metric, seed)


def _external_agreement(
    labels: np.ndarray, external: np.ndarray
) -> tuple[float | None, float | None, int]:
    """AMI and purity against a label assigned outside this pipeline.

    Only the points that carry an external label take part, so a domain is not
    penalised for compounds WHO never classified.
    """
    from sklearn.metrics import adjusted_mutual_info_score

    have = np.array([bool(v) and v == v for v in external])  # non-empty, non-NaN
    if have.sum() < 10 or len(np.unique(labels[have])) < 2:
        return None, None, int(have.sum())

    lab, ext = labels[have], np.asarray(external, dtype=object)[have]
    ami = float(adjusted_mutual_info_score(ext.astype(str), lab))

    # purity: how much of each cluster sits in its own commonest label
    hits = 0
    for c in np.unique(lab):
        in_c = ext[lab == c]
        values, counts = np.unique(in_c.astype(str), return_counts=True)
        hits += int(counts.max())
    return ami, hits / len(lab), int(have.sum())


def coherence(
    features: DomainFeatures,
    placement: Placement,
    domain: str,
    external: np.ndarray | None = None,
    circular: bool = False,
    matched_k: int | None = None,
    seed: int = 0,
) -> Coherence:
    """Score one placement. ``external`` is row-aligned with ``features``.

    ``matched_k`` re-cuts the map at a fixed cluster count so the silhouette
    can be compared against a null that found a different number of clusters.
    """
    keep = _scored_mask(placement)
    labels = placement.labels[keep]
    n_scored = int(keep.sum())

    sil_coords = _silhouette(placement.coords[keep], labels, "euclidean", seed)
    sil_feats = (
        _silhouette(features.X[keep], labels, features.metric, seed)
        if features.X.shape[1]
        else float("nan")
    )

    sil_matched = (
        silhouette_at_k(
            features.X[keep], placement.coords[keep], features.metric, matched_k, seed
        )
        if matched_k
        else float("nan")
    )

    ami = purity = None
    n_labelled = 0
    if external is not None and n_scored:
        ami, purity, n_labelled = _external_agreement(labels, np.asarray(external)[keep])

    return Coherence(
        domain=domain,
        n=int(placement.labels.shape[0]),
        n_scored=n_scored,
        n_clusters=len(placement.cluster_ids),
        noise_frac=float((placement.labels == NOISE).mean()),
        silhouette_coords=sil_coords,
        silhouette_features=sil_feats,
        silhouette_matched=sil_matched,
        ami=ami,
        purity=purity,
        n_labelled=n_labelled,
        circular=circular,
    )


def shuffled_features(features: DomainFeatures, seed: int = 0) -> DomainFeatures:
    """The same matrix with every column independently permuted.

    Each feature keeps its prevalence -- the same number of compounds carry
    "MeSH:Hypertension", the same distribution of molecular weights -- and all
    co-occurrence between features is destroyed. Anything the pipeline still
    finds here, it would have found in nothing.
    """
    rng = np.random.default_rng(seed)
    X = np.array(features.X, copy=True)
    for j in range(X.shape[1]):
        rng.shuffle(X[:, j])
    return replace(features, X=X, notes={**features.notes, "shuffled": True})


def null_coherence(
    features: DomainFeatures,
    domain: str,
    embed: EmbedConfig | None = None,
    cluster: ClusterConfig | None = None,
    external: np.ndarray | None = None,
    matched_k: int | None = None,
    seed: int = 0,
) -> Coherence:
    """Run the whole pipeline on shuffled features and score it the same way."""
    shuffled = shuffled_features(features, seed=seed)
    return coherence(
        shuffled,
        place(shuffled, embed, cluster),
        domain=f"{domain} (null)",
        external=external,
        matched_k=matched_k,
        seed=seed,
    )
