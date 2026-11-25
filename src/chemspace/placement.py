"""Placement: turn a domain's feature matrix into 2-D coordinates and clusters.

Always fitted on the subset it is given.  Nothing is reused from a parent node,
which is exactly why a subset re-placed under the same domain can look
different from the region it was cut out of -- and the point of the tool.

Compounds the domain has no data for (``DomainFeatures.covered == False``) are
kept out of the fit and parked in a band below the cloud.  Leaving them in
would be worse than it sounds: under Jaccard every all-zero row sits at
distance 0 from every other all-zero row and 1 from everything else, so they
collapse into one dense fake cluster and drag the real structure around it.
Parking them keeps them selectable and visible without letting them vote.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field, replace

import numpy as np

from chemspace.domains.base import DomainFeatures

log = logging.getLogger(__name__)

__all__ = ["Placement", "EmbedConfig", "ClusterConfig", "place", "recluster", "UNCOVERED"]

#: Cluster id for compounds with no data in the current domain.
UNCOVERED = -2
#: Cluster id for density-based noise.
NOISE = -1


@dataclass(frozen=True)
class EmbedConfig:
    method: str = "umap"  # "umap" | "pca"
    n_neighbors: int = 15
    min_dist: float = 0.1
    random_state: int | None = 0
    """Fixed by default, but it does not buy full reproducibility.

    Measured by running the same chain in two processes: euclidean placements
    come out bit-identical, while **Jaccard placements do not** -- pynndescent's
    approximate nearest-neighbour search is not fully pinned by the seed. In
    that test the cluster labels still matched, but nothing guarantees they
    will, and repeated runs of the smoke test do show downstream cluster counts
    moving.  Reproducibility of a *recorded* exploration comes from
    ``session.save()`` keeping the coordinates, not from recomputation.

    The cost of setting it is speed: UMAP drops to a single thread, which is
    most of the ~75-105 s it takes to place all 12,804 compounds.  ``None`` uses
    every core (roughly 4x faster) and gives up coordinate stability entirely.
    """


#: Where the square-root rule stops growing.
#:
#: ``sqrt(n) / 2`` was tuned at 12,804 compounds, and it keeps demanding bigger
#: clusters as the map grows. Measured at 73,864 it asks for 136 and the maps
#: fall apart: the target domain leaves **51% of its compounds as noise**, and
#: the moiety domain collapses to 7 clusters with AMI 0.033 -- against 0.168 at
#: a min size of 40, which is *higher* than the 0.130 the same domain scored
#: before the expansion. Nothing had blurred; the rule had outgrown its tuning.
#:
#: Capping is the fix rather than a different exponent, because the parameter
#: means "the smallest group worth calling a cluster", and that is a judgement
#: about chemistry: a series of forty analogues is a series whether the map
#: holds ten thousand compounds or a million. The square root still governs
#: small subsets, where a fixed value would call everything noise; the cap only
#: binds above n = 6,400.
#:
#: Measured both ways -- 40 is better than 57 at 12,804 too (moiety AMI 0.179
#: vs 0.148, target noise 13.4% vs 22.1%, nothing worse) -- so this is not a
#: trade made for the larger set at the smaller one's expense.
AUTO_MIN_CLUSTER_CAP = 40


def auto_min_cluster_size(n: int) -> int:
    """The default ``min_cluster_size`` for a subset of ``n`` placed points.

    Kept as a function because the placement cache key has to resolve it: the
    stored key records ``min_cluster_size=None``, so two placements made under
    different rules would otherwise collide and the older one be served.
    """
    return int(min(max(5, round(np.sqrt(max(n, 1)) / 2)), AUTO_MIN_CLUSTER_CAP))


@dataclass(frozen=True)
class ClusterConfig:
    method: str = "hdbscan"  # "hdbscan" | "kmeans" | "none"
    on: str = "coords"
    """``"coords"`` clusters what the user sees (fast, and matches the picture);
    ``"features"`` clusters the full feature space (slower, metric-aware)."""
    min_cluster_size: int | None = None
    """``None`` scales with the subset as ``auto_min_cluster_size(n)``.

    A fixed value fails at the small end -- all-noise at 300.  Linear scaling
    (``n // 50``) is wrong at the large end: at 12,804 it asks for clusters of
    256+ and yields about six continent-sized groups, so selecting "a region"
    still leaves most of the set.  Square-root scaling keeps the cluster
    *count* growing slowly while the map stays selectable:
    1,200 -> 17, 300 -> 9, 40 -> 5.

    Past a few thousand compounds it stops being right, which is what
    ``AUTO_MIN_CLUSTER_CAP`` is for -- see there.
    """
    min_samples: int | None = None
    n_clusters: int = 8  # kmeans only


@dataclass
class Placement:
    """Where a subset sits, and how it groups, under one domain."""

    coords: np.ndarray  # (n, 2) float32
    labels: np.ndarray  # (n,) int32; -1 noise, -2 uncovered
    compound_ids: tuple[str, ...]
    domain: str
    embed_method: str
    cluster_method: str
    metric: str
    n_covered: int
    notes: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.compound_ids)

    @property
    def cluster_ids(self) -> list[int]:
        """Real clusters only, ascending -- no noise, no uncovered."""
        return sorted(int(c) for c in set(self.labels.tolist()) if c >= 0)

    def members(self, cluster_id: int) -> tuple[str, ...]:
        mask = self.labels == cluster_id
        return tuple(np.asarray(self.compound_ids)[mask].tolist())

    def sizes(self) -> dict[int, int]:
        ids, counts = np.unique(self.labels, return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, counts)}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Placement {self.domain}: {self.n} compounds, "
            f"{len(self.cluster_ids)} clusters, {self.n_covered} covered>"
        )


# --------------------------------------------------------------------------- #


def place(
    features: DomainFeatures,
    embed: EmbedConfig | None = None,
    cluster: ClusterConfig | None = None,
) -> Placement:
    """Embed and cluster one subset under one domain."""
    embed = embed or EmbedConfig()
    cluster = cluster or ClusterConfig()
    n = features.X.shape[0]
    notes: dict = dict(features.notes)

    coords = np.zeros((n, 2), dtype=np.float32)
    labels = np.full(n, UNCOVERED, dtype=np.int32)

    fit = features.covered.copy()
    if features.X.shape[1] == 0:
        # Every feature was pruned: nothing distinguishes these compounds here.
        notes["degenerate"] = "no features with support in this subset"
        fit[:] = False
    if fit.sum() < 3:
        notes.setdefault("degenerate", f"only {int(fit.sum())} compounds with data")
        coords = _ring(n)
        labels[:] = np.where(features.covered, 0, UNCOVERED)
        return Placement(
            coords=coords,
            labels=labels,
            compound_ids=features.compound_ids,
            domain="",
            embed_method="ring",
            cluster_method="none",
            metric=features.metric,
            n_covered=int(features.covered.sum()),
            notes=notes,
        )

    Xf = features.X[fit]
    yf = features.supervision[fit] if features.supervision is not None else None
    emb, method, enote = _embed(
        Xf, features.metric, embed, y=yf, target_weight=features.supervision_weight
    )
    notes.update(enote)
    emb, sanitized = _finite(emb)
    if sanitized:
        notes["non_finite_coords"] = sanitized

    if features.cluster_labels is not None:
        # The domain already knows the grouping (e.g. a curated family): use it
        # directly instead of re-discovering it with HDBSCAN. Remap the covered
        # rows' codes to contiguous 0..k-1, keeping -1 as noise.
        lab, cmethod = _labels_as_clusters(features.cluster_labels[fit])
    else:
        lab, cmethod = _cluster(emb, Xf, features.metric, cluster)
    coords[fit] = emb
    labels[fit] = lab

    if (~fit).any():
        coords[~fit] = _uncovered_band(emb, int((~fit).sum()))
        notes["uncovered"] = int((~fit).sum())

    return Placement(
        coords=coords.astype(np.float32),
        labels=labels.astype(np.int32),
        compound_ids=features.compound_ids,
        domain="",
        embed_method=method,
        cluster_method=cmethod,
        metric=features.metric,
        n_covered=int(features.covered.sum()),
        notes=notes,
    )


# ------------------------------------------------------------------ embedding


def _embed(
    X: np.ndarray,
    metric: str,
    cfg: EmbedConfig,
    y: np.ndarray | None = None,
    target_weight: float = 0.5,
) -> tuple[np.ndarray, str, dict]:
    """Embed ``X`` to 2-D. When ``y`` is given (int label codes, ``-1`` unknown),
    UMAP runs semi-supervised: same-label points are drawn together while the
    within-label geometry still comes from ``X``. PCA has no supervised form, so
    the fallback path ignores ``y``."""
    n = X.shape[0]
    if cfg.method == "pca" or n < 10:
        return _pca(X), "pca", {"embed_fallback": "n too small for umap"} if n < 10 else {}

    n_neighbors = int(min(cfg.n_neighbors, max(2, n - 1)))
    try:
        import umap

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=cfg.min_dist,
                metric=metric,
                random_state=cfg.random_state,
                # Stated rather than left to UMAP's own override, which only
                # kicks in when random_state is set and is easy to lose track
                # of. Even so, the Jaccard path is not bit-reproducible across
                # processes -- see the reproducibility note in the README.
                n_jobs=1 if cfg.random_state is not None else -1,
                target_weight=target_weight,
            )
            emb = reducer.fit_transform(
                np.ascontiguousarray(X, dtype=np.float32), y=y
            )
        tag = f"umap({metric},k={n_neighbors}{',y' if y is not None else ''})"
        return np.asarray(emb, dtype=np.float32), tag, {}
    except Exception as exc:  # noqa: BLE001 - UMAP can fail on degenerate input
        log.warning("UMAP failed (%s); falling back to PCA", exc)
        return _pca(X), "pca", {"embed_fallback": f"umap failed: {exc}"}


def _finite(emb: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace any non-finite coordinate, and say how many there were.

    UMAP occasionally emits NaN or inf on small, very sparse Jaccard inputs.
    Left alone these spread quietly: the point vanishes from the plot, and any
    later arithmetic over the coordinate range (the uncovered band's geometry,
    an axis fit) turns into NaN for *every* point. Clamping here keeps the
    damage to the handful of affected rows and records that it happened.
    """
    bad = ~np.isfinite(emb)
    if not bad.any():
        return emb, 0
    cleaned = emb.copy()
    for col in range(cleaned.shape[1]):
        good = cleaned[np.isfinite(cleaned[:, col]), col]
        fill = float(np.median(good)) if good.size else 0.0
        cleaned[~np.isfinite(cleaned[:, col]), col] = fill
    return cleaned, int(bad.any(axis=1).sum())


def _pca(X: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    if X.shape[1] < 2:
        pad = np.zeros((X.shape[0], 2 - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    emb = PCA(n_components=2, random_state=0).fit_transform(X)
    return np.asarray(emb, dtype=np.float32)


def _ring(n: int) -> np.ndarray:
    """Deterministic layout for sets too small to embed."""
    t = np.linspace(0, 2 * np.pi, max(n, 1), endpoint=False)
    return np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float32)


def _uncovered_band(emb: np.ndarray, k: int) -> np.ndarray:
    """A compact block under the embedded cloud.

    Sized against the *core* of the embedding (5th-95th percentile), not its
    min/max.  UMAP routinely strands a handful of points far from everything;
    measuring against those stretched the band across the full outlier range,
    so with 619 uncovered compounds out of 1,200 it covered more of the panel
    than the real clusters did and squashed them into a corner.

    The block is held to roughly the width of that core and a quarter of its
    height, and sits just below the lowest real point -- visible and
    selectable, but plainly subordinate to the data.
    """
    x_lo, x_hi = (float(v) for v in np.percentile(emb[:, 0], [5, 95]))
    y_lo, y_hi = (float(v) for v in np.percentile(emb[:, 1], [5, 95]))
    width = max(x_hi - x_lo, 1e-6)
    core_height = max(y_hi - y_lo, 1e-6)
    block_height = 0.25 * core_height

    # Choose the row length so the block's aspect matches width : block_height.
    ratio = k * width / block_height
    per_row = (
        max(int(round(np.sqrt(ratio))), 1)
        if np.isfinite(ratio) and ratio > 0
        else max(int(np.ceil(np.sqrt(k * 4))), 1)
    )
    per_row = min(per_row, max(k, 1))  # a row longer than k would waste the width
    rows = max(int(np.ceil(k / per_row)), 1)
    dx = width / max(per_row - 1, 1)
    dy = block_height / max(rows - 1, 1)
    top = float(emb[:, 1].min()) - 0.12 * core_height

    idx = np.arange(k)
    xs = x_lo + dx * (idx % per_row)
    ys = top - dy * (idx // per_row)
    return np.stack([xs, ys], axis=1).astype(np.float32)


# ----------------------------------------------------------------- clustering


def _labels_as_clusters(codes: np.ndarray) -> tuple[np.ndarray, str]:
    """Use given per-point codes as clusters: remap to contiguous 0..k-1, with
    any negative code kept as noise (-1). For a domain that already knows the
    grouping, so the clusters match it exactly instead of being re-discovered."""
    codes = np.asarray(codes)
    uniq = sorted({int(c) for c in codes if int(c) >= 0})
    remap = {c: i for i, c in enumerate(uniq)}
    lab = np.array([remap.get(int(c), -1) for c in codes], dtype=np.int32)
    return lab, f"labels(k={len(uniq)})"


def recluster(p: Placement, cfg: ClusterConfig) -> Placement:
    """Re-cluster an existing placement's covered points on their 2-D coords,
    reusing the embedding (no re-UMAP) -- for switching cluster method from the
    UI cheaply. Mutates and returns ``p``. Uncovered points stay uncovered."""
    fit = p.labels != UNCOVERED
    labels = np.full(p.coords.shape[0], UNCOVERED, dtype=np.int32)
    cmethod = "none"
    if int(fit.sum()) >= 1:
        sub = np.ascontiguousarray(p.coords[fit], dtype=np.float64)
        lab, cmethod = _cluster(sub, sub, "euclidean", replace(cfg, on="coords"))
        labels[fit] = lab
    p.labels = labels
    p.cluster_method = cmethod
    p.notes = {**p.notes, "reclustered": cfg.method}
    return p


def _cluster(
    emb: np.ndarray, X: np.ndarray, metric: str, cfg: ClusterConfig
) -> tuple[np.ndarray, str]:
    n = emb.shape[0]
    if cfg.method == "none":
        return np.zeros(n, dtype=np.int32), "none"

    mcs = cfg.min_cluster_size or auto_min_cluster_size(n)
    mcs = int(min(max(mcs, 2), max(n // 2, 2)))

    if cfg.method == "kmeans":
        from sklearn.cluster import KMeans

        k = int(min(cfg.n_clusters, max(2, n)))
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(
            emb if cfg.on == "coords" else X
        )
        return lab.astype(np.int32), f"kmeans(k={k})"

    if cfg.method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        # Hierarchical, so cutting at a cluster count gives the same tree's coarse
        # level -- a natural fit for the L1/L2 drill-down. Ward needs Euclidean,
        # which the 2-D coords are; on raw features it falls back to the metric.
        k = int(min(cfg.n_clusters, max(2, n)))
        if cfg.on == "features" and metric == "jaccard":
            lab = AgglomerativeClustering(
                n_clusters=k, metric="jaccard", linkage="average"
            ).fit_predict(np.ascontiguousarray(X, dtype=np.float64))
            return lab.astype(np.int32), f"agglomerative(k={k})"
        data = np.ascontiguousarray(emb if cfg.on == "coords" else X, dtype=np.float64)
        # A kNN connectivity graph keeps it feasible at scale: the unconstrained
        # tree is O(n^2) in memory and hangs past a few thousand points. The
        # constraint also makes the clusters spatially contiguous on the map.
        conn = None
        if n > 2000:
            from sklearn.neighbors import kneighbors_graph

            conn = kneighbors_graph(data, n_neighbors=min(15, n - 1), include_self=False)
        lab = AgglomerativeClustering(
            n_clusters=k, linkage="ward", connectivity=conn
        ).fit_predict(data)
        return lab.astype(np.int32), f"agglomerative(k={k})"

    if cfg.method == "gmm":
        from sklearn.mixture import GaussianMixture

        # Soft, elliptical clusters hardened to a label. Fast at scale on 2-D.
        k = int(min(cfg.n_clusters, max(2, n)))
        data = np.ascontiguousarray(emb if cfg.on == "coords" else X, dtype=np.float64)
        lab = GaussianMixture(
            n_components=k, covariance_type="full", random_state=0
        ).fit_predict(data)
        return lab.astype(np.int32), f"gmm(k={k})"

    if cfg.method == "spectral":
        from sklearn.cluster import SpectralClustering

        # Graph-based. The default RBF affinity is a dense O(n^2) matrix that is
        # infeasible at scale, so use a kNN affinity graph; even so the Laplacian
        # eigendecomposition is the heaviest option here.
        k = int(min(cfg.n_clusters, max(2, n)))
        data = np.ascontiguousarray(emb if cfg.on == "coords" else X, dtype=np.float64)
        lab = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=min(10, n - 1),
            assign_labels="kmeans",
            random_state=0,
        ).fit_predict(data)
        return lab.astype(np.int32), f"spectral(k={k})"

    from sklearn.cluster import HDBSCAN

    # copy=False is safe and deliberate: fit_predict gets a freshly built
    # contiguous array below, so letting HDBSCAN work in place costs nothing.
    opts = dict(min_cluster_size=mcs, min_samples=cfg.min_samples, copy=False)
    if cfg.on == "features":
        data = X
        model = HDBSCAN(**opts, metric="jaccard") if metric == "jaccard" else HDBSCAN(**opts)
    else:
        data = emb
        model = HDBSCAN(**opts)

    lab = model.fit_predict(np.ascontiguousarray(data, dtype=np.float64))
    return lab.astype(np.int32), f"hdbscan(min_size={mcs},on={cfg.on})"
