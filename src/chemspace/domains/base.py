"""The domain abstraction.

A **domain** is one way of looking at a compound set: it turns compounds into a
feature matrix, and that matrix is what the embedding and the clustering see.
Switching the domain keeps the compound set and changes the features; selecting
a region keeps the features and shrinks the compound set.

Two rules make a domain *subset-relative*, which is the whole point of this
tool.  Both are applied inside ``featurize``, over the subset it is handed:

1. **Continuous features are standardised within the subset.** Once you have
   narrowed to 1,200 drug-like compounds, the spread that matters is the spread
   *among those 1,200*, not among all 16,809.
2. **Binary features with no support in the subset are dropped.** A MeSH term
   no selected compound carries contributes nothing but noise to a Jaccard
   distance.

Because of those two rules, re-placing a subset is genuinely different from
zooming into the full map -- compounds that sat together globally can separate,
which is the structure the user is looking for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "DomainFeatures",
    "Domain",
    "standardize_within_subset",
    "prune_unsupported",
    "multi_hot",
    "auto_min_support",
]


@dataclass(frozen=True)
class DomainFeatures:
    """Feature matrix for one subset under one domain."""

    X: np.ndarray
    """(n, d) float32. Row i is ``compound_ids[i]``."""

    compound_ids: tuple[str, ...]
    """Row order, so coordinates can always be traced back to compounds."""

    covered: np.ndarray
    """(n,) bool -- True where this domain has real data for the compound.

    A compound with no indication rows is not "at the origin of disease space",
    it is *unknown* there. Keeping that explicit stops a false cluster of
    zero-vectors from forming, and lets the session report honest coverage.
    """

    feature_names: tuple[str, ...]
    metric: str
    """Distance metric the embedding and clustering should use."""

    kind: str
    """``"continuous"`` or ``"binary"``."""

    notes: dict = field(default_factory=dict)
    """Diagnostics worth showing the user, e.g. how many features were pruned."""

    def __post_init__(self) -> None:
        n, d = self.X.shape
        if n != len(self.compound_ids):
            raise ValueError(f"X has {n} rows but {len(self.compound_ids)} compound ids")
        if n != self.covered.shape[0]:
            raise ValueError(f"X has {n} rows but covered has {self.covered.shape[0]}")
        if d != len(self.feature_names):
            raise ValueError(f"X has {d} cols but {len(self.feature_names)} feature names")

    @property
    def n_covered(self) -> int:
        return int(self.covered.sum())

    @property
    def coverage(self) -> float:
        return float(self.covered.mean()) if len(self.covered) else 0.0


class Domain(ABC):
    """One lens on the compound set."""

    name: str
    """Machine name used in session records, e.g. ``"physchem"``."""

    label: str
    """Display name, e.g. ``"물성"``."""

    description: str = ""

    @abstractmethod
    def featurize(self, compound_ids: Sequence[str]) -> DomainFeatures:
        """Features for exactly ``compound_ids``, in that order, standardised
        or pruned *relative to this subset*."""

    @abstractmethod
    def profile(self, compound_ids: Sequence[str]) -> dict:
        """What characterises this group of compounds in this domain.

        Called per cluster to build the cluster table, and per node to describe
        a whole selection. Returns small, display-ready values.
        """

    def color_series(self, compound_ids: Sequence[str]):
        """Categorical values worth colouring points by, or ``None``.

        Used where the domain has a meaningful label that is *not* what the
        coordinates encode -- modality being the clearest case.
        """
        return None

    def composition(self, compound_ids: Sequence[str], top: int = 25) -> list[dict]:
        """Full breakdown of the discrete kinds in this group, for the "구성
        종류" view -- the long list a top-3 summary only hints at.

        Returns a list of sections, each ``{"title", "total", "items": [{"name",
        "count"}]}``: ``total`` is how many distinct kinds there are, ``items``
        the first ``top`` of them. Default empty, so a domain with no
        categorical structure (or a test stub) simply contributes nothing.
        """
        return []

    def term_counts(self, compound_ids: Sequence[str]) -> dict | None:
        """How many of these compounds carry each discrete term, for enrichment
        (a cluster card asks what is over-represented here versus the node).

        ``None`` for a continuous domain with no terms. Terms keep their source
        prefix so the caller can strip it for display.
        """
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Domain {self.name} ({self.label})>"


# --------------------------------------------------------------------------- #
# Shared helpers implementing the two subset-relative rules.
# --------------------------------------------------------------------------- #


def standardize_within_subset(
    X: np.ndarray, eps: float = 1e-9
) -> tuple[np.ndarray, dict]:
    """Z-score each column using this subset's own mean and spread.

    Constant columns (zero variance inside the subset) carry no information
    here and are dropped -- after narrowing to a region, several descriptors
    often become near-constant, and keeping them just dilutes the distance.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.size == 0:
        return X.astype(np.float32), {"dropped_constant": 0}
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    keep = sd > eps
    Z = np.zeros_like(X)
    np.divide(X - mu, np.where(sd > eps, sd, 1.0), out=Z)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z[:, keep].astype(np.float32), {
        "dropped_constant": int((~keep).sum()),
        "kept": int(keep.sum()),
    }


def auto_min_support(n: int, floor: int = 1, cap: int = 25) -> int:
    """Support threshold that scales with the subset: about 0.2% of it.

    A fixed threshold fails at both ends.  At 1 it keeps every one-off MeSH
    term, and across all 12,804 compounds that is a ~5,000-column matrix of
    mostly noise -- hundreds of MB, slow, and no better structured.  Fixed at 25
    it deletes every feature once you are down to 300 compounds.  Tying it to
    the subset says the same thing at every scale: a term fewer than ~0.2% of
    these compounds carry cannot be what separates them.
    """
    return int(min(max(round(n / 500), floor), cap))


def prune_unsupported(
    X: np.ndarray, names: Sequence[str], min_support: int | None = 1
) -> tuple[np.ndarray, tuple[str, ...], dict]:
    """Drop binary columns carried by fewer than ``min_support`` compounds.

    ``min_support=None`` picks the threshold from the subset size via
    :func:`auto_min_support`.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        return X, tuple(names), {"dropped_unsupported": 0, "kept": 0, "min_support": 0}
    threshold = auto_min_support(X.shape[0]) if min_support is None else int(min_support)
    support = (X > 0).sum(axis=0)
    keep = support >= threshold
    if not keep.any() and X.shape[1]:
        # Never hand back a zero-column matrix just because the threshold bit
        # too hard -- fall back to whatever has any support at all.
        keep = support >= 1
    return (
        X[:, keep],
        tuple(n for n, k in zip(names, keep) if k),
        {
            "dropped_unsupported": int((~keep).sum()),
            "kept": int(keep.sum()),
            "min_support": threshold,
        },
    )


def multi_hot(
    compound_ids: Sequence[str],
    memberships: dict[str, Iterable[str]],
    min_support: int | None = None,
) -> tuple[np.ndarray, tuple[str, ...], dict]:
    """Build a (n_compounds, n_terms) binary matrix from id -> terms.

    Support is counted *before* the matrix is allocated, so a rare-term tail
    never becomes a column.  That matters at the root: all 16.8k compounds carry
    about 5,000 distinct MeSH terms between them, and allocating that densely
    costs ~340 MB to then throw most of it away.
    """
    counts = Counter(t for cid in compound_ids for t in memberships.get(cid, ()))
    threshold = auto_min_support(len(compound_ids)) if min_support is None else int(min_support)
    vocab = sorted(t for t, c in counts.items() if c >= threshold)
    if not vocab and counts:
        vocab = sorted(counts)  # threshold too aggressive for this subset
        threshold = 1
    index = {t: j for j, t in enumerate(vocab)}
    X = np.zeros((len(compound_ids), len(vocab)), dtype=np.float32)
    for i, cid in enumerate(compound_ids):
        for t in memberships.get(cid, ()):
            j = index.get(t)
            if j is not None:
                X[i, j] = 1.0
    return (
        X,
        tuple(vocab),
        {
            "vocab_full": len(counts),
            "vocab_kept": len(vocab),
            "min_support": threshold,
        },
    )
