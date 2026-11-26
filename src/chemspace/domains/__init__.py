"""Domains: the interchangeable lenses on a compound set."""

from __future__ import annotations

from typing import Iterator

from chemspace.data.dataset import Dataset
from chemspace.domains.base import Domain, DomainFeatures
from chemspace.domains.builtin import (
    DiseaseDomain,
    ModalityDomain,
    MoietyDomain,
    PhysChemDomain,
    TargetActivityDomain,
    TargetAnchoredCooccurDomain,
    TargetAnchoredDomain,
    TargetAtlasDomain,
    TargetCooccurDomain,
    TargetDomain,
    TargetFamilyActivityDomain,
    TargetFamilyDomain,
    TargetFamilyMapDomain,
    TargetKnowledgeDomain,
    TargetSeqDomain,
    TargetSvdDomain,
    TargetTerritoryDomain,
)

__all__ = [
    "Domain",
    "DomainFeatures",
    "DomainRegistry",
    "PhysChemDomain",
    "DiseaseDomain",
    "TargetDomain",
    "TargetFamilyDomain",
    "TargetFamilyMapDomain",
    "TargetActivityDomain",
    "TargetFamilyActivityDomain",
    "TargetSvdDomain",
    "TargetCooccurDomain",
    "TargetTerritoryDomain",
    "TargetAtlasDomain",
    "TargetAnchoredDomain",
    "TargetAnchoredCooccurDomain",
    "TargetSeqDomain",
    "TargetKnowledgeDomain",
    "MoietyDomain",
    "ModalityDomain",
    "default_registry",
]

BUILTIN = (
    PhysChemDomain,
    DiseaseDomain,
    TargetDomain,
    TargetFamilyDomain,
    TargetFamilyMapDomain,
    TargetActivityDomain,
    TargetFamilyActivityDomain,
    TargetSvdDomain,
    TargetCooccurDomain,
    TargetTerritoryDomain,
    TargetAtlasDomain,
    TargetAnchoredDomain,
    TargetAnchoredCooccurDomain,
    MoietyDomain,
    ModalityDomain,
)


class DomainRegistry:
    """Name -> domain instance, so a session can store a domain as a string.

    Add a domain of your own with :meth:`register`; the trained predictors in
    the project plan slot in here without the session or viewer changing.
    """

    def __init__(self, domains: dict[str, Domain] | None = None):
        self._domains: dict[str, Domain] = dict(domains or {})

    def register(self, domain: Domain) -> Domain:
        self._domains[domain.name] = domain
        return domain

    def __getitem__(self, name: str) -> Domain:
        try:
            return self._domains[name]
        except KeyError:
            raise KeyError(
                f"unknown domain {name!r}; available: {sorted(self._domains)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._domains

    def __iter__(self) -> Iterator[Domain]:
        return iter(self._domains.values())

    def __len__(self) -> int:
        return len(self._domains)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._domains)

    @property
    def labels(self) -> dict[str, str]:
        return {n: d.label for n, d in self._domains.items()}

    def __repr__(self) -> str:  # pragma: no cover
        return f"<DomainRegistry {list(self._domains)}>"


def default_registry(dataset: Dataset) -> DomainRegistry:
    """The built-in domains, bound to one dataset.

    The ESM-2 sequence view is registered only when the snapshot ships the
    precomputed embeddings (``target_esm.npz``); otherwise it would be a dead
    entry in the domain picker. The knowledge view is gated the same way on
    ``target_knowledge.npz``.
    """
    reg = DomainRegistry()
    for cls in BUILTIN:
        reg.register(cls(dataset))
    if getattr(dataset, "target_embeddings", None) is not None:
        reg.register(TargetSeqDomain(dataset))
    # The knowledge view, likewise, only when its embedding cache is present.
    if getattr(dataset, "target_knowledge", None) is not None:
        reg.register(TargetKnowledgeDomain(dataset))
    return reg
