"""Structural-alert labels (PAINS / NIH / Brenk) from SMILES, via RDKit.

ChEMBL ships no such flag, so 구조알림 is a *computed* overlay: a pure function
of structure, the one axis of the xlsx taxonomy that collection cannot fill.
Each compound gets the single most-severe family it matches -- PAINS first
(assay-interference, the most actionable call), then NIH, then Brenk -- and
blank when it matches none. The three catalogs are RDKit's published SMARTS
sets, so the label is deterministic and reproducible.

RDKit is imported lazily: ``Dataset.load`` reads the cached result and never
touches this module, so loading the app does not pay the import.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Iterator

#: Most actionable first -- the label is the first family that matches.
PRIORITY = ("PAINS", "NIH", "BRENK")
#: Display label per catalog (PAINS/NIH stay upper, Brenk is a name).
LABEL = {"PAINS": "PAINS", "NIH": "NIH", "BRENK": "Brenk"}


@lru_cache(maxsize=1)
def _catalogs():
    """One FilterCatalog per family, built once (SMARTS compilation is slow)."""
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    cats = []
    for name in PRIORITY:
        params = FilterCatalogParams()
        params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
        cats.append((name, FilterCatalog(params)))
    return cats


def alert_flags(smiles: str) -> dict[str, bool]:
    """``{"PAINS": ..., "NIH": ..., "BRENK": ...}`` for one SMILES.

    An unparseable or empty SMILES is all-False -- the structural domain still
    places such a compound, this only declines to label it.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return {name: False for name in PRIORITY}
    return {name: cat.HasMatch(mol) for name, cat in _catalogs()}


def label_from(flags: dict[str, bool]) -> str:
    """Single most-severe label from a flags dict (blank if none match)."""
    for name in PRIORITY:
        if flags.get(name):
            return LABEL[name]
    return ""


def alert_label(smiles: str) -> str:
    """The single most-severe structural-alert label for one SMILES."""
    return label_from(alert_flags(smiles))


def alert_rows(smiles_iter: Iterable[str]) -> Iterator[dict]:
    """Stream ``{pains, nih, brenk, struct_alert}`` dicts over many SMILES."""
    for smi in smiles_iter:
        flags = alert_flags(smi)
        yield {
            "pains": flags["PAINS"],
            "nih": flags["NIH"],
            "brenk": flags["BRENK"],
            "struct_alert": label_from(flags),
        }
