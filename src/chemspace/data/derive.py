"""RDKit-derived per-compound columns: descriptors, scaffolds, modality rules.

Computed once per snapshot and cached, because the domains need them on every
re-placement and parsing 17k molecules is not free.

The physicochemical numbers come from RDKit rather than from ChEMBL's
``molecule_properties``.  ChEMBL's are fine but have gaps, and a descriptor that
is missing for some compounds quietly distorts a z-score; recomputing gives one
consistent set with no holes.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

log = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")

FP_BITS = 2048
FP_RADIUS = 2  # ECFP4

#: Electrophilic groups, as (label, SMARTS, tier).
#:
#: ``strong`` groups are covalent *by design* -- finding one is good evidence
#: the compound is meant to form a bond with its target.
#:
#: ``weak`` groups are electrophilic but far too common in chemotypes nobody
#: calls covalent.  The generic Michael acceptor is the worst offender: on
#: ChEMBL_37 it matches 693 compounds, mostly prostaglandins, steroidal enones
#: and chalcones, and letting it set the label put 654 of those in the covalent
#: modality.  So weak matches are recorded in ``covalent_warhead`` for
#: inspection but never decide the modality.
#:
#: Order matters: ``next()`` takes the first match, so strong patterns come
#: first and a compound carrying both an acrylamide and a stray enone is
#: reported as the acrylamide.
COVALENT_WARHEADS: tuple[tuple[str, str, str], ...] = (
    ("beta-lactam", "[$([CX3](=O)[NX3]);r4]", "strong"),
    ("acrylamide", "[CX3]=[CX3][CX3](=O)[NX3]", "strong"),
    ("haloacetamide", "[F,Cl,Br,I][CH2][CX3](=O)[NX3]", "strong"),
    ("vinyl sulfone", "[CX3]=[CX3][SX4](=O)=O", "strong"),
    ("sulfonyl fluoride", "[SX4](=O)(=O)[F]", "strong"),
    ("isocyanate", "[NX2]=[CX2]=[OX1]", "strong"),
    ("epoxide", "[OX2;r3]1[CX4;r3][CX4;r3]1", "strong"),
    ("aziridine", "[NX3;r3]1[CX4;r3][CX4;r3]1", "strong"),
    ("boronic acid/ester", "[BX3]([OX2])[OX2]", "strong"),
    ("nitrile warhead", "[NX3,OX2][CX4][CX2]#[NX1]", "strong"),
    ("disulfide", "[SX2][SX2]", "weak"),
    ("michael acceptor", "[CX3]=[CX3][CX3](=O)[#6,OX2]", "weak"),
    ("aldehyde", "[CX3H1](=O)[#6]", "weak"),
)

STRONG_WARHEADS = frozenset(n for n, _, tier in COVALENT_WARHEADS if tier == "strong")

#: E3 ligase ligand motifs -- the strongest available signal for a degrader.
E3_LIGANDS: tuple[tuple[str, str], ...] = (
    ("glutarimide (CRBN)", "O=C1NC(=O)CCC1"),
    ("hydroxyproline amide (VHL)", "[OX2H][CX4]1[CX4][CX4]([NX3][CX4]1)[CX3](=O)[NX3]"),
)

_DESCRIPTORS = (
    "mw",
    "clogp",
    "tpsa",
    "hbd",
    "hba",
    "rotb",
    "fsp3",
    "qed",
    "ring_count",
    "aromatic_rings",
    "heavy_atoms",
    "formal_charge",
    "n_stereocenters",
    "max_ring_size",
    "n_amide_bonds",
    "n_ring_systems",
    "ro5_violations",
)

#: Columns the physchem domain embeds. Counts and continuous values together --
#: all are z-scored within the subset, so the scales do not need to match.
PHYSCHEM_FEATURES = (
    "mw",
    "clogp",
    "tpsa",
    "hbd",
    "hba",
    "rotb",
    "fsp3",
    "qed",
    "ring_count",
    "aromatic_rings",
    "heavy_atoms",
    "formal_charge",
    "n_stereocenters",
)

BIOLOGIC_TYPES = frozenset(
    {"Protein", "Antibody", "Oligonucleotide", "Oligosaccharide", "Enzyme", "Cell", "Gene"}
)

_compiled_warheads = [
    (name, Chem.MolFromSmarts(s), tier) for name, s, tier in COVALENT_WARHEADS
]
_compiled_e3 = [(name, Chem.MolFromSmarts(s)) for name, s in E3_LIGANDS]
_amide = Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")

for _n, _p in ((n, p) for n, p, _ in _compiled_warheads):
    if _p is None:  # pragma: no cover - guards a typo in the SMARTS tables
        raise ValueError(f"invalid warhead SMARTS for {_n}")
for _n, _p in _compiled_e3:
    if _p is None:  # pragma: no cover
        raise ValueError(f"invalid E3 ligand SMARTS for {_n}")


def _n_ring_systems(mol: Chem.Mol) -> int:
    """Number of fused-ring systems (rings sharing atoms count as one)."""
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    merged: list[set[int]] = []
    for r in rings:
        hits = [m for m in merged if m & r]
        for m in hits:
            merged.remove(m)
        merged.append(set().union(r, *hits))
    return len(merged)


def _max_ring_size(mol: Chem.Mol) -> int:
    rings = mol.GetRingInfo().AtomRings()
    return max((len(r) for r in rings), default=0)


def _descriptors(mol: Chem.Mol) -> dict:
    mw = Descriptors.MolWt(mol)
    clogp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    violations = sum((mw > 500, clogp > 5, hbd > 5, hba > 10))
    return {
        "mw": mw,
        "clogp": clogp,
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": hbd,
        "hba": hba,
        "rotb": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "qed": QED.qed(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "formal_charge": Chem.GetFormalCharge(mol),
        "n_stereocenters": rdMolDescriptors.CalcNumAtomStereoCenters(mol),
        "max_ring_size": _max_ring_size(mol),
        "n_amide_bonds": len(mol.GetSubstructMatches(_amide)),
        "n_ring_systems": _n_ring_systems(mol),
        "ro5_violations": violations,
    }


def _scaffolds(mol: Chem.Mol) -> dict:
    try:
        murcko = MurckoScaffold.GetScaffoldForMol(mol)
        murcko_smiles = Chem.MolToSmiles(murcko) if murcko.GetNumAtoms() else ""
        generic = (
            Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(murcko))
            if murcko.GetNumAtoms()
            else ""
        )
    except Exception:  # noqa: BLE001 - a few exotic structures defeat Murcko
        murcko_smiles, generic = "", ""
    return {"murcko_smiles": murcko_smiles, "murcko_generic": generic}


def _modality(row: dict, molecule_type: str | None) -> tuple[str, str]:
    """Rule-based modality label. Returns (modality, reason).

    First build only: this is the placeholder for the trained modality
    classifier in the project plan. ChEMBL's own ``molecule_type`` is trusted
    where it is informative, then small molecules are refined by structure.
    """
    if molecule_type in BIOLOGIC_TYPES:
        return molecule_type, "ChEMBL molecule_type"
    if row["max_ring_size"] >= 12:
        return "거대고리", f"max ring size {row['max_ring_size']}"
    if row["n_amide_bonds"] >= 4 and row["mw"] >= 500:
        return "펩타이드", f"{row['n_amide_bonds']} amide bonds, MW {row['mw']:.0f}"
    if row["e3_ligand"] and row["mw"] >= 600:
        return "PROTAC/분자접착제 후보", f"E3 motif: {row['e3_ligand']}"
    if row["mw"] >= 700 and row["rotb"] >= 10 and row["n_ring_systems"] >= 3:
        return "PROTAC/분자접착제 후보", "MW/flexibility/ring-system heuristic"
    # Only a by-design warhead sets this label; a bare enone or aldehyde does not.
    if row["covalent_tier"] == "strong":
        return "공유결합", f"warhead: {row['covalent_warhead']}"
    if row["mw"] > 500 and row["ro5_violations"] >= 2:
        return "bRo5", f"{row['ro5_violations']} Ro5 violations"
    return "저분자", "default"


def derive_table(
    molecules: pd.DataFrame, log_every: int = 2000
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute descriptors, scaffolds, modality and fingerprints.

    Returns ``(derived, fps, scaffold_fps)``; array rows align with
    ``derived["chembl_id"]``.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    rows: list[dict] = []
    fps: list[np.ndarray] = []
    sfps: list[np.ndarray] = []
    n_failed = 0

    for i, rec in enumerate(molecules.itertuples(index=False)):
        if log_every and i and i % log_every == 0:
            log.info("  derived %d/%d", i, len(molecules))
        mol = Chem.MolFromSmiles(rec.canonical_smiles)
        if mol is None:
            n_failed += 1
            continue
        row: dict = {"chembl_id": rec.chembl_id}
        row.update(_descriptors(mol))
        row.update(_scaffolds(mol))
        hit = next(
            (
                (name, tier)
                for name, patt, tier in _compiled_warheads
                if mol.HasSubstructMatch(patt)
            ),
            ("", ""),
        )
        row["covalent_warhead"], row["covalent_tier"] = hit
        row["e3_ligand"] = next(
            (name for name, patt in _compiled_e3 if mol.HasSubstructMatch(patt)), ""
        )
        row["modality"], row["modality_reason"] = _modality(
            row, getattr(rec, "molecule_type", None)
        )
        rows.append(row)

        fps.append(gen.GetFingerprintAsNumPy(mol).astype(np.uint8))
        scaf = Chem.MolFromSmiles(row["murcko_smiles"]) if row["murcko_smiles"] else None
        sfps.append(
            gen.GetFingerprintAsNumPy(scaf).astype(np.uint8)
            if scaf is not None
            else np.zeros(FP_BITS, dtype=np.uint8)
        )

    if n_failed:
        log.warning("%d molecules failed RDKit parsing and were dropped", n_failed)
    derived = pd.DataFrame(rows)
    return (
        derived,
        np.vstack(fps) if fps else np.zeros((0, FP_BITS), dtype=np.uint8),
        np.vstack(sfps) if sfps else np.zeros((0, FP_BITS), dtype=np.uint8),
    )


def descriptor_columns() -> tuple[str, ...]:
    return _DESCRIPTORS


def warhead_names(tier: str | None = None) -> Iterable[str]:
    """Warhead labels, optionally restricted to ``"strong"`` or ``"weak"``."""
    return (n for n, _, t in COVALENT_WARHEADS if tier is None or t == tier)
