"""화학적 신규성 (chemical novelty) from SMILES, via nearest-neighbour Tanimoto.

Another xlsx-taxonomy axis collection cannot fill: there is no "novel?" field
in ChEMBL. Novelty here is *distance from the known*, made concrete as the
Tanimoto (ECFP4) similarity to the nearest **approved drug** (``max_phase>=4``)
-- a compound far from every marketed drug sits in unexplored space, one on top
of one is a known chemotype. This is a structure-only proxy for "newness", not
a legal freedom-to-operate call (that needs a patent DB like SureChEMBL).

The fingerprint matches the rest of the project (ECFP4, 2048 bit) so the
similarity means the same thing the domains' do. RDKit is imported lazily;
``Dataset.load`` reads the cached result and never touches this module.
"""
from __future__ import annotations

import numpy as np

from chemspace.data.derive import FP_BITS, FP_RADIUS

#: Approved = marketed; the reference the distance is measured from.
APPROVED_MIN_PHASE = 4.0
#: Tier cut points on the nearest-approved Tanimoto. Above HIGH is a known
#: chemotype; below LOW is genuinely far from any marketed drug.
SIM_HIGH = 0.7
SIM_LOW = 0.4
LABEL_KNOWN = "기존 (≥0.7)"
LABEL_MID = "중간 (0.4–0.7)"
LABEL_NOVEL = "신규 (<0.4)"


def novelty_tier(nn_sim: float) -> str:
    """Tier label from the nearest-approved similarity (blank if unknown)."""
    if nn_sim is None or (isinstance(nn_sim, float) and np.isnan(nn_sim)):
        return ""
    if nn_sim >= SIM_HIGH:
        return LABEL_KNOWN
    if nn_sim >= SIM_LOW:
        return LABEL_MID
    return LABEL_NOVEL


def morgan_fps(smiles_iter):
    """Morgan (ECFP4, 2048-bit) fingerprints; ``None`` for an unparseable SMILES."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    fps = []
    for smi in smiles_iter:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
        fps.append(gen.GetFingerprint(mol) if mol is not None else None)
    return fps


def nearest_similarity(fps, ref_fps) -> np.ndarray:
    """Max Tanimoto of each fp to the reference set.

    ``NaN`` where a compound has no fingerprint, and all-``NaN`` when the
    reference set is empty (nothing to be near). The reference includes the
    approved drugs themselves, so an approved drug scores 1.0 against itself --
    correctly the least novel thing on the map.
    """
    from rdkit import DataStructs

    ref = [f for f in ref_fps if f is not None]
    out = np.full(len(fps), np.nan, dtype=np.float64)
    if not ref:
        return out
    for i, fp in enumerate(fps):
        if fp is None:
            continue
        out[i] = max(DataStructs.BulkTanimotoSimilarity(fp, ref))
    return out
