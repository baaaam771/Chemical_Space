"""Curated two-level protein-class hierarchy for the 표적 overlay.

ChEMBL's own ``protein_classification`` top level is *Enzyme · Membrane
receptor · Ion channel · Transporter · Transcription factor · Epigenetic
regulator* -- which does not match how a medicinal chemist groups targets.
Kinase and Protease are both *under* Enzyme (L2); GPCR is a branch of Membrane
receptor; Nuclear receptor is a branch of Transcription factor. Kinase
subfamilies (the Manning kinome groups: TK, CMGC, AGC, …) sit a further two
levels down (L4).

So this module folds ChEMBL's variable-depth path into a fixed, pharmacology-
first two-level map: eight curated L1 families, each with a readable L2
subfamily. It reaches a *different depth per branch* -- that is the whole point,
and why it cannot be done by picking one ``class_level``.

The curated hierarchy is a *label* laid over the existing target map, exactly
like the MoA/modality overlays. It is NOT a new feature space -- one-hot
encoding eight family labels would collapse each family to one stacked point
(the modality lesson). The map's geometry still comes from the individual
target tokens; this only colours and rolls it up.

Anything that does not fall into one of the eight families -- the ChEMBL
"Other*/Unclassified/Auxiliary" tail, non-GPCR membrane receptors, non-nuclear
transcription factors, and unannotated targets -- maps to ``("", "")`` and the
UI shows it as 미분류.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["curate", "curate_frame", "CURATED_L1"]

#: The eight curated top-level families, in the order a legend should list them.
CURATED_L1 = (
    "Kinase",
    "Protease",
    "Enzyme (기타)",
    "GPCR",
    "Ion Channel",
    "Nuclear Receptor",
    "Transporter",
    "Epigenetic",
)

#: Manning kinome groups (ChEMBL L4) -> readable kinase subfamily (curated L2).
_KINASE_GROUPS = {
    "TK protein kinase group": "Tyrosine kinase (TK)",
    "TKL protein kinase group": "Tyrosine kinase-like (TKL)",
    "CMGC protein kinase group": "CMGC (CDK·MAPK·GSK3)",
    "AGC protein kinase group": "AGC (PKA·PKC·PKG)",
    "CAMK protein kinase group": "CAMK",
    "STE protein kinase group": "STE",
    "CK1 protein kinase group": "CK1",
    "Other protein kinase group": "기타 protein kinase",
    "Atypical protein kinase group": "비정형 kinase",
}


def _norm(v) -> str:
    """A path level as a clean string -- '' for NaN/None/blank."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return "" if s.lower() == "nan" else s


def _gpcr_family(l2: str) -> str:
    """'Family A G protein-coupled receptor' -> 'Family A GPCR'."""
    s = l2.replace("G protein-coupled receptor", "GPCR").replace(" family", "")
    return " ".join(s.split())


def curate(l1, l2, l3, l4) -> tuple[str, str]:
    """One protein class's ``(l1..l4)`` path -> ``(curated_l1, curated_l2)``.

    Returns ``("", "")`` for anything outside the eight curated families.
    """
    l1, l2, l3, l4 = _norm(l1), _norm(l2), _norm(l3), _norm(l4)

    if l1 == "Enzyme":
        if l2 == "Kinase":
            sub = _KINASE_GROUPS.get(l4)
            if not sub:
                sub = (
                    "조절 subunit"
                    if l3 == "Protein kinase regulatory subunit"
                    else "기타 kinase"
                )
            return ("Kinase", sub)
        if l2 == "Protease":
            return ("Protease", l3 or "기타 protease")
        if l2:
            return ("Enzyme (기타)", l2)
        return ("Enzyme (기타)", "미상 효소")

    if l1 == "Membrane receptor":
        if "G protein-coupled receptor" in l2:
            return ("GPCR", _gpcr_family(l2))
        return ("", "")  # non-GPCR membrane receptors fall to 미분류

    if l1 == "Ion channel":
        return ("Ion Channel", l2 or "기타 채널")

    if l1 == "Transcription factor":
        if l2 == "Nuclear receptor":
            return ("Nuclear Receptor", l3 or "Nuclear receptor")
        return ("", "")

    if l1 == "Transporter":
        return ("Transporter", l2 or "기타 수송체")

    if l1 == "Epigenetic regulator":
        return ("Epigenetic", l2 or "기타")

    return ("", "")


def curate_frame(targets: pd.DataFrame, idx: pd.Index) -> dict[str, pd.Series]:
    """Per-compound curated ``target_l1``/``target_l2``, aligned to ``idx``.

    A compound hits several targets across several classes, so it carries a
    *distribution* of curated families. The label is the compound's most common
    curated L1, and -- within that L1 -- its most common L2. Ties break by name
    so the result is deterministic. Compounds with no curated target get ``""``.
    """
    cols = ["chembl_id", "l1", "l2", "l3", "l4"]
    have = [c for c in cols if c in targets.columns]
    if "chembl_id" not in have or targets.empty:
        blank = pd.Series("", index=idx, dtype=object)
        return {"target_l1": blank, "target_l2": blank.copy()}

    tc = targets[have].copy()
    for c in ("l1", "l2", "l3", "l4"):
        if c not in tc.columns:
            tc[c] = ""

    # Curate once per distinct path, then map back -- there are a few hundred
    # distinct class paths, not one per row.
    combos = tc[["l1", "l2", "l3", "l4"]].drop_duplicates()
    curated = combos.apply(
        lambda r: curate(r["l1"], r["l2"], r["l3"], r["l4"]), axis=1, result_type="expand"
    )
    combos = combos.assign(cl1=curated[0].values, cl2=curated[1].values)
    tc = tc.merge(combos, on=["l1", "l2", "l3", "l4"], how="left")
    tc = tc[tc["cl1"].astype(str) != ""]
    if tc.empty:
        blank = pd.Series("", index=idx, dtype=object)
        return {"target_l1": blank, "target_l2": blank.copy()}

    # Most common curated L1 per compound (ties -> alphabetical).
    g1 = tc.groupby(["chembl_id", "cl1"]).size().reset_index(name="n")
    g1 = g1.sort_values(["n", "cl1"], ascending=[False, True])
    top1 = g1.drop_duplicates("chembl_id").set_index("chembl_id")["cl1"]

    # Most common L2 within the chosen L1.
    sel = tc.merge(top1.rename("sel1"), left_on="chembl_id", right_index=True)
    sel = sel[sel["cl1"] == sel["sel1"]]
    g2 = sel.groupby(["chembl_id", "cl2"]).size().reset_index(name="n")
    g2 = g2.sort_values(["n", "cl2"], ascending=[False, True])
    top2 = g2.drop_duplicates("chembl_id").set_index("chembl_id")["cl2"]

    return {
        "target_l1": top1.reindex(idx).fillna("").astype(object),
        "target_l2": top2.reindex(idx).fillna("").astype(object),
    }
