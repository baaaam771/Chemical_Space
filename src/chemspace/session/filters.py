"""Named subsets -- narrowing by what a compound *is*, not by where you dragged.

``select`` answers "this region of the map"; a filter answers "only the ones
that have X". Both narrow and both re-place, so a filter is the same kind of
move as a lasso and belongs in the tree the same way.

The expansion to 73,864 compounds is what made these necessary. It moved the
first usable narrowing axis from disease to target: a 1,200-compound region of
the property map carries disease annotation on 91 of its compounds (7.6%), so
switching to disease there is a cliff rather than a step. Restricting to the
drug layer *first* puts that back to 48.3% -- the behaviour of the old 12,804
snapshot, reproduced inside the big one instead of kept in a second copy.

A filter must stay cheap enough to run before the user commits to a placement,
because its whole purpose is to be the thing you do *instead of* discovering
the cliff after a 20-second wait.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

__all__ = ["Filter", "FILTERS", "drug_layer_mask", "resolve"]


@dataclass(frozen=True)
class Filter:
    """One named subset rule.

    ``mask`` returns a boolean array over the compound ids it is given, in the
    same order. ``needs_domain`` says whether the rule is about one domain's
    annotation, which the caller has to name.
    """

    name: str
    label: str
    description: str
    needs_domain: bool = False


#: The annotation-dense layer the snapshots are anchored on.
#:
#: ``max_phase`` is ChEMBL's own highest clinical phase reached. The snapshot
#: manifest records the threshold it was built with (``min_phase``), so the
#: rule is read from the data rather than hard-coded: on ``chembl_37_t50``
#: this selects exactly the 12,804 compounds of the ``chembl_37`` snapshot.
DRUG_LAYER = Filter(
    name="drug",
    label="약물층",
    description="임상 단계에 오른 화합물 (max_phase ≥ 임계값) — 적응증·ATC 주석이 두터운 층",
)

ANNOTATED = Filter(
    name="annotated",
    label="주석 보유",
    description="고른 도메인의 데이터를 실제로 가진 화합물만 — 미커버 띠를 걷어냅니다",
    needs_domain=True,
)

FILTERS: dict[str, Filter] = {f.name: f for f in (DRUG_LAYER, ANNOTATED)}


def resolve(kind: str) -> Filter:
    try:
        return FILTERS[kind]
    except KeyError:
        raise KeyError(
            f"unknown filter {kind!r}; have {sorted(FILTERS)}"
        ) from None


def drug_layer_mask(dataset, compound_ids: Sequence[str]) -> np.ndarray:
    """Which of ``compound_ids`` belong to the clinical-phase drug layer.

    Missing and sentinel phases are *not* in the layer. ChEMBL writes -1 for
    "withdrawn or unknown", which is a smaller number than any real phase, so
    a plain ``>=`` comparison would be right by accident and wrong if the
    sentinel ever changed; ``fillna`` plus the explicit threshold says what is
    meant.
    """
    threshold = float(dataset.manifest.get("min_phase", 0.5))
    phase = pd.to_numeric(
        dataset.compounds["max_phase"].reindex(list(compound_ids)), errors="coerce"
    )
    return (phase.fillna(-99) >= threshold).to_numpy()
