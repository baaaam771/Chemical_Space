"""chemspace -- iterative chemical space exploration.

The loop: place a compound set with one domain's features, select a region, then
re-place *only the selected compounds* under a different domain. Every step is a
node on an exploration tree, so you can walk back and branch differently.

    import chemspace

    ds = chemspace.load()                    # newest cached ChEMBL snapshot
    s = chemspace.ExplorationSession(ds)
    s.root("physchem")                       # n1: all compounds, physchem
    s.select("n1", clusters=[3, 4])          # n2: a region, still physchem
    s.switch_domain("n2", "disease")         # n3: same compounds, disease
    print(s.breadcrumb("n3"))
    print(s.tree())

In a notebook, drive the same session by dragging::

    chemspace.SpaceExplorer(s).show()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "ExplorationSession",
    "SpaceExplorer",
    "DomainRegistry",
    "EmbedConfig",
    "ClusterConfig",
    "load",
    "fetch_snapshot",
    "data_path",
]

if TYPE_CHECKING:  # pragma: no cover
    from chemspace.data.chembl import fetch_snapshot
    from chemspace.data.dataset import Dataset
    from chemspace.domains import DomainRegistry
    from chemspace.placement import ClusterConfig, EmbedConfig
    from chemspace.session import ExplorationSession
    from chemspace.viewer import SpaceExplorer

_LAZY = {
    "Dataset": ("chemspace.data.dataset", "Dataset"),
    "ExplorationSession": ("chemspace.session", "ExplorationSession"),
    "SpaceExplorer": ("chemspace.viewer", "SpaceExplorer"),
    "DomainRegistry": ("chemspace.domains", "DomainRegistry"),
    "EmbedConfig": ("chemspace.placement", "EmbedConfig"),
    "ClusterConfig": ("chemspace.placement", "ClusterConfig"),
    "fetch_snapshot": ("chemspace.data.chembl", "fetch_snapshot"),
    "data_path": ("chemspace.data.chembl", "resolve_data_path"),
}


def __getattr__(name: str) -> Any:
    """Import viewer/RDKit machinery only when it is actually asked for."""
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


def load(snapshot: str | None = None, cache_dir: str = "data/cache", **kw):
    """Load the newest cached ChEMBL snapshot (or a named one)."""
    from chemspace.data.dataset import Dataset

    return Dataset.load(snapshot=snapshot, cache_dir=cache_dir, **kw)
