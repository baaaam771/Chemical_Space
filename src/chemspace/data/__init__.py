"""Data acquisition and loading."""

from chemspace.data.chembl import (
    ChemblClient,
    fetch_snapshot,
    project_root,
    resolve_data_path,
    snapshot_dir,
)

__all__ = [
    "ChemblClient",
    "fetch_snapshot",
    "project_root",
    "resolve_data_path",
    "snapshot_dir",
]
