"""Disk cache for placements.

Placing all 12,804 compounds takes 75-105 s, almost all of it UMAP running
single-threaded for the seed.  In a notebook you pay that once and move on; in
a web UI every visitor would pay it before seeing anything.

A placement is a pure function of (domain, compound set, embed config, cluster
config), so it can be cached safely: a hit returns exactly what recomputation
would.  The one wrinkle is that Jaccard embeddings are not bit-reproducible
across processes (see the README), which makes the cache *more* useful, not
less -- it is the only way two runs get identical coordinates for those domains.

The key covers the compound ids in order, so a different selection, a different
ordering, or a changed config all miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from chemspace.placement import ClusterConfig, EmbedConfig, Placement

log = logging.getLogger(__name__)

__all__ = ["PlacementCache"]

#: Bumped when a change makes previously cached coordinates wrong.
#:
#: 2: ``min_cluster_size=None`` resolves through ``auto_min_cluster_size``,
#:    which now caps at 40. The stored key records the *unresolved* config, so
#:    entries written under the old rule would be served for an identical key
#:    and silently keep the old labels. The resolved value cannot go into the
#:    key -- it depends on how many compounds the domain covers, which is not
#:    known until the features are built -- so the version carries it instead.
CACHE_VERSION = 2


class PlacementCache:
    """Memoise placements under ``directory``, one ``.npz`` per key."""

    def __init__(self, directory: str | Path, enabled: bool = True):
        self.directory = Path(directory)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def key(
        self,
        dataset_id: str,
        domain: str,
        compound_ids: Sequence[str],
        embed: EmbedConfig,
        cluster: ClusterConfig,
    ) -> str:
        h = hashlib.blake2b(digest_size=16)
        # dataset_id must identify the snapshot's *content*, not just its
        # release name -- see Dataset.fingerprint.
        h.update(f"v{CACHE_VERSION}|{dataset_id}|{domain}|".encode())
        h.update(json.dumps(asdict(embed), sort_keys=True).encode())
        h.update(json.dumps(asdict(cluster), sort_keys=True).encode())
        h.update(b"|")
        for cid in compound_ids:
            h.update(cid.encode())
            h.update(b",")
        return h.hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.npz"

    def get(self, key: str, compound_ids: Sequence[str], domain: str) -> Placement | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            data = np.load(path, allow_pickle=False)
            meta = json.loads(str(data["meta"]))
            placement = Placement(
                coords=data["coords"],
                labels=data["labels"],
                compound_ids=tuple(compound_ids),
                domain=domain,
                embed_method=meta["embed_method"],
                cluster_method=meta["cluster_method"],
                metric=meta["metric"],
                n_covered=meta["n_covered"],
                notes={**meta["notes"], "from_cache": True},
            )
        except Exception as exc:  # noqa: BLE001 - a bad cache file must not be fatal
            log.warning("ignoring unreadable cache entry %s (%s)", path.name, exc)
            path.unlink(missing_ok=True)
            self.misses += 1
            return None
        if placement.coords.shape[0] != len(compound_ids):
            log.warning("cache entry %s has the wrong row count; dropping", path.name)
            path.unlink(missing_ok=True)
            self.misses += 1
            return None
        self.hits += 1
        return placement

    def put(self, key: str, placement: Placement) -> None:
        if not self.enabled:
            return
        meta = {
            "embed_method": placement.embed_method,
            "cluster_method": placement.cluster_method,
            "metric": placement.metric,
            "n_covered": placement.n_covered,
            "notes": {k: v for k, v in placement.notes.items() if k != "from_cache"},
        }
        # The temp name must itself end in .npz: savez_compressed appends .npz
        # to any path that does not, so a ".npz.tmp" target silently becomes
        # ".npz.tmp.npz" and the rename below then fails on a missing file --
        # which looked exactly like a cache that never stored anything.
        tmp = self.directory / f"{key}.{os.getpid()}.tmp.npz"
        try:
            np.savez_compressed(
                tmp,
                coords=placement.coords,
                labels=placement.labels,
                meta=np.array(json.dumps(meta)),
            )
            tmp.replace(self._path(key))
        except Exception as exc:  # noqa: BLE001 - caching is best-effort
            log.warning("could not write cache entry (%s)", exc)
            tmp.unlink(missing_ok=True)

    def clear(self) -> int:
        n = 0
        for f in self.directory.glob("*.npz"):
            f.unlink()
            n += 1
        for stray in self.directory.glob("*.tmp.npz"):  # interrupted writes
            stray.unlink(missing_ok=True)
        return n

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<PlacementCache {self.directory} hits={self.hits} misses={self.misses}>"
        )
