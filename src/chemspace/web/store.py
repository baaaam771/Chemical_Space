"""Server-side exploration sessions.

The dataset is loaded once and shared: it is read-only after construction and
costs ~1.5 s plus a few hundred MB, so one copy serves every browser session.
Each browser session gets its own :class:`ExplorationSession` -- its own tree --
on top of that shared data.

Sessions live in memory. Losing them on restart is survivable because the
expensive part is the placements, and those are in the disk cache: a rebuilt
session re-places instantly. Anything a user wants to keep goes through
``save`` to a named directory on disk.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from chemspace.cache import PlacementCache
from chemspace.data.chembl import resolve_data_path
from chemspace.data.dataset import Dataset
from chemspace.domains import default_registry
from chemspace.placement import ClusterConfig, EmbedConfig
from chemspace.session import ExplorationSession

log = logging.getLogger(__name__)

__all__ = ["SessionStore", "StoredSession"]


@dataclass
class StoredSession:
    session_id: str
    session: ExplorationSession
    created_at: float = field(default_factory=time.time)
    touched_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.touched_at = time.time()


class SessionStore:
    def __init__(
        self,
        snapshot: str | Path | None = None,
        cache_dir: str | Path = "data/cache/placements",
        use_cache: bool = True,
        max_sessions: int = 64,
    ):
        log.info("loading dataset ...")
        self.dataset: Dataset = Dataset.load(snapshot=snapshot)
        self.registry = default_registry(self.dataset)
        self.cache = PlacementCache(resolve_data_path(cache_dir), enabled=use_cache)
        self.embed_cfg = EmbedConfig()
        self.cluster_cfg = ClusterConfig()
        self._sessions: dict[str, StoredSession] = {}
        self._lock = threading.Lock()
        self._max = max_sessions
        log.info("ready: %s, cache %s", self.dataset, self.cache.directory)

    # ------------------------------------------------------------------ CRUD

    def create(self) -> StoredSession:
        sid = uuid.uuid4().hex[:12]
        session = ExplorationSession(
            self.dataset,
            # One registry shared across sessions: domains hold only the
            # dataset and config, never per-exploration state.
            self.registry,
            self.embed_cfg,
            self.cluster_cfg,
            cache=self.cache,
        )
        stored = StoredSession(session_id=sid, session=session)
        with self._lock:
            self._sessions[sid] = stored
            self._evict()
        log.info("session %s created (%d live)", sid, len(self._sessions))
        return stored

    def get(self, session_id: str) -> StoredSession | None:
        stored = self._sessions.get(session_id)
        if stored:
            stored.touch()
        return stored

    def _evict(self) -> None:
        while len(self._sessions) > self._max:
            oldest = min(self._sessions.values(), key=lambda s: s.touched_at)
            self._sessions.pop(oldest.session_id, None)
            log.info("evicted idle session %s", oldest.session_id)

    @property
    def domains(self) -> list[dict]:
        return [
            {"name": d.name, "label": d.label, "description": d.description}
            for d in self.registry
        ]

    def stats(self) -> dict:
        return {
            "chembl_version": self.dataset.version,
            "compounds": len(self.dataset.compounds),
            "sessions": len(self._sessions),
            "cache_hits": self.cache.hits,
            "cache_misses": self.cache.misses,
            "cache_dir": str(self.cache.directory),
        }
