"""Background jobs, because a placement can take over a minute.

Placing all 12,804 compounds takes 75-105 s. That cannot happen inside a
request: browsers and proxies time out, and the user stares at a blank page
with no idea whether anything is happening.

So every call that places compounds returns a job id immediately and the
browser polls. Jobs run on a small thread pool -- UMAP and HDBSCAN spend their
time in NumPy/numba, which release the GIL, so threads are enough and avoid
shipping the dataset to worker processes.

The pool is deliberately narrow (two workers). These jobs are CPU-bound; more
workers would not finish sooner, they would just make every one of them slower
and compete with whoever is waiting.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

__all__ = ["Job", "JobRunner"]


@dataclass
class Job:
    job_id: str
    kind: str
    state: str = "pending"  # pending | running | done | error
    detail: str = ""
    result: Any = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return end - (self.started_at or self.created_at)

    def as_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "state": self.state,
            "detail": self.detail,
            "result": self.result,
            "error": self.error,
            "elapsed": round(self.elapsed, 1),
        }


class JobRunner:
    def __init__(self, max_workers: int = 2, keep: int = 200):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="place")
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._keep = keep
        self._lock = threading.Lock()

    def submit(self, kind: str, fn: Callable[[], Any], detail: str = "") -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], kind=kind, detail=detail)
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            self._evict()
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[], Any]) -> None:
        job.state = "running"
        job.started_at = time.time()
        try:
            job.result = fn()
            job.state = "done"
        except Exception as exc:  # noqa: BLE001 - surfaced to the browser
            job.state = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            log.error("job %s (%s) failed: %s", job.job_id, job.kind, job.error)
            log.debug("%s", traceback.format_exc())
        finally:
            job.finished_at = time.time()
            log.info("job %s (%s) %s in %.1fs", job.job_id, job.kind, job.state, job.elapsed)

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def _evict(self) -> None:
        while len(self._order) > self._keep:
            old = self._order.pop(0)
            self._jobs.pop(old, None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
