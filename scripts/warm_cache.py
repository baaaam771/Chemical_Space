"""Precompute the root placement for every domain, so nobody waits for it.

The first visitor to a cold server waits 75-105 s for the full-set placement.
Run this once after deploying (or after a ChEMBL refresh) and that wait becomes
a cache read.

    uv run python scripts/warm_cache.py [--domains physchem disease] [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import chemspace
from chemspace.cache import PlacementCache
from chemspace.data.chembl import resolve_data_path
from chemspace.domains import default_registry
from chemspace.session import ExplorationSession


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domains", nargs="*", default=None, help="default: all of them")
    ap.add_argument("--cache-dir", default="data/cache/placements")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stdout
    )
    logging.getLogger("numba").setLevel(logging.WARNING)

    ds = chemspace.load()
    registry = default_registry(ds)
    cache = PlacementCache(resolve_data_path(args.cache_dir))
    if args.force:
        print(f"cleared {cache.clear()} cached placements")

    names = args.domains or list(registry.names)
    unknown = [n for n in names if n not in registry]
    if unknown:
        print(f"unknown domain(s): {unknown}; available: {registry.names}")
        return 2

    print(f"{ds} · warming {len(names)} domain(s) into {cache.directory}\n")
    for name in names:
        session = ExplorationSession(ds, registry, cache=cache)
        before = cache.hits
        t = time.perf_counter()
        node = session.root(name)
        elapsed = time.perf_counter() - t
        hit = cache.hits > before
        print(
            f"  {registry[name].label:6} ({name:9}) "
            f"{node.n:>7,}개 · {len(node.placement.cluster_ids):>3}군집 · "
            f"{elapsed:6.1f}s {'(캐시 적중)' if hit else '(계산 후 저장)'}"
        )

    print(f"\ncache: {cache.hits} hits, {cache.misses} misses -> {cache.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
