"""Drop a domain's cached root placement after its embedding cache was rebuilt.

The placement cache keys on (dataset fingerprint, domain, embed/cluster config,
compound ids) -- deliberately NOT on the embedding vectors, since those are not
known until the features are built (see chemspace.cache). So regenerating a
gitignored embedding cache (``target_esm.npz`` / ``target_knowledge.npz``) is
invisible to the cache, and the *old* placement would be served for the new
embeddings. Run this for the affected domain right after rebuilding its cache.

    uv run python scripts/invalidate_placement.py --domain target_knowledge

Only the root (whole-snapshot) placement is dropped -- that is what a domain
switch rebuilds. Any narrowed sub-placements a session made are niche and will
be re-placed when next visited; clear the whole ``placements`` dir to force that.
"""
from __future__ import annotations

import argparse

from chemspace.cache import PlacementCache
from chemspace.data.chembl import resolve_data_path
from chemspace.data.dataset import Dataset
from chemspace.placement import ClusterConfig, EmbedConfig


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--domain", required=True, help="domain name, e.g. target_knowledge")
    ap.add_argument("--cache-dir", default="data/cache/placements")
    args = ap.parse_args()

    ds = Dataset.load()
    cache = PlacementCache(resolve_data_path(args.cache_dir))
    dataset_id = getattr(ds, "fingerprint", None) or ds.version
    key = cache.key(dataset_id, args.domain, ds.all_ids, EmbedConfig(), ClusterConfig())
    path = cache.directory / f"{key}.npz"
    existed = path.exists()
    path.unlink(missing_ok=True)
    print(
        f"{'deleted' if existed else 'no cached root placement for'} "
        f"{args.domain}: {path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
