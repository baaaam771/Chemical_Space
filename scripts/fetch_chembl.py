"""Download the ChEMBL drug layer into data/cache/<release>/.

    uv run python scripts/fetch_chembl.py [--force] [--min-phase 0.5]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from chemspace.data.chembl import fetch_snapshot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--min-phase", type=float, default=0.5)
    ap.add_argument(
        "--min-pchembl",
        type=float,
        default=7.0,
        help="measured-potency cutoff for target annotation (7 = 100 nM)",
    )
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    out = fetch_snapshot(
        cache_dir=args.cache_dir,
        min_phase=args.min_phase,
        min_pchembl=args.min_pchembl,
        force=args.force,
    )
    manifest = json.loads((Path(out) / "manifest.json").read_text(encoding="utf-8"))
    print("\nsnapshot:", out)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
