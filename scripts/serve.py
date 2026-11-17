"""Run the web UI.

    uv run python scripts/serve.py [--port 8000] [--no-cache]

The dataset loads once at startup and is shared by every browser session.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--snapshot", default=None, help="ChEMBL snapshot dir (default: newest)")
    ap.add_argument(
        "--no-cache",
        action="store_true",
        help="recompute every placement instead of reusing the disk cache",
    )
    ap.add_argument("--workers", type=int, default=2, help="placement threads")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("numba").setLevel(logging.WARNING)

    import uvicorn

    from chemspace.web import create_app

    app = create_app(
        snapshot=args.snapshot,
        use_cache=not args.no_cache,
        max_workers=args.workers,
    )
    print(f"\n  chemspace → http://{args.host}:{args.port}\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
