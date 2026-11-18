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

    # Windows consoles default to a legacy code page, which turns the arrow in
    # the ready line and every Korean message into mojibake. The other scripts
    # already do this; serve.py was missed, and it is the one a new user runs
    # first.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("numba").setLevel(logging.WARNING)

    # The web stack is an optional dependency group, so a fresh clone that ran
    # a plain `uv sync` has the engine but not the server. The bare
    # ModuleNotFoundError that produces names uvicorn and nothing else, which
    # reads as a broken install rather than an install that stopped one step
    # short -- so say which step.
    try:
        import uvicorn

        from chemspace.web import create_app
    except ModuleNotFoundError as exc:
        if exc.name not in {"uvicorn", "fastapi"}:
            raise
        for line in (
            "",
            f"  웹 서버 의존성이 없습니다 ({exc.name}).",
            "  설치: uv sync --python 3.12 --all-extras",
            "  (또는 uv pip install 'fastapi>=0.115' 'uvicorn[standard]>=0.30')",
            "",
        ):
            print(line, file=sys.stderr)
        return 1

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
