"""Build the 표적(지식) knowledge-embedding cache for a snapshot.

The 표적(지식) domain re-places compounds by a language model's *knowledge* of
the proteins they hit: each target's description (name + protein-class path +
organism) is turned into a text embedding, so targets a model treats as
biologically related -- same pathway, disease or mechanism -- land near each
other even when they share no ligand and little sequence. A compound's vector is
the mean over its targets' vectors.

Like the ESM cache, the embeddings ship as a per-snapshot ``target_knowledge.npz``
(keyed by ChEMBL target id) and are gitignored; the app only reads them, and the
domain is simply unavailable until this script has produced the file. Generating
them needs an embedding model, which the app's own runtime does NOT require.

Two backends (pick with ``--backend``):

  local  sentence-transformers on this box (default). Needs ``sentence-transformers``
         installed. A biomedical model reads target text better than a generic one:
             --backend local --model pritamdeka/S-PubMedBert-MS-MARCO
         The default (all-MiniLM-L6-v2) is small and general.

  api    an OpenAI-compatible ``/embeddings`` endpoint -- e.g. an internal LLM
         server. Only public ChEMBL target text is sent. No extra dependency.
             --backend api --api-base http://HOST:PORT/v1 --api-model <name> \
                 [--api-key $KEY]   (or env OPENAI_API_KEY)

Usage:

    uv run python scripts/build_target_knowledge.py                 # newest snapshot, local
    uv run python scripts/build_target_knowledge.py --all
    uv run python scripts/build_target_knowledge.py --snapshot chembl_37_t50 \
        --backend api --api-base http://10.13.22.171:8000/v1 --api-model text-embedding

The description text is a pure function of ChEMBL fields; swapping the model or
backend changes only the geometry, so re-running is safe and idempotent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from chemspace.data.chembl import resolve_data_path, snapshot_dir

OUT_NAME = "target_knowledge.npz"


def build_descriptions(snap: Path) -> tuple[list[str], list[str]]:
    """One description per ChEMBL target id: name + class path + organism."""
    tg = pd.read_parquet(snap / "targets.parquet")
    pcs = pd.read_parquet(
        snap / "protein_classes.parquet", columns=["protein_class_id", "class_path"]
    )
    df = tg.merge(pcs, on="protein_class_id", how="left")

    ids: list[str] = []
    texts: list[str] = []
    for tid, g in df.groupby("target_chembl_id"):
        if not isinstance(tid, str) or not tid:
            continue
        name = next((str(v) for v in g.get("target_pref_name", []) if isinstance(v, str) and v), "")
        organism = next((str(v) for v in g.get("organism", []) if isinstance(v, str) and v), "")
        # The richest (longest) class path this target carries.
        paths = [str(v) for v in g.get("class_path", []) if isinstance(v, str) and v]
        path = max(paths, key=len) if paths else ""
        parts = [name or tid]
        if path:
            parts.append(f"Protein class: {path.replace(' > ', ' / ')}")
        if organism:
            parts.append(f"Organism: {organism}")
        ids.append(tid)
        texts.append(". ".join(parts) + ".")
    return ids, texts


def embed_local(texts: list[str], model: str, batch: int) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "sentence-transformers is not installed. Install it on a build box "
            "(`uv pip install sentence-transformers`) or use --backend api."
        ) from exc
    st = SentenceTransformer(model)
    emb = st.encode(
        texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=True
    )
    return np.ascontiguousarray(emb, dtype=np.float32)


def embed_api(texts: list[str], base: str, model: str, key: str, batch: int) -> np.ndarray:
    """Call an OpenAI-compatible POST {base}/embeddings in batches."""
    url = base.rstrip("/") + "/embeddings"
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        body = json.dumps({"model": model, "input": chunk}).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        rows = sorted(data["data"], key=lambda d: d["index"])
        out.extend(r["embedding"] for r in rows)
        print(f"  embedded {min(i + batch, len(texts)):,}/{len(texts):,}", flush=True)
    return np.ascontiguousarray(out, dtype=np.float32)


def build_one(snap: Path, embfn) -> Path:
    ids, texts = build_descriptions(snap)
    print(f"{snap.name}: {len(ids):,} targets", flush=True)
    t0 = time.perf_counter()
    emb = embfn(texts)
    if emb.shape[0] != len(ids):
        raise SystemExit(f"got {emb.shape[0]} vectors for {len(ids)} targets")
    out = snap / OUT_NAME
    np.savez_compressed(out, target_ids=np.array(ids, dtype=object), emb=emb)
    dt = time.perf_counter() - t0
    print(f"  wrote {out}  ({emb.shape[1]}-dim, {dt:.0f}s)", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--snapshot", default=None, help="snapshot dir name (default: newest)")
    ap.add_argument("--all", action="store_true", help="build for every cached snapshot")
    ap.add_argument("--backend", choices=["local", "api"], default="local")
    ap.add_argument("--model", default="all-MiniLM-L6-v2", help="local sentence-transformers model")
    ap.add_argument("--api-base", default=None, help="OpenAI-compatible base URL, e.g. http://host:port/v1")
    ap.add_argument("--api-model", default=None, help="embedding model name on the endpoint")
    ap.add_argument("--api-key", default=None, help="bearer token (or env OPENAI_API_KEY)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    if args.backend == "api":
        if not args.api_base or not args.api_model:
            print("--backend api needs --api-base and --api-model")
            return 2
        key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        embfn = lambda t: embed_api(t, args.api_base, args.api_model, key, args.batch_size)
    else:
        embfn = lambda t: embed_local(t, args.model, args.batch_size)

    base = resolve_data_path(args.cache_dir)
    if args.all:
        snaps = sorted(d for d in base.iterdir() if (d / "targets.parquet").exists())
    elif args.snapshot:
        snaps = [base / args.snapshot]
    else:
        s = snapshot_dir(args.cache_dir)
        if s is None:
            print("no snapshot found")
            return 2
        snaps = [s]

    for snap in snaps:
        if not (snap / "targets.parquet").exists():
            print(f"skip {snap} (no targets.parquet)")
            continue
        build_one(snap, embfn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
