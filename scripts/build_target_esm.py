"""Rebuild the ESM-2 target-sequence embeddings that feed the 표적(서열) view.

The 표적(서열) domain re-places compounds by the *sequence* of the proteins they
hit, using ESM-2 (``esm2_t33_650M_UR50D``) protein-language-model embeddings.
Those embeddings ship as a static ``target_esm.npz`` per snapshot and are gitignored
(~24 MB), so this script regenerates them from scratch after a ChEMBL refresh.

Three steps, each resumable via ``--from``:

  1. maps   enumerate the UniProt accessions hit by covered compounds across the
            snapshots, writing ``<work>/acc_union.txt`` (and per-snapshot
            ``<snap>_comp_acc.parquet`` link tables, handy for re-measuring).
  2. fetch  download the sequences for that union from UniProtKB (small batches
            with a one-by-one fallback, so a stray Ensembl id can't sink a chunk)
            into ``<work>/seqs.parquet``.
  3. embed  mean-pool ESM-2 last-layer per-residue representations into a 1280-dim
            vector per accession and write ``target_esm.npz`` into each snapshot's
            cache dir. Needs a GPU box with ``torch`` + ``fair-esm`` installed
            (CPU works but is slow); the other two steps need neither.

    # whole pipeline (both snapshots), GPU box:
    uv run python scripts/build_target_esm.py

    # just re-embed from an already-fetched seqs.parquet:
    uv run python scripts/build_target_esm.py --from embed

    # fetch sequences on a networked box, embed later elsewhere (share --work):
    uv run python scripts/build_target_esm.py --only maps fetch --work /shared/esm
    uv run python scripts/build_target_esm.py --only embed        --work /shared/esm

See README's "루트에서 표적·모이어티가 동그란 이유" item 7 for why this encoding
earns a place alongside the Jaccard 표적 view rather than replacing it.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

STEPS = ("maps", "fetch", "embed")
DEFAULT_SNAPSHOTS = ("chembl_37", "chembl_37_t50")
UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/accessions"
USER_AGENT = "chemspace-research/1.0"
MAXLEN = 1022  # ESM-2 position limit; longer chains are truncated
EMB_DIM = 1280
LAYER = 33


# --------------------------------------------------------------------------- maps
def step_maps(work: Path, snapshots: tuple[str, ...]) -> None:
    """compound->accession links + the accession union to embed."""
    import chemspace

    union: set[str] = set()
    for snap in snapshots:
        cache = Path("data/cache") / snap
        tg = pd.read_parquet(cache / "targets.parquet")[
            ["target_chembl_id", "accession"]
        ].dropna().drop_duplicates()
        frames = []
        for name in ("mechanisms", "activities"):
            f = pd.read_parquet(cache / f"{name}.parquet")
            frames.append(f[["chembl_id", "target_chembl_id"]].dropna())
        link = pd.concat(frames).drop_duplicates()

        ds = chemspace.load(snapshot=str(cache))
        link = link[link["chembl_id"].isin(set(ds.compounds.index))]
        comp_acc = link.merge(tg, on="target_chembl_id", how="inner")[
            ["chembl_id", "accession"]
        ].drop_duplicates()
        comp_acc.to_parquet(work / f"{snap}_comp_acc.parquet", index=False)

        accs = set(comp_acc["accession"])
        union |= accs
        print(
            f"  {snap}: compounds {len(ds.compounds):,} · covered "
            f"{comp_acc['chembl_id'].nunique():,} · pairs {len(comp_acc):,} "
            f"· unique acc {len(accs):,}",
            flush=True,
        )

    (work / "acc_union.txt").write_text("\n".join(sorted(union)) + "\n")
    print(f"  union: {len(union):,} unique accessions -> {work/'acc_union.txt'}", flush=True)


# -------------------------------------------------------------------------- fetch
def _fetch_fasta(chunk: list[str], retries: int = 4) -> str:
    q = urllib.parse.urlencode({"accessions": ",".join(chunk), "format": "fasta"})
    req = urllib.request.Request(f"{UNIPROT_BASE}?{q}", headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"    retry {attempt+1}: {exc}", flush=True)
            time.sleep(2 * (attempt + 1))
    return ""


def _parse_fasta(text: str):
    acc: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if acc:
                yield acc, "".join(buf)
            parts = line[1:].split("|")  # >sp|ACC|NAME  or  >tr|ACC|NAME
            acc = parts[1] if len(parts) >= 2 else parts[0].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if acc:
        yield acc, "".join(buf)


def step_fetch(work: Path, batch: int = 100) -> None:
    """Download sequences for the union; batch, falling back one-by-one on failure.

    A whole batch can 400 when it contains a non-UniProt id (ChEMBL sometimes
    carries Ensembl gene ids in the accession field); the per-accession fallback
    isolates the offender instead of losing the batch. Ids that never resolve
    (obsolete / demerged / non-UniProt) are recorded as missing and their targets
    simply stay uncovered in the view.
    """
    accs = [a.strip() for a in (work / "acc_union.txt").read_text().splitlines() if a.strip()]
    print(f"  to fetch: {len(accs):,}", flush=True)

    seqs: dict[str, str] = {}
    bad: list[str] = []
    for i in range(0, len(accs), batch):
        chunk = accs[i : i + batch]
        try:
            seqs.update(dict(_parse_fasta(_fetch_fasta(chunk))))
        except Exception:  # noqa: BLE001 — isolate the offender
            for a in chunk:
                try:
                    seqs.update(dict(_parse_fasta(_fetch_fasta([a]))))
                except Exception:  # noqa: BLE001
                    bad.append(a)
                time.sleep(0.1)
        if (i // batch) % 10 == 0:
            print(f"    {min(i+batch, len(accs)):,}/{len(accs):,} · have {len(seqs):,}", flush=True)
        time.sleep(0.3)

    missing = [a for a in accs if a not in seqs]
    df = pd.DataFrame({"accession": list(seqs), "sequence": list(seqs.values())})
    df["length"] = df["sequence"].str.len()
    df = df[df["length"] > 0].reset_index(drop=True)
    df.to_parquet(work / "seqs.parquet", index=False)
    (work / "missing_acc.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    print(
        f"  fetched {len(df):,}/{len(accs):,} · missing {len(missing):,} "
        f"(unresolvable {len(bad):,}) -> {work/'seqs.parquet'}",
        flush=True,
    )
    if len(df):
        print(
            f"  length: min {int(df.length.min())} median {int(df.length.median())} "
            f"max {int(df.length.max())} · >{MAXLEN}aa truncated: {(df.length>MAXLEN).sum():,}",
            flush=True,
        )


# -------------------------------------------------------------------------- embed
def step_embed(work: Path, snapshots: tuple[str, ...], token_budget: int = 20000) -> None:
    """ESM-2 per-accession embeddings -> target_esm.npz in each snapshot cache."""
    import torch
    import esm

    df = pd.read_parquet(work / "seqs.parquet")
    df = df[df["sequence"].str.len() > 0].copy()
    df["trunc"] = df["sequence"].str.slice(0, MAXLEN)
    df["tlen"] = df["trunc"].str.len()
    df = df.sort_values("tlen").reset_index(drop=True)  # tight batches by length

    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    print(
        f"  sequences: {len(df):,} · device {device} "
        f"({torch.cuda.get_device_name(0) if cuda else 'cpu'})",
        flush=True,
    )

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device).eval()
    if cuda:
        model = model.half()

    accs = df["accession"].tolist()
    seqs = df["trunc"].tolist()
    tlens = df["tlen"].tolist()
    emb = np.zeros((len(df), EMB_DIM), dtype=np.float32)

    t0 = time.perf_counter()
    i = 0
    done = 0
    while i < len(df):
        j, budget = i, 0  # greedily pack a batch under the token budget
        while j < len(df):
            cost = tlens[j] + 2
            if j > i and (budget + cost > token_budget or j - i >= 64):
                break
            budget += cost
            j += 1
        _, _, toks = batch_converter([(accs[k], seqs[k]) for k in range(i, j)])
        with torch.no_grad():
            out = model(toks.to(device), repr_layers=[LAYER], return_contacts=False)
        reps = out["representations"][LAYER].float()  # (B, L, 1280)
        for b, k in enumerate(range(i, j)):
            emb[k] = reps[b, 1 : tlens[k] + 1].mean(0).cpu().numpy()  # drop BOS/EOS
        done += j - i
        if done % 512 < (j - i):
            print(f"    {done:,}/{len(df):,} ({time.perf_counter()-t0:4.0f}s)", flush=True)
        i = j

    accs_arr = np.array(accs, dtype=object)
    for snap in snapshots:
        out_path = Path("data/cache") / snap / "target_esm.npz"
        np.savez(out_path, accessions=accs_arr, emb=emb)
        print(f"  wrote {len(df):,}x{EMB_DIM} -> {out_path}", flush=True)
    print(f"  embed done in {time.perf_counter()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------- cli
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="data/cache/esm_build", help="scratch dir for intermediates")
    ap.add_argument("--snapshots", nargs="+", default=list(DEFAULT_SNAPSHOTS))
    ap.add_argument("--from", dest="start", choices=STEPS, default="maps", help="resume from this step")
    ap.add_argument("--only", nargs="+", choices=STEPS, default=None, help="run exactly these steps")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

    if args.only is not None:
        steps = [s for s in STEPS if s in args.only]
    else:
        steps = list(STEPS[STEPS.index(args.start):])

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    snaps = tuple(args.snapshots)
    print(f"snapshots: {snaps} · work: {work} · steps: {steps}", flush=True)

    for step in steps:
        print(f"\n[{step}]", flush=True)
        if step == "maps":
            step_maps(work, snaps)
        elif step == "fetch":
            step_fetch(work)
        elif step == "embed":
            step_embed(work, snaps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
