"""FastAPI app exposing the exploration engine.

Every endpoint is a thin wrapper over :class:`ExplorationSession`. The browser
holds no exploration state -- it knows a session id and a node id, and asks the
server for everything else. That is the same contract the Jupyter viewer works
to, which is why this layer is small.

Calls that place compounds (``root``, ``select``, ``switch``) return a job id
rather than a result; see :mod:`chemspace.web.jobs`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chemspace.data.chembl import resolve_data_path
from chemspace.placement import NOISE, UNCOVERED
from chemspace.session.filters import FILTERS, drug_layer_mask, resolve as resolve_filter
from chemspace.session.tree import Node
from chemspace.web.jobs import JobRunner
from chemspace.web.store import SessionStore

log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"

#: Coordinates are rounded before being sent. Three decimals is far finer than
#: a screen pixel on any sane zoom, and it cuts the payload for 12,804 points
#: roughly in half.
COORD_DECIMALS = 3


# ----------------------------------------------------------------- payloads


class RootRequest(BaseModel):
    domain: str = "physchem"
    note: str = ""


class SelectRequest(BaseModel):
    node_id: str
    point_indices: list[int] | None = None
    clusters: list[int] | None = None
    polygon: list[list[float]] | None = None
    polygon_partial: bool = False
    how: str = "드래그 선택"
    note: str = ""


class RestrictRequest(BaseModel):
    node_id: str
    kind: str = "drug"
    domain: str | None = None
    note: str = ""


class SwitchRequest(BaseModel):
    node_id: str
    domain: str
    note: str = ""


class SaveRequest(BaseModel):
    name: str = Field(default="web", pattern=r"^[A-Za-z0-9._-]{1,64}$")


# ------------------------------------------------------------- serialisation


def node_brief(session, node: Node) -> dict:
    p = node.placement
    return {
        "node_id": node.node_id,
        "parent_id": node.parent_id,
        "domain": node.domain,
        "domain_label": session.domains[node.domain].label,
        "step": node.origin.kind,
        "n": node.n,
        "n_covered": p.n_covered,
        "clusters": len(p.cluster_ids),
        "noise": p.sizes().get(NOISE, 0),
        "uncovered": p.sizes().get(UNCOVERED, 0),
        "embed_method": p.embed_method,
        "cluster_method": p.cluster_method,
        "note": node.note,
        "children": list(node.children),
        "description": node.origin.describe(
            None, session.domains[node.domain].label, node.n
        ),
        "from_cache": bool(p.notes.get("from_cache")),
        "degenerate": p.notes.get("degenerate", ""),
    }


def node_points(session, node: Node) -> dict:
    """Coordinates plus the per-point fields the UI colours and hovers with."""
    df = session.frame(node.node_id)
    p = node.placement
    coords = np.round(p.coords.astype(float), COORD_DECIMALS)
    names = df["pref_name"].astype("object").where(df["pref_name"].notna(), "")
    out: dict[str, Any] = {
        "node_id": node.node_id,
        "x": coords[:, 0].tolist(),
        "y": coords[:, 1].tolist(),
        "cluster": p.labels.astype(int).tolist(),
        "chembl_id": list(df.index.astype(str)),
        "name": [str(v) for v in names],
        "modality": [str(v) for v in df["modality"]],
        "mw": [round(float(v), 1) for v in df["mw"]],
        "clogp": [round(float(v), 2) for v in df["clogp"]],
    }
    if "domain_label" in df.columns:
        out["domain_label"] = [str(v) for v in df["domain_label"]]
    return out


def frame_records(session, node_id: str, limit: int) -> list[dict]:
    cols = ["pref_name", "modality", "mw", "clogp", "qed", "murcko_smiles", "cluster"]
    df = session.frame(node_id)
    df = df[[c for c in cols if c in df.columns]].head(limit).reset_index()
    return df.replace({np.nan: None}).to_dict(orient="records")


def table_records(df) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    return df.reset_index().replace({np.nan: ""}).astype(object).to_dict(orient="records")


# --------------------------------------------------------------------- app


def create_app(
    snapshot: str | Path | None = None,
    use_cache: bool = True,
    max_workers: int = 2,
) -> FastAPI:
    store = SessionStore(snapshot=snapshot, use_cache=use_cache)
    runner = JobRunner(max_workers=max_workers)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        runner.shutdown()

    app = FastAPI(title="chemspace", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.runner = runner

    def need(session_id: str):
        stored = store.get(session_id)
        if stored is None:
            raise HTTPException(404, f"no session {session_id} (it may have expired)")
        return stored.session

    def need_node(session, node_id: str) -> Node:
        try:
            return session.node(node_id)
        except KeyError:
            raise HTTPException(404, f"no node {node_id}") from None

    # ------------------------------------------------------------- metadata

    @app.get("/api/meta")
    def meta():
        return {
            "domains": store.domains,
            "filters": [
                {
                    "name": f.name,
                    "label": f.label,
                    "description": f.description,
                    "needs_domain": f.needs_domain,
                }
                for f in FILTERS.values()
            ],
            **store.stats(),
        }

    @app.get("/api/coverage")
    def coverage():
        return table_records(store.dataset.coverage())

    # ------------------------------------------------------------- sessions

    @app.post("/api/sessions")
    def create_session():
        stored = store.create()
        return {"session_id": stored.session_id}

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        session = need(session_id)
        return {
            "session_id": session_id,
            "root_ids": session.root_ids,
            "nodes": [node_brief(session, n) for n in session.nodes.values()],
            "tree_text": session.tree(),
        }

    @app.post("/api/sessions/{session_id}/save")
    def save_session(session_id: str, req: SaveRequest):
        session = need(session_id)
        if not session.nodes:
            raise HTTPException(400, "nothing to save yet")
        out = session.save(resolve_data_path(f"data/sessions/{req.name}"))
        return {"path": str(out), "nodes": len(session.nodes)}

    # --------------------------------------------------------------- moves

    @app.post("/api/sessions/{session_id}/root")
    def make_root(session_id: str, req: RootRequest):
        session = need(session_id)
        if req.domain not in session.domains:
            raise HTTPException(400, f"unknown domain {req.domain}")
        n = len(store.dataset.compounds)
        job = runner.submit(
            "root",
            lambda: session.root(req.domain, note=req.note).node_id,
            detail=f"{n:,}개를 {session.domains[req.domain].label} 기준으로 배치",
        )
        return job.as_dict()

    @app.post("/api/sessions/{session_id}/select")
    def select(session_id: str, req: SelectRequest):
        session = need(session_id)
        parent = need_node(session, req.node_id)
        given = [x is not None for x in (req.point_indices, req.clusters)]
        if sum(given) != 1:
            raise HTTPException(400, "pass exactly one of point_indices or clusters")
        count = len(req.point_indices) if req.point_indices is not None else None

        def run() -> str:
            return session.select(
                req.node_id,
                point_indices=req.point_indices,
                clusters=req.clusters,
                polygon=req.polygon,
                polygon_partial=req.polygon_partial,
                how=req.how,
                note=req.note,
            ).node_id

        job = runner.submit(
            "select",
            run,
            detail=f"{count:,}개 선택 후 재배치" if count else "군집 선택 후 재배치",
        )
        return job.as_dict()

    @app.post("/api/sessions/{session_id}/restrict")
    def restrict(session_id: str, req: RestrictRequest):
        session = need(session_id)
        node = need_node(session, req.node_id)
        try:
            f = resolve_filter(req.kind)
        except KeyError as exc:
            raise HTTPException(400, str(exc)) from None
        if req.domain is not None and req.domain not in session.domains:
            raise HTTPException(400, f"unknown domain {req.domain}")
        # Count before submitting: a filter that keeps everything or nothing is
        # a 90-second placement the user did not want, and the mask is cheap.
        try:
            kept = int(session.filter_mask(req.node_id, req.kind, req.domain).sum())
        except KeyError as exc:
            raise HTTPException(400, str(exc)) from None
        if kept == 0 or kept == node.n:
            raise HTTPException(
                400,
                f"{f.label} 필터가 {node.n:,}개 중 {kept:,}개를 남깁니다 — 좁혀지지 않습니다",
            )

        def run() -> str:
            return session.restrict(
                req.node_id, req.kind, domain=req.domain, note=req.note
            ).node_id

        job = runner.submit(
            "restrict",
            run,
            detail=f"{f.label}만 남긴 {kept:,}개를 재배치",
        )
        return job.as_dict()

    @app.post("/api/sessions/{session_id}/switch")
    def switch(session_id: str, req: SwitchRequest):
        session = need(session_id)
        node = need_node(session, req.node_id)
        if req.domain not in session.domains:
            raise HTTPException(400, f"unknown domain {req.domain}")
        if req.domain == node.domain:
            raise HTTPException(400, "already in that domain")
        label = session.domains[req.domain].label
        job = runner.submit(
            "switch",
            lambda: session.switch_domain(req.node_id, req.domain, note=req.note).node_id,
            detail=f"{node.n:,}개를 {label} 기준으로 재배치",
        )
        return job.as_dict()

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(404, f"no job {job_id}")
        return job.as_dict()

    # --------------------------------------------------------------- nodes

    @app.get("/api/sessions/{session_id}/nodes/{node_id}")
    def get_node(session_id: str, node_id: str):
        session = need(session_id)
        node = need_node(session, node_id)
        return {
            **node_brief(session, node),
            "breadcrumb": session.breadcrumb(node_id),
            "path": [n.node_id for n in session.path(node_id)],
        }

    @app.get("/api/sessions/{session_id}/nodes/{node_id}/points")
    def get_points(session_id: str, node_id: str):
        session = need(session_id)
        return node_points(session, need_node(session, node_id))

    @app.get("/api/sessions/{session_id}/nodes/{node_id}/domain-coverage")
    def get_domain_coverage(session_id: str, node_id: str):
        """How many of this node's compounds each domain actually covers.

        The UI labels its domain picker with this, so a switch that would land
        on a near-empty map is visible before the click rather than after the
        wait. Featurising every domain is seconds even at 73,864 compounds --
        it is the *placement* that is slow, and this does not place.
        """
        session = need(session_id)
        node = need_node(session, node_id)
        cov = session.domain_coverage(node_id)
        return {
            "node_id": node_id,
            "n": node.n,
            "domains": [
                {
                    "name": name,
                    "label": session.domains[name].label,
                    "covered": covered,
                    "fraction": round(covered / node.n, 4) if node.n else 0.0,
                }
                for name, covered in cov.items()
            ],
            "drug_layer": int(drug_layer_mask(store.dataset, node.compound_ids).sum()),
        }

    @app.get("/api/sessions/{session_id}/nodes/{node_id}/clusters")
    def get_clusters(session_id: str, node_id: str):
        session = need(session_id)
        need_node(session, node_id)
        return table_records(session.cluster_table(node_id))

    @app.get("/api/sessions/{session_id}/nodes/{node_id}/describe")
    def get_describe(session_id: str, node_id: str):
        session = need(session_id)
        need_node(session, node_id)
        return table_records(session.describe(node_id))

    @app.get("/api/sessions/{session_id}/nodes/{node_id}/compounds")
    def get_compounds(session_id: str, node_id: str, limit: int = 200):
        session = need(session_id)
        need_node(session, node_id)
        return frame_records(session, node_id, min(max(limit, 1), 2000))

    # -------------------------------------------------------------- static

    if STATIC.exists():

        class NoCacheStatic(StaticFiles):
            """Serve the UI without letting the browser hold on to it.

            StaticFiles answers a conditional request with 304 based on mtime
            and ETag, which is right for fingerprinted assets but wrong here:
            app.js and style.css are served under fixed names, so an edited
            file reaches a browser that already has one only after a manual
            cache-bypassing reload. That failure is silent -- the page loads,
            it just runs the old code -- so pay the re-download instead.
            """

            def is_not_modified(self, response_headers, request_headers) -> bool:
                return False

            async def get_response(self, path: str, scope):
                resp = await super().get_response(path, scope)
                resp.headers["Cache-Control"] = "no-store, must-revalidate"
                return resp

        app.mount("/static", NoCacheStatic(directory=STATIC), name="static")

        @app.get("/")
        def index():
            """Serve the page with the asset URLs stamped by mtime.

            Cache-Control alone does not rescue a browser that cached app.js
            *before* the header existed -- with no caching directive at all it
            is free to reuse the file without ever revalidating, which is
            exactly how an edited UI silently fails to load. Changing the URL
            makes an edited asset a different resource, so it is always
            fetched, poisoned cache entry or not.
            """
            html = (STATIC / "index.html").read_text(encoding="utf-8")
            for asset in ("app.js", "style.css"):
                f = STATIC / asset
                if f.exists():
                    html = html.replace(
                        f"/static/{asset}", f"/static/{asset}?v={int(f.stat().st_mtime)}"
                    )
            return HTMLResponse(
                html, headers={"Cache-Control": "no-store, must-revalidate"}
            )

    return app
