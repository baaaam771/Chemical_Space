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
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chemspace.data.chembl import resolve_data_path
from chemspace.placement import NOISE, UNCOVERED
from chemspace.web.auth import (
    COOKIE_NAME,
    SESSION_TTL,
    AuthError,
    UserStore,
    safe_subdir,
    valid_name,
)
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
    # No dots, so ".." is not a legal name -- the old pattern allowed it and a
    # save could be aimed outside the sessions directory. The path is also
    # checked for containment when it is built; see auth.safe_subdir.
    name: str = Field(default="session", pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")


class LoginRequest(BaseModel):
    name: str
    password: str


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
    # Label overlays the UI can colour/filter any map by (MoA, stage, safety).
    for col in ("moa", "moa_l2", "dev_stage", "dev_stage_l2", "modality_l2", "physchem_l2", "safety", "struct_alert"):
        if col in df.columns:
            out[col] = [str(v) for v in df[col]]
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


#: (label, column) pairs shown in the compound detail modal's value grid.
_DETAIL_PROPS = (
    ("MW", "mw"),
    ("cLogP", "clogp"),
    ("TPSA", "tpsa"),
    ("QED", "qed"),
    ("HBD", "hbd"),
    ("HBA", "hba"),
    ("RotB", "rotb"),
    ("Fsp3", "fsp3"),
    ("고리수", "ring_count"),
    ("방향족고리", "aromatic_rings"),
    ("중원자수", "heavy_atoms"),
)


def _clean(v) -> Any:
    """A JSON-safe builtin number: NaN/None -> null, else a rounded float.

    numpy scalars go through ``float()`` so the response never carries an
    ``np.float64`` that the JSON encoder would choke on."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return None if np.isnan(f) else round(f, 2)


def compound_record(dataset, chembl_id: str) -> dict | None:
    """Per-compound reference fields for the hover card. Global, not node-scoped
    -- a compound's structure is the same wherever it is placed."""
    df = dataset.compounds
    if chembl_id not in df.index:
        return None
    r = df.loc[chembl_id]
    name = r.get("pref_name")
    name = "" if name is None or (isinstance(name, float) and np.isnan(name)) else str(name)

    props = {label: _clean(r.get(col)) for label, col in _DETAIL_PROPS if col in df.columns}

    def _strip(t: str) -> str:
        return t.split(":", 1)[1] if ":" in t else t

    dterms = getattr(dataset, "disease_terms", {}).get(chembl_id, ())
    tterms = getattr(dataset, "target_terms", {}).get(chembl_id, ())
    return {
        "chembl_id": chembl_id,
        "name": name,
        "smiles": str(r.get("canonical_smiles") or ""),
        "murcko_smiles": str(r.get("murcko_smiles") or ""),
        "modality": str(r.get("modality") or ""),
        # Back-compat flat keys (older callers); `props` is the full set.
        "mw": _clean(r.get("mw")),
        "clogp": _clean(r.get("clogp")),
        "qed": _clean(r.get("qed")),
        "props": props,
        "diseases": [_strip(t) for t in dterms if t.startswith(("MeSH:", "EFO:"))],
        "areas": [_strip(t) for t in dterms if t.startswith("ATC:")],
        "targets": [_strip(t) for t in tterms if t.startswith("TGT:")],
        "families": sorted({_strip(t) for t in tterms if t.startswith("L1:")}),
    }


@lru_cache(maxsize=8192)
def _svg_for_smiles(smiles: str) -> str | None:
    """Render a SMILES to a compact, transparent-background 2-D SVG.

    Cached by SMILES: the same structure is drawn once however many points or
    sessions hover it. RDKit is imported lazily so process start stays light.
    """
    if not smiles:
        return None
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    d = rdMolDraw2D.MolDraw2DSVG(260, 200)
    opts = d.drawOptions()
    opts.clearBackground = False  # let the hover card's own background show
    opts.bondLineWidth = 1
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return d.GetDrawingText()


# --------------------------------------------------------------------- app


def create_app(
    snapshot: str | Path | None = None,
    use_cache: bool = True,
    max_workers: int = 2,
    users_path: str | Path = "data/users.json",
    secret_path: str | Path = "data/cookie_secret",
    secure_cookie: bool = False,
    require_users: bool = True,
) -> FastAPI:
    users = UserStore(
        resolve_data_path(users_path), resolve_data_path(secret_path)
    )
    if require_users and not len(users):
        # Refusing to start is the safe default. The alternative -- running
        # open until somebody remembers to add a user -- is the state that
        # ships to a server and stays there.
        raise SystemExit(
            chr(10).join((
                "",
                f"  등록된 사용자가 없습니다 ({users.users_path}).",
                "  먼저 만드세요: python scripts/manage_users.py add <이름>",
                "",
            ))
        )
    store = SessionStore(snapshot=snapshot, use_cache=use_cache)
    runner = JobRunner(max_workers=max_workers)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        runner.shutdown()

    app = FastAPI(title="chemspace", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.runner = runner
    app.state.users = users

    #: Reachable without a cookie. Everything else is closed.
    #:
    #: An allow-list in one middleware rather than a dependency on each route:
    #: a route added later is protected by default, and the way to open one is
    #: to name it here, deliberately, in the file where the rule lives. The
    #: opposite arrangement -- guard each endpoint -- fails the first time
    #: somebody adds a route and forgets.
    OPEN_PATHS = {"/api/login", "/api/health", "/login", "/"}

    @app.middleware("http")
    async def require_login(request: Request, call_next):
        path = request.url.path
        if path in OPEN_PATHS or path.startswith("/static/") or path.startswith("/docs"):
            return await call_next(request)
        user = users.identify(request.cookies.get(COOKIE_NAME))
        if user is None:
            return JSONResponse({"detail": "로그인이 필요합니다"}, status_code=401)
        request.state.user = user
        return await call_next(request)

    def current_user(request: Request) -> str:
        """The logged-in user, trusting only what the middleware established."""
        user = getattr(request.state, "user", None)
        if user is None:  # pragma: no cover - middleware guarantees this
            raise HTTPException(401, "로그인이 필요합니다")
        return user

    def need(session_id: str, request: Request):
        stored = store.get(session_id, owner=current_user(request))
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

    # ------------------------------------------------------------- 로그인

    @app.get("/api/health")
    def health():
        """Open on purpose: a load balancer or a colleague checking the host
        should not need an account to learn that the process is up. It says
        nothing about the data."""
        return {"ok": True, "users": len(users)}

    @app.post("/api/login")
    def login(req: LoginRequest, response: Response):
        # One message for every failure. "No such user" and "wrong password"
        # told apart would turn the login form into a way to find out who has
        # an account here.
        if not valid_name(req.name) or not users.check(req.name, req.password):
            log.warning("failed login for %r", req.name[:32])
            raise HTTPException(401, "이름 또는 비밀번호가 맞지 않습니다")
        response.set_cookie(
            COOKIE_NAME,
            users.issue(req.name),
            max_age=SESSION_TTL,
            httponly=True,      # page scripts cannot read it
            samesite="lax",     # another site cannot ride it
            secure=secure_cookie,
        )
        log.info("login: %s", req.name)
        return {"user": req.name}

    @app.post("/api/logout")
    def logout(response: Response):
        response.delete_cookie(COOKIE_NAME, httponly=True, samesite="lax")
        return {"ok": True}

    @app.get("/api/me")
    def me(request: Request):
        return {"user": current_user(request)}

    # ------------------------------------------------------------- sessions

    @app.post("/api/sessions")
    def create_session(request: Request):
        stored = store.create(owner=current_user(request))
        return {"session_id": stored.session_id}

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str, request: Request):
        session = need(session_id, request)
        return {
            "session_id": session_id,
            "root_ids": session.root_ids,
            "nodes": [node_brief(session, n) for n in session.nodes.values()],
            "tree_text": session.tree(),
        }

    @app.post("/api/sessions/{session_id}/save")
    def save_session(session_id: str, req: SaveRequest, request: Request):
        session = need(session_id, request)
        if not session.nodes:
            raise HTTPException(400, "nothing to save yet")
        # The user name is part of the path, so two people pressing 저장 with
        # the same name write to different places. Before this every browser
        # sent "web" and the second save overwrote the first one's tree.
        user = current_user(request)
        root = resolve_data_path("data/sessions")
        root.mkdir(parents=True, exist_ok=True)
        try:
            target = safe_subdir(root, user, req.name)
        except AuthError as exc:
            raise HTTPException(400, str(exc)) from None
        out = session.save(target)
        return {"path": str(out), "name": req.name, "user": user,
                "nodes": len(session.nodes)}

    @app.get("/api/saves")
    def list_saves(request: Request):
        """This user's saved trees. Only this user's -- the directory listing
        is scoped to their own folder, so the endpoint cannot be used to see
        who else works here."""
        user = current_user(request)
        root = resolve_data_path("data/sessions") / user
        if not root.exists():
            return []
        saves = [
            {"name": d.name,
             "nodes": len(list((d / "placements").glob("*.npz")))}
            for d in root.iterdir()
            if (d / "session.json").exists()
        ]
        return sorted(saves, key=lambda r: r["name"])

    # --------------------------------------------------------------- moves

    @app.post("/api/sessions/{session_id}/root")
    def make_root(session_id: str, req: RootRequest, request: Request):
        session = need(session_id, request)
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
    def select(session_id: str, req: SelectRequest, request: Request):
        session = need(session_id, request)
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
    def restrict(session_id: str, req: RestrictRequest, request: Request):
        session = need(session_id, request)
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
    def switch(session_id: str, req: SwitchRequest, request: Request):
        session = need(session_id, request)
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
    def get_node(session_id: str, node_id: str, request: Request):
        session = need(session_id, request)
        node = need_node(session, node_id)
        return {
            **node_brief(session, node),
            "breadcrumb": session.breadcrumb(node_id),
            "path": [n.node_id for n in session.path(node_id)],
        }

    @app.get("/api/sessions/{session_id}/nodes/{node_id}/points")
    def get_points(session_id: str, node_id: str, request: Request):
        session = need(session_id, request)
        return node_points(session, need_node(session, node_id))

    @app.get("/api/sessions/{session_id}/nodes/{node_id}/domain-coverage")
    def get_domain_coverage(session_id: str, node_id: str, request: Request):
        """How many of this node's compounds each domain actually covers.

        The UI labels its domain picker with this, so a switch that would land
        on a near-empty map is visible before the click rather than after the
        wait. Featurising every domain is seconds even at 73,864 compounds --
        it is the *placement* that is slow, and this does not place.
        """
        session = need(session_id, request)
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
    def get_clusters(session_id: str, node_id: str, request: Request):
        session = need(session_id, request)
        need_node(session, node_id)
        return table_records(session.cluster_table(node_id))

    @app.get("/api/sessions/{session_id}/nodes/{node_id}/describe")
    def get_describe(session_id: str, node_id: str, request: Request):
        session = need(session_id, request)
        need_node(session, node_id)
        return table_records(session.describe(node_id))

    @app.get("/api/sessions/{session_id}/nodes/{node_id}/clusters/{cluster_id}")
    def get_cluster_card(session_id: str, node_id: str, cluster_id: int, request: Request):
        session = need(session_id, request)
        need_node(session, node_id)
        try:
            return session.cluster_card(node_id, cluster_id)
        except KeyError:
            raise HTTPException(404, f"no cluster {cluster_id} in {node_id}") from None

    @app.get("/api/sessions/{session_id}/nodes/{node_id}/compounds")
    def get_compounds(session_id: str, node_id: str, request: Request, limit: int = 200):
        session = need(session_id, request)
        need_node(session, node_id)
        return frame_records(session, node_id, min(max(limit, 1), 2000))

    @app.get("/api/sessions/{session_id}/nodes/{node_id}/composition")
    def get_composition(session_id: str, node_id: str, request: Request, top: int = 25):
        """Every discrete kind present in this node, per domain -- the full list
        the top-3 summary only previews."""
        session = need(session_id, request)
        node = need_node(session, node_id)
        t = min(max(top, 1), 200)
        out = []
        for name in session.domains.names:
            dom = session.domains[name]
            sections = dom.composition(node.compound_ids, top=t)
            if sections:
                out.append(
                    {
                        "domain": dom.label,
                        "name": name,
                        "current": name == node.domain,
                        "sections": sections,
                    }
                )
        return out

    # ------------------------------------------------------------- compound

    # Not session-scoped: a compound's structure and properties are the same
    # wherever it sits. Hover fetches these lazily, so the big coordinate
    # payload never has to carry a SMILES string per point.

    @app.get("/api/compound/{chembl_id}")
    def compound_detail(chembl_id: str):
        rec = compound_record(store.dataset, chembl_id)
        if rec is None:
            raise HTTPException(404, f"no compound {chembl_id}")
        return rec

    @app.get("/api/compound/{chembl_id}/structure.svg")
    def compound_structure(chembl_id: str):
        rec = compound_record(store.dataset, chembl_id)
        if rec is None:
            raise HTTPException(404, f"no compound {chembl_id}")
        svg = _svg_for_smiles(rec["smiles"])
        if svg is None:
            raise HTTPException(404, f"no drawable structure for {chembl_id}")
        return Response(
            svg,
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=86400"},
        )

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
