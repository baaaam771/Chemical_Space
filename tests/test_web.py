"""API-level checks against a tiny fake dataset.

Uses a stub store so the suite stays fast and does not need a ChEMBL snapshot:
the engine's own behaviour is covered in test_core, and what matters here is
the HTTP contract -- job lifecycle, the two moves, error codes, payload shape.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from chemspace.domains import DomainRegistry  # noqa: E402
from chemspace.placement import ClusterConfig, EmbedConfig  # noqa: E402
from chemspace.session import ExplorationSession  # noqa: E402
from chemspace.web import app as app_module  # noqa: E402

from test_core import ContinuousDomain, FakeDataset, SparseDomain  # noqa: E402


class FakeStore:
    """Same surface as SessionStore, over the in-memory FakeDataset."""

    def __init__(self):
        ds = FakeDataset(n=80)
        # The web layer reads these columns for hover text and tables.
        ds.compounds["pref_name"] = [f"CMPD {i}" for i in range(len(ds.compounds))]
        ds.compounds["modality"] = ["저분자" if i % 3 else "공유결합" for i in range(len(ds.compounds))]
        ds.compounds["mw"] = np.linspace(200, 600, len(ds.compounds))
        ds.compounds["clogp"] = np.linspace(-1, 6, len(ds.compounds))
        ds.compounds["qed"] = np.linspace(0.1, 0.9, len(ds.compounds))
        ds.compounds["murcko_smiles"] = "c1ccccc1"
        self.dataset = ds
        self.registry = DomainRegistry()
        self.registry.register(ContinuousDomain(ds))
        self.registry.register(SparseDomain(ds))
        self.embed_cfg = EmbedConfig(method="pca")
        self.cluster_cfg = ClusterConfig(method="kmeans", n_clusters=3)
        self._sessions = {}
        self._n = 0

    def create(self):
        self._n += 1
        sid = f"s{self._n}"
        session = ExplorationSession(
            self.dataset, self.registry, self.embed_cfg, self.cluster_cfg
        )

        class Stored:
            session_id = sid

        stored = Stored()
        stored.session = session
        self._sessions[sid] = stored
        return stored

    def get(self, sid):
        return self._sessions.get(sid)

    @property
    def domains(self):
        return [{"name": d.name, "label": d.label, "description": ""} for d in self.registry]

    def stats(self):
        return {
            "chembl_version": self.dataset.version,
            "compounds": len(self.dataset.compounds),
            "sessions": len(self._sessions),
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_dir": "",
        }


@pytest.fixture
def client(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(app_module, "SessionStore", lambda **kw: store)
    app = app_module.create_app()
    with TestClient(app) as c:
        c.store = store  # for the few tests that inspect engine state directly
        yield c


def finish(client, job, timeout=30.0):
    """Jobs run on a real thread pool; wait for one to settle.

    Polls with a pause rather than spinning -- a tight loop both burns through
    its attempts in milliseconds and competes with the worker for the GIL.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = client.get(f"/api/jobs/{job['job_id']}").json()
        if res["state"] in ("done", "error"):
            return res
        time.sleep(0.02)
    raise AssertionError(f"job {job['job_id']} did not finish within {timeout}s")


def make_root(client, domain="cont"):
    sid = client.post("/api/sessions").json()["session_id"]
    job = client.post(f"/api/sessions/{sid}/root", json={"domain": domain}).json()
    res = finish(client, job)
    assert res["state"] == "done", res
    return sid, res["result"]


# ------------------------------------------------------------------- tests


def test_meta_lists_domains(client):
    body = client.get("/api/meta").json()
    assert {d["name"] for d in body["domains"]} == {"cont", "sparse"}
    assert body["compounds"] == 80


def test_root_runs_as_a_job_and_returns_a_node(client):
    sid, node_id = make_root(client)
    node = client.get(f"/api/sessions/{sid}/nodes/{node_id}").json()
    assert node["n"] == 80
    assert node["step"] == "root"
    assert "전체 80개" in node["breadcrumb"]


def test_points_payload_is_row_aligned(client):
    sid, node_id = make_root(client)
    pts = client.get(f"/api/sessions/{sid}/nodes/{node_id}/points").json()
    n = 80
    for key in ("x", "y", "cluster", "chembl_id", "name", "modality", "mw", "clogp"):
        assert len(pts[key]) == n, f"{key} has {len(pts[key])} entries, expected {n}"


def test_select_narrows_and_switch_preserves(client):
    sid, root = make_root(client)

    job = client.post(f"/api/sessions/{sid}/select", json={
        "node_id": root, "point_indices": list(range(20)),
        "polygon": [[0, 0], [1, 0], [1, 1]],
    }).json()
    narrowed = finish(client, job)["result"]
    assert client.get(f"/api/sessions/{sid}/nodes/{narrowed}").json()["n"] == 20

    job = client.post(f"/api/sessions/{sid}/switch", json={
        "node_id": narrowed, "domain": "sparse",
    }).json()
    switched = finish(client, job)["result"]
    body = client.get(f"/api/sessions/{sid}/nodes/{switched}").json()
    assert body["n"] == 20, "switching a domain must not change the compound set"
    assert body["domain"] == "sparse"


def test_select_accepts_a_mixed_gesture_and_records_it_as_partial(client):
    """What the browser sends after a lasso drag plus a few shift-clicks: one
    index list, the lasso trace, and a flag saying the trace alone no longer
    reproduces the selection."""
    sid, root = make_root(client)
    job = client.post(f"/api/sessions/{sid}/select", json={
        "node_id": root,
        "point_indices": [0, 1, 2, 3, 61, 77],
        "polygon": [[0, 0], [1, 0], [1, 1]],
        "polygon_partial": True,
        "how": "드래그 + 개별 선택",
    }).json()
    nid = finish(client, job)["result"]
    assert client.get(f"/api/sessions/{sid}/nodes/{nid}").json()["n"] == 6

    detail = client.store.get(sid).session.node(nid).origin.detail
    assert detail["polygon_partial"] is True
    assert detail["how"] == "드래그 + 개별 선택"


def test_polygon_without_the_flag_is_left_unmarked(client):
    sid, root = make_root(client)
    job = client.post(f"/api/sessions/{sid}/select", json={
        "node_id": root, "point_indices": list(range(5)),
        "polygon": [[0, 0], [1, 0], [1, 1]],
    }).json()
    nid = finish(client, job)["result"]
    assert "polygon_partial" not in client.store.get(sid).session.node(nid).origin.detail


def test_going_back_and_selecting_again_branches(client):
    sid, root = make_root(client)
    a = finish(client, client.post(f"/api/sessions/{sid}/select", json={
        "node_id": root, "point_indices": list(range(10))}).json())["result"]
    b = finish(client, client.post(f"/api/sessions/{sid}/select", json={
        "node_id": root, "point_indices": list(range(10, 30))}).json())["result"]
    tree = client.get(f"/api/sessions/{sid}").json()
    children = next(n for n in tree["nodes"] if n["node_id"] == root)["children"]
    assert set(children) == {a, b}
    assert len(tree["nodes"]) == 3


def test_tables_render(client):
    sid, root = make_root(client)
    for path in ("clusters", "describe", "compounds"):
        rows = client.get(f"/api/sessions/{sid}/nodes/{root}/{path}").json()
        assert isinstance(rows, list) and rows, path
    describe = client.get(f"/api/sessions/{sid}/nodes/{root}/describe").json()
    assert {r["도메인"] for r in describe} == {"물성", "질환"}


def test_unknown_session_and_node_are_404(client):
    assert client.get("/api/sessions/nope").status_code == 404
    sid, root = make_root(client)
    assert client.get(f"/api/sessions/{sid}/nodes/n999").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_bad_requests_are_400(client):
    sid, root = make_root(client)
    both = client.post(f"/api/sessions/{sid}/select", json={
        "node_id": root, "point_indices": [1], "clusters": [0]})
    assert both.status_code == 400
    neither = client.post(f"/api/sessions/{sid}/select", json={"node_id": root})
    assert neither.status_code == 400
    same = client.post(f"/api/sessions/{sid}/switch", json={"node_id": root, "domain": "cont"})
    assert same.status_code == 400
    unknown = client.post(f"/api/sessions/{sid}/switch", json={"node_id": root, "domain": "zzz"})
    assert unknown.status_code == 400


def test_engine_errors_surface_as_a_failed_job(client):
    """A selection that cannot be placed fails the job, not the request."""
    sid, root = make_root(client)
    job = client.post(f"/api/sessions/{sid}/select", json={
        "node_id": root, "clusters": [999]}).json()
    res = finish(client, job)
    assert res["state"] == "error"
    assert "empty" in res["error"]


def test_save_writes_a_session(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "resolve_data_path", lambda p: tmp_path / p)
    sid, root = make_root(client)
    res = client.post(f"/api/sessions/{sid}/save", json={"name": "run1"}).json()
    assert (tmp_path / "data/sessions/run1/session.json").exists()
    assert res["nodes"] == 1


def test_save_rejects_a_path_traversing_name(client):
    sid, _ = make_root(client)
    bad = client.post(f"/api/sessions/{sid}/save", json={"name": "../../etc/passwd"})
    assert bad.status_code == 422


# -------------------------------------------------------- filters over http


def test_meta_advertises_the_filters(client):
    body = client.get("/api/meta").json()
    names = {f["name"] for f in body["filters"]}
    assert names == {"drug", "annotated"}
    annotated = next(f for f in body["filters"] if f["name"] == "annotated")
    assert annotated["needs_domain"] is True


def test_restrict_runs_as_a_job_and_lands_as_its_own_node(client):
    sid, node_id = make_root(client)
    job = client.post(
        f"/api/sessions/{sid}/restrict", json={"node_id": node_id, "kind": "drug"}
    ).json()
    res = finish(client, job)
    assert res["state"] == "done", res
    node = client.get(f"/api/sessions/{sid}/nodes/{res['result']}").json()
    assert node["n"] == 40
    assert node["step"] == "filter"
    assert node["parent_id"] == node_id
    assert "약물층만 남긴 40개" in node["breadcrumb"]


def test_a_filter_that_would_change_nothing_is_refused_before_the_job(client):
    """Refused at the request, not inside a job.

    The point of a 400 here is that the user finds out now instead of watching
    a progress overlay for a placement identical to the one on screen.
    """
    sid, node_id = make_root(client)
    r = client.post(
        f"/api/sessions/{sid}/restrict",
        json={"node_id": node_id, "kind": "annotated", "domain": "cont"},
    )
    assert r.status_code == 400
    assert "좁혀지지 않습니다" in r.json()["detail"]


def test_unknown_filter_and_domain_are_400(client):
    sid, node_id = make_root(client)
    assert client.post(
        f"/api/sessions/{sid}/restrict", json={"node_id": node_id, "kind": "nope"}
    ).status_code == 400
    assert client.post(
        f"/api/sessions/{sid}/restrict",
        json={"node_id": node_id, "kind": "annotated", "domain": "nope"},
    ).status_code == 400


def test_domain_coverage_reports_every_domain_and_the_drug_layer(client):
    sid, node_id = make_root(client)
    body = client.get(
        f"/api/sessions/{sid}/nodes/{node_id}/domain-coverage"
    ).json()
    assert body["n"] == 80
    assert body["drug_layer"] == 40
    by_name = {d["name"]: d for d in body["domains"]}
    assert by_name["cont"]["covered"] == 80
    assert by_name["cont"]["fraction"] == 1.0
    assert by_name["sparse"]["covered"] == 40
    assert by_name["sparse"]["fraction"] == 0.5


def test_coverage_is_reported_against_the_node_not_the_snapshot(client):
    """What the UI labels the picker with has to follow the narrowing.

    A coverage figure for the whole snapshot would say 50% forever, which is
    exactly the number that misleads: after restricting to the drug layer the
    sparse domain covers everything standing there.
    """
    sid, node_id = make_root(client)
    job = client.post(
        f"/api/sessions/{sid}/restrict", json={"node_id": node_id, "kind": "drug"}
    ).json()
    filtered = finish(client, job)["result"]
    body = client.get(
        f"/api/sessions/{sid}/nodes/{filtered}/domain-coverage"
    ).json()
    by_name = {d["name"]: d for d in body["domains"]}
    assert body["n"] == 40
    assert by_name["sparse"]["covered"] == 40
    assert by_name["sparse"]["fraction"] == 1.0
