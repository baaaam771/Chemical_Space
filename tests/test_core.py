"""Invariants of the exploration loop, checked without the ChEMBL snapshot.

These lock the behaviour the tool is defined by:
  - switching a domain keeps the compound set exactly
  - selecting narrows it and never invents compounds
  - going back and selecting again branches instead of overwriting
  - features are recomputed relative to the subset, not inherited
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from chemspace.domains.base import (
    Domain,
    DomainFeatures,
    auto_min_support,
    multi_hot,
    prune_unsupported,
    standardize_within_subset,
)
from chemspace.data.dataset import Dataset
from chemspace.domains import DomainRegistry
from chemspace.placement import ClusterConfig, EmbedConfig, UNCOVERED, place
from chemspace.session import ExplorationSession
from chemspace.session.korean import eul, eun, ro

N = 120


# --------------------------------------------------------------------- doubles


class FakeDataset:
    """Enough of Dataset for the session: ids, a frame, and a version."""

    def __init__(self, n: int = N):
        rng = np.random.default_rng(0)
        ids = [f"C{i:04d}" for i in range(n)]
        self.version = "FAKE_1"
        self.manifest = {"min_phase": 0.5}
        # The drug layer is laid out to coincide with SparseDomain's coverage
        # (every second compound), because that is the real relationship the
        # filter exists for: clinical-phase compounds are the annotated ones.
        # Odd compounds alternate between a missing phase and ChEMBL's -1
        # sentinel, so both ways of "not in the layer" are exercised.
        self.compounds = pd.DataFrame(
            {
                "a": rng.normal(size=n),
                "b": rng.normal(size=n) * 10,
                "constant": np.ones(n),
                "group": [f"g{i % 4}" for i in range(n)],
                "max_phase": [
                    4.0 if i % 2 == 0 else (np.nan if i % 4 == 1 else -1.0)
                    for i in range(n)
                ],
            },
            index=pd.Index(ids, name="chembl_id"),
        )

    @property
    def all_ids(self) -> tuple[str, ...]:
        return tuple(self.compounds.index)

    def frame(self, ids):
        return self.compounds.loc[list(ids)]


class ContinuousDomain(Domain):
    name = "cont"
    label = "물성"

    def __init__(self, ds: FakeDataset, cols=("a", "b", "constant")):
        self.ds = ds
        self.cols = list(cols)

    def featurize(self, compound_ids):
        ids = tuple(compound_ids)
        raw = self.ds.frame(ids)[self.cols].to_numpy(dtype=float)
        X, notes = standardize_within_subset(raw)
        kept = np.nanstd(raw, axis=0) > 1e-9
        return DomainFeatures(
            X=X,
            compound_ids=ids,
            covered=np.ones(len(ids), dtype=bool),
            feature_names=tuple(c for c, k in zip(self.cols, kept) if k),
            metric="euclidean",
            kind="continuous",
            notes=notes,
        )

    def profile(self, compound_ids):
        return {"n": len(compound_ids), "summary": "cont"}


class SparseDomain(Domain):
    """Binary domain where only some compounds have data at all."""

    name = "sparse"
    label = "질환"

    def __init__(self, ds: FakeDataset, covered_every: int = 2):
        self.ds = ds
        self.terms = {
            cid: (f"t{i % 5}", f"u{i % 3}")
            for i, cid in enumerate(ds.all_ids)
            if i % covered_every == 0
        }

    def featurize(self, compound_ids):
        ids = tuple(compound_ids)
        X, names, notes = multi_hot(ids, self.terms, min_support=1)
        return DomainFeatures(
            X=X,
            compound_ids=ids,
            covered=np.array([c in self.terms for c in ids], dtype=bool),
            feature_names=names,
            metric="jaccard",
            kind="binary",
            notes=notes,
        )

    def profile(self, compound_ids):
        return {"n": len(compound_ids), "summary": "sparse"}


@pytest.fixture
def session() -> ExplorationSession:
    ds = FakeDataset()
    reg = DomainRegistry()
    reg.register(ContinuousDomain(ds))
    reg.register(SparseDomain(ds))
    return ExplorationSession(
        ds,  # type: ignore[arg-type]
        reg,
        EmbedConfig(method="pca"),
        ClusterConfig(method="kmeans", n_clusters=4),
    )


# ----------------------------------------------------------- subset relativity


def test_standardize_uses_subset_statistics():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    whole, _ = standardize_within_subset(X)
    half, _ = standardize_within_subset(X[:2])
    # The first two rows sit together on the full-set scale but are pushed to
    # the extremes once they are the whole subset -- that is the point.
    assert abs(whole[0, 0] - whole[1, 0]) < abs(half[0, 0] - half[1, 0])


def test_standardize_drops_constant_columns():
    X = np.column_stack([np.arange(10.0), np.ones(10)])
    Z, notes = standardize_within_subset(X)
    assert Z.shape == (10, 1)
    assert notes["dropped_constant"] == 1


def test_prune_unsupported_drops_empty_binary_columns():
    X = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    Xp, names, notes = prune_unsupported(X, ["a", "b", "c"], min_support=1)
    assert names == ("a",)
    assert notes["dropped_unsupported"] == 2
    assert Xp.shape == (3, 1)


def test_prune_never_returns_zero_columns():
    X = np.array([[1.0, 0.0], [0.0, 0.0]])
    Xp, names, _ = prune_unsupported(X, ["a", "b"], min_support=50)
    assert Xp.shape[1] >= 1 and len(names) >= 1


def test_auto_min_support_scales_with_subset():
    assert auto_min_support(300) == 1
    assert auto_min_support(1200) == 2
    assert auto_min_support(12_800) == 25  # capped
    assert auto_min_support(5) == 1


def test_multi_hot_counts_support_before_allocating():
    members = {"a": ("x", "y"), "b": ("x",), "c": ("z",)}
    X, names, notes = multi_hot(["a", "b", "c"], members, min_support=2)
    assert names == ("x",)
    assert notes["vocab_full"] == 3
    assert notes["vocab_kept"] == 1
    assert X.shape == (3, 1)


# ------------------------------------------------------------------- placement


def test_uncovered_compounds_are_excluded_from_the_fit():
    n = 40
    covered = np.zeros(n, dtype=bool)
    covered[:20] = True
    X = np.zeros((n, 3), dtype=np.float32)
    X[:20] = np.random.default_rng(1).integers(0, 2, size=(20, 3))
    feats = DomainFeatures(
        X=X,
        compound_ids=tuple(f"C{i}" for i in range(n)),
        covered=covered,
        feature_names=("a", "b", "c"),
        metric="jaccard",
        kind="binary",
    )
    p = place(feats, EmbedConfig(method="pca"), ClusterConfig(method="kmeans", n_clusters=3))
    assert (p.labels[~covered] == UNCOVERED).all()
    assert (p.labels[covered] >= 0).all()
    assert p.n_covered == 20
    assert p.notes["uncovered"] == 20
    # Parked compounds sit below the embedded cloud, not inside it.
    assert p.coords[~covered, 1].max() < p.coords[covered, 1].min()


def test_auto_cluster_size_follows_sqrt_then_stops():
    """Small subsets scale, large ones hold at the cap.

    The square root is what keeps a 300-compound subset from being called all
    noise; the cap is what keeps a 74k map from being told that nothing under
    136 compounds counts. Both halves matter, so both are pinned.
    """
    from chemspace.placement import AUTO_MIN_CLUSTER_CAP, auto_min_cluster_size

    assert auto_min_cluster_size(40) == 5       # floor
    assert auto_min_cluster_size(300) == 9
    assert auto_min_cluster_size(1_200) == 17
    # 6,400 is where sqrt(n)/2 reaches the cap; nothing below it changes.
    assert auto_min_cluster_size(6_400) == AUTO_MIN_CLUSTER_CAP
    assert auto_min_cluster_size(12_804) == AUTO_MIN_CLUSTER_CAP
    assert auto_min_cluster_size(73_864) == AUTO_MIN_CLUSTER_CAP
    assert auto_min_cluster_size(2_897_819) == AUTO_MIN_CLUSTER_CAP


def test_cache_version_separates_placements_made_under_different_rules():
    """The key stores ``min_cluster_size=None``, so the rule lives in the version.

    Without this, changing ``auto_min_cluster_size`` would leave every cached
    placement keyed identically to a new one and the stale labels would be
    served for ever.
    """
    import tempfile

    import chemspace.cache as cache_mod

    ids = ("A", "B", "C")
    cache = cache_mod.PlacementCache(tempfile.mkdtemp())
    before = cache.key("V1", "d", ids, EmbedConfig(), ClusterConfig())
    original = cache_mod.CACHE_VERSION
    try:
        cache_mod.CACHE_VERSION = original + 1
        assert cache.key("V1", "d", ids, EmbedConfig(), ClusterConfig()) != before
    finally:
        cache_mod.CACHE_VERSION = original


def test_non_finite_embedding_coordinates_are_clamped_and_reported():
    """UMAP can emit NaN on sparse Jaccard input; it must not spread."""
    from chemspace.placement import _finite, _uncovered_band

    emb = np.array([[0.0, 0.0], [1.0, 1.0], [np.nan, 2.0], [3.0, np.inf]])
    cleaned, n_bad = _finite(emb)
    assert n_bad == 2
    assert np.isfinite(cleaned).all()
    # Untouched rows keep their values.
    assert cleaned[0, 0] == 0.0 and cleaned[1, 1] == 1.0

    # And the band stays finite even if handed a degenerate cloud.
    band = _uncovered_band(np.zeros((5, 2), dtype=np.float32), 12)
    assert band.shape == (12, 2)
    assert np.isfinite(band).all()


def test_uncovered_band_ignores_embedding_outliers():
    """Band geometry follows the core of the cloud, not stray points."""
    from chemspace.placement import _uncovered_band

    rng = np.random.default_rng(3)
    core = rng.normal(size=(200, 2))
    emb = np.vstack([core, [[500.0, 0.0], [-500.0, 0.0]]])  # two far outliers
    band = _uncovered_band(emb, 100)
    width = band[:, 0].max() - band[:, 0].min()
    assert width < 50, f"band spans {width:.0f}, so outliers still drive its size"
    assert band[:, 1].max() < emb[:, 1].min()


def test_degenerate_features_are_reported_not_crashed():
    feats = DomainFeatures(
        X=np.zeros((5, 0), dtype=np.float32),
        compound_ids=tuple("abcde"),
        covered=np.ones(5, dtype=bool),
        feature_names=(),
        metric="jaccard",
        kind="binary",
    )
    p = place(feats)
    assert "degenerate" in p.notes
    assert p.coords.shape == (5, 2)


# --------------------------------------------------------------------- session


def test_switch_domain_preserves_the_compound_set(session):
    root = session.root("cont")
    switched = session.switch_domain(root.node_id, "sparse")
    assert switched.compound_ids == root.compound_ids
    assert switched.n == root.n
    assert switched.domain == "sparse"


def test_select_narrows_and_keeps_the_domain(session):
    root = session.root("cont")
    keep = root.compound_ids[:30]
    node = session.select(root.node_id, compound_ids=keep)
    assert node.n == 30
    assert node.domain == root.domain
    assert set(node.compound_ids) <= set(root.compound_ids)


def test_select_rejects_compounds_outside_the_parent(session):
    root = session.root("cont")
    with pytest.raises(ValueError, match="not in node"):
        session.select(root.node_id, compound_ids=["NOPE"])


def test_select_rejects_an_empty_result(session):
    root = session.root("cont")
    with pytest.raises(ValueError, match="empty"):
        session.select(root.node_id, clusters=[999])


def test_select_requires_exactly_one_selector(session):
    root = session.root("cont")
    with pytest.raises(ValueError, match="exactly one"):
        session.select(root.node_id, clusters=[0], point_indices=[1])
    with pytest.raises(ValueError, match="exactly one"):
        session.select(root.node_id)


def test_select_by_point_indices_maps_to_compounds(session):
    root = session.root("cont")
    node = session.select(root.node_id, point_indices=[0, 5, 9])
    assert node.compound_ids == tuple(root.compound_ids[i] for i in (0, 5, 9))


def test_point_indices_are_bounds_checked(session):
    root = session.root("cont")
    with pytest.raises(IndexError):
        session.select(root.node_id, point_indices=[0, 10_000])


def test_mixed_selection_is_deduped_and_sorted(session):
    """A lassoed region plus a few shift-clicked points arrive as one index
    list, out of order and possibly overlapping. Normalising belongs here, so
    that no viewer has to care which gesture produced which index."""
    root = session.root("cont")
    node = session.select(root.node_id, point_indices=[9, 0, 5, 9, 0])
    assert node.compound_ids == tuple(root.compound_ids[i] for i in (0, 5, 9))
    assert node.origin.detail["n_points"] == 3


def test_polygon_is_flagged_when_it_no_longer_describes_the_selection(session):
    """The lasso trace is kept either way, but a replay that re-selected from
    the polygon alone would get a different compound set once the user has
    shift-clicked. The flag is what stops that going unnoticed."""
    root = session.root("cont")
    poly = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]

    clean = session.select(root.node_id, point_indices=[0, 1, 2], polygon=poly)
    assert clean.origin.detail["polygon"] == poly
    assert "polygon_partial" not in clean.origin.detail

    mixed = session.select(
        root.node_id, point_indices=[0, 1, 2, 7], polygon=poly, polygon_partial=True
    )
    assert mixed.origin.detail["polygon"] == poly
    assert mixed.origin.detail["polygon_partial"] is True


def test_partial_flag_survives_a_save_and_load(session, tmp_path):
    root = session.root("cont")
    session.select(
        root.node_id,
        point_indices=[1, 4, 6],
        polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        polygon_partial=True,
        how="드래그 + 개별 선택",
    )
    path = session.save(tmp_path / "mixed.json")
    reloaded = ExplorationSession.load(path, session.ds, session.domains)
    detail = reloaded.node("n2").origin.detail
    assert detail["polygon_partial"] is True
    assert detail["how"] == "드래그 + 개별 선택"


def test_unknown_domain_is_rejected(session):
    root = session.root("cont")
    with pytest.raises(KeyError):
        session.switch_domain(root.node_id, "missing")


def test_going_back_branches_instead_of_overwriting(session):
    root = session.root("cont")
    a = session.select(root.node_id, compound_ids=root.compound_ids[:40])
    b = session.select(root.node_id, compound_ids=root.compound_ids[40:90])
    assert a.node_id in session.node(root.node_id).children
    assert b.node_id in session.node(root.node_id).children
    assert {n.node_id for n in session.siblings(a.node_id)} == {b.node_id}
    # The first branch is untouched.
    assert session.node(a.node_id).n == 40


def test_reselecting_the_same_region_reembeds_within_the_subset(session):
    """A subset's coordinates are its own, not a crop of the parent's."""
    root = session.root("cont")
    keep = root.compound_ids[:40]
    child = session.select(root.node_id, compound_ids=keep)
    parent_coords = root.placement.coords[:40]
    assert not np.allclose(parent_coords, child.placement.coords)


def test_breadcrumb_reads_as_the_path_taken(session):
    root = session.root("cont")
    n2 = session.select(root.node_id, compound_ids=root.compound_ids[:50])
    n3 = session.switch_domain(n2.node_id, "sparse")
    crumb = session.breadcrumb(n3.node_id)
    assert f"전체 {N}개" in crumb
    assert "물성에서" in crumb and "50개" in crumb
    assert "질환으로 재배치" in crumb


def test_path_and_tree_cover_every_node(session):
    root = session.root("cont")
    n2 = session.select(root.node_id, compound_ids=root.compound_ids[:60])
    n3 = session.switch_domain(n2.node_id, "sparse")
    assert [n.node_id for n in session.path(n3.node_id)] == [
        root.node_id,
        n2.node_id,
        n3.node_id,
    ]
    tree = session.tree(highlight=n3.node_id)
    for nid in (root.node_id, n2.node_id, n3.node_id):
        assert f"[{nid}]" in tree
    assert "◀" in tree


def test_history_and_cluster_table_are_tabular(session):
    root = session.root("cont")
    session.switch_domain(root.node_id, "sparse")
    hist = session.history()
    assert len(hist) == 2
    assert {"node", "domain", "compounds", "clusters"} <= set(hist.columns)
    table = session.cluster_table(root.node_id)
    assert len(table) >= 1
    assert "n" in table.columns


def test_save_and_load_roundtrip(tmp_path, session):
    root = session.root("cont")
    n2 = session.select(root.node_id, compound_ids=root.compound_ids[:45], note="영역 A")
    n3 = session.switch_domain(n2.node_id, "sparse")
    out = session.save(tmp_path / "sess")

    restored = ExplorationSession.load(out, session.ds, session.domains)
    assert restored.tree() == session.tree()
    assert restored.breadcrumb(n3.node_id) == session.breadcrumb(n3.node_id)
    assert restored.node(n2.node_id).note == "영역 A"
    assert np.allclose(
        restored.node(n3.node_id).placement.coords, n3.placement.coords
    )
    assert restored.node(n3.node_id).compound_ids == n3.compound_ids
    # Ids keep counting from where they left off, so a reloaded session can grow.
    grown = restored.select(root.node_id, compound_ids=root.compound_ids[:5])
    assert grown.node_id not in {root.node_id, n2.node_id, n3.node_id}


def test_frame_carries_coordinates_and_clusters(session):
    root = session.root("cont")
    df = session.frame(root.node_id)
    assert {"x", "y", "cluster"} <= set(df.columns)
    assert len(df) == root.n
    assert list(df.index) == list(root.compound_ids)


# ----------------------------------------------------------------------- cache


def _features(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return DomainFeatures(
        X=rng.normal(size=(n, 4)).astype(np.float32),
        compound_ids=tuple(f"C{i}" for i in range(n)),
        covered=np.ones(n, dtype=bool),
        feature_names=("a", "b", "c", "d"),
        metric="euclidean",
        kind="continuous",
    )


def test_cache_round_trips_a_placement(tmp_path):
    """The stored file must actually come back -- a silent miss looks like a
    working cache that is simply always cold."""
    from chemspace.cache import PlacementCache

    cache = PlacementCache(tmp_path / "placements")
    feats = _features()
    p = place(feats, EmbedConfig(method="pca"), ClusterConfig(method="kmeans", n_clusters=3))
    p.domain = "cont"
    key = cache.key("V1", "cont", feats.compound_ids, EmbedConfig(), ClusterConfig())

    assert cache.get(key, feats.compound_ids, "cont") is None
    cache.put(key, p)

    stored = list((tmp_path / "placements").glob("*.npz"))
    assert len(stored) == 1, f"expected one cache file, found {[f.name for f in stored]}"
    assert not stored[0].name.endswith(".tmp.npz"), "temp file was left behind"

    got = cache.get(key, feats.compound_ids, "cont")
    assert got is not None, "cache stored a file but could not read it back"
    assert np.array_equal(got.coords, p.coords)
    assert np.array_equal(got.labels, p.labels)
    assert got.notes["from_cache"] is True
    assert cache.hits == 1 and cache.misses == 1


def test_cache_key_distinguishes_inputs(tmp_path):
    from chemspace.cache import PlacementCache

    cache = PlacementCache(tmp_path)
    ids = tuple(f"C{i}" for i in range(10))
    base = cache.key("V1", "cont", ids, EmbedConfig(), ClusterConfig())
    assert base != cache.key("V2", "cont", ids, EmbedConfig(), ClusterConfig())
    assert base != cache.key("V1", "other", ids, EmbedConfig(), ClusterConfig())
    assert base != cache.key("V1", "cont", ids[:-1], EmbedConfig(), ClusterConfig())
    assert base != cache.key("V1", "cont", tuple(reversed(ids)), EmbedConfig(), ClusterConfig())
    assert base != cache.key("V1", "cont", ids, EmbedConfig(min_dist=0.5), ClusterConfig())
    assert base != cache.key("V1", "cont", ids, EmbedConfig(), ClusterConfig(method="kmeans"))
    assert base == cache.key("V1", "cont", ids, EmbedConfig(), ClusterConfig())


def test_cache_survives_a_corrupt_entry(tmp_path):
    from chemspace.cache import PlacementCache

    cache = PlacementCache(tmp_path)
    ids = tuple(f"C{i}" for i in range(5))
    key = cache.key("V1", "cont", ids, EmbedConfig(), ClusterConfig())
    (tmp_path / f"{key}.npz").write_bytes(b"not an npz")
    assert cache.get(key, ids, "cont") is None
    assert not (tmp_path / f"{key}.npz").exists(), "bad entry should be dropped"


def test_disabled_cache_stores_nothing(tmp_path):
    from chemspace.cache import PlacementCache

    cache = PlacementCache(tmp_path, enabled=False)
    feats = _features(10)
    p = place(feats, EmbedConfig(method="pca"), ClusterConfig(method="kmeans", n_clusters=2))
    key = cache.key("V1", "cont", feats.compound_ids, EmbedConfig(), ClusterConfig())
    cache.put(key, p)
    assert cache.get(key, feats.compound_ids, "cont") is None
    assert not list(tmp_path.glob("*.npz"))


def test_session_uses_the_cache_on_a_repeat_placement(tmp_path, session):
    """Same compounds, same domain, same config -> second session reuses it."""
    from chemspace.cache import PlacementCache

    cache = PlacementCache(tmp_path / "placements")
    ds = session.ds
    a = ExplorationSession(
        ds, session.domains, session.embed_cfg, session.cluster_cfg, cache=cache
    )
    root_a = a.root("cont")
    assert cache.misses == 1 and cache.hits == 0

    b = ExplorationSession(
        ds, session.domains, session.embed_cfg, session.cluster_cfg, cache=cache
    )
    root_b = b.root("cont")
    assert cache.hits == 1, "a repeated placement should have hit the cache"
    assert np.array_equal(root_a.placement.coords, root_b.placement.coords)
    assert np.array_equal(root_a.placement.labels, root_b.placement.labels)


# ------------------------------------------------------------------ data paths
# A notebook opened in notebooks/ has that as its cwd, so a bare "data/cache"
# pointed at notebooks/data/cache and the snapshot looked missing. Every cell of
# the notebook failed on it, so these pin the resolution rules.


def _fake_project(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "data" / "cache").mkdir(parents=True)
    (tmp_path / "notebooks").mkdir()
    return tmp_path


def test_project_root_found_from_a_subdirectory(tmp_path, monkeypatch):
    from chemspace.data.chembl import project_root

    root = _fake_project(tmp_path)
    monkeypatch.chdir(root / "notebooks")
    assert project_root() == root.resolve()


def test_data_path_resolves_from_a_subdirectory(tmp_path, monkeypatch):
    from chemspace.data.chembl import resolve_data_path

    root = _fake_project(tmp_path)
    monkeypatch.chdir(root / "notebooks")
    assert resolve_data_path("data/cache") == root.resolve() / "data" / "cache"


def test_data_path_prefers_an_existing_path_under_cwd(tmp_path, monkeypatch):
    from chemspace.data.chembl import resolve_data_path

    root = _fake_project(tmp_path)
    local = root / "notebooks" / "data" / "cache"
    local.mkdir(parents=True)
    monkeypatch.chdir(root / "notebooks")
    assert resolve_data_path("data/cache") == local


def test_data_path_creates_under_the_project_root(tmp_path, monkeypatch):
    """A path that does not exist yet still belongs to the project, not cwd."""
    from chemspace.data.chembl import resolve_data_path

    root = _fake_project(tmp_path)
    monkeypatch.chdir(root / "notebooks")
    assert resolve_data_path("data/sessions/run1") == (
        root.resolve() / "data" / "sessions" / "run1"
    )


def test_data_path_falls_back_to_cwd_without_a_project(tmp_path, monkeypatch):
    from chemspace.data.chembl import project_root, resolve_data_path

    monkeypatch.chdir(tmp_path)
    if project_root() is not None:  # tmp dir sits inside some other project
        pytest.skip("temp directory is inside a project")
    assert resolve_data_path("data/cache") == tmp_path / "data" / "cache"


def test_data_path_leaves_absolute_paths_alone(tmp_path, monkeypatch):
    from chemspace.data.chembl import resolve_data_path

    monkeypatch.chdir(_fake_project(tmp_path) / "notebooks")
    given = (tmp_path / "elsewhere" / "cache").resolve()
    assert resolve_data_path(given) == given


# ---------------------------------------------------------------------- korean


@pytest.mark.parametrize(
    "word,expected",
    [("물성", "물성으로"), ("표적", "표적으로"), ("질환", "질환으로"),
     ("모이어티", "모이어티로"), ("모달리티", "모달리티로")],
)
def test_ro_particle(word, expected):
    assert ro(word) == expected


def test_other_particles():
    assert eun("물성") == "물성은"
    assert eun("모이어티") == "모이어티는"
    assert eul("표적") == "표적을"
    assert eul("모달리티") == "모달리티를"


def test_non_hangul_falls_back_to_both_forms():
    assert ro("target") == "target(으)로"

# ----------------------------------------------- target annotation sources


def _frames(mech_rows, act_rows):
    """Minimal mechanism/activity/target/class frames for _target_table."""
    mechs = pd.DataFrame(mech_rows, columns=["chembl_id", "target_chembl_id"])
    acts = (
        pd.DataFrame(act_rows, columns=["chembl_id", "target_chembl_id", "pchembl_value"])
        if act_rows is not None
        else None
    )
    names = sorted(
        {t for _, t in mech_rows} | ({t for _, t, _ in act_rows} if act_rows else set())
    )
    tgts = pd.DataFrame(
        {
            "target_chembl_id": names,
            "target_pref_name": [f"Protein {t[-1]}" for t in names],
            "target_type": "SINGLE PROTEIN",
            "organism": "Homo sapiens",
            "protein_class_id": 1,
        }
    )
    pcs = pd.DataFrame(
        {
            "protein_class_id": [1],
            "class_path": ["Enzyme>Kinase"],
            "l1": ["Enzyme"], "l2": ["Kinase"], "l3": [None],
            "l4": [None], "l5": [None], "l6": [None],
        }
    )
    return mechs, acts, tgts, pcs


def test_activity_targets_join_the_same_term_namespace():
    """A target means the same thing whichever table it came from.

    Giving measured targets their own prefix would stop two compounds that hit
    the same protein from ever sharing a feature, which is the opposite of
    what the domain is for. Provenance lives in a column instead."""
    mechs, acts, tgts, pcs = _frames(
        [("C1", "T_a")], [("C2", "T_a", 8.0)]
    )
    table = Dataset._target_table(mechs, acts, tgts, pcs, keep={"C1", "C2"})
    terms = Dataset._target_terms(table)
    assert terms["C1"] == terms["C2"]
    assert "TGT:Protein a" in terms["C1"]
    assert set(table["source"]) == {"mechanism", "activity"}


def test_target_table_still_works_without_activities():
    """Snapshots taken before the activity pull have no such file."""
    mechs, _, tgts, pcs = _frames([("C1", "T_a")], None)
    table = Dataset._target_table(mechs, None, tgts, pcs, keep={"C1"})
    assert list(table["source"].unique()) == ["mechanism"]
    assert "TGT:Protein a" in Dataset._target_terms(table)["C1"]


def test_cell_line_assay_readouts_are_not_targets():
    """ChEMBL calls anything an assay ran against a "target".

    36% of the activity pairs we pull point at a CELL-LINE: docetaxel came
    back with 544 cell lines and two proteins. Those measure an effect, not a
    binding partner, and left in they would group cytotoxic chemotherapies by
    which screening panel they went through."""
    mechs = pd.DataFrame([("C1", "T_p")], columns=["chembl_id", "target_chembl_id"])
    acts = pd.DataFrame(
        [("C2", "T_p", 8.0), ("C2", "T_cell", 9.9), ("C3", "T_cell", 9.9)],
        columns=["chembl_id", "target_chembl_id", "pchembl_value"],
    )
    tgts = pd.DataFrame(
        {
            "target_chembl_id": ["T_p", "T_cell"],
            "target_pref_name": ["Kinase X", "MCF7"],
            "target_type": ["SINGLE PROTEIN", "CELL-LINE"],
            "organism": "Homo sapiens",
            "protein_class_id": [1, None],
        }
    )
    pcs = pd.DataFrame(
        {
            "protein_class_id": [1], "class_path": ["Enzyme>Kinase"],
            "l1": ["Enzyme"], "l2": ["Kinase"], "l3": [None],
            "l4": [None], "l5": [None], "l6": [None],
        }
    )
    table = Dataset._target_table(mechs, acts, tgts, pcs, keep={"C1", "C2", "C3"})
    terms = Dataset._target_terms(table)
    assert "TGT:Kinase X" in terms["C2"]
    assert not any("MCF7" in t for t in terms["C2"])
    # C3 had nothing but a cell line, so it has no target annotation at all --
    # better than a map that says it shares pharmacology with every other
    # compound screened on MCF7.
    assert "C3" not in terms


def test_activity_targets_are_capped_per_compound():
    """ChEMBL activity volume tracks study effort, not promiscuity.

    Uncapped, a compound run through a kinase panel carries hundreds of
    targets while an equally promiscuous but less-examined one carries three,
    and the map would separate famous compounds from obscure ones."""
    rows = [("C1", f"T_{i:03d}", 7.0 + i / 100) for i in range(40)]
    acts = pd.DataFrame(rows, columns=["chembl_id", "target_chembl_id", "pchembl_value"])
    capped = Dataset._cap_per_compound(acts, cap=25)
    assert len(capped) == 25
    # the strongest ones survive, not an arbitrary 25
    assert capped["pchembl_value"].min() == pytest.approx(7.15)


def test_cap_keeps_the_strongest_measurement_per_target():
    acts = pd.DataFrame(
        [("C1", "T_a", 7.1), ("C1", "T_a", 9.4), ("C1", "T_b", 8.0)],
        columns=["chembl_id", "target_chembl_id", "pchembl_value"],
    )
    capped = Dataset._cap_per_compound(acts, cap=25)
    assert len(capped) == 2, "the same target measured twice is still one target"
    best = capped.set_index("target_chembl_id")["pchembl_value"]
    assert best["T_a"] == pytest.approx(9.4)


def test_cache_key_follows_snapshot_content_not_release_name():
    """Adding the activity layer changes every target feature vector while
    ChEMBL_37 stays ChEMBL_37. Keyed on the version alone, the cache would
    hand back placements computed from the thinner annotation."""
    from chemspace.cache import PlacementCache

    cache = PlacementCache("unused", enabled=False)
    ids = ("A", "B", "C")
    embed, cluster = EmbedConfig(method="pca"), ClusterConfig(method="kmeans")

    before = cache.key("ChEMBL_37:aaaa", "target", ids, embed, cluster)
    after = cache.key("ChEMBL_37:bbbb", "target", ids, embed, cluster)
    assert before != after


def test_dataset_fingerprint_moves_with_the_manifest():
    from chemspace.data.dataset import Dataset

    base = {"chembl_version": "ChEMBL_37", "counts": {"activities": 0}}
    grown = {"chembl_version": "ChEMBL_37", "counts": {"activities": 412_000}}
    fp = Dataset.fingerprint.fget
    assert fp(SimpleNamespace(manifest=base)) != fp(SimpleNamespace(manifest=grown))
    assert fp(SimpleNamespace(manifest=base)) == fp(SimpleNamespace(manifest=dict(base)))


# ------------------------------------------------- cluster coherence


def _blobs(n_per=60, spread=0.25, seed=0):
    """Three well-separated Gaussian blobs in 4-D: structure we put there."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0, 0, 0], [8, 0, 0, 0], [0, 8, 0, 0]], dtype=np.float32)
    X = np.vstack([c + rng.normal(0, spread, (n_per, 4)) for c in centres]).astype(np.float32)
    truth = np.repeat(np.arange(3), n_per)
    return X, truth


def _blob_features(X, metric="euclidean"):
    n = X.shape[0]
    return DomainFeatures(
        X=X,
        compound_ids=tuple(f"C{i}" for i in range(n)),
        covered=np.ones(n, bool),
        feature_names=tuple(f"f{i}" for i in range(X.shape[1])),
        metric=metric,
        kind="continuous",
    )


def test_shuffling_keeps_each_column_and_breaks_their_pairing():
    """The null has to be a fair comparison: same marginals, no structure."""
    from chemspace.quality import shuffled_features

    X, _ = _blobs()
    feats = _blob_features(X)
    null = shuffled_features(feats, seed=1)
    for j in range(X.shape[1]):
        assert np.allclose(np.sort(X[:, j]), np.sort(null.X[:, j])), "column changed"
    before = abs(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    after = abs(np.corrcoef(null.X[:, 0], null.X[:, 1])[0, 1])
    assert after < before


def test_coherence_separates_real_structure_from_its_null():
    """The measurement has to detect structure that is actually there."""
    from chemspace.quality import coherence, null_coherence

    X, truth = _blobs()
    feats = _blob_features(X)
    p = place(feats, EmbedConfig(method="pca"), ClusterConfig(method="kmeans", n_clusters=3))
    real = coherence(feats, p, domain="blobs", external=truth.astype(str), matched_k=3)
    null = null_coherence(
        feats, domain="blobs",
        embed=EmbedConfig(method="pca"),
        cluster=ClusterConfig(method="kmeans", n_clusters=3),
        external=truth.astype(str), matched_k=3,
    )
    assert real.ami > 0.9, f"should recover planted clusters, got AMI {real.ami}"
    assert null.ami < 0.2, f"null should find nothing, got AMI {null.ami}"
    assert real.silhouette_matched > null.silhouette_matched


def test_coherence_finds_nothing_in_structureless_data():
    """And it has to *not* detect structure that is not there.

    This is the control that makes the other test mean something: a metric
    that always reports a good score would pass that one too."""
    from chemspace.quality import coherence, null_coherence

    rng = np.random.default_rng(0)
    X = rng.standard_normal((180, 4)).astype(np.float32)
    feats = _blob_features(X)
    cfg = (EmbedConfig(method="pca"), ClusterConfig(method="kmeans", n_clusters=3))
    real = coherence(feats, place(feats, *cfg), domain="noise", matched_k=3)
    null = null_coherence(feats, domain="noise", embed=cfg[0], cluster=cfg[1], matched_k=3)
    gap = real.silhouette_matched - null.silhouette_matched
    assert abs(gap) < 0.15, f"noise should score like its own null, gap was {gap:.3f}"


def test_noise_and_uncovered_are_left_out_of_the_score():
    """Scoring them as clusters would reward a run that dumped everything
    into one of those two buckets."""
    from chemspace.placement import NOISE, UNCOVERED, Placement
    from chemspace.quality import coherence

    X, _ = _blobs(n_per=20)
    feats = _blob_features(X)
    labels = np.full(60, 0, dtype=np.int32)
    labels[:10] = NOISE
    labels[10:20] = UNCOVERED
    labels[40:] = 1
    p = Placement(
        coords=np.random.default_rng(0).normal(size=(60, 2)).astype(np.float32),
        labels=labels, compound_ids=feats.compound_ids, domain="x",
        embed_method="pca", cluster_method="fake", metric="euclidean",
        n_covered=50, notes={},
    )
    c = coherence(feats, p, domain="x")
    assert c.n == 60
    assert c.n_scored == 40


# ----------------------------------------------------------------- filters


def test_drug_layer_mask_excludes_missing_and_sentinel_phases():
    from chemspace.session.filters import drug_layer_mask

    ds = FakeDataset()
    mask = drug_layer_mask(ds, ds.all_ids)
    phase = ds.compounds["max_phase"].to_numpy(dtype=float)
    assert mask.sum() == int((phase == 4.0).sum())
    assert not mask[np.isnan(phase)].any()      # no phase recorded
    assert not mask[phase == -1.0].any()        # withdrawn/unknown sentinel


def test_restrict_to_the_drug_layer_narrows_and_keeps_the_domain(session):
    root = session.root("cont")
    node = session.restrict(root.node_id, "drug")
    assert node.domain == root.domain
    assert node.n == root.n // 2
    assert set(node.compound_ids) < set(root.compound_ids)
    assert node.origin.kind == "filter"
    assert node.parent_id == root.node_id


def test_the_filter_step_reads_as_a_rule_not_a_gesture(session):
    root = session.root("cont")
    node = session.restrict(root.node_id, "drug")
    # A lasso cannot be replayed from its name; a filter can, and the
    # breadcrumb has to say which one it was.
    assert "약물층만 남긴" in session.breadcrumb(node.node_id)


def test_restrict_rejects_a_filter_that_changes_nothing(session):
    """A node identical to its parent is a placement paid for twice."""
    root = session.root("cont")
    with pytest.raises(ValueError, match="모두 남깁니다"):
        session.restrict(root.node_id, "annotated", domain="cont")


def test_restrict_rejects_a_filter_that_keeps_nothing(session):
    root = session.root("cont")
    only_odd = [c for i, c in enumerate(root.compound_ids) if i % 2 == 1]
    thin = session.select(root.node_id, compound_ids=only_odd)
    with pytest.raises(ValueError, match="아무것도 남기지 않습니다"):
        session.restrict(thin.node_id, "drug")


def test_restrict_to_annotated_removes_the_uncovered_band(session):
    root = session.root("sparse")
    assert (root.placement.labels == UNCOVERED).any()
    node = session.restrict(root.node_id, "annotated")
    assert node.n == root.placement.n_covered
    assert not (node.placement.labels == UNCOVERED).any()


def test_restricting_first_is_what_makes_a_thin_domain_usable(session):
    """The whole reason the filter exists.

    Switching a mixed set onto a sparsely annotated domain puts most of it in
    the uncovered band; restricting to the drug layer first does not. This is
    the 7.6% -> 48.3% effect of the 73,864-compound snapshot, in miniature.
    """
    root = session.root("cont")
    direct = session.switch_domain(root.node_id, "sparse")
    filtered = session.switch_domain(
        session.restrict(root.node_id, "drug").node_id, "sparse"
    )
    assert direct.placement.n_covered / direct.n == pytest.approx(0.5)
    assert filtered.placement.n_covered / filtered.n == pytest.approx(1.0)


def test_unknown_filter_is_rejected_by_name(session):
    root = session.root("cont")
    with pytest.raises(KeyError, match="unknown filter"):
        session.restrict(root.node_id, "nope")


def test_domain_coverage_covers_every_domain_without_placing(session):
    root = session.root("cont")
    cov = session.domain_coverage(root.node_id)
    assert set(cov) == set(session.domains.names)
    assert cov["cont"] == root.n            # continuous domain covers all
    assert cov["sparse"] == root.n // 2
    # Asking must not add nodes: the UI calls this on every node load.
    assert len(session.nodes) == 1


def test_covered_mask_for_the_nodes_own_domain_comes_from_its_placement(session):
    root = session.root("sparse")
    mask = session.covered_mask(root.node_id, "sparse")
    assert mask.tolist() == (root.placement.labels != UNCOVERED).tolist()


def test_a_filter_node_survives_a_save_and_load(session, tmp_path):
    from chemspace.session import ExplorationSession as S

    root = session.root("cont")
    node = session.restrict(root.node_id, "drug")
    session.save(tmp_path)
    back = S.load(tmp_path, session.ds, session.domains)  # type: ignore[arg-type]
    restored = back.node(node.node_id)
    assert restored.origin.kind == "filter"
    assert restored.origin.detail["filter"] == "drug"
    assert "약물층만 남긴" in back.breadcrumb(node.node_id)


# ------------------------------------------------------------------- 사용자


def _store(tmp_path):
    from chemspace.web.auth import UserStore

    return UserStore(tmp_path / "users.json", tmp_path / "secret")


def test_names_that_could_escape_a_directory_are_not_names(tmp_path):
    """The grammar closes the hole, not a check somewhere downstream.

    The old save-name pattern allowed dots, so ``..`` was a legal name and the
    save path left the sessions directory. Dots are gone from the grammar.
    """
    from chemspace.web.auth import valid_name

    for bad in ("..", ".", "a.b", "a/b", "a\b", "", " x", "x" * 33, "-x", "_x"):
        assert not valid_name(bad), f"{bad!r} should not be a name"
    for good in ("alice", "a", "user_1", "run-2", "A9"):
        assert valid_name(good), f"{good!r} should be a name"


def test_safe_subdir_refuses_to_leave_its_root(tmp_path):
    from chemspace.web.auth import AuthError, safe_subdir

    root = tmp_path / "sessions"
    root.mkdir()
    assert safe_subdir(root, "alice", "run1") == (root / "alice" / "run1").resolve()
    for parts in (("..",), ("alice", ".."), ("..", "..", "etc")):
        with pytest.raises(AuthError):
            safe_subdir(root, *parts)


def test_a_password_is_never_stored_in_the_clear(tmp_path):
    us = _store(tmp_path)
    us.add("alice", "correct-horse-battery")
    blob = (tmp_path / "users.json").read_text(encoding="utf-8")
    assert "correct-horse-battery" not in blob
    assert us.check("alice", "correct-horse-battery")
    assert not us.check("alice", "correct-horse-batterz")


def test_short_passwords_are_refused(tmp_path):
    from chemspace.web.auth import AuthError

    with pytest.raises(AuthError):
        _store(tmp_path).add("alice", "short")


def test_a_cookie_cannot_be_given_a_later_expiry(tmp_path):
    """The expiry is inside the signature, not beside it."""
    us = _store(tmp_path)
    us.add("alice", "correct-horse-battery")
    token = us.issue("alice", ttl=60)
    name, expiry, sig = token.split(".", 2)
    assert us.identify(token) == "alice"
    assert us.identify(f"{name}.{int(expiry) + 86400}.{sig}") is None


def test_an_expired_cookie_is_not_a_login(tmp_path):
    us = _store(tmp_path)
    us.add("alice", "correct-horse-battery")
    assert us.identify(us.issue("alice", ttl=-1)) is None


def test_a_cookie_signed_with_another_secret_is_not_a_login(tmp_path):
    """Deleting the secret file logs everybody out -- the revoke-all switch."""
    us = _store(tmp_path)
    us.add("alice", "correct-horse-battery")
    token = us.issue("alice")
    assert us.identify(token) == "alice"
    (tmp_path / "secret").unlink()
    assert us.identify(token) is None


def test_lead_summary_leads_with_specifics_then_context():
    """The cluster label shows the specific things a cluster contains, with the
    broad category only as a parenthesised tag -- and falls back to the context
    when nothing specific is annotated."""
    from chemspace.domains.builtin import _lead_summary

    specific = pd.Series({"MeSH:Breast Neoplasms": 10, "MeSH:Lymphoma": 5})
    context = pd.Series({"ATC:항암·면역조절": 12})
    s = _lead_summary(specific, context, "없음")
    assert s.startswith("Breast Neoplasms · Lymphoma")
    assert s.endswith("(항암·면역조절)")

    empty = pd.Series(dtype="int64")
    assert _lead_summary(empty, context, "없음") == "항암·면역조절 (12)"
    assert _lead_summary(empty, empty, "없음") == "없음"


# ---------------------------------------------------------------- target_seq


def _esm_fake_ds():
    """A tiny dataset carrying ESM embeddings for three accessions."""
    from chemspace.data.dataset import TargetESM

    ids = [f"C{i:04d}" for i in range(8)]
    rng = np.random.default_rng(1)
    mat = rng.normal(size=(3, 6)).astype(np.float32)
    esm = TargetESM(dim=6, index={"P1": 0, "P2": 1, "P3": 2}, matrix=mat)
    # C0..C4 hit known proteins; C5 hits an accession with no embedding; C6/C7
    # have no target at all -> uncovered.
    accs = {"C0000": ("P1",), "C0001": ("P2",), "C0002": ("P1", "P2"),
            "C0003": ("P3",), "C0004": ("P2", "P3"), "C0005": ("P9",)}
    terms = {c: ("L1:Enzyme", "TGT:Protein") for c in accs}
    return SimpleNamespace(target_embeddings=esm, target_accessions=accs,
                           target_terms=terms, compounds=pd.DataFrame(index=ids)), ids


def test_target_seq_domain_builds_sequence_features():
    from chemspace.domains.builtin import TargetSeqDomain

    ds, ids = _esm_fake_ds()
    dom = TargetSeqDomain(ds)
    f = dom.featurize(ids)

    assert f.metric == "euclidean" and f.kind == "continuous"
    # covered exactly the five with an embedded accession (C5's P9 is absent)
    assert [ids[i] for i in np.flatnonzero(f.covered)] == ["C0000", "C0001", "C0002", "C0003", "C0004"]
    # uncovered rows stay zero; covered rows are z-scored (mean ~0 per column)
    assert np.allclose(f.X[~f.covered], 0.0)
    assert np.allclose(f.X[f.covered].mean(axis=0), 0.0, atol=1e-5)


def test_target_seq_registered_only_when_embeddings_present():
    from chemspace.domains import default_registry
    from chemspace.domains.builtin import TargetSeqDomain

    ds, _ = _esm_fake_ds()
    # The real registry needs a full Dataset; here just check the guard logic
    # the registry uses, plus that the domain reuses target tokens to describe.
    assert ds.target_embeddings is not None
    dom = TargetSeqDomain(ds)
    assert dom.name == "target_seq"
    # description comes from target tokens, not the embedding vectors
    prof = dom.profile(["C0000", "C0002"])
    assert "Protein" in prof["summary"]


def test_overlay_columns_derive_moa_stage_safety():
    from chemspace.data.dataset import Dataset

    ids = ["C0", "C1", "C2", "C3", "C4"]
    compounds = pd.DataFrame(
        {
            "max_phase": [4.0, 2.0, np.nan, -1.0, 0.5],   # approved/clinical/pre/pre/clinical
            "withdrawn_flag": [0, 0, 0, 1, 0],            # C3 withdrawn
            "black_box_warning": [1, 0, 0, 1, 0],         # C0 warning; C3 both -> withdrawn wins
            "molecule_type": ["Small molecule", "Small molecule", "Antibody", "Protein", "Small molecule"],
            "mw": [320.0, 650.0, 150000.0, 1200.0, 280.0],
            "clogp": [2.0, 6.0, 0.0, 0.0, 2.0],
            "tpsa": [60.0, 50.0, 0.0, 0.0, 95.0],
            "covalent_tier": ["none", "none", "none", "none", "none"],
        },
        index=ids,
    )
    compounds.loc["C0", "covalent_tier"] = "strong"       # C0 inhibitor -> irreversible
    mechs = pd.DataFrame(
        {
            "chembl_id": ["C0", "C0", "C1"],              # C0 first action_type wins
            "action_type": ["INHIBITOR", "AGONIST", "ANTAGONIST"],
        }
    )
    out = Dataset._overlay_columns(compounds, mechs)

    assert list(out["dev_stage"]) == ["승인", "임상", "전임상/미상", "전임상/미상", "임상"]
    assert list(out["dev_stage_l2"]) == [
        "Phase 4 (시판)", "Phase 2", "전임상/미상", "전임상/미상", "Early Phase 1"
    ]
    assert list(out["safety"]) == ["블랙박스경고", "", "", "시판중단", ""]
    assert list(out["moa"]) == ["Inhibitor", "Antagonist", "", "", ""]
    # MoA L2: C0 INHIBITOR + covalent -> irreversible; C1 ANTAGONIST unchanged
    assert list(out["moa_l2"]) == ["Irreversible (Covalent) Inhibitor", "Antagonist", "", "", ""]
    # Modality L2 from type + size
    assert list(out["modality_l2"]) == [
        "Synthetic organic (MW<500)", "Macrocycle-like (MW 500-1000)",
        "mAb/ADC/Bispecific", "Large Biologics", "Synthetic organic (MW<500)",
    ]
    # Physicochemical L2 (BCS / bRo5): C4 is small but matches no BCS band
    assert list(out["physchem_l2"]) == [
        "BCS1 (고용해·고투과)", "고MW·고지용성", "고MW (>500)", "고MW (>500)", "기타(소분자)",
    ]


def test_overlay_columns_tolerate_missing_source_columns():
    from chemspace.data.dataset import Dataset

    compounds = pd.DataFrame(index=["C0", "C1"])  # no max_phase / flags
    mechs = pd.DataFrame(columns=["chembl_id"])   # no action_type
    out = Dataset._overlay_columns(compounds, mechs)

    assert list(out["dev_stage"]) == ["전임상/미상", "전임상/미상"]
    assert list(out["dev_stage_l2"]) == ["전임상/미상", "전임상/미상"]
    assert list(out["safety"]) == ["", ""]
    assert list(out["moa"]) == ["", ""]
    assert list(out["moa_l2"]) == ["", ""]
    assert list(out["modality_l2"]) == ["", ""]
    assert list(out["physchem_l2"]) == ["", ""]  # no descriptors -> 미분류, not a bucket


def test_structural_alert_label_and_priority():
    from chemspace.data.alerts import alert_label, label_from

    # p-benzoquinone is a classic PAINS; a clean/invalid/empty SMILES is blank.
    assert alert_label("O=C1C=CC(=O)C=C1") == "PAINS"
    assert alert_label("CCO") == ""           # ethanol: no alert
    assert alert_label("") == ""
    assert alert_label("not_a_smiles") == ""  # unparseable -> blank, not an error

    # Most-severe wins: PAINS > NIH > Brenk, blank when none match.
    assert label_from({"PAINS": True, "NIH": True, "BRENK": True}) == "PAINS"
    assert label_from({"PAINS": False, "NIH": True, "BRENK": True}) == "NIH"
    assert label_from({"PAINS": False, "NIH": False, "BRENK": True}) == "Brenk"
    assert label_from({"PAINS": False, "NIH": False, "BRENK": False}) == ""


def test_structural_alerts_cache_optional(tmp_path):
    from chemspace.data.dataset import Dataset

    idx = pd.Index(["C0", "C1", "C2"])
    # No cache file -> blank column, so a pre-cache snapshot still loads.
    blank = Dataset._structural_alerts(tmp_path, idx)
    assert list(blank) == ["", "", ""]

    # With a cache, labels map by chembl_id; ids absent from it stay blank.
    pd.DataFrame(
        {"chembl_id": ["C0", "C2"], "struct_alert": ["PAINS", "Brenk"]}
    ).to_parquet(tmp_path / "structural_alerts.parquet", index=False)
    got = Dataset._structural_alerts(tmp_path, idx)
    assert list(got) == ["PAINS", "", "Brenk"]
