"""The exploration tree -- the record of where you have looked.

Every step is a node holding ``(compound set, domain, placement)``.  Two moves
create one:

``select``         fewer compounds, same domain  -- re-placed within the selection
``switch_domain``  same compounds, new domain    -- re-placed under new features
``restrict``       fewer compounds, same domain  -- by a named rule, not a gesture

All three re-fit the embedding, so all three change the picture; ``select`` and
``restrict`` also change the compound count.  That distinction is what the
breadcrumb shows.  ``restrict`` is split out from ``select`` only so the step
reads as what it was -- "약물층만 남긴 12,804개" rather than "선택한 12,804개" --
since a filter is reproducible from its name while a lasso is not.

It is a **tree, not a log.** Going back to an earlier node and selecting a
different region adds a sibling instead of overwriting, so "disease region A"
and "disease region B" both stay available for comparison. That is the point of
keeping the path at all.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
import pandas as pd

from chemspace.data.dataset import Dataset

if TYPE_CHECKING:  # pragma: no cover
    from chemspace.cache import PlacementCache
from chemspace.domains import DomainRegistry, default_registry
from chemspace.placement import (
    ClusterConfig,
    EmbedConfig,
    Placement,
    UNCOVERED,
    place,
    recluster,
)
from chemspace.session.filters import drug_layer_mask, resolve as resolve_filter
from chemspace.session.korean import ro

log = logging.getLogger(__name__)

__all__ = ["Origin", "Node", "ExplorationSession"]


@dataclass(frozen=True)
class Origin:
    """How a node came to exist -- enough to replay or explain the step."""

    kind: str  # "root" | "select" | "filter" | "domain"
    detail: dict = field(default_factory=dict)

    def describe(self, parent: "Node | None", domain_label: str, n: int) -> str:
        if self.kind == "root":
            return f"전체 {n:,}개 ({domain_label})"
        if self.kind == "select":
            how = self.detail.get("how", "선택")
            parent_label = self.detail.get("parent_domain_label", domain_label)
            return f"{parent_label}에서 {how}한 {n:,}개"
        if self.kind == "filter":
            what = self.detail.get("filter_label", "필터")
            return f"{what}만 남긴 {n:,}개"
        if self.kind == "domain":
            return f"{ro(domain_label)} 재배치 ({n:,}개)"
        return f"{self.kind} ({n:,}개)"


@dataclass
class Node:
    node_id: str
    parent_id: str | None
    domain: str
    compound_ids: tuple[str, ...]
    origin: Origin
    placement: Placement
    created_at: str
    children: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def n(self) -> int:
        return len(self.compound_ids)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Node {self.node_id} {self.domain} n={self.n}>"


class ExplorationSession:
    """Holds the dataset, the domains, and the tree of steps taken."""

    def __init__(
        self,
        dataset: Dataset,
        registry: DomainRegistry | None = None,
        embed: EmbedConfig | None = None,
        cluster: ClusterConfig | None = None,
        cache: "PlacementCache | None" = None,
    ):
        self.ds = dataset
        self.domains = registry or default_registry(dataset)
        self.embed_cfg = embed or EmbedConfig()
        self.cluster_cfg = cluster or ClusterConfig()
        self.cache = cache
        """Optional :class:`~chemspace.cache.PlacementCache`.

        A placement is a pure function of (domain, compound set, configs), so
        reusing one is not a shortcut -- it returns what recomputation would.
        Mostly it saves the 75-105 s root placement, which matters in a UI.
        """
        self.nodes: dict[str, Node] = {}
        self.root_ids: list[str] = []
        self._counter = 0

    # --------------------------------------------------------------- building

    def _new_id(self) -> str:
        self._counter += 1
        return f"n{self._counter}"

    def _place(self, domain_name: str, compound_ids: Sequence[str]) -> Placement:
        ids = tuple(compound_ids)
        key = None
        if self.cache is not None:
            # Fingerprint, not version: re-pulling the snapshot with a
            # different potency threshold changes the target features while
            # the release name stays put.
            dataset_id = getattr(self.ds, "fingerprint", None) or self.ds.version
            key = self.cache.key(
                dataset_id, domain_name, ids, self.embed_cfg, self.cluster_cfg
            )
            cached = self.cache.get(key, ids, domain_name)
            if cached is not None:
                log.info("cache hit for %s on %d compounds", domain_name, len(ids))
                return cached

        domain = self.domains[domain_name]
        feats = domain.featurize(ids)
        p = place(feats, self.embed_cfg, self.cluster_cfg)
        p.domain = domain_name
        if self.cache is not None and key is not None:
            self.cache.put(key, p)
        return p

    def _add(
        self,
        parent: Node | None,
        domain: str,
        compound_ids: Sequence[str],
        origin: Origin,
        note: str = "",
    ) -> Node:
        ids = tuple(compound_ids)
        if not ids:
            raise ValueError("a node needs at least one compound")
        node = Node(
            node_id=self._new_id(),
            parent_id=parent.node_id if parent else None,
            domain=domain,
            compound_ids=ids,
            origin=origin,
            placement=self._place(domain, ids),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            note=note,
        )
        self.nodes[node.node_id] = node
        if parent is not None:
            parent.children.append(node.node_id)
        else:
            self.root_ids.append(node.node_id)
        log.info(
            "%s: %s n=%d clusters=%d",
            node.node_id,
            domain,
            node.n,
            len(node.placement.cluster_ids),
        )
        return node

    def root(
        self,
        domain: str = "physchem",
        compound_ids: Sequence[str] | None = None,
        note: str = "",
    ) -> Node:
        """Place the starting set. Defaults to every compound in the snapshot."""
        ids = tuple(compound_ids) if compound_ids is not None else self.ds.all_ids
        return self._add(None, domain, ids, Origin("root", {"n": len(ids)}), note)

    def select(
        self,
        node_id: str,
        *,
        compound_ids: Iterable[str] | None = None,
        clusters: Iterable[int] | None = None,
        point_indices: Iterable[int] | None = None,
        polygon: Sequence[Sequence[float]] | None = None,
        polygon_partial: bool = False,
        how: str = "선택",
        note: str = "",
    ) -> Node:
        """Narrow to part of ``node_id``, keeping the same domain.

        Give exactly one of ``compound_ids``, ``clusters`` or ``point_indices``
        (what the viewer hands over after a lasso drag). ``polygon`` is recorded
        alongside for replay but does not itself select.

        Set ``polygon_partial`` when the selection is *not* simply "the points
        inside that polygon" -- a second lassoed region, or compounds added and
        removed one at a time by shift-clicking. The trace is still worth
        keeping as a record of the gesture, but a later replay that re-selected
        from the polygon alone would quietly produce a different compound set,
        so the flag says not to trust it that way.
        """
        parent = self.node(node_id)
        given = [x is not None for x in (compound_ids, clusters, point_indices)]
        if sum(given) != 1:
            raise ValueError("pass exactly one of compound_ids, clusters, point_indices")

        if clusters is not None:
            wanted = {int(c) for c in clusters}
            mask = np.isin(parent.placement.labels, list(wanted))
            ids = tuple(np.asarray(parent.compound_ids)[mask].tolist())
            detail: dict = {"clusters": sorted(wanted)}
        elif point_indices is not None:
            idx = np.asarray(sorted({int(i) for i in point_indices}), dtype=int)
            if idx.size and (idx.min() < 0 or idx.max() >= parent.n):
                raise IndexError(f"point index out of range for node {node_id}")
            ids = tuple(np.asarray(parent.compound_ids)[idx].tolist())
            detail = {"n_points": int(idx.size)}
        else:
            ids = tuple(compound_ids)  # type: ignore[arg-type]
            unknown = set(ids) - set(parent.compound_ids)
            if unknown:
                raise ValueError(
                    f"{len(unknown)} compounds are not in node {node_id} "
                    f"(e.g. {sorted(unknown)[:3]})"
                )
            detail = {"explicit": len(ids)}

        if not ids:
            raise ValueError("that selection is empty -- nothing to place")

        detail.update(
            how=how,
            parent_domain=parent.domain,
            parent_domain_label=self.domains[parent.domain].label,
            parent_n=parent.n,
        )
        if polygon is not None:
            detail["polygon"] = [[float(x), float(y)] for x, y in polygon]
            if polygon_partial:
                detail["polygon_partial"] = True
        return self._add(parent, parent.domain, ids, Origin("select", detail), note)

    def covered_mask(self, node_id: str, domain: str) -> np.ndarray:
        """Which of the node's compounds carry ``domain``'s annotation.

        Free when ``domain`` is the node's own -- the placement already marks
        the uncovered ones -- and one featurisation otherwise. Deliberately
        does *not* place anything: the point is to answer "is this switch worth
        waiting for" before paying for the switch.
        """
        node = self.node(node_id)
        if domain == node.domain:
            return node.placement.labels != UNCOVERED
        if domain not in self.domains:
            raise KeyError(f"unknown domain {domain!r}; have {self.domains.names}")
        return np.asarray(self.domains[domain].featurize(node.compound_ids).covered)

    def domain_coverage(self, node_id: str) -> dict[str, int]:
        """``{domain name: how many of this node's compounds it covers}``.

        What the UI needs to label its domain picker. Switching to a domain
        that covers 91 of 1,200 compounds is a cliff, not a step, and the only
        honest moment to say so is before the click.
        """
        return {
            name: int(self.covered_mask(node_id, name).sum())
            for name in self.domains.names
        }

    def filter_mask(self, node_id: str, kind: str, domain: str | None = None) -> np.ndarray:
        """The boolean mask :meth:`restrict` would apply, without placing."""
        node = self.node(node_id)
        f = resolve_filter(kind)
        if f.needs_domain:
            return self.covered_mask(node_id, domain or node.domain)
        return drug_layer_mask(self.ds, node.compound_ids)

    def restrict(
        self,
        node_id: str,
        kind: str = "drug",
        *,
        domain: str | None = None,
        note: str = "",
    ) -> Node:
        """Narrow ``node_id`` by a named rule, keeping the same domain.

        ``drug`` keeps the clinical-phase layer; ``annotated`` keeps the
        compounds that carry a domain's data (``domain`` names which, default
        the node's own). Raises when the rule would keep everything or nothing,
        because a node identical to its parent is a placement paid for twice
        and an empty one cannot be placed at all.
        """
        node = self.node(node_id)
        f = resolve_filter(kind)
        target_domain = (domain or node.domain) if f.needs_domain else None
        mask = self.filter_mask(node_id, kind, target_domain)
        kept = int(mask.sum())
        if kept == 0:
            raise ValueError(
                f"{f.label} 필터가 {node_id}에서 아무것도 남기지 않습니다"
            )
        if kept == node.n:
            raise ValueError(
                f"{f.label} 필터가 {node_id}의 {node.n:,}개를 모두 남깁니다 — "
                "좁혀지지 않으므로 새 노드를 만들지 않습니다"
            )
        ids = tuple(np.asarray(node.compound_ids)[mask].tolist())
        label = f.label
        if f.needs_domain and target_domain is not None:
            label = f"{self.domains[target_domain].label} {f.label}"
        detail = {
            "filter": kind,
            "filter_label": label,
            "parent_n": node.n,
            "kept": kept,
        }
        if target_domain is not None:
            detail["domain"] = target_domain
        return self._add(node, node.domain, ids, Origin("filter", detail), note)

    def switch_domain(self, node_id: str, domain: str, note: str = "") -> Node:
        """Re-place the same compounds under a different domain."""
        parent = self.node(node_id)
        if domain not in self.domains:
            raise KeyError(f"unknown domain {domain!r}; have {self.domains.names}")
        if domain == parent.domain:
            log.warning("node %s is already in domain %s", node_id, domain)
        return self._add(
            parent,
            domain,
            parent.compound_ids,
            Origin(
                "domain",
                {
                    "from_domain": parent.domain,
                    "from_domain_label": self.domains[parent.domain].label,
                    "n": parent.n,
                },
            ),
            note,
        )

    def recluster_node(self, node_id: str, method: str) -> Node:
        """Re-cluster a node in place with a different method, reusing its coords
        (no re-embed). The choice becomes the session default, so later nodes use
        it too. ``method`` is one of hdbscan / agglomerative / kmeans / none."""
        allowed = {"hdbscan", "agglomerative", "kmeans", "none"}
        if method not in allowed:
            raise ValueError(f"unknown cluster method {method!r}; use one of {sorted(allowed)}")
        node = self.node(node_id)
        self.cluster_cfg = replace(self.cluster_cfg, method=method)
        recluster(node.placement, self.cluster_cfg)
        return node

    # ---------------------------------------------------------------- reading

    def node(self, node_id: str) -> Node:
        try:
            return self.nodes[node_id]
        except KeyError:
            raise KeyError(f"no node {node_id!r}; have {sorted(self.nodes)}") from None

    def path(self, node_id: str) -> list[Node]:
        """Root to ``node_id``."""
        out: list[Node] = []
        cur: str | None = node_id
        while cur is not None:
            n = self.node(cur)
            out.append(n)
            cur = n.parent_id
        return list(reversed(out))

    def breadcrumb(self, node_id: str, sep: str = "  →  ") -> str:
        """``전체 16,809개 (물성)  →  물성에서 선택한 1,200개  →  질환으로 재배치 (1,200개)``"""
        parts = []
        prev: Node | None = None
        for n in self.path(node_id):
            parts.append(n.origin.describe(prev, self.domains[n.domain].label, n.n))
            prev = n
        return sep.join(parts)

    def siblings(self, node_id: str) -> list[Node]:
        node = self.node(node_id)
        if node.parent_id is None:
            return [self.node(r) for r in self.root_ids if r != node_id]
        parent = self.node(node.parent_id)
        return [self.node(c) for c in parent.children if c != node_id]

    def tree(self, highlight: str | None = None) -> str:
        """ASCII tree of the whole session, so branches are visible."""
        lines: list[str] = []

        def walk(nid: str, prefix: str, last: bool) -> None:
            node = self.node(nid)
            marker = "└─ " if last else "├─ "
            star = " ◀" if nid == highlight else ""
            label = self.domains[node.domain].label
            desc = node.origin.describe(None, label, node.n)
            note = f'  "{node.note}"' if node.note else ""
            nclus = len(node.placement.cluster_ids)
            lines.append(
                f"{prefix}{marker}[{nid}] {desc} · {nclus}군집{note}{star}"
            )
            kids = node.children
            child_prefix = prefix + ("   " if last else "│  ")
            for i, k in enumerate(kids):
                walk(k, child_prefix, i == len(kids) - 1)

        for i, r in enumerate(self.root_ids):
            walk(r, "", i == len(self.root_ids) - 1)
        return "\n".join(lines)

    def history(self) -> pd.DataFrame:
        rows = []
        for nid, node in self.nodes.items():
            p = node.placement
            rows.append(
                {
                    "node": nid,
                    "parent": node.parent_id or "",
                    "step": node.origin.kind,
                    "domain": self.domains[node.domain].label,
                    "compounds": node.n,
                    "covered": p.n_covered,
                    "clusters": len(p.cluster_ids),
                    "noise": p.sizes().get(-1, 0),
                    "uncovered": p.sizes().get(UNCOVERED, 0),
                    "embed": p.embed_method,
                    "note": node.note,
                }
            )
        return pd.DataFrame(rows)

    def cluster_table(self, node_id: str, domains: Sequence[str] | None = None) -> pd.DataFrame:
        """Per-cluster profile of a node.

        By default each cluster is described in its *own* domain plus the two
        that usually matter for reading a map -- physchem and modality. Pass
        ``domains`` to profile against any others.
        """
        node = self.node(node_id)
        want = list(domains) if domains else _default_profile_domains(node.domain, self.domains)
        rows = []
        for cid in node.placement.cluster_ids:
            members = node.placement.members(cid)
            row: dict = {"cluster": cid, "n": len(members)}
            for dname in want:
                prof = self.domains[dname].profile(members)
                row[self.domains[dname].label] = prof.get("summary", "")
            rows.append(row)
        extra = node.placement.sizes()
        specials = [
            {"cluster": special, "n": extra[special], "비고": name}
            for special, name in ((-1, "noise"), (UNCOVERED, "데이터 없음"))
            if special in extra
        ]
        if not rows and not specials:
            return pd.DataFrame()
        df = pd.DataFrame(rows + specials).set_index("cluster")
        if "비고" in df.columns:
            df["비고"] = df["비고"].fillna("")
        return df.fillna("")

    def cluster_card(self, node_id: str, cluster_id: int) -> dict:
        """One cluster read in depth, for the cluster card: what is
        over-represented in it versus the whole node (the thing that actually
        characterises it), a few central example compounds, and its physchem
        range.
        """
        node = self.node(node_id)
        p = node.placement
        cid = int(cluster_id)
        mask = p.labels == cid
        member_ids = [c for c, m in zip(p.compound_ids, mask) if m]
        if not member_ids:
            raise KeyError(f"no cluster {cid} in node {node_id}")
        kind = "noise" if cid == -1 else "uncovered" if cid == UNCOVERED else "cluster"

        reps: list[str] = []
        if kind == "cluster":
            # The compounds nearest the cluster's centre read as its typical
            # examples; show their structures.
            coords = p.coords[mask]
            centroid = coords.mean(axis=0)
            order = np.argsort(((coords - centroid) ** 2).sum(axis=1))[:6]
            reps = [member_ids[int(i)] for i in order]

        enriched = []
        for name in self.domains.names:
            if name in ("physchem", "moiety"):
                continue  # continuous, or shown as the representative structures
            items = _enriched_terms(self.domains[name], member_ids, node.compound_ids)
            if items:
                enriched.append({"domain": self.domains[name].label, "items": items})

        return {
            "cluster": cid,
            "n": len(member_ids),
            "kind": kind,
            "representatives": reps,
            "enriched": enriched,
            "physchem": _physchem_range(self.ds.frame(member_ids)),
        }

    def describe(self, node_id: str, domains: Sequence[str] | None = None) -> pd.DataFrame:
        """What this whole selection looks like under *every* domain at once.

        The question right after a drag is "what did I just grab?", and the
        answer is rarely confined to the domain you were looking at -- a region
        picked on the physchem map is also some mix of therapeutic areas,
        target families and scaffolds.  One row per domain, with that domain's
        own coverage so a thin answer is visible as thin rather than read as
        fact.
        """
        node = self.node(node_id)
        want = list(domains) if domains else list(self.domains.names)
        rows = []
        for name in want:
            dom = self.domains[name]
            prof = dom.profile(node.compound_ids)
            covered = prof.get("covered", node.n)
            top = (
                prof.get("top_indications")
                or prof.get("top_families")
                or prof.get("top_scaffolds")
                or prof.get("modalities")
                or []
            )
            # The specific things, for the summary line: proteins for target,
            # indications for disease -- not the coarse families the "상위 항목"
            # column already lists. Up to five, where the one-line cluster
            # summary stops at three.
            specific = (
                prof.get("top_indications")
                or prof.get("top_targets")
                or prof.get("top_scaffolds")
                or prof.get("modalities")
                or []
            )
            summary = " · ".join(k.split(":", 1)[-1] for k, _ in specific[:5]) or prof.get(
                "summary", ""
            )
            rows.append(
                {
                    "도메인": dom.label,
                    "요약": summary,
                    "데이터 있음": f"{covered:,}/{node.n:,}",
                    "커버리지": f"{covered / node.n:.0%}" if node.n else "-",
                    "상위 항목": ", ".join(
                        f"{k.split(':', 1)[-1]}({v})" for k, v in top[:5]
                    ),
                    "현재 도메인": "◀" if name == node.domain else "",
                }
            )
        return pd.DataFrame(rows).set_index("도메인")

    def frame(self, node_id: str) -> pd.DataFrame:
        """Compounds of a node with their coordinates and cluster labels."""
        node = self.node(node_id)
        df = self.ds.frame(node.compound_ids).copy()
        df["x"] = node.placement.coords[:, 0]
        df["y"] = node.placement.coords[:, 1]
        df["cluster"] = node.placement.labels
        colors = self.domains[node.domain].color_series(node.compound_ids)
        if colors is not None:
            df["domain_label"] = np.asarray(colors)
        return df

    # ------------------------------------------------------------ persistence

    def save(self, directory: str | Path) -> Path:
        """Write the tree to ``directory``: one JSON index plus arrays per node."""
        out = Path(directory)
        (out / "placements").mkdir(parents=True, exist_ok=True)
        index = {
            "chembl_version": self.ds.version,
            "embed": asdict(self.embed_cfg),
            "cluster": asdict(self.cluster_cfg),
            "root_ids": self.root_ids,
            "counter": self._counter,
            "nodes": [],
        }
        for nid, node in self.nodes.items():
            p = node.placement
            np.savez_compressed(
                out / "placements" / f"{nid}.npz",
                coords=p.coords,
                labels=p.labels,
            )
            index["nodes"].append(
                {
                    "node_id": nid,
                    "parent_id": node.parent_id,
                    "domain": node.domain,
                    "compound_ids": list(node.compound_ids),
                    "origin": {"kind": node.origin.kind, "detail": node.origin.detail},
                    "created_at": node.created_at,
                    "children": node.children,
                    "note": node.note,
                    "placement_meta": {
                        "embed_method": p.embed_method,
                        "cluster_method": p.cluster_method,
                        "metric": p.metric,
                        "n_covered": p.n_covered,
                        "notes": _jsonable(p.notes),
                    },
                }
            )
        (out / "session.json").write_text(
            json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return out

    @classmethod
    def load(
        cls,
        directory: str | Path,
        dataset: Dataset,
        registry: DomainRegistry | None = None,
    ) -> ExplorationSession:
        src = Path(directory)
        index = json.loads((src / "session.json").read_text(encoding="utf-8"))
        if index["chembl_version"] != dataset.version:
            log.warning(
                "session was built on %s but dataset is %s",
                index["chembl_version"],
                dataset.version,
            )
        s = cls(
            dataset,
            registry,
            EmbedConfig(**index["embed"]),
            ClusterConfig(**index["cluster"]),
        )
        s.root_ids = list(index["root_ids"])
        s._counter = int(index["counter"])
        for rec in index["nodes"]:
            nid = rec["node_id"]
            arr = np.load(src / "placements" / f"{nid}.npz")
            meta = rec["placement_meta"]
            ids = tuple(rec["compound_ids"])
            s.nodes[nid] = Node(
                node_id=nid,
                parent_id=rec["parent_id"],
                domain=rec["domain"],
                compound_ids=ids,
                origin=Origin(rec["origin"]["kind"], rec["origin"]["detail"]),
                placement=Placement(
                    coords=arr["coords"],
                    labels=arr["labels"],
                    compound_ids=ids,
                    domain=rec["domain"],
                    embed_method=meta["embed_method"],
                    cluster_method=meta["cluster_method"],
                    metric=meta["metric"],
                    n_covered=meta["n_covered"],
                    notes=meta["notes"],
                ),
                created_at=rec["created_at"],
                children=list(rec["children"]),
                note=rec["note"],
            )
        return s

    def __repr__(self) -> str:  # pragma: no cover
        return f"<ExplorationSession {len(self.nodes)} nodes on {self.ds.version}>"


def _default_profile_domains(current: str, registry: DomainRegistry) -> list[str]:
    want = [current]
    for extra in ("physchem", "modality"):
        if extra in registry and extra not in want:
            want.append(extra)
    return want


def _enriched_terms(
    dom,
    cluster_ids: Sequence[str],
    node_ids: Sequence[str],
    top: int = 5,
    min_frac: float = 0.12,
    min_count: int = 2,
    min_ratio: float = 1.3,
) -> list[dict]:
    """Terms that are over-represented in the cluster versus the whole node.

    The top term of a cluster is often the top term of the node too (every
    cluster in an oncology set leads with "Neoplasms"); the *ratio* of the
    cluster's rate to the node's rate is what tells clusters apart. Guarded by a
    support floor so a term carried by two compounds does not top the list.
    """
    cc = dom.term_counts(cluster_ids)
    if not cc:
        return []
    nc = dom.term_counts(node_ids) or {}
    ncl = max(len(cluster_ids), 1)
    nnode = max(len(node_ids), 1)
    scored = []
    for term, c in cc.items():
        if c < min_count:
            continue
        cfrac = c / ncl
        if cfrac < min_frac:
            continue
        nfrac = nc.get(term, 0) / nnode
        ratio = cfrac / nfrac if nfrac > 0 else float("inf")
        if ratio < min_ratio:
            continue
        scored.append((term, cfrac, nfrac, ratio))
    scored.sort(key=lambda r: (r[3], r[1]), reverse=True)
    return [
        {
            "name": t.split(":", 1)[-1] if ":" in t else t,
            "cluster_pct": round(cfrac * 100),
            "node_pct": round(nfrac * 100, 1),
        }
        for t, cfrac, nfrac, _ in scored[:top]
    ]


_CARD_PHYSCHEM = ("mw", "clogp", "tpsa", "qed", "hbd", "hba", "rotb", "fsp3")


def _physchem_range(df) -> list[dict]:
    """Median and inter-quartile range of each key descriptor over a cluster."""
    out = []
    for col in _CARD_PHYSCHEM:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        out.append(
            {
                "name": col,
                "median": round(float(s.median()), 2),
                "p25": round(float(s.quantile(0.25)), 2),
                "p75": round(float(s.quantile(0.75)), 2),
            }
        )
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
