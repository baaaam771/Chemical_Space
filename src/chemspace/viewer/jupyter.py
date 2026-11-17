"""Jupyter viewer: drag a region, switch the domain, walk the tree.

Deliberately a thin shell over :class:`~chemspace.session.tree.ExplorationSession`
-- it holds no exploration state of its own, it only calls ``select`` and
``switch_domain`` and redraws.  Everything it does is available from plain
Python, which is what makes a web front end later a re-skin rather than a
rewrite.

One trace, not one per cluster: plotly reports lasso selections as point indices
*within a trace*, so a single trace means the indices it hands back are already
the node's compound indices. Colour goes in a marker colour array and the
legend is drawn as HTML beside the plot.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from chemspace.placement import NOISE, UNCOVERED
from chemspace.session.tree import ExplorationSession

log = logging.getLogger(__name__)

__all__ = ["SpaceExplorer"]

#: Qualitative palette; cycled when a node has more clusters than colours.
PALETTE = (
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#7F7F7F",
    "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
    "#E6AB02", "#A6761D", "#666666", "#8DD3C7", "#FB8072",
)
NOISE_COLOR = "#C9CDD4"
UNCOVERED_COLOR = "#E8EAED"


def _cluster_colors(labels: np.ndarray) -> tuple[list[str], list[tuple[str, str, int]]]:
    """Per-point colours plus legend entries ``(label, color, count)``."""
    colors: list[str] = []
    for lab in labels:
        if lab == UNCOVERED:
            colors.append(UNCOVERED_COLOR)
        elif lab == NOISE:
            colors.append(NOISE_COLOR)
        else:
            colors.append(PALETTE[int(lab) % len(PALETTE)])
    legend: list[tuple[str, str, int]] = []
    ids, counts = np.unique(labels, return_counts=True)
    for lab, cnt in zip(ids, counts):
        lab = int(lab)
        if lab >= 0:
            legend.append((f"군집 {lab}", PALETTE[lab % len(PALETTE)], int(cnt)))
    for lab, cnt in zip(ids, counts):
        lab = int(lab)
        if lab == NOISE:
            legend.append(("noise", NOISE_COLOR, int(cnt)))
        elif lab == UNCOVERED:
            legend.append(("데이터 없음", UNCOVERED_COLOR, int(cnt)))
    return colors, legend


def _category_colors(values: Sequence[str]) -> tuple[list[str], list[tuple[str, str, int]]]:
    s = pd.Series(list(values), dtype="object").fillna("미분류")
    order = s.value_counts()
    mapping = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(order.index)}
    legend = [(str(k), mapping[k], int(v)) for k, v in order.items()]
    return [mapping[v] for v in s], legend


class SpaceExplorer:
    """Interactive front end for one :class:`ExplorationSession`.

    Usage in a notebook::

        explorer = SpaceExplorer(session)
        explorer.show()

    The session keeps every step, so ``session.tree()`` after a few drags shows
    the branches you created.
    """

    def __init__(
        self,
        session: ExplorationSession,
        node_id: str | None = None,
        max_points: int = 40_000,
        height: int = 620,
    ):
        import ipywidgets as W
        import plotly.graph_objects as go

        self.session = session
        self.W = W
        self.max_points = max_points
        self.current = node_id or (session.root_ids[0] if session.root_ids else None)
        if self.current is None:
            raise ValueError("session has no nodes yet -- call session.root(...) first")

        # A set, because a lasso drag and individual shift-clicks both write
        # here and the engine takes whatever indices come out (see
        # ExplorationSession.select).
        self._selected: set[int] = set()
        self._polygon: list[list[float]] | None = None
        self._polygon_partial = False
        self._click_toggles = 0
        self._color_mode = "cluster"

        self.fig = go.FigureWidget(
            data=[
                go.Scattergl(
                    x=[],
                    y=[],
                    mode="markers",
                    marker=dict(size=5, opacity=0.8, line=dict(width=0)),
                    selected=dict(marker=dict(opacity=1.0)),
                    unselected=dict(marker=dict(opacity=0.15)),
                    hovertemplate="%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
                    customdata=[],
                )
            ],
            layout=go.Layout(
                height=height,
                margin=dict(l=8, r=8, t=8, b=8),
                dragmode="lasso",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False, scaleanchor="x"),
                showlegend=False,
                plot_bgcolor="white",
                newselection=dict(line=dict(color="#333", width=1.5)),
            ),
        )
        self.fig.data[0].on_selection(self._on_selection)
        self.fig.data[0].on_deselect(self._on_deselect)
        self.fig.data[0].on_click(self._on_click)

        self._build_controls()
        self._render()

    # ------------------------------------------------------------------- UI

    def _build_controls(self) -> None:
        W = self.W
        labels = self.session.domains.labels
        self.domain_dd = W.Dropdown(
            options=[(f"{lab} ({name})", name) for name, lab in labels.items()],
            description="도메인",
            style={"description_width": "48px"},
            layout=W.Layout(width="230px"),
        )
        self.switch_btn = W.Button(
            description="도메인 전환", button_style="primary", icon="refresh",
            tooltip="같은 화합물을 다른 도메인 기준으로 다시 배치",
            layout=W.Layout(width="130px"),
        )
        self.narrow_btn = W.Button(
            description="선택 영역으로 좁히기", button_style="success", icon="filter",
            disabled=True, layout=W.Layout(width="180px"),
        )
        self.back_btn = W.Button(
            description="이전 단계", icon="arrow-left", layout=W.Layout(width="110px")
        )
        self.node_dd = W.Dropdown(
            description="단계", style={"description_width": "36px"},
            layout=W.Layout(width="420px"),
        )
        self.color_dd = W.Dropdown(
            options=[("군집", "cluster"), ("도메인 라벨", "domain"), ("모달리티", "modality")],
            value="cluster", description="색", style={"description_width": "28px"},
            layout=W.Layout(width="170px"),
        )
        self.sel_label = W.HTML()
        self.crumb = W.HTML()
        self.legend = W.HTML()
        self.status = W.HTML()
        self.tree_out = W.Output()
        self.table_out = W.Output()

        self.switch_btn.on_click(self._do_switch)
        self.narrow_btn.on_click(self._do_narrow)
        self.back_btn.on_click(self._do_back)
        self.node_dd.observe(self._on_node_pick, names="value")
        self.color_dd.observe(self._on_color_pick, names="value")

        self.box = W.VBox(
            [
                self.crumb,
                W.HBox([self.node_dd, self.back_btn]),
                W.HBox([self.domain_dd, self.switch_btn, self.narrow_btn, self.color_dd]),
                self.sel_label,
                W.HBox(
                    [self.fig, W.VBox([self.legend], layout=W.Layout(width="230px"))]
                ),
                self.status,
                W.Accordion(
                    children=[self.table_out, self.tree_out],
                    titles=("군집 프로파일", "탐색 경로(트리)"),
                    selected_index=0,
                ),
            ]
        )

    def show(self):
        return self.box

    def _ipython_display_(self):  # pragma: no cover - notebook integration
        from IPython.display import display

        display(self.box)

    # --------------------------------------------------------------- drawing

    def _render(self) -> None:
        node = self.session.node(self.current)
        df = self.session.frame(self.current)
        self._reset_selection()

        warnings: list[str] = []
        if len(df) > self.max_points:
            warnings.append(
                f"{len(df):,}개는 한 번에 그리기엔 많습니다 "
                f"({self.max_points:,}개까지 권장). 먼저 좁히거나 max_points를 올리세요."
            )

        labels = node.placement.labels
        if self._color_mode == "cluster":
            colors, legend = _cluster_colors(labels)
        elif self._color_mode == "domain" and "domain_label" in df.columns:
            colors, legend = _category_colors(df["domain_label"].tolist())
        else:
            colors, legend = _category_colors(df["modality"].tolist())

        custom = np.stack([_hover_title(df), _hover_detail(df)], axis=1)

        with self.fig.batch_update():
            t = self.fig.data[0]
            t.x = df["x"].to_numpy()
            t.y = df["y"].to_numpy()
            t.marker.color = colors
            t.marker.size = 4 if len(df) > 6000 else 6
            t.customdata = custom
            t.selectedpoints = None

        self.legend.value = _legend_html(legend)
        self.crumb.value = (
            "<div style='font-size:13px;line-height:1.7'>"
            f"<b>경로</b>  {self.session.breadcrumb(self.current)}</div>"
        )
        p = node.placement
        dom = self.session.domains[node.domain]
        bits = [
            f"<b>{dom.label}</b>",
            f"{node.n:,}개 화합물",
            f"{len(p.cluster_ids)}군집",
            f"배치 {p.embed_method}",
            f"군집 {p.cluster_method}",
        ]
        if p.n_covered < node.n:
            bits.append(
                f"<span style='color:#B45309'>이 도메인에 데이터 있는 화합물 "
                f"{p.n_covered:,}/{node.n:,}</span>"
            )
        if p.notes.get("degenerate"):
            bits.append(f"<span style='color:#B91C1C'>{p.notes['degenerate']}</span>")
        warn_html = "".join(
            f"<div style='color:#B45309;font-size:12px'>{w}</div>" for w in warnings
        )
        self.status.value = (
            "<div style='font-size:12px;color:#444'>"
            + " · ".join(bits)
            + "</div>"
            + warn_html
        )

        # Assign without unobserve_all(): nothing observes this dropdown (the
        # switch is driven by the button), and unobserve_all() would also strip
        # the widget's own traitlet wiring.
        self.domain_dd.value = node.domain
        self._refresh_nodes()
        self._refresh_selection_label()
        self.back_btn.disabled = node.parent_id is None

        self.table_out.clear_output(wait=True)
        with self.table_out:
            from IPython.display import display

            tbl = self.session.cluster_table(self.current)
            display(tbl if len(tbl) else "군집 없음")
        self.tree_out.clear_output(wait=True)
        with self.tree_out:
            print(self.session.tree(highlight=self.current))

    def _refresh_nodes(self) -> None:
        opts = []
        for nid in self.session.nodes:
            node = self.session.node(nid)
            lab = self.session.domains[node.domain].label
            desc = node.origin.describe(None, lab, node.n)
            opts.append((f"[{nid}] {desc}", nid))
        self.node_dd.unobserve(self._on_node_pick, names="value")
        self.node_dd.options = opts
        self.node_dd.value = self.current
        self.node_dd.observe(self._on_node_pick, names="value")

    def _reset_selection(self) -> None:
        self._selected = set()
        self._polygon = None
        self._polygon_partial = False
        self._click_toggles = 0

    def _describe_selection(self) -> str:
        n = len(self._selected)
        if not self._click_toggles:
            return f"영역 선택: <b>{n:,}개</b>"
        if self._polygon is None:
            return f"개별 선택: <b>{n:,}개</b>"
        return f"선택: <b>{n:,}개</b> (올가미 + 개별 {self._click_toggles:,}회 조정)"

    def _selection_how(self) -> str:
        if not self._click_toggles:
            return "드래그 선택"
        return "드래그 + 개별 선택" if self._polygon is not None else "개별 선택"

    def _refresh_selection_label(self) -> None:
        n = len(self._selected)
        self.narrow_btn.disabled = n == 0
        # Drive the highlight from our set: a lasso drag sets selectedpoints by
        # itself, a shift-click does not.
        self.fig.data[0].selectedpoints = sorted(self._selected) if n else None
        if n:
            self.sel_label.value = (
                f"<div style='font-size:13px'>{self._describe_selection()} "
                "— ‘선택 영역으로 좁히기’를 누르면 다음 단계가 만들어집니다.</div>"
            )
        else:
            self.sel_label.value = (
                "<div style='font-size:13px;color:#666'>올가미(lasso)로 영역을 "
                "드래그하세요. shift+클릭으로 화합물을 하나씩 더하거나 뺄 수 "
                "있습니다.</div>"
            )

    # -------------------------------------------------------------- handlers

    def _on_click(self, trace, points, state) -> None:
        """shift+click toggles one compound in or out of the selection.

        A plain click is left alone: in lasso mode it is how a drag starts, and
        wiping a careful selection on a mis-click is expensive to redo.
        """
        if not points.point_inds or not getattr(state, "shift", False):
            return
        i = int(points.point_inds[0])
        self._selected.symmetric_difference_update({i})
        self._click_toggles += 1
        # The polygon is now a record of the gesture, not the selection itself.
        self._polygon_partial = self._polygon is not None
        self._refresh_selection_label()

    def _on_selection(self, trace, points, selector) -> None:
        self._selected = set(int(i) for i in points.point_inds)
        self._polygon_partial = False
        self._click_toggles = 0
        xs = getattr(selector, "xs", None)
        ys = getattr(selector, "ys", None)
        if xs is not None and ys is not None:
            self._polygon = [[float(a), float(b)] for a, b in zip(xs, ys)]
        else:
            xr, yr = getattr(selector, "xrange", None), getattr(selector, "yrange", None)
            self._polygon = (
                [
                    [float(xr[0]), float(yr[0])],
                    [float(xr[1]), float(yr[0])],
                    [float(xr[1]), float(yr[1])],
                    [float(xr[0]), float(yr[1])],
                ]
                if xr and yr
                else None
            )
        self._refresh_selection_label()

    def _on_deselect(self, trace, points) -> None:
        self._reset_selection()
        self._refresh_selection_label()

    def _do_narrow(self, _btn) -> None:
        if not self._selected:
            return
        how = self._selection_how()
        try:
            node = self.session.select(
                self.current,
                point_indices=self._selected,
                polygon=self._polygon,
                polygon_partial=self._polygon_partial,
                how=how,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the notebook
            self.status.value = f"<span style='color:#B91C1C'>좁히기 실패: {exc}</span>"
            return
        self.current = node.node_id
        self._render()

    def _do_switch(self, _btn) -> None:
        target = self.domain_dd.value
        node = self.session.node(self.current)
        if target == node.domain:
            self.status.value = (
                "<span style='color:#B45309'>이미 같은 도메인입니다. "
                "다른 도메인을 고르세요.</span>"
            )
            return
        try:
            new = self.session.switch_domain(self.current, target)
        except Exception as exc:  # noqa: BLE001
            self.status.value = f"<span style='color:#B91C1C'>전환 실패: {exc}</span>"
            return
        self.current = new.node_id
        self._render()

    def _do_back(self, _btn) -> None:
        node = self.session.node(self.current)
        if node.parent_id:
            self.current = node.parent_id
            self._render()

    def _on_node_pick(self, change) -> None:
        if change["new"] and change["new"] != self.current:
            self.current = change["new"]
            self._render()

    def _on_color_pick(self, change) -> None:
        self._color_mode = change["new"]
        self._render()


def _text(df: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    """A plain string Series for ``column``, present or not.

    Built from ``df.index`` rather than by arithmetic on mixed Index/Series
    objects -- that combination changes behaviour between pandas versions and
    silently misaligns.
    """
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    return df[column].astype("object").where(df[column].notna(), default).astype(str)


def _hover_title(df: pd.DataFrame) -> np.ndarray:
    ids = pd.Series(df.index.astype(str), index=df.index, dtype="object")
    names = _text(df, "pref_name", "(이름 없음)")
    return (ids + " · " + names).to_numpy(dtype=object)


def _hover_detail(df: pd.DataFrame) -> np.ndarray:
    parts = [_text(df, "modality", "?")]
    for col, label, digits in (("mw", "MW", 0), ("clogp", "cLogP", 1)):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").round(digits)
            shown = vals.map(lambda v: "?" if pd.isna(v) else f"{v:.{digits}f}")
            parts.append(label + " " + shown.astype(str))
    parts.append("군집 " + _text(df, "cluster", "?"))
    out = parts[0]
    for p in parts[1:]:
        out = out + " · " + p
    return out.to_numpy(dtype=object)


def _legend_html(entries: list[tuple[str, str, int]]) -> str:
    rows = "".join(
        f"<div style='display:flex;align-items:center;gap:6px;margin:2px 0'>"
        f"<span style='width:11px;height:11px;border-radius:2px;background:{c};"
        f"display:inline-block;flex:none'></span>"
        f"<span style='font-size:12px'>{name}</span>"
        f"<span style='font-size:11px;color:#888;margin-left:auto'>{cnt:,}</span></div>"
        for name, c, cnt in entries[:28]
    )
    more = (
        f"<div style='font-size:11px;color:#888;margin-top:4px'>… 외 {len(entries) - 28}개</div>"
        if len(entries) > 28
        else ""
    )
    return f"<div style='padding:4px 2px'>{rows}{more}</div>"
