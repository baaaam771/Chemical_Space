"""Render a saved exploration session as one static HTML page.

Every node becomes a panel, in tree order, so the whole walk is visible at once
-- including the moment a subset is re-placed under a new domain and regroups.

    python scripts/make_figure.py [session_dir] [-o out.html]

Reads coordinates straight from the saved session; nothing is recomputed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import chemspace
from chemspace.placement import NOISE, UNCOVERED
from chemspace.viewer.jupyter import NOISE_COLOR, PALETTE, UNCOVERED_COLOR


def tree_order(session) -> list[str]:
    out: list[str] = []

    def walk(nid: str) -> None:
        out.append(nid)
        for child in session.node(nid).children:
            walk(child)

    for root in session.root_ids:
        walk(root)
    return out


def point_colors(labels: np.ndarray) -> list[str]:
    return [
        UNCOVERED_COLOR
        if lab == UNCOVERED
        else NOISE_COLOR
        if lab == NOISE
        else PALETTE[int(lab) % len(PALETTE)]
        for lab in labels
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", nargs="?", default="data/sessions/smoke")
    ap.add_argument("-o", "--out", default="data/figures/exploration.html")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument(
        "--png", action="store_true", help="also write a static PNG beside the HTML"
    )
    args = ap.parse_args()

    ds = chemspace.load()
    session = chemspace.ExplorationSession.load(args.session_dir, ds)

    order = tree_order(session)
    cols = max(1, args.cols)
    rows = math.ceil(len(order) / cols)

    titles = []
    for nid in order:
        node = session.node(nid)
        dom = session.domains[node.domain]
        step = {"root": "시작", "select": "영역 선택", "domain": "도메인 전환"}[node.origin.kind]
        titles.append(
            f"[{nid}] {step} · {dom.label} · {node.n:,}개 · "
            f"{len(node.placement.cluster_ids)}군집"
        )

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        horizontal_spacing=0.045,
        vertical_spacing=max(0.10 / max(rows - 1, 1), 0.045),
    )

    for i, nid in enumerate(order):
        node = session.node(nid)
        p = node.placement
        r, c = i // cols + 1, i % cols + 1
        df = session.frame(nid)
        hover = [
            f"{cid}<br>{name}<br>군집 {lab}"
            for cid, name, lab in zip(
                df.index,
                df["pref_name"].astype("object").where(df["pref_name"].notna(), "(이름 없음)"),
                p.labels,
            )
        ]
        fig.add_trace(
            go.Scattergl(
                x=p.coords[:, 0],
                y=p.coords[:, 1],
                mode="markers",
                marker=dict(
                    size=3 if node.n > 4000 else 5,
                    color=point_colors(p.labels),
                    opacity=0.75,
                    line=dict(width=0),
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            row=r,
            col=c,
        )
        fig.update_xaxes(visible=False, row=r, col=c)
        fig.update_yaxes(visible=False, scaleanchor=f"x{i + 1 if i else ''}", row=r, col=c)

    crumbs = "<br>".join(
        f"<b>[{nid}]</b> {session.breadcrumb(nid)}" for nid in order
    )
    # Reserve room for one breadcrumb line per node plus a gap, or the last
    # paths run off the bottom of the page.
    line_px, gap_px = 19, 40
    bottom = gap_px + line_px * len(order) + 20
    height = 300 * rows + 130 + bottom
    fig.update_layout(
        title=dict(
            text=(
                f"Chemical space 반복 탐색 · {ds.version} · {len(ds.compounds):,}개 화합물"
                "<br><sub>각 패널은 그 단계의 화합물만으로 다시 임베딩·군집화한 결과입니다. "
                "색은 군집, 회색은 해당 도메인에 데이터가 없는 화합물입니다.</sub>"
            ),
            x=0.01,
            font=dict(size=17),
        ),
        height=height,
        margin=dict(l=20, r=20, t=130, b=bottom),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=11),
        annotations=list(fig.layout.annotations)
        + [
            dict(
                text=f"<span style='font-size:11px;line-height:1.7'>{crumbs}</span>",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0,
                y=-gap_px / (height - 130 - bottom),
                xanchor="left",
                yanchor="top",
                align="left",
            )
        ],
    )
    for ann in fig.layout.annotations[: len(order)]:
        ann.font.size = 11

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
    print(f"wrote {out}  ({len(order)} panels)")

    if args.png:
        png = out.with_suffix(".png")
        try:
            fig.write_image(str(png), width=1500, height=height, scale=2)
            print(f"wrote {png}")
        except Exception as exc:  # noqa: BLE001 - kaleido/Chrome may be absent
            print(f"PNG export skipped ({type(exc).__name__}: {exc}); HTML is still written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
