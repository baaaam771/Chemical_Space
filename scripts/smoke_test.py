"""End-to-end check of the exploration loop, headless.

Walks the flow from the project brief -- place everything by physchem, select a
region, re-place it by disease, select again, re-place by target -- and then
*measures* the thing the tool is for: whether re-placing a subset under a new
domain really regroups it, or just reshuffles the same clusters.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import chemspace
from chemspace.placement import UNCOVERED

# A Korean Windows console is cp949, which cannot encode the check marks and
# arrows this script prints -- it would die on its own output partway through a
# run that was otherwise fine.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def timed(label: str):
    class _T:
        def __enter__(self):
            self.t = time.perf_counter()
            return self

        def __exit__(self, *a):
            print(f"    [{label}: {time.perf_counter() - self.t:.1f}s]")

    return _T()


def lasso_like(session, node_id: str, target: int) -> tuple[list[int], list[list[float]]]:
    """A circular region of exactly ``target`` points, as a lasso drag would give.

    Selecting whole clusters is the wrong stand-in here: HDBSCAN on a 12.8k-point
    map finds ~10 density peaks, so the largest single cluster already overshoots
    1,200 and the "narrowing" step would not narrow. Taking the N points nearest
    a centroid reproduces what dragging a loop actually does, and lets the walk
    hit the brief's 1,200 -> 300 numbers exactly.

    Returns ``(point_indices, polygon)``.
    """
    node = session.node(node_id)
    coords = node.placement.coords
    labels = node.placement.labels

    # Centre on the densest real cluster so the region sits in populated space.
    real = [c for c in node.placement.cluster_ids]
    centre = (
        coords[labels == max(real, key=lambda c: int((labels == c).sum()))].mean(axis=0)
        if real
        else coords[labels >= -1].mean(axis=0)
    )
    eligible = np.flatnonzero(labels != UNCOVERED)
    d = np.linalg.norm(coords[eligible] - centre, axis=1)
    take = eligible[np.argsort(d)[: min(target, eligible.size)]]
    radius = float(d[np.argsort(d)[: take.size]].max())

    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    polygon = [
        [float(centre[0] + radius * np.cos(a)), float(centre[1] + radius * np.sin(a))]
        for a in t
    ]
    return take.tolist(), polygon


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stdout
    )
    logging.getLogger("numba").setLevel(logging.WARNING)

    rule("1. 스냅샷 적재")
    with timed("load"):
        ds = chemspace.load()
    print(ds)
    print("\nmanifest:", ds.manifest["counts"])
    print("\n도메인별 데이터 커버리지:")
    print(ds.coverage().to_string(index=False))

    print("\n모달리티 분포 (규칙 기반):")
    print(ds.compounds["modality"].value_counts().to_string())

    rule("2. 루트: 전체를 물성 도메인으로 배치")
    s = chemspace.ExplorationSession(ds)
    with timed("root/physchem"):
        n1 = s.root("physchem", note="전체 집합")
    print(s.breadcrumb(n1.node_id))
    print(f"군집 {len(n1.placement.cluster_ids)}개, notes={n1.placement.notes}")

    rule("3. 관심 영역 선택 -> 물성 기준 부분집합 (목표 1,200개)")
    idx, poly = lasso_like(s, n1.node_id, 1200)
    with timed("select"):
        n2 = s.select(
            n1.node_id, point_indices=idx, polygon=poly, how="드래그 선택", note="물성 관심 영역"
        )
    print(s.breadcrumb(n2.node_id))
    print(f"선택 {n2.n:,}개 (부모 {n1.n:,}개에서)")
    print("선택 영역 물성 프로파일:", s.domains["physchem"].profile(n2.compound_ids)["summary"])

    rule("3b. 전환 전에 비용을 재기 + 필터")
    # The question a user asks right before switching is "will there be
    # anything there", and it is answerable without placing anything.
    cov = s.domain_coverage(n2.node_id)
    for name, covered in cov.items():
        flag = "  ◀ 얇음" if covered / n2.n < 0.4 else ""
        print(f"  {s.domains[name].label:6} {covered:6,}/{n2.n:,}  "
              f"({covered / n2.n:5.1%}){flag}")
    drug = int(s.filter_mask(n2.node_id, "drug").sum())
    print()
    print(f"이 선택 안의 약물층: {drug:,}/{n2.n:,}")
    if 0 < drug < n2.n:
        with timed("restrict/drug"):
            nf = s.restrict(n2.node_id, "drug", note="약물층")
        print(s.breadcrumb(nf.node_id))
        after = s.domain_coverage(nf.node_id)["disease"]
        print(f"약물층으로 제한하면 질환 커버리지 "
              f"{cov['disease'] / n2.n:.1%} -> {after / nf.n:.1%}")
    else:
        print("(이 스냅샷에서는 선택 전체가 약물층이라 필터가 좁히지 않습니다)")

    rule("4. 도메인 전환: 같은 화합물을 질환 기준으로 재배치")
    with timed("switch/disease"):
        n3 = s.switch_domain(n2.node_id, "disease", note="질환 관점")
    print(s.breadcrumb(n3.node_id))
    assert n3.n == n2.n, "도메인 전환은 화합물 수를 바꾸지 않아야 한다"
    print(f"화합물 수 유지: {n2.n:,} -> {n3.n:,}  ✓")
    print(f"질환 데이터 있는 화합물: {n3.placement.n_covered:,}/{n3.n:,}")
    print(f"notes={n3.placement.notes}")

    rule("5. 핵심 검증: 물성 군집이 질환 관점에서 갈라지는가")
    phys = n2.placement.labels
    dis = n3.placement.labels
    both = (phys >= 0) & (dis >= 0)
    ari = adjusted_rand_score(phys[both], dis[both])
    ami = adjusted_mutual_info_score(phys[both], dis[both])
    print(f"비교 대상 {both.sum():,}개 (양쪽 모두 군집에 속한 화합물)")
    print(f"조정 랜드 지수 ARI = {ari:.3f}   (0에 가까우면 두 관점이 서로 독립)")
    print(f"조정 상호정보 AMI = {ami:.3f}")

    ct = pd.crosstab(
        pd.Series(phys[both], name="물성 군집"),
        pd.Series(dis[both], name="질환 군집"),
    )
    print("\n교차표 (행=물성 군집, 열=질환 군집):")
    print(ct.to_string())
    spread = (ct > 0).sum(axis=1)
    print("\n물성 군집별로 흩어진 질환 군집 수:")
    for cid, k in spread.items():
        print(f"  물성 군집 {cid} ({int(ct.loc[cid].sum()):,}개) -> 질환 군집 {k}개로 분리")
    print(
        f"\n=> 물성에서 한 덩어리였던 군집이 질환 관점에서 평균 "
        f"{spread.mean():.1f}개로 갈라짐"
    )

    rule("6. 질환 지도에서 다시 선택 -> 표적으로 재배치 (목표 300개)")
    idx2, poly2 = lasso_like(s, n3.node_id, 300)
    with timed("select"):
        n4 = s.select(
            n3.node_id, point_indices=idx2, polygon=poly2, how="드래그 선택", note="질환 관심 영역"
        )
    print(s.breadcrumb(n4.node_id))
    print("선택 영역 질환 프로파일:", s.domains["disease"].profile(n4.compound_ids)["summary"])
    with timed("switch/target"):
        n5 = s.switch_domain(n4.node_id, "target", note="표적 관점")
    print(s.breadcrumb(n5.node_id))
    print(f"표적 데이터 있는 화합물: {n5.placement.n_covered:,}/{n5.n:,}")

    rule("7. 모이어티로 한 번 더")
    with timed("switch/moiety"):
        n6 = s.switch_domain(n5.node_id, "moiety", note="모이어티 관점")
    print(s.breadcrumb(n6.node_id))
    print(f"notes={n6.placement.notes}")

    rule("8. 분기: n3으로 돌아가 다른 영역을 선택")
    # The opposite side of the same disease map: a genuinely different region,
    # so the two branches are worth comparing.
    taken = set(idx2)
    rest = [i for i in range(n3.n) if i not in taken]
    far = sorted(
        rest,
        key=lambda i: -float(
            np.linalg.norm(n3.placement.coords[i] - n3.placement.coords[idx2].mean(axis=0))
        ),
    )[:300]
    nb = s.select(n3.node_id, point_indices=far, how="드래그 선택", note="반대쪽 질환 영역")
    s.switch_domain(nb.node_id, "modality", note="모달리티 관점")
    print(s.breadcrumb(nb.node_id))
    print(f"n4의 형제 노드: {[x.node_id for x in s.siblings(n4.node_id)]}")
    print(f"두 분기는 서로 겹치지 않음: {not (set(n4.compound_ids) & set(nb.compound_ids))}")
    print(f"n4는 그대로 유지: {s.node(n4.node_id).n}개")

    rule("9. 탐색 경로 트리")
    print(s.tree(highlight=n6.node_id))

    rule("10. 단계 기록")
    print(s.history().to_string(index=False))

    rule("11. 군집 프로파일 (현재 노드: 모이어티)")
    print(s.cluster_table(n6.node_id).to_string())

    rule("11b. 이 선택은 무엇인가 — 다섯 도메인 동시 요약")
    print(s.describe(n4.node_id).to_string())

    rule("11c. 군집이 의미 있는가 — 귀무값과 비교")
    check_coherence(s, n5.node_id)

    rule("12. 저장/복원 왕복")
    out = s.save("data/sessions/smoke")
    s2 = chemspace.ExplorationSession.load(out, ds)
    same_tree = s2.tree() == s.tree()
    same_crumb = s2.breadcrumb(n6.node_id) == s.breadcrumb(n6.node_id)
    coords_match = np.allclose(
        s2.node(n6.node_id).placement.coords, s.node(n6.node_id).placement.coords
    )
    print(f"노드 수 {len(s2.nodes)} · 트리 일치 {same_tree} · 경로 일치 {same_crumb} "
          f"· 좌표 일치 {coords_match}")
    assert same_tree and same_crumb and coords_match, "저장/복원이 일치하지 않음"

    rule("13. 방어 동작 확인")
    checks = []
    try:
        s.select(n1.node_id, compound_ids=["NOT_A_REAL_ID"])
    except ValueError as e:
        checks.append(f"외부 화합물 선택 거부: {str(e)[:60]}")
    try:
        s.select(n1.node_id, clusters=[9_999])
    except ValueError as e:
        checks.append(f"빈 선택 거부: {str(e)[:60]}")
    try:
        s.switch_domain(n1.node_id, "nope")
    except KeyError as e:
        checks.append(f"없는 도메인 거부: {str(e)[:60]}")
    try:
        s.select(n1.node_id, clusters=[0], compound_ids=["x"])
    except ValueError as e:
        checks.append(f"선택 방식 중복 거부: {str(e)[:60]}")
    for c in checks:
        print("  ✓", c)
    assert len(checks) == 4, f"방어 동작 {len(checks)}/4만 작동"

    rule("14. 뷰어 코드 경로 (브라우저 없이)")
    check_viewer(s, n4.node_id)

    rule("완료")
    print(f"노드 {len(s.nodes)}개, 도메인 {len(s.domains)}종. ARI={ari:.3f}")
    return 0


class _Points:
    def __init__(self, inds):
        self.point_inds = list(inds)


class _Lasso:
    def __init__(self, xs, ys):
        self.xs, self.ys = xs, ys


class _Click:
    """The modifier state plotly hands a click callback."""

    def __init__(self, shift):
        self.shift = shift


def check_coherence(session, node_id: str) -> None:
    """ARI near zero says the domains disagree; it does not say either is real.

    Two random partitions score ARI 0 as well. This measures the other half on
    the node at hand: agreement with WHO's ATC classification, beside the same
    pipeline run on the same matrix with its feature columns shuffled.

    ATC is used because nobody involved in building these features chose it.
    The assertion is deliberately weak -- a few hundred compounds is a small
    sample and the point here is to catch a regression that turns the
    clustering into noise, not to re-derive the numbers in
    scripts/cluster_quality.py.
    """
    from chemspace.quality import coherence, null_coherence

    node = session.node(node_id)
    ids = node.compound_ids
    col = session.ds.compounds["atc_level1"].reindex(list(ids))
    ext = np.array(
        ["" if (v is None or v != v or ";" in str(v)) else str(v) for v in col],
        dtype=object,
    )
    feats = session.domains[node.domain].featurize(ids)
    real = coherence(feats, node.placement, domain=node.domain, external=ext)
    null = null_coherence(feats, domain=node.domain, external=ext)

    print(f"  도메인 {node.domain} · {node.n}개 · ATC 라벨 {real.n_labelled}개")
    if real.ami is None or null.ami is None:
        print("  ! ATC 라벨이 너무 적어 판정을 건너뜁니다")
        return
    print(f"    진짜  군집 {real.n_clusters:>2}  AMI {real.ami:+.3f}")
    print(f"    귀무  군집 {null.n_clusters:>2}  AMI {null.ami:+.3f}")
    assert real.ami > null.ami, (
        f"군집이 귀무값을 못 이깁니다 ({real.ami:.3f} <= {null.ami:.3f}) "
        f"-- 이 도메인의 군집이 잡음일 수 있습니다"
    )
    print(f"  ✓ 귀무값을 이깁니다 (+{real.ami - null.ami:.3f})")


def check_viewer(session, node_id: str) -> None:
    """Build the Jupyter viewer and drive a lasso selection through its handlers.

    No browser is involved: the point indices a real drag would deliver are fed
    straight to the callback, so the drag -> narrow -> redraw path is covered.
    """
    from chemspace.viewer import SpaceExplorer

    ex = SpaceExplorer(session, node_id=node_id)
    trace = ex.fig.data[0]
    n = session.node(node_id).n
    assert len(trace.x) == n, f"figure has {len(trace.x)} points, node has {n}"
    assert len(trace.customdata) == n
    print(f"  ✓ 뷰어 생성: {n:,}개 점, hover 데이터 {len(trace.customdata):,}행")

    assert ex.narrow_btn.disabled, "선택 전에는 좁히기 버튼이 비활성이어야 한다"
    picked = list(range(0, n, 3))[:200]
    ex._on_selection(trace, _Points(picked), _Lasso([0.0, 1.0, 1.0], [0.0, 0.0, 1.0]))
    assert not ex.narrow_btn.disabled, "선택 후에는 좁히기 버튼이 활성이어야 한다"
    print(f"  ✓ 드래그 선택 콜백: {len(picked)}개 선택, 버튼 활성화")

    # shift+click adds one compound to the lassoed region, clicking it again
    # takes it back out, and a plain click must leave the selection alone.
    outside = next(i for i in range(n) if i not in set(picked))
    ex._on_click(trace, _Points([outside]), _Click(shift=True))
    assert len(ex._selected) == len(picked) + 1, "shift+click did not add a compound"
    assert ex._polygon_partial, "the lasso trace should be marked as no longer complete"
    assert ex.fig.data[0].selectedpoints is not None, "the highlight was not updated"

    ex._on_click(trace, _Points([outside]), _Click(shift=True))
    assert len(ex._selected) == len(picked), "shift+click did not remove it again"

    ex._on_click(trace, _Points([outside]), _Click(shift=False))
    assert len(ex._selected) == len(picked), "a plain click must not change the selection"

    ex._on_click(trace, _Points([outside]), _Click(shift=True))
    picked_mixed = sorted(ex._selected)
    print(
        f"  ✓ shift+클릭: 추가/제거/무시 동작, 최종 {len(picked_mixed)}개 "
        f"(개별 조정 {ex._click_toggles}회)"
    )

    before = len(session.nodes)
    ex._do_narrow(None)
    assert len(session.nodes) == before + 1, "좁히기가 노드를 만들지 않았다"
    new_node = session.node(ex.current)
    assert new_node.n == len(picked_mixed)
    assert new_node.origin.detail.get("polygon"), "올가미 좌표가 기록되지 않았다"
    assert new_node.origin.detail.get("polygon_partial") is True, (
        "개별 조정이 섞인 선택인데 polygon_partial이 기록되지 않았다"
    )
    assert new_node.origin.detail["how"] == "드래그 + 개별 선택"
    print(
        f"  ✓ 좁히기: {ex.current} 생성 ({new_node.n}개), "
        f"올가미 좌표 + polygon_partial 기록됨"
    )

    ex.domain_dd.value = "modality"
    ex._do_switch(None)
    assert session.node(ex.current).domain == "modality"
    assert session.node(ex.current).n == new_node.n
    print(f"  ✓ 도메인 전환: {ex.current} (모달리티, {new_node.n}개 유지)")

    ex._on_color_pick({"new": "modality"})
    ex._do_back(None)
    assert ex.current == new_node.node_id
    print("  ✓ 색 변경 / 이전 단계 이동")


if __name__ == "__main__":
    raise SystemExit(main())
