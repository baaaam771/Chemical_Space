"use strict";

/* Front end for the exploration engine.
 *
 * It holds a session id, the current node id, and whatever the user has
 * lassoed. Everything else -- the tree, the placements, the profiles -- lives
 * on the server, same as in the Jupyter viewer. Narrowing and switching are
 * two POSTs; the page just redraws whatever node comes back. */

const PALETTE = [
  "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
  "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#7F7F7F",
  "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
  "#E6AB02", "#A6761D", "#666666", "#8DD3C7", "#FB8072",
];
const NOISE = -1, UNCOVERED = -2;
const NOISE_COLOR = "#C9CDD4", UNCOVERED_COLOR = "#E8EAED";

const state = {
  sessionId: null,
  nodeId: null,
  node: null,
  points: null,
  /* The selection is a Set of point indices within the current node, and that
   * is the entire contract the server cares about (`point_indices`) -- it
   * never asks how they were picked. So a lasso drag, a shift-click and a
   * cluster pick all write into the same place and mix freely. */
  selection: new Set(),
  polygon: null,
  polygonPartial: false,   // true once the polygon no longer *is* the selection
  clickToggles: 0,         // how many individual shift-clicks went into it
  domains: [],
  colorBy: "cluster",
  /* Per-domain coverage of the *current* node, fetched after it loads.
   * The whole point of holding it is to label the domain picker before the
   * click: a switch onto a domain that covers 91 of 1,200 compounds costs the
   * same 20 seconds as a useful one and leaves an empty map. */
  domainCoverage: null,
  drugLayer: null,
  /* Per-cluster characteristic label (the cluster's own-domain summary), keyed
   * by cluster id. Populated after the cluster profile loads, so the legend can
   * say "군집 3 · Breast Neoplasms …" instead of a bare "군집 3". */
  clusterLabels: null,
};

const $ = (id) => document.getElementById(id);

/* Plotly's click event carries the DOM event, but its selection events do not
 * reliably, so track the modifier ourselves rather than depending on it.
 * Clearing on blur matters: alt-tabbing away while holding shift would
 * otherwise leave every later drag in "add to selection" mode. */
let shiftHeld = false;
addEventListener("keydown", (e) => { if (e.key === "Shift") shiftHeld = true; });
addEventListener("keyup", (e) => { if (e.key === "Shift") shiftHeld = false; });
addEventListener("blur", () => { shiftHeld = false; });
const withShift = (ev) => shiftHeld || !!(ev && ev.event && ev.event.shiftKey);

/* Set while we are writing the selection back into the plot, so our own
 * relayout/restyle cannot come back round as if the user had done it. */
let applying = false;

/* ----------------------------------------------------------------- http */

async function api(path, options) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* keep the status line */ }
    /* A cookie that ran out mid-session is not an error to report, it is a
     * login to ask for. Without this the page would show "로그인이 필요합니다"
     * in a toast and otherwise sit there, looking broken. */
    if (res.status === 401 && !path.endsWith("/api/login")) {
      showGate();
      throw new Error("로그인이 만료되었습니다. 다시 로그인하세요.");
    }
    throw new Error(msg);
  }
  return res.json();
}

const post = (path, body) =>
  api(path, { method: "POST", body: JSON.stringify(body || {}) });

/* Poll a job to completion. Placement can take 75-105s, so this shows what is
 * running and how long it has been going rather than an opaque spinner. */
async function awaitJob(job, label) {
  showBusy(true, label || job.detail);
  try {
    let current = job;
    while (current.state === "pending" || current.state === "running") {
      $("busyElapsed").textContent = `${current.elapsed.toFixed(0)}초 경과`;
      await new Promise((r) => setTimeout(r, 700));
      current = await api(`/api/jobs/${job.job_id}`);
    }
    if (current.state === "error") throw new Error(current.error);
    return current.result;
  } finally {
    showBusy(false);
  }
}

function showBusy(on, text) {
  $("busy").classList.toggle("hidden", !on);
  if (text) $("busyText").textContent = text;
  if (!on) $("busyElapsed").textContent = "";
}

let toastTimer = null;
function toast(msg, isError) {
  const el = $("toast");
  el.textContent = msg;
  el.classList.toggle("error", !!isError);
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), isError ? 7000 : 3500);
}

/* ---------------------------------------------------------------- colour */

function colourPoints(pts) {
  const legend = new Map();
  let colors;

  if (state.colorBy === "cluster") {
    colors = pts.cluster.map((c) => {
      if (c === UNCOVERED) return UNCOVERED_COLOR;
      if (c === NOISE) return NOISE_COLOR;
      return PALETTE[c % PALETTE.length];
    });
    const counts = new Map();
    pts.cluster.forEach((c) => counts.set(c, (counts.get(c) || 0) + 1));
    [...counts.keys()].filter((c) => c >= 0).sort((a, b) => a - b)
      .forEach((c) => {
        const label = state.clusterLabels ? state.clusterLabels[c] : "";
        legend.set(`군집 ${c}`, [PALETTE[c % PALETTE.length], counts.get(c), label || ""]);
      });
    if (counts.has(NOISE)) legend.set("noise", [NOISE_COLOR, counts.get(NOISE)]);
    if (counts.has(UNCOVERED)) legend.set("데이터 없음", [UNCOVERED_COLOR, counts.get(UNCOVERED)]);
  } else {
    const field = pts[state.colorBy] ? state.colorBy : "modality";
    const values = pts[field].map((v) => v || "미분류");
    const counts = new Map();
    values.forEach((v) => counts.set(v, (counts.get(v) || 0) + 1));
    const order = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    const map = new Map(order.map(([k], i) => [k, PALETTE[i % PALETTE.length]]));
    colors = values.map((v) => map.get(v));
    order.forEach(([k, n]) => legend.set(k, [map.get(k), n]));
  }
  return { colors, legend };
}

function renderLegend(legend) {
  const rows = [...legend.entries()].slice(0, 30).map(
    ([name, [color, count, label]]) => {
      const tag = label
        ? `<span class="legend-label" title="${escapeHtml(label)}">${escapeHtml(label)}</span>`
        : "";
      return (
        `<div class="legend-row" title="${escapeHtml(label || name)}">` +
        `<span class="swatch" style="background:${color}"></span>` +
        `<span class="legend-name">${escapeHtml(name)}</span>` +
        tag +
        `<span class="count">${count.toLocaleString()}</span></div>`
      );
    }
  );
  if (legend.size > 30) rows.push(`<div class="muted small">… 외 ${legend.size - 30}개</div>`);
  $("legend").innerHTML = rows.join("");
}

/* The cluster profile carries each cluster's own-domain summary; lift it into
 * state so the legend can show what each cluster actually is, then repaint the
 * legend (the map was drawn before the profile arrived). */
function applyClusterLabels(clusters) {
  const dom = state.domains.find((d) => d.name === (state.node && state.node.domain));
  const col = dom && dom.label;
  const labels = {};
  if (col) {
    for (const row of clusters || []) {
      if (typeof row.cluster === "number" && row[col]) labels[row.cluster] = String(row[col]);
    }
  }
  state.clusterLabels = labels;
  if (state.colorBy === "cluster" && state.points) {
    renderLegend(colourPoints(state.points).legend);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* 으로 / 로, by whether the last syllable has a final consonant.
 * Mirrors chemspace/session/korean.py -- the server gets this right in the
 * breadcrumb, and a toast reading "질환로" right beside it is just wrong. */
function josaRo(word) {
  const ch = String(word).trim().slice(-1);
  const code = ch.charCodeAt(0);
  if (!(code >= 0xac00 && code <= 0xd7a3)) return `${word}(으)로`;
  const jong = (code - 0xac00) % 28;
  return jong === 0 || jong === 8 ? `${word}로` : `${word}으로`;  // 8 = ㄹ
}

/* ------------------------------------------------------------ hover card */

/* Plotly's native tooltip is text-only. To show the 2-D structure we turn its
 * hover text off (hoverinfo: "none") and position our own card, which pulls the
 * depiction from /api/compound/<id>/structure.svg -- rendered by RDKit on the
 * server and cached, so the coordinate payload carries no SMILES per point. */
let hoverHideTimer = null;
/* A plain click on a point, in lasso mode, arrives only as an empty
 * plotly_selected -- no plotly_click, and the hover events are too flaky to
 * rely on. So we track the raw mousedown position and whether the cursor then
 * dragged; on an empty selection that was a click (not a drag), we find the
 * nearest point to that pixel ourselves and open its detail. */
let clickDown = null;         // {x, y, t} of the last mousedown over the plot
let draggedSinceDown = false; // true once the cursor moved far enough to be a drag

function hoverCardEl() {
  let el = $("hovercard");
  if (!el) {
    el = document.createElement("div");
    el.id = "hovercard";
    el.hidden = true;
    document.body.appendChild(el);
  }
  return el;
}

function showHoverCard(ev) {
  const pt = ev && ev.points && ev.points[0];
  const pts = state.points;
  if (!pt || !pts) return;
  const i = pt.pointIndex ?? pt.pointNumber;
  if (i == null) return;
  const id = pts.chembl_id[i];
  if (!id) return;

  clearTimeout(hoverHideTimer);
  const el = hoverCardEl();
  const name = pts.name[i];
  const cl = pts.cluster[i];
  const clLabel = cl === UNCOVERED ? "미커버" : cl === NOISE ? "잡음" : `군집 ${cl}`;
  // Rebuild only when the compound changes, so the <img> is not reloaded while
  // the cursor merely moves within one point's hover region.
  if (el.dataset.id !== id) {
    el.dataset.id = id;
    el.innerHTML =
      `<img class="hc-struct" alt="" ` +
      `src="/api/compound/${encodeURIComponent(id)}/structure.svg" ` +
      `onerror="this.style.display='none'">` +
      `<div class="hc-body">` +
      `<div class="hc-title">${escapeHtml(id)}</div>` +
      (name ? `<div class="hc-name">${escapeHtml(name)}</div>` : "") +
      `<div class="hc-props">${escapeHtml(pts.modality[i])} · ` +
      `MW ${pts.mw[i]} · cLogP ${pts.clogp[i]}</div>` +
      `<div class="hc-cluster">${clLabel}</div>` +
      `</div>`;
  }
  el.hidden = false;
  positionHoverCard(el, ev.event);
}

function positionHoverCard(el, mouse) {
  if (!mouse) return;
  const pad = 14;
  const w = el.offsetWidth || 280;
  const h = el.offsetHeight || 180;
  let x = mouse.clientX + pad;
  let y = mouse.clientY + pad;
  if (x + w > window.innerWidth - 8) x = mouse.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = mouse.clientY - h - pad;
  el.style.left = `${Math.max(8, x)}px`;
  el.style.top = `${Math.max(8, y)}px`;
}

function hideHoverCard() {
  const el = $("hovercard");
  if (!el) return;
  // A short delay stops the card flickering as the cursor crosses between
  // adjacent points: plotly fires unhover an instant before the next hover.
  hoverHideTimer = setTimeout(() => {
    el.hidden = true;
    el.dataset.id = "";
  }, 90);
}

/* --------------------------------------------------------- compound modal */

/* Click a point (or a 화합물 row) to inspect one compound: a large structure
 * plus every value and its disease / target annotations. The hover card is the
 * glance; this is the full read. */
async function openCompoundModal(cid) {
  if (!cid) return;
  const backdrop = $("cmodal-backdrop");
  const box = $("cmodal");
  box.innerHTML = '<div class="muted">불러오는 중…</div>';
  backdrop.hidden = false;
  try {
    const rec = await api(`/api/compound/${encodeURIComponent(cid)}`);
    box.innerHTML = renderCompoundDetail(rec);
  } catch (e) {
    box.innerHTML = `<div class="muted">불러오지 못했습니다: ${escapeHtml(cid)}</div>`;
  }
}

function closeCompoundModal() {
  const b = $("cmodal-backdrop");
  if (b) b.hidden = true;
}

function renderCompoundDetail(r) {
  const chips = (arr) =>
    arr && arr.length
      ? `<div class="cmodal-chips">${arr
          .map((x) => `<span class="chip">${escapeHtml(x)}</span>`)
          .join("")}</div>`
      : '<div class="cmodal-empty">데이터 없음</div>';
  const props = Object.entries(r.props || {})
    .map(
      ([k, v]) =>
        `<div><div class="k">${escapeHtml(k)}</div>` +
        `<div class="v">${v == null ? "–" : escapeHtml(String(v))}</div></div>`
    )
    .join("");
  const sec = (title, body) => `<div class="cmodal-sec"><h4>${title}</h4>${body}</div>`;
  return (
    `<div class="cmodal-head">` +
    `<img class="cmodal-struct" alt="" ` +
    `src="/api/compound/${encodeURIComponent(r.chembl_id)}/structure.svg" ` +
    `onerror="this.style.display='none'">` +
    `<div class="cmodal-id"><h3>${escapeHtml(r.name || r.chembl_id)}</h3>` +
    `<div class="sub">${escapeHtml(r.chembl_id)}` +
    `${r.modality ? " · " + escapeHtml(r.modality) : ""}</div>` +
    (r.smiles ? `<div class="smiles">${escapeHtml(r.smiles)}</div>` : "") +
    `</div><button id="cmodal-close" title="닫기">×</button></div>` +
    `<div class="cmodal-grid">${props}</div>` +
    sec("질환 (적응증)", chips(r.diseases)) +
    (r.areas && r.areas.length ? sec("치료영역 (ATC)", chips(r.areas)) : "") +
    sec("표적 단백질", chips(r.targets)) +
    (r.families && r.families.length ? sec("단백질 계열", chips(r.families)) : "") +
    (r.murcko_smiles
      ? sec("스캐폴드", `<div class="smiles">${escapeHtml(r.murcko_smiles)}</div>`)
      : "")
  );
}

/* ------------------------------------------------------------------ plot */

/* Pan and zoom belong to the node, not to the redraw. A new node is a fresh
 * placement with its own coordinate range, so it opens fitted to everything;
 * redrawing the *same* node -- recolouring -- has to leave the user where
 * they were, or every colour change throws away the zoom they just made. */
let drawnNode = null;

function currentView() {
  const lay = $("plot").layout;
  if (drawnNode !== state.nodeId || !lay || !lay.xaxis || !lay.xaxis.range) return null;
  return { x: lay.xaxis.range.slice(), y: lay.yaxis.range.slice() };
}

function drawPlot(pts) {
  const { colors, legend } = colourPoints(pts);
  renderLegend(legend);

  const trace = {
    type: "scattergl",
    mode: "markers",
    x: pts.x,
    y: pts.y,
    marker: {
      size: pts.x.length > 6000 ? 4 : 6,
      color: colors,
      opacity: 0.8,
      line: { width: 0 },
    },
    // Native hover text is off: a custom card (showHoverCard) draws the 2-D
    // structure alongside the id and properties, which a text-only tooltip
    // cannot do.
    hoverinfo: "none",
    // Stated rather than left to plotly's defaults, because a shift-click
    // selection has to look the same as a lassoed one.
    selected: { marker: { opacity: 1 } },
    unselected: { marker: { opacity: 0.15 } },
    selectedpoints: state.selection.size ? [...state.selection] : null,
  };

  const view = currentView();

  const layout = {
    dragmode: "lasso",
    margin: { l: 8, r: 8, t: 8, b: 8 },
    // autorange is stated rather than left out: a new node is a fresh
    // placement with its own coordinate range, so a zoom left over from the
    // previous node would open the new map on an empty patch. Redrawing the
    // same node puts the view back instead -- see currentView().
    xaxis: view
      ? { visible: false, range: view.x, autorange: false }
      : { visible: false, autorange: true },
    yaxis: view
      ? { visible: false, range: view.y, autorange: false, scaleanchor: "x" }
      : { visible: false, autorange: true, scaleanchor: "x" },
    showlegend: false,
    plot_bgcolor: "#fff",
    paper_bgcolor: "#fff",
    hoverlabel: { font: { size: 12 } },
  };

  Plotly.react("plot", [trace], layout, {
    responsive: true,
    displaylogo: false,
    // autoScale2d is the "fit everything" button. Zooming into an empty patch
    // is easy and looks exactly like a broken plot, so leave the way back.
    // Wheel zoom. Plotly calls preventDefault on the wheel event while the
    // cursor is over the plot, so the page behind it does not scroll away.
    scrollZoom: true,
    modeBarButtonsToRemove: ["toggleSpikelines"],
  });
  drawnNode = state.nodeId;

  const plot = $("plot");
  [
    "plotly_selected", "plotly_deselect", "plotly_click", "plotly_doubleclick",
    "plotly_hover", "plotly_unhover",
  ].forEach((e) => plot.removeAllListeners?.(e));

  plot.on("plotly_hover", (ev) => showHoverCard(ev));
  plot.on("plotly_unhover", () => hideHoverCard());
  if (!plot._hoverLeaveWired) {
    plot.addEventListener("mouseleave", () => hideHoverCard());
    plot.addEventListener("mousedown", (e) => {
      clickDown = { x: e.clientX, y: e.clientY, t: performance.now() };
      draggedSinceDown = false;
      hideHoverCard();
    });
    plot.addEventListener("mousemove", (e) => {
      if (clickDown && Math.abs(e.clientX - clickDown.x) + Math.abs(e.clientY - clickDown.y) > 6) {
        draggedSinceDown = true;
      }
    });
    plot._hoverLeaveWired = true;
  }

  /* Plotly's own double-click reset goes back to whatever range was in force
   * when the plot was last built -- and a recolour rebuilds it at the current
   * zoom, so after one recolour the gesture stops zooming out. Ask for
   * autorange instead: it always fits everything, whatever happened before. */
  plot.on("plotly_doubleclick", () => {
    Plotly.relayout(plot, { "xaxis.autorange": true, "yaxis.autorange": true });
  });

  plot.on("plotly_selected", (ev) => {
    if (applying) return;   // our own restyle, not a gesture
    // Plotly fires this with no payload when a selection is cleared -- and, in
    // lasso mode, a plain click on a point lands here too (empty selection,
    // never plotly_click). If a point is under the cursor, treat it as
    // click-to-inspect: open the detail and keep the current lasso selection.
    if (!ev || !ev.points) {
      // An empty selection that was a plain click (no drag), landing on a
      // point, is a request to inspect that compound -- open it and keep the
      // current selection rather than clearing.
      if (
        !shiftHeld &&
        clickDown &&
        !draggedSinceDown &&
        performance.now() - clickDown.t < 500
      ) {
        const i = nearestPoint(clickDown.x, clickDown.y);
        clickDown = null;
        if (i >= 0) {
          const id = state.points && state.points.chembl_id[i];
          if (id) {
            openCompoundModal(id);
            return;
          }
        }
      }
      return shiftHeld ? undefined : clearSelection();
    }
    const idx = ev.points.map((p) => p.pointIndex ?? p.pointNumber);
    const poly = lassoPolygon(ev);
    if (withShift(ev) && state.selection.size) {
      // A second region instead of a replacement. Plotly may already have
      // unioned them itself; taking the union again is harmless either way.
      setSelection([...state.selection, ...idx], {
        polygon: state.polygon || poly,
        partial: true,
        toggles: state.clickToggles,
      });
    } else {
      setSelection(idx, { polygon: poly });
    }
  });

  plot.on("plotly_click", (ev) => {
    const pt = ev && ev.points && ev.points[0];
    if (applying || !pt) return;
    const i = pt.pointIndex ?? pt.pointNumber;
    if (withShift(ev)) {
      // Shift-click toggles one point into/out of the selection.
      const next = new Set(state.selection);
      next.has(i) ? next.delete(i) : next.add(i);
      setSelection(next, {
        polygon: state.polygon,
        partial: state.polygon !== null,
        toggles: state.clickToggles + 1,
      });
      return;
    }
    // Plain-click inspection is handled in plotly_selected: in lasso mode plotly
    // does not fire plotly_click for a plain click on a point, only an empty
    // plotly_selected.
  });

  /* A click with no drag counts as "clear the selection" in box/lasso mode. */
  plot.on("plotly_deselect", () => {
    if (!applying && !shiftHeld) clearSelection();
  });
}

/* The index of the plotted point nearest a viewport pixel, or -1 if none is
 * within ~12px. Used to turn a click into a compound to inspect without relying
 * on plotly's own (flaky, in lasso mode) hit testing. */
function nearestPoint(clientX, clientY) {
  const plot = $("plot");
  const pts = state.points;
  if (!plot._fullLayout || !pts || !pts.x.length) return -1;
  const r = plot.getBoundingClientRect();
  const xa = plot._fullLayout.xaxis, ya = plot._fullLayout.yaxis;
  const px = clientX - r.left - xa._offset;
  const py = clientY - r.top - ya._offset;
  let best = -1, bestD = 12 * 12;
  for (let i = 0; i < pts.x.length; i++) {
    const dx = xa.l2p(pts.x[i]) - px;
    const dy = ya.l2p(pts.y[i]) - py;
    const d = dx * dx + dy * dy;
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

function lassoPolygon(ev) {
  if (ev.lassoPoints && ev.lassoPoints.x) {
    return ev.lassoPoints.x.map((x, i) => [x, ev.lassoPoints.y[i]]);
  }
  if (ev.range && ev.range.x) {
    const [x0, x1] = ev.range.x, [y0, y1] = ev.range.y;
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  }
  return null;
}

/* ------------------------------------------------------------- selection */

function setSelection(indices, { polygon = null, partial = false, toggles = 0 } = {}) {
  state.selection = new Set(indices);
  state.polygon = polygon;
  state.polygonPartial = partial;
  state.clickToggles = toggles;
  applySelection();
}

const clearSelection = () => setSelection([]);

function applySelection() {
  const n = state.selection.size;
  $("narrowBtn").disabled = n === 0;
  $("clearSelBtn").disabled = n === 0;

  /* Drive plotly's highlight from our Set. A lasso drag sets selectedpoints
   * internally but a shift-click does not, so state it every time rather than
   * trusting whichever gesture happened last. */
  const gd = $("plot");
  if (gd && gd.data && gd.data.length) {
    const pts = n ? [...state.selection] : null;
    /* While a drawn selection shape is still on the plot, plotly re-derives
     * selectedpoints from that shape and overwrites whatever we restyle --
     * so a shift-clicked point would be in the selection but not highlighted,
     * and a cleared selection would leave every point dimmed behind a stale
     * outline. Dropping the shape hands the highlight back to us. The outline
     * goes with it, which is the right trade: an outline that no longer
     * matches the selection is worse than no outline.
     *
     * A plain lasso result keeps its shape -- there plotly is right and the
     * outline is worth having. */
    const shaped = (gd.layout.selections || []).length > 0;
    const takeOver = shaped && (state.polygonPartial || n === 0);
    applying = true;
    const step = takeOver
      ? Plotly.relayout(gd, { selections: [] })
      : Promise.resolve();
    step
      .then(() => Plotly.restyle(gd, { selectedpoints: [pts] }))
      .finally(() => { applying = false; });
  }

  const note = $("selnote");
  note.classList.toggle("active", n > 0);
  note.textContent = n
    ? `${describeSelection(n)} — ‘선택 영역으로 좁히기’를 누르면 다음 단계가 만들어집니다.`
    : "올가미(lasso)로 영역을 드래그하세요. shift+클릭으로 화합물을 하나씩 더하거나 뺄 수 있고, shift+드래그로 영역을 덧붙일 수 있습니다. 휠로 확대·축소, 더블클릭으로 전체 보기.  권장 경로는 물성·모이어티 → 표적 → (좁아진 뒤) 질환입니다. 질환을 먼저 보려면 약물층만으로 제한하세요 — 적응증 주석은 그 층에만 있습니다.";
}

function describeSelection(n) {
  const total = n.toLocaleString();
  if (!state.clickToggles) return `영역 선택: ${total}개`;
  if (!state.polygon) return `개별 선택: ${total}개`;
  return `선택: ${total}개 (올가미 + 개별 ${state.clickToggles.toLocaleString()}회 조정)`;
}

function selectionHow() {
  if (!state.clickToggles) return "드래그 선택";
  return state.polygon ? "드래그 + 개별 선택" : "개별 선택";
}

/* ---------------------------------------------------------------- tables */

function renderTable(rows, container) {
  const el = $(container);
  if (!rows || !rows.length) {
    el.innerHTML = '<div class="muted">표시할 내용이 없습니다.</div>';
    return;
  }
  const cols = Object.keys(rows[0]);
  const head = cols.map((c) => `<th>${escapeHtml(c)}</th>`).join("");
  const body = rows.map((r) => {
    const special = r.cluster === NOISE || r.cluster === UNCOVERED;
    const tds = cols.map((c) => {
      const v = r[c];
      const num = typeof v === "number";
      return `<td class="${num ? "num" : ""}">${escapeHtml(v === null ? "" : v)}</td>`;
    }).join("");
    // Rows that carry a compound id (the 화합물 table) open the detail modal.
    const cid = r.chembl_id;
    const cls = `${special ? "special" : ""}${cid ? " row-compound" : ""}`.trim();
    const attr = cid ? ` data-cid="${escapeHtml(String(cid))}"` : "";
    return `<tr class="${cls}"${attr}>${tds}</tr>`;
  }).join("");
  el.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

/* The 구성 종류 view: per domain, every discrete kind present in this node with
 * its count -- the full list the top-3 summary only previews. */
function renderComposition(data, container) {
  const el = $(container);
  if (!data || !data.length) {
    el.innerHTML = '<div class="muted">표시할 내용이 없습니다.</div>';
    return;
  }
  el.innerHTML = data
    .map((d) => {
      const sections = d.sections
        .map((s) => {
          const rows = s.items
            .map(
              (it) =>
                `<div class="comp-row"><span class="comp-name" title="${escapeHtml(it.name)}">` +
                `${escapeHtml(it.name)}</span>` +
                `<span class="comp-count">${Number(it.count).toLocaleString()}</span></div>`
            )
            .join("");
          const rest = s.total - s.items.length;
          const more = rest > 0 ? `<div class="muted small">… 외 ${rest.toLocaleString()}종</div>` : "";
          return (
            `<div class="comp-section"><div class="comp-title">${escapeHtml(s.title)} ` +
            `<span class="muted">${s.total.toLocaleString()}종</span></div>${rows}${more}</div>`
          );
        })
        .join("");
      return (
        `<div class="comp-domain"><h4>${escapeHtml(d.domain)}${d.current ? " ◀" : ""}</h4>` +
        `${sections}</div>`
      );
    })
    .join("");
}

/* ----------------------------------------------------------------- views */

async function loadNode(nodeId) {
  state.nodeId = nodeId;
  const base = `/api/sessions/${state.sessionId}/nodes/${nodeId}`;
  const [node, points] = await Promise.all([api(base), api(`${base}/points`)]);
  state.node = node;
  state.points = points;
  state.clusterLabels = null;  // stale from the previous node until clusters reload

  $("breadcrumb").textContent = node.breadcrumb;
  $("domainSelect").value = node.domain;
  $("backBtn").disabled = !node.parent_id;
  clearSelection();
  drawPlot(points);
  renderStatus(node);
  remember();
  await Promise.all([
    refreshSessionViews(),
    refreshPanels(nodeId),
    loadDomainCoverage(nodeId),
  ]);
}

/* Coverage is a separate request on purpose. It featurises every domain, which
 * is seconds even at 73,864 compounds, but the map should already be on screen
 * by then -- so it must not sit in front of the first paint. A failure here
 * leaves the picker unlabelled rather than breaking the node. */
async function loadDomainCoverage(nodeId) {
  state.domainCoverage = null;
  state.drugLayer = null;
  renderDomainOptions();
  try {
    const cov = await api(
      `/api/sessions/${state.sessionId}/nodes/${nodeId}/domain-coverage`
    );
    if (state.nodeId !== nodeId) return;   // user moved on while we waited
    state.domainCoverage = cov;
    state.drugLayer = cov.drug_layer;
  } catch (err) {
    console.warn("domain coverage unavailable", err);
  }
  renderDomainOptions();
}

function coverageOf(name) {
  if (!state.domainCoverage) return null;
  return state.domainCoverage.domains.find((d) => d.name === name) || null;
}

/* Write the coverage into the option labels, and the warning beside them.
 * Thresholds are deliberately blunt: under 15% the next map is mostly the
 * uncovered band, 15-40% is thin enough to mislead. They are a nudge, never a
 * block -- a 7.6% disease map is still the honest picture of that selection. */
function renderDomainOptions() {
  const sel = $("domainSelect");
  const chosen = sel.value;
  sel.innerHTML = state.domains
    .map((d) => {
      const c = coverageOf(d.name);
      const suffix = c ? `  ·  ${(c.fraction * 100).toFixed(1)}%` : "";
      return `<option value="${d.name}" title="${escapeHtml(d.description)}">` +
             `${escapeHtml(d.label)} (${d.name})${suffix}</option>`;
    })
    .join("");
  sel.value = chosen || (state.node ? state.node.domain : "");
  renderDomainNote();

  const node = state.node;
  const drug = state.drugLayer;
  $("drugBtn").disabled = !(node && drug !== null && drug > 0 && drug < node.n);
  if (node && drug !== null) {
    $("drugBtn").textContent =
      drug < node.n ? `약물층만 (${drug.toLocaleString()})` : "약물층만";
  }
  const own = node ? coverageOf(node.domain) : null;
  $("annotatedBtn").disabled =
    !(node && own && own.covered > 0 && own.covered < node.n);
  $("annotatedBtn").textContent =
    own && own.covered < (node ? node.n : 0)
      ? `주석 있는 것만 (${own.covered.toLocaleString()})`
      : "주석 있는 것만";
}

function renderDomainNote() {
  const el = $("domcov");
  const target = $("domainSelect").value;
  const node = state.node;
  if (!node || !state.domainCoverage || target === node.domain) {
    el.textContent = "";
    el.className = "muted small";
    return;
  }
  const c = coverageOf(target);
  if (!c) { el.textContent = ""; return; }
  const pct = (c.fraction * 100).toFixed(1);
  const head =
    `전환하면 ${c.covered.toLocaleString()}/${node.n.toLocaleString()}개 (${pct}%)`;
  // "만 ... 오릅니다" reads as a complaint, so it is only right when the
  // number is one: at 98.4% the same sentence sounds like a warning about
  // nothing.
  el.textContent =
    c.fraction < 0.4
      ? `${head}만 지도에 오릅니다 — 나머지는 미커버 띠로 갑니다`
      : `${head}가 지도에 오릅니다`;
  el.className =
    c.fraction < 0.15 ? "danger small" : c.fraction < 0.4 ? "warn small" : "muted small";
}

function renderStatus(node) {
  const bits = [
    `<b>${escapeHtml(node.domain_label)}</b>`,
    `${node.n.toLocaleString()}개 화합물`,
    `${node.clusters}군집`,
    `배치 ${escapeHtml(node.embed_method)}`,
    `군집 ${escapeHtml(node.cluster_method)}`,
  ];
  if (node.noise) bits.push(`noise ${node.noise.toLocaleString()}개`);
  if (node.n_covered < node.n) {
    bits.push(
      `<span class="warn">이 도메인에 데이터 있는 화합물 ` +
      `${node.n_covered.toLocaleString()}/${node.n.toLocaleString()}</span>`
    );
  }
  if (node.degenerate) bits.push(`<span class="danger">${escapeHtml(node.degenerate)}</span>`);
  if (node.from_cache) bits.push(`<span class="muted">캐시에서 불러옴</span>`);
  $("status").innerHTML = bits.join(" · ");
}

async function refreshSessionViews() {
  const s = await api(`/api/sessions/${state.sessionId}`);
  const sel = $("nodeSelect");
  sel.innerHTML = s.nodes
    .map((n) => `<option value="${n.node_id}">[${n.node_id}] ${escapeHtml(n.description)}</option>`)
    .join("");
  sel.value = state.nodeId;
  $("treeText").textContent = s.tree_text;
}

async function refreshPanels(nodeId) {
  const base = `/api/sessions/${state.sessionId}/nodes/${nodeId}`;
  const [describe, composition, clusters, compounds] = await Promise.all([
    api(`${base}/describe`),
    api(`${base}/composition`),
    api(`${base}/clusters`),
    api(`${base}/compounds?limit=200`),
  ]);
  renderTable(describe, "panel-describe");
  renderComposition(composition, "panel-composition");
  renderTable(clusters, "panel-clusters");
  renderTable(compounds, "panel-compounds");
  applyClusterLabels(clusters);
}

/* --------------------------------------------------------------- actions */

async function guarded(fn) {
  try {
    await fn();
  } catch (err) {
    toast(err.message, true);
    console.error(err);
  }
}

function wire() {
  $("narrowBtn").onclick = () => guarded(async () => {
    if (!state.selection.size) return;
    const job = await post(`/api/sessions/${state.sessionId}/select`, {
      node_id: state.nodeId,
      point_indices: [...state.selection],
      polygon: state.polygon,
      polygon_partial: state.polygonPartial,
      how: selectionHow(),
    });
    const newId = await awaitJob(job);
    await loadNode(newId);
    toast(`${newId} 생성 — ${state.node.n.toLocaleString()}개`);
  });

  $("clearSelBtn").onclick = () => clearSelection();

  /* A filter is the same kind of move as a lasso -- fewer compounds, same
   * domain, re-placed -- so it goes through the same job/await path and lands
   * as its own node in the tree. */
  const restrict = (kind, domain) => guarded(async () => {
    const job = await post(`/api/sessions/${state.sessionId}/restrict`, {
      node_id: state.nodeId, kind, domain: domain || null,
    });
    const newId = await awaitJob(job);
    await loadNode(newId);
    toast(`${newId} 생성 — ${state.node.n.toLocaleString()}개`);
  });
  $("drugBtn").onclick = () => restrict("drug");
  $("annotatedBtn").onclick = () => restrict("annotated", state.node.domain);

  /* Read the modifier off the mouse event itself, in the capture phase so it
   * is set before plotly's own handlers run. Keyboard tracking alone is not
   * enough -- a shift-click can carry shiftKey on the mouse event without any
   * keydown reaching this window -- and plotly_deselect, which fires on
   * mouseup before plotly_click, has to know. The listener goes here rather
   * than in drawPlot because #plot outlives every re-render. */
  ["mousedown", "mouseup"].forEach((t) =>
    $("plot").addEventListener(t, (e) => { shiftHeld = e.shiftKey; }, true)
  );

  $("switchBtn").onclick = () => guarded(async () => {
    const domain = $("domainSelect").value;
    if (domain === state.node.domain) return toast("이미 같은 도메인입니다.", true);
    const job = await post(`/api/sessions/${state.sessionId}/switch`, {
      node_id: state.nodeId, domain,
    });
    const newId = await awaitJob(job);
    await loadNode(newId);
    toast(
      `${newId} — ${josaRo(state.node.domain_label)} 재배치 ` +
      `(${state.node.n.toLocaleString()}개 유지)`
    );
  });

  $("backBtn").onclick = () => guarded(async () => {
    if (state.node.parent_id) await loadNode(state.node.parent_id);
  });

  $("nodeSelect").onchange = (e) => guarded(() => loadNode(e.target.value));

  $("domainSelect").onchange = () => renderDomainNote();

  $("colorSelect").onchange = (e) => {
    state.colorBy = e.target.value;
    if (state.points) drawPlot(state.points);
  };

  $("saveBtn").onclick = () => guarded(async () => {
    /* Ask for a name. It used to send "web" for everybody, which on a shared
     * server meant the second person to save destroyed the first person's
     * tree -- same path, no warning. Saves now live under the logged-in
     * user, so a name only has to be unique to that person. */
    const suggested = `run-${new Date().toISOString().slice(0, 10).replace(/-/g, "")}`;
    const name = prompt("저장 이름 (영문·숫자·_·-)", suggested);
    if (name === null) return;
    const res = await post(`/api/sessions/${state.sessionId}/save`, { name: name.trim() });
    toast(`저장됨: ${res.user}/${res.name} (노드 ${res.nodes}개)`);
  });

  document.querySelectorAll(".tab").forEach((btn) => {
    btn.onclick = () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".panel").forEach((p) => p.classList.add("hidden"));
      btn.classList.add("active");
      $(`panel-${btn.dataset.tab}`).classList.remove("hidden");
      setTabHelp(btn.dataset.tab);
    };
  });
  setTabHelp("describe");  // the tab that starts active

  // A row in the 화합물 table opens the same detail modal as a map point click.
  $("panel-compounds").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-cid]");
    if (tr) openCompoundModal(tr.dataset.cid);
  });

  // Close the detail modal: backdrop click, the × button, or Escape.
  $("cmodal-backdrop").addEventListener("click", (e) => {
    if (e.target.id === "cmodal-backdrop") closeCompoundModal();
  });
  $("cmodal").addEventListener("click", (e) => {
    if (e.target.id === "cmodal-close") closeCompoundModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeCompoundModal();
  });
}

/* What each tab is for and on what basis -- shown under the tab row so the
 * criteria are not a mystery. The two altitudes: describe/composition/clusters
 * summarise the *group*, compounds lists the *individuals*. */
const TAB_HELP = {
  describe:
    "이 선택(노드) 전체를 도메인별로 한 줄 요약합니다. 각 도메인에서 가장 흔한 상위 5개와 커버리지(데이터 있는 비율)를 보여줍니다.",
  composition:
    "이 선택에 들어 있는 모든 종류를 도메인별로 빠짐없이 나열합니다 — 질병·치료영역·표적·계열·스캐폴드·모달리티(개수 순), 물성은 기술자 중앙값.",
  clusters:
    "지도에서 밀도로 자동으로 나뉜 각 군집이 무엇인지 한 줄씩 보여줍니다. 군집마다 그 도메인·물성·모달리티의 대표 특징.",
  tree: "지금까지 좁혀온 단계(루트 → 선택 → 전환 → 필터)를 트리로 보여줍니다.",
  compounds:
    "이 노드의 개별 화합물을 앞에서부터 200개까지 나열합니다(정렬·선별 없는 미리보기). 행을 클릭하면 구조와 세부 값을 크게 봅니다.",
};

function setTabHelp(tab) {
  const el = $("tab-help");
  if (el) el.textContent = TAB_HELP[tab] || "";
}

/* ------------------------------------------------------------------ boot */

/* Remember where we were, so a reload does not throw the exploration away.
 * Sessions live in server memory, so a resume can legitimately fail (restart,
 * eviction); that is not an error, it just means starting over. */
const STORE_KEY = "chemspace.session";

function remember() {
  try {
    localStorage.setItem(
      STORE_KEY, JSON.stringify({ sessionId: state.sessionId, nodeId: state.nodeId })
    );
  } catch (_) { /* private mode, quota -- resuming is a convenience, not a feature */ }
}

async function tryResume() {
  let saved;
  try {
    saved = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
  } catch (_) { return false; }
  if (!saved || !saved.sessionId) return false;
  try {
    const s = await api(`/api/sessions/${saved.sessionId}`);
    if (!s.nodes.length) return false;
    state.sessionId = saved.sessionId;
    const known = new Set(s.nodes.map((n) => n.node_id));
    await loadNode(known.has(saved.nodeId) ? saved.nodeId : s.nodes[s.nodes.length - 1].node_id);
    toast(`이전 세션을 이어받았습니다 (노드 ${s.nodes.length}개)`);
    return true;
  } catch (_) {
    return false;
  }
}

/* ---------------------------------------------------------------- 로그인 */

function showGate(message) {
  $("gate").classList.remove("hidden");
  $("loginError").textContent = message || "";
  const name = $("loginName");
  if (name && !name.value) name.focus();
}

function hideGate() {
  $("gate").classList.add("hidden");
  $("loginPassword").value = "";
  $("loginError").textContent = "";
}

function wireLogin() {
  $("loginForm").onsubmit = async (ev) => {
    ev.preventDefault();
    const btn = $("loginBtn");
    btn.disabled = true;
    try {
      const { user } = await post("/api/login", {
        name: $("loginName").value.trim(),
        password: $("loginPassword").value,
      });
      hideGate();
      $("whoami").textContent = user;
      await startSession();
    } catch (err) {
      $("loginError").textContent = err.message;
      $("loginPassword").value = "";
      $("loginPassword").focus();
    } finally {
      btn.disabled = false;
    }
  };

  $("logoutBtn").onclick = () => guarded(async () => {
    await post("/api/logout");
    /* Drop the remembered node too: the next person at this browser must not
     * land on someone else's exploration, and the ids would 404 for them
     * anyway now that sessions are owned. */
    try { localStorage.removeItem(STORE_KEY); } catch (_) {}
    state.sessionId = null;
    state.nodeId = null;
    $("whoami").textContent = "";
    showGate("로그아웃했습니다.");
  });
}

/* Everything after a successful login: resume what this browser was doing,
 * or place the root. Split out of boot() because logging in again has to run
 * it without reloading the page. */
async function startSession() {
  const meta = await api("/api/meta");
  state.domains = meta.domains;
  $("meta").textContent =
    `${meta.chembl_version} · ${meta.compounds.toLocaleString()}개 화합물 · 도메인 ${meta.domains.length}종`;
  renderDomainOptions();

  if (await tryResume()) return;

  const { session_id } = await post("/api/sessions");
  state.sessionId = session_id;

  const job = await post(`/api/sessions/${session_id}/root`, { domain: "physchem", note: "전체 집합" });
  const rootId = await awaitJob(job);
  await loadNode(rootId);
}

async function boot() {
  wire();
  wireLogin();

  /* Ask who we are before anything else. A valid cookie from a previous visit
   * means the page can go straight to the map; a 401 is the normal first
   * visit, not a failure, so it raises the gate instead of a red toast. */
  let user = null;
  try {
    ({ user } = await api("/api/me"));
  } catch (_) {
    showGate();
    return;
  }
  $("whoami").textContent = user;
  hideGate();

  try {
    await startSession();
  } catch (err) {
    showBusy(false);
    toast(`시작 실패: ${err.message}`, true);
    console.error(err);
  }
}

boot();
