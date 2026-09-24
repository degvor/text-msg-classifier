/* Dashboard behaviour.
 *
 * Three things the previous version got wrong, all of them here on purpose:
 *
 *  - Not one of its eleven fetch call sites had a `.catch()` or an `ok` check,
 *    so any 4xx/5xx froze a table silently and the config-save button gave no
 *    feedback at all. Every request here goes through `api()`, which checks the
 *    status, surfaces failures, and never inserts a server message into HTML.
 *  - It interpolated message text and a user-supplied colour straight into
 *    `innerHTML` and a `style=` attribute. Everything here is set with
 *    `textContent`, or built with `createElement`.
 *  - It ran three `setInterval`s regardless of whether the previous response
 *    had landed or the tab was even visible. Polling here is a single loop that
 *    awaits each cycle and stops when the page is hidden.
 */
"use strict";

const CSRF = document.querySelector('input[name="csrfmiddlewaretoken"]')?.value ?? "";
const POLL_MS = 5000;

/* ---------------------------------------------------------------- transport */
async function api(path, { method = "GET", body } = {}) {
  const options = { method, headers: { Accept: "application/json" } };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.headers["X-CSRFToken"] = CSRF;
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  if (!response.ok || payload.ok === false) {
    const detail = payload?.details
      ? Object.entries(payload.details).map(([k, v]) => `${k}: ${v}`).join("; ")
      : "";
    throw new Error([payload?.error ?? `HTTP ${response.status}`, detail].filter(Boolean).join(" — "));
  }
  return payload.data;
}

function flash(message, kind = "ok") {
  const node = document.getElementById("flash");
  if (!node) return;
  node.textContent = message;          // never innerHTML: this can carry server text
  node.className = `flash is-${kind}`;
  node.hidden = false;
  clearTimeout(flash._timer);
  flash._timer = setTimeout(() => { node.hidden = true; }, 4000);
}

/* ------------------------------------------------------------------ helpers */
const pct = (v) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(1)}%`);
const num = (v, d = 0) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
const int = (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString());

function el(tag, { text, cls, attrs } = {}) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;   // escaping by construction
  if (cls) node.className = cls;
  if (attrs) for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

function fillTable(tableId, rows, buildCells) {
  const body = document.querySelector(`#${tableId} tbody`);
  if (!body) return;
  const fragment = document.createDocumentFragment();
  if (!rows.length) {
    const tr = el("tr");
    const td = el("td", { text: "No data yet.", cls: "hint" });
    td.colSpan = document.querySelectorAll(`#${tableId} thead th`).length || 1;
    tr.append(td);
    fragment.append(tr);
  }
  for (const row of rows) {
    const tr = el("tr");
    for (const cell of buildCells(row)) tr.append(cell);
    fragment.append(tr);
  }
  body.replaceChildren(fragment);      // one reflow, not one per row
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* ------------------------------------------------------------------- charts */
const charts = {};

function upsertChart(id, config) {
  const canvas = document.getElementById(id);
  if (!canvas) return null;
  if (charts[id]) {
    charts[id].data = config.data;
    charts[id].options = config.options;
    charts[id].update("none");
    return charts[id];
  }
  charts[id] = new Chart(canvas.getContext("2d"), config);
  return charts[id];
}

/** Draws the injected change points and the model-version boundaries.
 *  Without these the accuracy chart is unreadable: a reader cannot tell an
 *  adaptation from noise. */
const annotationPlugin = {
  id: "annotations",
  afterDatasetsDraw(chart, _args, opts) {
    const { ctx, chartArea, scales } = chart;
    if (!chartArea) return;
    const marks = opts?.marks ?? [];
    ctx.save();
    for (const mark of marks) {
      const x = scales.x.getPixelForValue(mark.index);
      if (Number.isNaN(x)) continue;
      ctx.strokeStyle = mark.color;
      ctx.lineWidth = mark.width ?? 2;
      ctx.setLineDash(mark.dash ?? []);
      ctx.beginPath();
      ctx.moveTo(x, chartArea.top);
      ctx.lineTo(x, chartArea.bottom);
      ctx.stroke();
      if (mark.label) {
        ctx.setLineDash([]);
        ctx.fillStyle = mark.color;
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillText(mark.label, x + 4, chartArea.top + 12);
      }
    }
    ctx.restore();
  },
};
Chart.register(annotationPlugin);

function versionChangeMarks(series) {
  const marks = [];
  const versions = series.model_versions ?? [];
  for (let i = 1; i < versions.length; i += 1) {
    if (versions[i] !== versions[i - 1]) {
      marks.push({ index: i, color: cssVar("--ok"), dash: [5, 4], label: versions[i] });
    }
  }
  for (const batch of series.change_point_batches ?? []) {
    const index = (series.batches ?? []).indexOf(batch);
    if (index >= 0) marks.push({ index, color: cssVar("--bad"), label: "drift injected" });
  }
  return marks;
}

const baseOptions = (extra = {}) => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  interaction: { mode: "index", intersect: false },
  plugins: { legend: { labels: { boxWidth: 12, color: cssVar("--muted") } } },
  scales: {
    x: { grid: { display: false }, ticks: { color: cssVar("--muted"), maxTicksLimit: 14 } },
    y: { grid: { color: cssVar("--border") }, ticks: { color: cssVar("--muted") } },
  },
  ...extra,
});

/* -------------------------------------------------------------------- pages */
async function renderOverview() {
  const overview = await api("/api/overview/");
  paintBadge(overview);
  paintStats(overview);
  if (!overview.run) return;

  const [series, categories, messages] = await Promise.all([
    api("/api/series/?limit=200"),
    api("/api/categories/"),
    api("/api/messages/?limit=10"),
  ]);

  upsertChart("accuracy-chart", {
    type: "line",
    data: {
      labels: series.batches,
      datasets: [
        { label: "Accuracy", data: series.accuracy, borderColor: cssVar("--accent"),
          backgroundColor: "transparent", tension: 0.25, pointRadius: 0, borderWidth: 2 },
        { label: "Cumulative", data: series.cumulative_accuracy, borderColor: cssVar("--muted"),
          borderDash: [4, 4], backgroundColor: "transparent", pointRadius: 0, borderWidth: 1.5 },
        { label: "Mean confidence", data: series.mean_confidence, borderColor: cssVar("--warn"),
          backgroundColor: "transparent", pointRadius: 0, borderWidth: 1.5 },
      ],
    },
    options: baseOptions({
      plugins: {
        legend: { labels: { boxWidth: 12, color: cssVar("--muted") } },
        annotations: { marks: versionChangeMarks(series) },
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: cssVar("--muted"), maxTicksLimit: 14 },
             title: { display: true, text: "batch", color: cssVar("--muted") } },
        y: { min: 0, max: 1, grid: { color: cssVar("--border") },
             ticks: { color: cssVar("--muted"), callback: (v) => `${(v * 100).toFixed(0)}%` } },
      },
    }),
  });

  upsertChart("category-chart", {
    type: "bar",
    data: {
      labels: categories.map((c) => c.label),
      datasets: [
        { label: "Actual", data: categories.map((c) => c.actual), backgroundColor: cssVar("--muted") },
        { label: "Predicted", data: categories.map((c) => c.predicted),
          backgroundColor: categories.map((c) => c.color) },
      ],
    },
    options: baseOptions(),
  });

  fillTable("latest-table", messages.results, (m) => [
    el("td", { text: m.msg_index, cls: "num" }),
    el("td", { text: m.text, cls: "truncate" }),
    el("td", { text: m.true_label }),
    el("td", { text: m.predicted_label, cls: m.is_correct ? "" : "pill-bad" }),
    el("td", { text: pct(m.confidence), cls: "num" }),
  ]);
}

function paintBadge(overview) {
  const badge = document.getElementById("stream-badge");
  if (!badge) return;
  const active = overview.is_active;
  badge.textContent = active ? "Stream active" : "Stream stopped";
  badge.className = `badge ${active ? "badge-ok" : "badge-off"}`;
}

function paintStats(overview) {
  const host = document.getElementById("stats");
  if (!host) return;
  const totals = overview.totals ?? {};
  const cards = [
    ["Live accuracy", pct(totals.accuracy), overview.run ? `run #${overview.run.id}` : "no active run"],
    ["Classified", int(totals.classified), `${int(totals.dropped)} dropped`],
    ["Batches", int(totals.batches), overview.latest_batch ? `last #${overview.latest_batch.batch_id}` : ""],
    ["Active model", overview.active_model?.version ?? "—",
      overview.active_model ? `test ${pct(overview.active_model.test_accuracy)}` : ""],
    ["Throughput", `${num(totals.mean_throughput, 1)}/s`, "mean per batch"],
  ];
  const fragment = document.createDocumentFragment();
  for (const [label, value, sub] of cards) {
    const card = el("div", { cls: "stat" });
    card.append(el("div", { text: label, cls: "stat-label" }));
    card.append(el("div", { text: value, cls: "stat-value" }));
    if (sub) card.append(el("div", { text: sub, cls: "stat-sub" }));
    fragment.append(card);
  }
  host.replaceChildren(fragment);
}

async function renderDrift() {
  const overview = await api("/api/overview/");
  paintBadge(overview);
  if (!overview.run) return;
  const [series, events, jobs] = await Promise.all([
    api("/api/series/?limit=200"), api("/api/drift/?limit=100"), api("/api/jobs/"),
  ]);

  upsertChart("drift-chart", {
    type: "line",
    data: {
      labels: series.batches,
      datasets: [
        { label: "Error rate", data: series.accuracy.map((a) => (a === null ? null : 1 - a)),
          borderColor: cssVar("--bad"), backgroundColor: "transparent", pointRadius: 0, borderWidth: 2 },
        { label: "PSI vocabulary", data: series.psi_vocabulary, borderColor: cssVar("--accent"),
          backgroundColor: "transparent", pointRadius: 0, borderWidth: 1.5 },
        { label: "PSI confidence", data: series.psi_confidence, borderColor: cssVar("--warn"),
          backgroundColor: "transparent", pointRadius: 0, borderWidth: 1.5 },
        { label: "PSI predicted class", data: series.psi_predicted, borderColor: cssVar("--ok"),
          backgroundColor: "transparent", pointRadius: 0, borderWidth: 1.5 },
      ],
    },
    options: baseOptions({
      plugins: {
        legend: { labels: { boxWidth: 12, color: cssVar("--muted") } },
        annotations: { marks: versionChangeMarks(series) },
      },
    }),
  });

  fillTable("drift-table", events, (e) => [
    el("td", { text: e.batch_id, cls: "num" }),
    el("td", { text: e.detector }),
    el("td", { text: num(e.statistic, 4), cls: "num" }),
    el("td", { text: e.detection_delay_batches ?? "—", cls: "num" }),
    el("td", { text: e.action_taken,
      cls: e.action_taken === "retrain_queued" ? "pill pill-ok" : "" }),
  ]);

  fillTable("jobs-table", jobs, (j) => [
    el("td", { text: j.id, cls: "num" }),
    el("td", { text: j.status, cls: j.status === "failed" ? "pill pill-bad" : "" }),
    el("td", { text: j.model_version ?? "—" }),
    el("td", { text: j.trigger_batch ?? "—", cls: "num" }),
    el("td", { text: j.finished_at ? new Date(j.finished_at).toLocaleTimeString() : "—" }),
  ]);
}

async function renderModels() {
  const [models, confusion] = await Promise.all([api("/api/models/"), api("/api/confusion/")]);
  fillTable("models-table", models, (m) => [
    el("td", { text: m.version }),
    el("td", { text: m.status, cls: m.status === "active" ? "pill pill-ok" : "pill" }),
    el("td", { text: m.algorithm }),
    el("td", { text: pct(m.test_accuracy), cls: "num" }),
    el("td", { text: pct(m.recent_accuracy), cls: "num" }),
    el("td", { text: int(m.train_rows), cls: "num" }),
    el("td", { text: new Date(m.trained_at).toLocaleString() }),
  ]);

  const table = document.getElementById("confusion-table");
  if (!table || !confusion) return;
  const fragment = document.createDocumentFragment();
  const head = el("tr");
  head.append(el("th", { text: "true \\ predicted" }));
  for (const label of confusion.labels) head.append(el("th", { text: label }));
  fragment.append(head);
  const flat = confusion.matrix.flat();
  const peak = Math.max(1, ...flat);
  confusion.matrix.forEach((row, i) => {
    const tr = el("tr");
    tr.append(el("th", { text: confusion.labels[i] }));
    row.forEach((value, j) => {
      const cls = i === j ? "num diag" : value > peak * 0.08 ? "num hot" : "num";
      tr.append(el("td", { text: int(value), cls }));
    });
    fragment.append(tr);
  });
  table.replaceChildren(fragment);
}

async function renderRuns() {
  const runs = await api("/api/runs/");
  fillTable("runs-table", runs, (r) => [
    el("td", { text: r.id, cls: "num" }),
    el("td", { text: r.name }),
    el("td", { text: r.scenario || "—" }),
    el("td", { text: r.seed, cls: "num" }),
    el("td", { text: r.adaptation_enabled ? "on" : "off",
      cls: r.adaptation_enabled ? "pill pill-ok" : "pill" }),
    el("td", { text: r.status }),
    el("td", { text: int(r.batches), cls: "num" }),
    el("td", { text: pct(r.accuracy), cls: "num" }),
    el("td", { text: pct(r.accuracy_after_drift), cls: "num" }),
    el("td", { text: int(r.drift_events), cls: "num" }),
    el("td", { text: int(r.retrains), cls: "num" }),
  ]);
}

/* Messages page keeps its own paging state. */
const messageState = { offset: 0, limit: 50, total: 0 };

async function renderMessages() {
  const params = new URLSearchParams({
    limit: String(messageState.limit),
    offset: String(messageState.offset),
  });
  const q = document.getElementById("f-q")?.value.trim();
  const category = document.getElementById("f-category")?.value;
  if (q) params.set("q", q);
  if (category) params.set("category", category);
  if (document.getElementById("f-errors")?.checked) params.set("errors", "1");

  const page = await api(`/api/messages/?${params}`);
  messageState.total = page.total;
  fillTable("messages-table", page.results, (m) => [
    el("td", { text: m.msg_index, cls: "num" }),
    el("td", { text: m.text, cls: "truncate" }),
    el("td", { text: m.true_label }),
    el("td", { text: m.predicted_label, cls: m.is_correct ? "" : "pill pill-bad" }),
    el("td", { text: pct(m.confidence), cls: "num" }),
    el("td", { text: m.model_version }),
  ]);

  const from = page.total ? messageState.offset + 1 : 0;
  const to = Math.min(messageState.offset + messageState.limit, page.total);
  // The old endpoint truncated to 100 rows and reported neither the total nor
  // that truncation had happened.
  const info = `${from}–${to} of ${int(page.total)}`;
  const infoNode = document.getElementById("page-info");
  if (infoNode) infoNode.textContent = info;
  const count = document.getElementById("result-count");
  if (count) count.textContent = `${int(page.total)} matching messages`;
  const prev = document.getElementById("prev-page");
  const next = document.getElementById("next-page");
  if (prev) prev.disabled = messageState.offset === 0;
  if (next) next.disabled = to >= page.total;
}

/* ------------------------------------------------------------------ actions */
function wireActions() {
  document.getElementById("stream-toggle")?.addEventListener("click", async (event) => {
    event.target.disabled = true;
    try {
      const result = await api("/api/stream/toggle/", { method: "POST", body: {} });
      flash(result.is_active ? "Stream started." : "Stream stopped.");
      await refresh();
    } catch (error) {
      flash(error.message, "error");
    } finally {
      event.target.disabled = false;
    }
  });

  document.getElementById("save-config")?.addEventListener("click", async () => {
    const form = document.getElementById("config-form");
    const body = {};
    for (const field of form.querySelectorAll("input[name]")) {
      if (field.dataset.multi !== undefined) {
        // A checkbox group posts as a list of the checked values.
        body[field.name] = body[field.name] || [];
        if (field.checked) body[field.name].push(field.value);
      } else {
        body[field.name] = field.type === "checkbox" ? field.checked : field.value;
      }
    }
    try {
      await api("/api/config/", { method: "POST", body });
      flash("Configuration saved.");
    } catch (error) {
      flash(error.message, "error");
    }
  });

  for (const button of document.querySelectorAll(".cat-save")) {
    button.addEventListener("click", async (event) => {
      const row = event.target.closest("tr");
      const id = row.dataset.category;
      try {
        await api(`/api/categories/${id}/`, {
          method: "POST",
          body: {
            display_name: row.querySelector(".cat-name").value,
            color: row.querySelector(".cat-color").value,
            description: "",
          },
        });
        row.querySelector(".swatch").style.setProperty("--swatch", row.querySelector(".cat-color").value);
        flash("Category saved.");
      } catch (error) {
        flash(error.message, "error");
      }
    });
  }

  document.getElementById("f-apply")?.addEventListener("click", () => {
    messageState.offset = 0;
    renderMessages().catch((error) => flash(error.message, "error"));
  });
  document.getElementById("prev-page")?.addEventListener("click", () => {
    messageState.offset = Math.max(0, messageState.offset - messageState.limit);
    renderMessages().catch((error) => flash(error.message, "error"));
  });
  document.getElementById("next-page")?.addEventListener("click", () => {
    messageState.offset += messageState.limit;
    renderMessages().catch((error) => flash(error.message, "error"));
  });
}

/* --------------------------------------------------------------- poll loop */
const RENDERERS = {
  "accuracy-chart": renderOverview,
  "drift-chart": renderDrift,
  "models-table": renderModels,
  "runs-table": renderRuns,
  "messages-table": renderMessages,
};

function pageRenderer() {
  for (const [anchor, renderer] of Object.entries(RENDERERS)) {
    if (document.getElementById(anchor)) return renderer;
  }
  return async () => paintBadge(await api("/api/overview/"));
}

let running = false;

async function refresh() {
  // A single in-flight guard. The old dashboard fired three intervals that did
  // not care whether the previous response had come back.
  if (running) return;
  running = true;
  const state = document.getElementById("poll-state");
  if (state) state.textContent = "refreshing…";
  try {
    await pageRenderer()();
    if (state) state.textContent = `updated ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    if (state) state.textContent = "error";
    flash(error.message, "error");
  } finally {
    running = false;
  }
}

async function pollLoop() {
  for (;;) {
    if (document.visibilityState === "visible") await refresh();
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
  }
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") refresh();
});

document.addEventListener("DOMContentLoaded", () => {
  wireActions();
  pollLoop();
});
