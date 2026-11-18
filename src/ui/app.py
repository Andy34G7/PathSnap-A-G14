"""Minimal web client for PathSnap two-laptop map-matching evaluation (Laptop 2)."""

from __future__ import annotations

import argparse
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import uvicorn

from src.eda.inventory_and_quality import BASE_DIR

ALLOWED_PARTITIONS = ("evaluator_test", "student_test")
TRIP_ID_PATTERN = re.compile(r"^[0-9A-Za-z_-]+$")


class ReconstructRequest(BaseModel):
    """UI request schema for forwarding a trip to Laptop 1."""

    partition: str = "evaluator_test"
    trip_id: str
    transmit_files: bool = True


def _get_api_url() -> str:
    """Resolve the Laptop 1 FastAPI base URL."""
    return os.environ.get("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _validate_partition(partition: str) -> str:
    """Validate allowed partition identifier."""
    if partition not in ALLOWED_PARTITIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported partition '{partition}'.")
    return partition


def _validate_trip_id(trip_id: str) -> str:
    """Sanitize trip identifier."""
    if not TRIP_ID_PATTERN.fullmatch(trip_id):
        raise HTTPException(status_code=400, detail=f"Invalid trip_id '{trip_id}'.")
    return trip_id


MINIMAL_UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PathSnap Evaluator (UE24CS352A | A-G14)</title>
  <style>
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      padding: 0;
      height: 100%;
      background: #ffffff;
      color: #111827;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 13px;
      line-height: 1.4;
    }
    .app {
      display: flex;
      flex-direction: column;
      height: 100vh;
      width: 100vw;
      overflow: hidden;
    }
    header {
      height: 44px;
      min-height: 44px;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 0 12px;
      border-bottom: 1px solid #d1d5db;
      background: #f8f9fa;
      white-space: nowrap;
      overflow-x: auto;
    }
    .title {
      font-weight: 600;
      font-size: 13px;
      margin-right: 6px;
    }
    label {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      font-size: 12px;
      color: #374151;
    }
    select, button {
      font-family: inherit;
      font-size: 12px;
      height: 28px;
      padding: 0 8px;
      border: 1px solid #d1d5db;
      border-radius: 2px;
      background: #ffffff;
      color: #111827;
    }
    button {
      cursor: pointer;
      background: #f3f4f6;
      font-weight: 500;
    }
    button.primary {
      background: #111827;
      color: #ffffff;
      border-color: #111827;
    }
    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    .status {
      margin-left: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 11px;
      color: #374151;
    }
    main {
      display: flex;
      flex: 1;
      min-height: 0;
    }
    aside {
      width: 390px;
      min-width: 390px;
      border-right: 1px solid #d1d5db;
      background: #ffffff;
      overflow-y: auto;
      padding: 10px;
    }
    section {
      margin-bottom: 10px;
    }
    h2 {
      margin: 0 0 5px 0;
      font-size: 12px;
      font-weight: 600;
      color: #111827;
      border-bottom: 1px solid #e5e7eb;
      padding-bottom: 3px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
    }
    th, td {
      border: 1px solid #e5e7eb;
      padding: 3px 6px;
      text-align: left;
      vertical-align: top;
    }
    th {
      background: #f8f9fa;
      font-weight: 600;
      color: #374151;
    }
    td.num, th.num {
      text-align: right;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    tr.clickable {
      cursor: pointer;
    }
    tr.clickable:hover {
      background: #f3f4f6;
    }
    tr.selected-row {
      background: #e5e7eb;
    }
    tr.agg-row td {
      font-weight: 600;
      background: #f8f9fa;
    }
    pre.arc-seq {
      margin: 0;
      padding: 6px;
      border: 1px solid #e5e7eb;
      background: #f8f9fa;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 10.5px;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 58px;
      overflow-y: auto;
    }
    .empty-note {
      color: #6b7280;
      font-size: 11.5px;
      margin: 4px 0;
    }
    .map-pane {
      flex: 1;
      min-width: 0;
      background: #f8f9fa;
      display: flex;
      flex-direction: column;
    }
    iframe {
      width: 100%;
      height: 100%;
      border: 0;
      flex: 1;
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <span class="title">PathSnap Evaluator (UE24CS352A | A-G14)</span>
      <label>
        Partition
        <select id="partitionSelect">
          <option value="evaluator_test">evaluator_test</option>
          <option value="student_test">student_test</option>
        </select>
      </label>
      <label>
        Trip
        <select id="tripSelect"></select>
      </label>
      <label title="Send .track, .nodes, and .arcs from Laptop 2 over HTTP without .route">
        <input type="checkbox" id="payloadToggle" checked />
        Send .track/.nodes/.arcs payload
      </label>
      <button id="reconstructBtn" class="primary" type="button">Reconstruct Trip</button>
      <button id="evaluateAllBtn" type="button">Evaluate All 10 Trips</button>
      <span id="statusText" class="status">Ready</span>
    </header>
    <main>
      <aside>
        <section>
          <h2>Selected Trip Summary &amp; Evaluation Metrics</h2>
          <div id="tripMetricsBox">
            <p class="empty-note">Select a trip and click Reconstruct Trip.</p>
          </div>
        </section>
        <section>
          <h2>Reconstructed Route Arc Sequence (.route prediction)</h2>
          <pre id="arcSeqBox" class="arc-seq">No route predicted yet.</pre>
        </section>
        <section>
          <h2>Partition Batch Summary (All 10 Trips)</h2>
          <div id="batchBox">
            <p class="empty-note">Click Evaluate All 10 Trips to run full partition inference.</p>
          </div>
        </section>
      </aside>
      <div class="map-pane">
        <iframe id="mapFrame" title="Folium Route Reconstruction Map"></iframe>
      </div>
    </main>
  </div>

  <script>
    const partitionSelect = document.getElementById("partitionSelect");
    const tripSelect = document.getElementById("tripSelect");
    const payloadToggle = document.getElementById("payloadToggle");
    const reconstructBtn = document.getElementById("reconstructBtn");
    const evaluateAllBtn = document.getElementById("evaluateAllBtn");
    const statusText = document.getElementById("statusText");
    const tripMetricsBox = document.getElementById("tripMetricsBox");
    const arcSeqBox = document.getElementById("arcSeqBox");
    const batchBox = document.getElementById("batchBox");
    const mapFrame = document.getElementById("mapFrame");

    function setBusy(busy, msg) {
      reconstructBtn.disabled = busy;
      evaluateAllBtn.disabled = busy;
      statusText.textContent = msg;
    }

    function fmtPct(v) {
      return (v === null || v === undefined || Number.isNaN(v)) ? "N/A" : Number(v).toFixed(4) + "%";
    }

    function renderTripDetails(data) {
      const m = data.metrics;
      const rows = [
        ["Trip", `${data.partition}/${data.trip_id}`],
        ["GPS Observations", `${data.n_obs.toLocaleString()} (${data.duration_s.toFixed(1)} s)`],
        ["Road Graph Size", `${data.n_nodes.toLocaleString()} nodes / ${data.n_arcs.toLocaleString()} arcs`],
        ["Predicted Route Length", `${data.n_reconstructed_arcs} arcs (${data.dijkstra_truncations} truncations)`],
        ["Inference Latency", `${(data.inference_time_s * 1000).toFixed(1)} ms`],
      ];
      if (m) {
        rows.push(
          ["Final HMM Top-1 Acc", fmtPct(m.top1_acc_pct)],
          ["Raw HMM Top-1 Acc", fmtPct(m.raw_hmm_top1_acc_pct)],
          ["Classifier-Only Top-1", fmtPct(m.clf_top1_acc_pct)],
          ["Route Arc F1 vs .route", fmtPct(m.route_arc_f1_pct)],
          ["Route Precision / Recall", `${Number(m.route_arc_precision_pct).toFixed(2)}% / ${Number(m.route_arc_recall_pct).toFixed(2)}%`],
          ["Route Arc Jaccard", fmtPct(m.route_arc_jaccard_pct)],
          ["Sequence Continuity", fmtPct(m.seq_continuity_pct)],
          ["Observation Continuity", fmtPct(m.obs_continuity_pct)],
          ["Candidate Recall (K=8)", fmtPct(m.cand_recall_pct)],
          ["Error Buckets (miss/twin/bnd/detour)", `${m.total_errors} (${m.cand_miss_errors}/${m.opp_twin_errors}/${m.adj_boundary_errors}/${m.other_detour_errors})`],
        );
      } else {
        rows.push(["Server Evaluation", "Inference only (.route absent)"]);
      }

      const bodyHtml = rows.map(([k, v]) => `<tr><th>${k}</th><td class="num">${v}</td></tr>`).join("");
      tripMetricsBox.innerHTML = `<table><tbody>${bodyHtml}</tbody></table>`;
      arcSeqBox.textContent = (data.reconstructed_route || []).join(" ");
      mapFrame.src = `/ui/map/${encodeURIComponent(data.partition)}/${encodeURIComponent(data.trip_id)}?t=${Date.now()}`;
    }

    function renderBatchSummary(payload) {
      const s = payload.summary;
      const selectedTid = tripSelect.value;
      const rowsHtml = (payload.trips || []).map((t) => {
        const m = t.metrics || {};
        const cls = t.trip_id === selectedTid ? "clickable selected-row" : "clickable";
        return `<tr class="${cls}" data-tid="${t.trip_id}">
          <td>${t.trip_id}</td>
          <td class="num">${t.n_obs}</td>
          <td class="num">${m.top1_acc_pct !== undefined ? Number(m.top1_acc_pct).toFixed(2) + "%" : "N/A"}</td>
          <td class="num">${m.route_arc_f1_pct !== undefined ? Number(m.route_arc_f1_pct).toFixed(2) + "%" : "N/A"}</td>
          <td class="num">${m.seq_continuity_pct !== undefined ? Number(m.seq_continuity_pct).toFixed(2) + "%" : "N/A"}</td>
          <td class="num">${(t.inference_time_s * 1000).toFixed(0)}</td>
        </tr>`;
      }).join("");

      const aggRow = `<tr class="agg-row">
        <td>ALL (${s.n_trips})</td>
        <td class="num">${s.total_obs}</td>
        <td class="num">${Number(s.weighted_top1_acc_pct).toFixed(2)}%</td>
        <td class="num">${Number(s.mean_route_arc_f1_pct).toFixed(2)}%</td>
        <td class="num">${Number(s.mean_seq_continuity_pct).toFixed(2)}%</td>
        <td class="num">${(s.total_inference_time_s * 1000).toFixed(0)}</td>
      </tr>`;

      batchBox.innerHTML = `<table>
        <thead>
          <tr>
            <th>Trip</th>
            <th class="num">Obs</th>
            <th class="num">Top-1</th>
            <th class="num">Route F1</th>
            <th class="num">Seq Cont</th>
            <th class="num">ms</th>
          </tr>
        </thead>
        <tbody>${rowsHtml}${aggRow}</tbody>
      </table>`;

      batchBox.querySelectorAll("tr.clickable").forEach((tr) => {
        tr.addEventListener("click", () => {
          tripSelect.value = tr.dataset.tid;
          reconstructCurrentTrip();
        });
      });
    }

    async function loadTrips() {
      const part = partitionSelect.value;
      setBusy(true, `Loading ${part} trips...`);
      try {
        const res = await fetch(`/ui/trips?partition=${encodeURIComponent(part)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        tripSelect.innerHTML = (data.trips || [])
          .map((t) => `<option value="${t.trip_id}">${t.trip_id} (${t.n_obs} obs)</option>`)
          .join("");
        setBusy(false, `Loaded ${data.trips.length} trips (${part})`);
      } catch (err) {
        setBusy(false, `Error loading trips: ${err.message}`);
      }
    }

    async function reconstructCurrentTrip() {
      const partition = partitionSelect.value;
      const trip_id = tripSelect.value;
      if (!trip_id) return;
      const t0 = performance.now();
      setBusy(true, `POST /api/predict ${partition}/${trip_id}...`);
      try {
        const res = await fetch("/ui/reconstruct", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            partition,
            trip_id,
            transmit_files: payloadToggle.checked,
          }),
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`HTTP ${res.status}: ${text}`);
        }
        const data = await res.json();
        renderTripDetails(data);
        const dt = Math.round(performance.now() - t0);
        setBusy(false, `200 OK | ${partition}/${trip_id} in ${dt} ms`);
      } catch (err) {
        setBusy(false, `Request failed: ${err.message}`);
      }
    }

    async function evaluateAllTrips() {
      const partition = partitionSelect.value;
      const t0 = performance.now();
      setBusy(true, `GET /api/evaluate_all ${partition}...`);
      try {
        const res = await fetch(`/ui/evaluate_all?partition=${encodeURIComponent(partition)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        renderBatchSummary(data);
        const dt = Math.round(performance.now() - t0);
        setBusy(false, `200 OK | Evaluated ${data.summary.n_trips} trips in ${dt} ms`);
      } catch (err) {
        setBusy(false, `Batch eval failed: ${err.message}`);
      }
    }

    partitionSelect.addEventListener("change", async () => {
      await loadTrips();
      tripMetricsBox.innerHTML = '<p class="empty-note">Select a trip and click Reconstruct Trip.</p>';
      arcSeqBox.textContent = "No route predicted yet.";
      batchBox.innerHTML = '<p class="empty-note">Click Evaluate All 10 Trips to run full partition inference.</p>';
      mapFrame.src = "about:blank";
    });
    reconstructBtn.addEventListener("click", reconstructCurrentTrip);
    evaluateAllBtn.addEventListener("click", evaluateAllTrips);

    loadTrips();
  </script>
</body>
</html>
"""

app = FastAPI(title="PathSnap Evaluator UI (Laptop 2)", version="2.0.0")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the minimal single-page evaluator interface."""
    return HTMLResponse(content=MINIMAL_UI_HTML)


@app.get("/ui/trips")
def ui_trips(partition: str = Query("evaluator_test")) -> dict[str, Any]:
    """List trips locally on Laptop 2 if present, otherwise query Laptop 1."""
    part = _validate_partition(partition)
    part_dir = BASE_DIR / part
    if part_dir.exists():
        trips = []
        for tdir in sorted(p for p in part_dir.iterdir() if p.is_dir()):
            tid = tdir.name
            track_file = tdir / f"{tid}.track"
            if track_file.exists():
                with track_file.open("r", encoding="utf-8") as fh:
                    n_obs = sum(1 for _ in fh)
            else:
                n_obs = 0
            trips.append({"trip_id": tid, "n_obs": n_obs})
        if trips:
            return {"partition": part, "trips": trips}

    api_url = _get_api_url()
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{api_url}/api/trips", params={"partition": part})
        resp.raise_for_status()
        return resp.json()


@app.post("/ui/reconstruct")
def ui_reconstruct(req: ReconstructRequest) -> dict[str, Any]:
    """Send selected trip (or raw .track/.nodes/.arcs files without .route) to Laptop 1."""
    part = _validate_partition(req.partition)
    tid = _validate_trip_id(req.trip_id)
    api_url = _get_api_url()

    payload: dict[str, Any] = {"partition": part, "trip_id": tid, "generate_map": True}
    local_trip_dir = BASE_DIR / part / tid
    if req.transmit_files and local_trip_dir.exists():
        # Strictly exclude .route when transmitting from Laptop 2 to Laptop 1.
        payload["raw_payload"] = {
            "track_text": (local_trip_dir / f"{tid}.track").read_text(encoding="utf-8"),
            "nodes_text": (local_trip_dir / f"{tid}.nodes").read_text(encoding="utf-8"),
            "arcs_text": (local_trip_dir / f"{tid}.arcs").read_text(encoding="utf-8"),
        }

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{api_url}/api/predict", json=payload)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()


@app.get("/ui/evaluate_all")
def ui_evaluate_all(partition: str = Query("evaluator_test")) -> dict[str, Any]:
    """Trigger full partition evaluation on Laptop 1."""
    part = _validate_partition(partition)
    api_url = _get_api_url()
    with httpx.Client(timeout=300.0) as client:
        resp = client.get(f"{api_url}/api/evaluate_all", params={"partition": part})
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()


@app.get("/ui/map/{partition}/{trip_id}", response_class=HTMLResponse)
def ui_map(partition: str, trip_id: str) -> Response:
    """Proxy the interactive Folium map HTML from Laptop 1 so the iframe remains same-origin."""
    part = _validate_partition(partition)
    tid = _validate_trip_id(trip_id)
    api_url = _get_api_url()
    with httpx.Client(timeout=60.0) as client:
        resp = client.get(f"{api_url}/api/map/{part}/{tid}")
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return HTMLResponse(content=resp.text)


def main() -> None:
    """CLI entrypoint for launching the Laptop 2 minimal UI server."""
    parser = argparse.ArgumentParser(description="PathSnap Laptop 2 Minimal Evaluator UI")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8501, help="Port number (default: 8501)")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Laptop 1 FastAPI base URL")
    args = parser.parse_args()

    os.environ["PATHSNAP_API_URL"] = args.api_url.rstrip("/")
    uvicorn.run("src.ui.app:app", host=args.host, port=args.port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
