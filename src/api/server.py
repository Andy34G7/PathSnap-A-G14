"""FastAPI model server, post-prediction evaluator, and batch benchmark for PathSnap (Laptop 1)."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import uvicorn

from src.eda.folium_visualisations import create_trip_folium_map
from src.eda.inventory_and_quality import BASE_DIR
from src.eda.trajectory_analysis import project_coords_to_utm
from src.features.candidate_generation import (
    align_route_to_observations,
    compute_smoothed_velocities,
)
from src.features.feature_engineering import extract_trip_features
from src.hmm.viterbi_decoder import (
    build_directed_graph,
    classify_error_buckets,
    compute_sequence_metrics,
    predict_trip_route,
    reconstruct_continuous_route,
)

ARTIFACTS_MODEL_PATH = BASE_DIR / "artifacts" / "models" / "best_lgbm_tuned.joblib"
ARTIFACTS_HMM_PATH = BASE_DIR / "artifacts" / "hmm" / "hmm_config.json"
EVAL_REPORTS_DIR = BASE_DIR / "reports" / "evaluation"
EVAL_FIGURES_DIR = EVAL_REPORTS_DIR / "figures"
EVAL_MAPS_DIR = EVAL_REPORTS_DIR / "maps"
ALLOWED_PARTITIONS = ("evaluator_test", "student_test")
TRIP_ID_PATTERN = re.compile(r"^[0-9A-Za-z_-]+$")


class RawTripPayload(BaseModel):
    """Optional raw GPS and road-network file strings uploaded over LAN from Laptop 2."""

    track_text: str
    nodes_text: str
    arcs_text: str


class PredictRequest(BaseModel):
    """Request schema for POST /api/predict."""

    partition: str = "evaluator_test"
    trip_id: str
    raw_payload: RawTripPayload | None = None
    generate_map: bool = True


_STATE: dict[str, Any] = {
    "clf": None,
    "hmm_config": None,
    "loaded_at": None,
}


def _ensure_artifacts_loaded() -> tuple[Any, dict[str, float]]:
    """Load the frozen Phase 1 LightGBM model and HMM configuration once."""
    if _STATE["clf"] is None or _STATE["hmm_config"] is None:
        _STATE["clf"] = joblib.load(ARTIFACTS_MODEL_PATH)
        _STATE["hmm_config"] = json.loads(ARTIFACTS_HMM_PATH.read_text(encoding="utf-8"))
        _STATE["loaded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return _STATE["clf"], _STATE["hmm_config"]


def _validate_partition(partition: str) -> str:
    """Validate partition name against allowed holdout directories."""
    if partition not in ALLOWED_PARTITIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported partition '{partition}'.")
    return partition


def _validate_trip_id(trip_id: str) -> str:
    """Sanitize trip identifiers to prevent path traversal."""
    if not TRIP_ID_PATTERN.fullmatch(trip_id):
        raise HTTPException(status_code=400, detail=f"Invalid trip_id '{trip_id}'.")
    return trip_id


def _load_route_file(route_path: Path) -> np.ndarray | None:
    """Load optional ground-truth .route file solely for post-prediction evaluation."""
    if not route_path.exists():
        return None
    df = pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"])
    return df["arc_id"].to_numpy(dtype=np.int64)


def _load_trip_inputs_only(trip_path: Path, trip_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load only .track, .nodes, and .arcs without reading any .route label file."""
    track = pd.read_csv(trip_path / f"{trip_id}.track", sep=r"\s+", header=None, names=["lon", "lat", "timestamp"]).to_numpy(dtype=np.float64)
    nodes = pd.read_csv(trip_path / f"{trip_id}.nodes", sep=r"\s+", header=None, names=["lon", "lat"]).to_numpy(dtype=np.float64)
    arcs = pd.read_csv(trip_path / f"{trip_id}.arcs", sep=r"\s+", header=None, names=["from_node", "to_node"]).to_numpy(dtype=np.int64)
    return track, nodes, arcs


def _list_partition_trips(partition: str) -> list[dict[str, Any]]:
    """Return lightweight metadata for all trip directories under BASE_DIR / partition."""
    part_dir = BASE_DIR / partition
    if not part_dir.exists():
        return []
    trips: list[dict[str, Any]] = []
    for tdir in sorted(p for p in part_dir.iterdir() if p.is_dir()):
        tid = tdir.name
        track, nodes, arcs = _load_trip_inputs_only(tdir, tid)
        duration_s = float(track[-1, 2] - track[0, 2]) if len(track) > 1 else 0.0
        trips.append(
            {
                "trip_id": tid,
                "n_obs": int(len(track)),
                "n_nodes": int(len(nodes)),
                "n_arcs": int(len(arcs)),
                "duration_s": round(duration_s, 2),
                "has_server_route": bool((tdir / f"{tid}.route").exists()),
            }
        )
    return trips


def _log_trip_response_cmd(result: dict[str, Any], source_mode: str) -> None:
    """Print structured diagnostic output to Laptop 1 terminal for CMD verification."""
    m = result.get("metrics")
    border = "=" * 78
    lines = [
        "",
        border,
        f"[PathSnap API] POST /api/predict | source={source_mode} | trip={result['partition']}/{result['trip_id']}",
        f"  Input Summary   : obs={result['n_obs']:,} | nodes={result['n_nodes']:,} | arcs={result['n_arcs']:,} | duration={result['duration_s']:.1f}s",
        f"  Inference       : time={result['inference_time_s'] * 1000.0:.1f} ms | pred_arcs={result['n_reconstructed_arcs']} | truncations={result['dijkstra_truncations']}",
    ]
    if m is not None:
        lines.extend(
            [
                f"  Server Eval Top1: Final={m['top1_acc_pct']:.4f}% | RawHMM={m['raw_hmm_top1_acc_pct']:.4f}% | ClfOnly={m['clf_top1_acc_pct']:.4f}% | CandRecall@8={m['cand_recall_pct']:.4f}%",
                f"  Route Sequence  : ArcF1={m['route_arc_f1_pct']:.4f}% (P={m['route_arc_precision_pct']:.2f}%, R={m['route_arc_recall_pct']:.2f}%, J={m['route_arc_jaccard_pct']:.2f}%) | SeqCont={m['seq_continuity_pct']:.4f}% | ObsCont={m['obs_continuity_pct']:.4f}%",
                f"  Error Buckets   : total={m['total_errors']} (cand_miss={m['cand_miss_errors']}, opp_twin={m['opp_twin_errors']}, adj_boundary={m['adj_boundary_errors']}, detour={m['other_detour_errors']})",
            ]
        )
    else:
        lines.append("  Server Eval     : No .route file on server (inference-only mode)")
    preview = " ".join(str(a) for a in result["reconstructed_route"][:14])
    if len(result["reconstructed_route"]) > 14:
        preview += f" ... (+{len(result['reconstructed_route']) - 14} more)"
    lines.append(f"  Predicted Route : [{preview}]")
    lines.append(border)
    print("\n".join(lines), flush=True)


def evaluate_single_trip(
    trip_dir: str | Path,
    partition: str,
    trip_id: str,
    clf: Any,
    hmm_config: dict[str, float],
    gt_route_path: Path | None = None,
    generate_map: bool = True,
) -> dict[str, Any]:
    """Run leak-free route prediction first, then compute server-side metrics and Folium map."""
    trip_path = Path(trip_dir)
    track, nodes, arcs = _load_trip_inputs_only(trip_path, trip_id)
    duration_s = float(track[-1, 2] - track[0, 2]) if len(track) > 1 else 0.0

    # Isolate .track/.nodes/.arcs during inference so .route is never on disk in infer_dir.
    route_on_disk = (trip_path / f"{trip_id}.route").exists()
    with tempfile.TemporaryDirectory() as infer_tmp:
        if route_on_disk:
            infer_dir = Path(infer_tmp) / partition / trip_id
            infer_dir.mkdir(parents=True, exist_ok=True)
            for ext in (".track", ".nodes", ".arcs"):
                shutil.copyfile(trip_path / f"{trip_id}{ext}", infer_dir / f"{trip_id}{ext}")
        else:
            infer_dir = trip_path

        t0 = time.perf_counter()
        pred = predict_trip_route(infer_dir, clf, hmm_config)
        inf_time_s = time.perf_counter() - t0

        feats_eval = extract_trip_features(infer_dir, k=8, include_labels=False) if gt_route_path is not None or route_on_disk else None

    # Read .route strictly after predict_trip_route finishes and stops the timer.
    resolved_route_path = gt_route_path if gt_route_path is not None else (trip_path / f"{trip_id}.route")
    gt_route = _load_route_file(resolved_route_path)
    metrics: dict[str, Any] | None = None
    if gt_route is not None and feats_eval is not None:
        X_3d = feats_eval["X"]
        n_obs, k, n_feat = X_3d.shape
        probs_2d = clf.predict_proba(X_3d.reshape(-1, n_feat))[:, 1].reshape(n_obs, k)
        cand_arcs = feats_eval["cand_arcs"]
        clf_obs_arcs = cand_arcs[np.arange(n_obs), np.argmax(probs_2d, axis=1)]

        tx, ty, epsg = project_coords_to_utm(track[:, 0], track[:, 1])
        nx, ny, _ = project_coords_to_utm(nodes[:, 0], nodes[:, 1], epsg=epsg)
        pts_xy = np.column_stack([tx, ty])
        nodes_xy = np.column_stack([nx, ny])
        adj, _ = build_directed_graph(nodes_xy, arcs)
        _, _, _, _, v_sm_u = compute_smoothed_velocities(pts_xy, track[:, 2])
        gt_obs_arcs, _ = align_route_to_observations(pts_xy, nodes_xy, arcs, gt_route, v_sm_u)

        cand_recall_pct = 100.0 * float(np.mean(np.any(cand_arcs == gt_obs_arcs[:, None], axis=1)))
        clf_top1_pct = 100.0 * float(np.mean(clf_obs_arcs == gt_obs_arcs))
        raw_top1_pct = 100.0 * float(np.mean(pred["raw_hmm_obs_arcs"] == gt_obs_arcs))
        final_top1_pct = 100.0 * float(np.mean(pred["obs_arcs"] == gt_obs_arcs))

        clf_seq_list = reconstruct_continuous_route(clf_obs_arcs, arcs, adj)

        clf_seq_m = compute_sequence_metrics(clf_obs_arcs, clf_seq_list, gt_route, arcs)
        raw_seq_m = compute_sequence_metrics(pred["raw_hmm_obs_arcs"], pred["reconstructed_route"], gt_route, arcs)
        seq_m = compute_sequence_metrics(pred["obs_arcs"], pred["reconstructed_route"], gt_route, arcs)
        err_m = classify_error_buckets(pred["obs_arcs"], gt_obs_arcs, cand_arcs, arcs)

        metrics = {
            "cand_recall_pct": round(cand_recall_pct, 4),
            "clf_top1_acc_pct": round(clf_top1_pct, 4),
            "clf_route_arc_f1_pct": round(float(clf_seq_m["route_arc_f1_pct"]), 4),
            "clf_obs_continuity_pct": round(float(clf_seq_m["obs_continuity_pct"]), 4),
            "raw_hmm_top1_acc_pct": round(raw_top1_pct, 4),
            "raw_hmm_obs_continuity_pct": round(float(raw_seq_m["obs_continuity_pct"]), 4),
            "top1_acc_pct": round(final_top1_pct, 4),
            "obs_continuity_pct": round(float(seq_m["obs_continuity_pct"]), 4),
            "seq_continuity_pct": round(float(seq_m["seq_continuity_pct"]), 4),
            "route_arc_precision_pct": round(float(seq_m["route_arc_precision_pct"]), 4),
            "route_arc_recall_pct": round(float(seq_m["route_arc_recall_pct"]), 4),
            "route_arc_f1_pct": round(float(seq_m["route_arc_f1_pct"]), 4),
            "route_arc_jaccard_pct": round(float(seq_m["route_arc_jaccard_pct"]), 4),
            "n_gt_route_arcs": int(len(gt_route)),
            **err_m,
        }

    EVAL_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    map_file = EVAL_MAPS_DIR / f"trip_{partition}_{trip_id}_predicted.html"
    if generate_map:
        if gt_route is not None and not (trip_path / f"{trip_id}.route").exists():
            with tempfile.TemporaryDirectory() as tmp_root:
                map_trip_dir = Path(tmp_root) / partition / trip_id
                map_trip_dir.mkdir(parents=True, exist_ok=True)
                for ext in (".track", ".nodes", ".arcs"):
                    src_f = trip_path / f"{trip_id}{ext}"
                    (map_trip_dir / f"{trip_id}{ext}").write_text(src_f.read_text(encoding="utf-8"), encoding="utf-8")
                (map_trip_dir / f"{trip_id}.route").write_text(
                    "\n".join(str(int(x)) for x in gt_route.tolist()) + "\n",
                    encoding="utf-8",
                )
                create_trip_folium_map(map_trip_dir, map_file, predicted_arcs=pred["reconstructed_route"])
        else:
            create_trip_folium_map(trip_path, map_file, predicted_arcs=pred["reconstructed_route"])

    return {
        "trip_id": trip_id,
        "partition": partition,
        "n_obs": int(len(track)),
        "n_nodes": int(len(nodes)),
        "n_arcs": int(len(arcs)),
        "duration_s": round(duration_s, 2),
        "inference_time_s": round(inf_time_s, 4),
        "n_reconstructed_arcs": int(len(pred["reconstructed_route"])),
        "dijkstra_truncations": int(pred["dijkstra_truncations"]),
        "reconstructed_route": [int(x) for x in pred["reconstructed_route"]],
        "metrics": metrics,
        "map_url": f"/api/map/{partition}/{trip_id}",
    }


def run_partition_evaluation(partition: str, save_outputs: bool = True, generate_maps: bool = True) -> dict[str, Any]:
    """Evaluate all trips in a partition using strictly frozen Phase 1 artifacts."""
    clf, hmm_config = _ensure_artifacts_loaded()
    part_dir = BASE_DIR / partition
    if not part_dir.exists():
        raise FileNotFoundError(f"Partition directory not found: {part_dir}")

    trip_dirs = sorted(p for p in part_dir.iterdir() if p.is_dir())
    results: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []

    for tdir in trip_dirs:
        tid = tdir.name
        res = evaluate_single_trip(
            trip_dir=tdir,
            partition=partition,
            trip_id=tid,
            clf=clf,
            hmm_config=hmm_config,
            generate_map=generate_maps,
        )
        results.append(res)
        m = res["metrics"] or {}
        flat_rows.append(
            {
                "partition": partition,
                "trip_id": tid,
                "n_obs": res["n_obs"],
                "n_nodes": res["n_nodes"],
                "n_arcs": res["n_arcs"],
                "duration_s": res["duration_s"],
                "inference_time_s": res["inference_time_s"],
                "n_reconstructed_arcs": res["n_reconstructed_arcs"],
                "dijkstra_truncations": res["dijkstra_truncations"],
                "cand_recall_pct": m.get("cand_recall_pct", math.nan),
                "clf_top1_acc_pct": m.get("clf_top1_acc_pct", math.nan),
                "clf_route_arc_f1_pct": m.get("clf_route_arc_f1_pct", math.nan),
                "clf_obs_continuity_pct": m.get("clf_obs_continuity_pct", math.nan),
                "raw_hmm_top1_acc_pct": m.get("raw_hmm_top1_acc_pct", math.nan),
                "raw_hmm_obs_continuity_pct": m.get("raw_hmm_obs_continuity_pct", math.nan),
                "top1_acc_pct": m.get("top1_acc_pct", math.nan),
                "obs_continuity_pct": m.get("obs_continuity_pct", math.nan),
                "seq_continuity_pct": m.get("seq_continuity_pct", math.nan),
                "route_arc_precision_pct": m.get("route_arc_precision_pct", math.nan),
                "route_arc_recall_pct": m.get("route_arc_recall_pct", math.nan),
                "route_arc_f1_pct": m.get("route_arc_f1_pct", math.nan),
                "route_arc_jaccard_pct": m.get("route_arc_jaccard_pct", math.nan),
                "n_gt_route_arcs": m.get("n_gt_route_arcs", 0),
                "cand_miss_errors": m.get("cand_miss_errors", 0),
                "opp_twin_errors": m.get("opp_twin_errors", 0),
                "adj_boundary_errors": m.get("adj_boundary_errors", 0),
                "other_detour_errors": m.get("other_detour_errors", 0),
                "total_errors": m.get("total_errors", 0),
            }
        )

    df = pd.DataFrame(flat_rows)
    total_obs = int(df["n_obs"].sum())
    weights = df["n_obs"].to_numpy(dtype=np.float64)

    def _wavg(col: str) -> float:
        vals = df[col].to_numpy(dtype=np.float64)
        if np.any(np.isnan(vals)):
            return float("nan")
        return float(np.sum(vals * weights) / np.sum(weights))

    summary = {
        "partition": partition,
        "n_trips": int(len(df)),
        "total_obs": total_obs,
        "total_inference_time_s": round(float(df["inference_time_s"].sum()), 4),
        "mean_inference_time_s": round(float(df["inference_time_s"].mean()), 4),
        "weighted_cand_recall_pct": round(_wavg("cand_recall_pct"), 4),
        "weighted_clf_top1_acc_pct": round(_wavg("clf_top1_acc_pct"), 4),
        "mean_clf_route_arc_f1_pct": round(float(df["clf_route_arc_f1_pct"].mean()), 4),
        "weighted_raw_hmm_top1_acc_pct": round(_wavg("raw_hmm_top1_acc_pct"), 4),
        "weighted_top1_acc_pct": round(_wavg("top1_acc_pct"), 4),
        "mean_top1_acc_pct": round(float(df["top1_acc_pct"].mean()), 4),
        "mean_obs_continuity_pct": round(float(df["obs_continuity_pct"].mean()), 4),
        "mean_seq_continuity_pct": round(float(df["seq_continuity_pct"].mean()), 4),
        "mean_route_arc_precision_pct": round(float(df["route_arc_precision_pct"].mean()), 4),
        "mean_route_arc_recall_pct": round(float(df["route_arc_recall_pct"].mean()), 4),
        "mean_route_arc_f1_pct": round(float(df["route_arc_f1_pct"].mean()), 4),
        "mean_route_arc_jaccard_pct": round(float(df["route_arc_jaccard_pct"].mean()), 4),
        "total_dijkstra_truncations": int(df["dijkstra_truncations"].sum()),
        "total_errors": int(df["total_errors"].sum()),
        "total_cand_miss_errors": int(df["cand_miss_errors"].sum()),
        "total_opp_twin_errors": int(df["opp_twin_errors"].sum()),
        "total_adj_boundary_errors": int(df["adj_boundary_errors"].sum()),
        "total_other_detour_errors": int(df["other_detour_errors"].sum()),
    }

    if save_outputs:
        EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(EVAL_REPORTS_DIR / f"{partition}_per_trip.csv", index=False)
        pd.DataFrame([summary]).to_csv(EVAL_REPORTS_DIR / f"{partition}_summary.csv", index=False)

    # Print formatted batch table to Laptop 1 CMD terminal
    border = "=" * 96
    print(f"\n{border}", flush=True)
    print(f"[PathSnap API] Batch Evaluation Summary | partition={partition} | trips={summary['n_trips']} | obs={total_obs:,}", flush=True)
    print(border, flush=True)
    header = f"{'Trip':<10} {'Obs':>6} {'ClfTop1%':>9} {'RawHMM%':>9} {'FinalTop1%':>11} {'RouteF1%':>9} {'SeqCont%':>9} {'Errs':>6} {'Time(ms)':>9}"
    print(header, flush=True)
    print("-" * 96, flush=True)
    for row in flat_rows:
        print(
            f"{row['trip_id']:<10} {row['n_obs']:>6d} {row['clf_top1_acc_pct']:>9.4f} "
            f"{row['raw_hmm_top1_acc_pct']:>9.4f} {row['top1_acc_pct']:>11.4f} "
            f"{row['route_arc_f1_pct']:>9.4f} {row['seq_continuity_pct']:>9.4f} "
            f"{row['total_errors']:>6d} {row['inference_time_s'] * 1000.0:>9.1f}",
            flush=True,
        )
    print("-" * 96, flush=True)
    print(
        f"{'AGGREGATE':<10} {total_obs:>6d} {summary['weighted_clf_top1_acc_pct']:>9.4f} "
        f"{summary['weighted_raw_hmm_top1_acc_pct']:>9.4f} {summary['weighted_top1_acc_pct']:>11.4f} "
        f"{summary['mean_route_arc_f1_pct']:>9.4f} {summary['mean_seq_continuity_pct']:>9.4f} "
        f"{summary['total_errors']:>6d} {summary['total_inference_time_s'] * 1000.0:>9.1f}",
        flush=True,
    )
    print(border, flush=True)

    return {"partition": partition, "summary": summary, "trips": results}


def plot_evaluator_test_benchmark(
    eval_per_trip_csv: Path | None = None,
    student_per_trip_csv: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Create 2-panel benchmark figure for blind evaluator_test vs. student_test."""
    EVAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    eval_csv = eval_per_trip_csv or (EVAL_REPORTS_DIR / "evaluator_test_per_trip.csv")
    stud_csv = student_per_trip_csv or (EVAL_REPORTS_DIR / "student_test_per_trip.csv")
    out_png = output_path or (EVAL_FIGURES_DIR / "fig1_evaluator_test_metrics.png")

    eval_df = pd.read_csv(eval_csv, dtype={"trip_id": str})
    stud_df = pd.read_csv(stud_csv, dtype={"trip_id": str})

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.8), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Panel A: Per-trip Top-1 and Route Arc F1 across evaluator_test
    ax1 = axes[0]
    trips = eval_df["trip_id"].astype(str).tolist()
    x = np.arange(len(trips))
    w = 0.27
    b1 = ax1.bar(x - w, eval_df["clf_top1_acc_pct"], width=w, color="#94a3b8", label="Classifier-Only Top-1")
    b2 = ax1.bar(x, eval_df["top1_acc_pct"], width=w, color="#1d4ed8", label="Final HMM + Realignment Top-1")
    b3 = ax1.bar(x + w, eval_df["route_arc_f1_pct"], width=w, color="#16a34a", label="Final HMM Route Arc F1")

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.5,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                rotation=90,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(trips, rotation=30, ha="right", fontsize=8.2)
    min_y = max(50.0, float(min(eval_df["clf_top1_acc_pct"].min(), eval_df["route_arc_f1_pct"].min())) - 8.0)
    ax1.set_ylim(min_y, 108.0)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("(a) Blind Holdout Performance Across 10 evaluator_test Trips", pad=22)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=3, fontsize=7.8, frameon=True)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel B: Partition comparison (student_test vs evaluator_test)
    ax2 = axes[1]
    def _agg(df: pd.DataFrame) -> list[float]:
        w_arr = df["n_obs"].to_numpy(dtype=np.float64)
        wavg = lambda c: float(np.sum(df[c].to_numpy(dtype=np.float64) * w_arr) / np.sum(w_arr))
        return [
            wavg("cand_recall_pct"),
            wavg("clf_top1_acc_pct"),
            wavg("top1_acc_pct"),
            float(df["route_arc_f1_pct"].mean()),
            float(df["seq_continuity_pct"].mean()),
        ]

    stud_vals = _agg(stud_df)
    eval_vals = _agg(eval_df)
    metric_labels = ["Cand Recall\n(K=8)", "Clf-Only\nTop-1", "Final HMM\nTop-1", "Route Arc\nF1", "Sequence\nContinuity"]
    xm = np.arange(len(metric_labels)) * 1.25
    wm = 0.44
    r1 = ax2.bar(xm - wm / 2.0, stud_vals, width=wm, color="#475569", label=f"student_test (10 trips, {int(stud_df['n_obs'].sum()):,} obs)")
    r2 = ax2.bar(xm + wm / 2.0, eval_vals, width=wm, color="#0f766e", label=f"evaluator_test (10 trips, {int(eval_df['n_obs'].sum()):,} obs)")

    for bars in (r1, r2):
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.25,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
            )

    ax2.set_xticks(xm)
    ax2.set_xticklabels(metric_labels, fontsize=8.5)
    min_y2 = max(75.0, float(min(min(stud_vals), min(eval_vals))) - 4.0)
    ax2.set_ylim(min_y2, 102.6)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("(b) Aggregate Holdout Consistency: student_test vs. evaluator_test", pad=22)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=1, fontsize=7.8, frameon=True)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_png


@asynccontextmanager
async def _lifespan(_: FastAPI):
    _ensure_artifacts_loaded()
    print(f"[PathSnap API] Loaded model: {ARTIFACTS_MODEL_PATH.name}", flush=True)
    print(f"[PathSnap API] Loaded HMM config: {ARTIFACTS_HMM_PATH.name}", flush=True)
    yield


app = FastAPI(title="PathSnap Map-Matching API", version="2.0.0", lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    """Return server readiness, model identity, and available test partitions."""
    _, hmm_cfg = _ensure_artifacts_loaded()
    available_parts = [p for p in ALLOWED_PARTITIONS if (BASE_DIR / p).exists()]
    return {
        "status": "ok",
        "model": ARTIFACTS_MODEL_PATH.name,
        "hmm_config": hmm_cfg,
        "loaded_at": _STATE["loaded_at"],
        "partitions": available_parts,
    }


@app.get("/api/trips")
def api_trips(partition: str = Query("evaluator_test")) -> dict[str, Any]:
    """List the available evaluation trips in a holdout partition."""
    part = _validate_partition(partition)
    return {"partition": part, "trips": _list_partition_trips(part)}


@app.post("/api/predict")
def api_predict(req: PredictRequest) -> dict[str, Any]:
    """Run leak-free route reconstruction and post-prediction server-side evaluation."""
    clf, hmm_config = _ensure_artifacts_loaded()
    part = _validate_partition(req.partition)
    tid = _validate_trip_id(req.trip_id)

    server_trip_dir = BASE_DIR / part / tid
    server_route_path = server_trip_dir / f"{tid}.route"

    if req.raw_payload is not None:
        with tempfile.TemporaryDirectory() as tmp_root:
            tmp_trip_dir = Path(tmp_root) / part / tid
            tmp_trip_dir.mkdir(parents=True, exist_ok=True)
            (tmp_trip_dir / f"{tid}.track").write_text(req.raw_payload.track_text, encoding="utf-8")
            (tmp_trip_dir / f"{tid}.nodes").write_text(req.raw_payload.nodes_text, encoding="utf-8")
            (tmp_trip_dir / f"{tid}.arcs").write_text(req.raw_payload.arcs_text, encoding="utf-8")
            res = evaluate_single_trip(
                trip_dir=tmp_trip_dir,
                partition=part,
                trip_id=tid,
                clf=clf,
                hmm_config=hmm_config,
                gt_route_path=server_route_path,
                generate_map=req.generate_map,
            )
        _log_trip_response_cmd(res, source_mode="lan_payload")
        return res

    if not server_trip_dir.exists():
        raise HTTPException(status_code=404, detail=f"Trip {part}/{tid} not found on server.")

    res = evaluate_single_trip(
        trip_dir=server_trip_dir,
        partition=part,
        trip_id=tid,
        clf=clf,
        hmm_config=hmm_config,
        generate_map=req.generate_map,
    )
    _log_trip_response_cmd(res, source_mode="server_partition")
    return res


@app.get("/api/evaluate_all")
def api_evaluate_all(partition: str = Query("evaluator_test")) -> dict[str, Any]:
    """Run batch inference across all trips in a holdout partition and log aggregate metrics."""
    part = _validate_partition(partition)
    try:
        return run_partition_evaluation(part, save_outputs=True, generate_maps=False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/map/{partition}/{trip_id}", response_class=HTMLResponse)
def api_map(partition: str, trip_id: str) -> Response:
    """Serve the generated interactive Folium HTML map for a predicted trip."""
    part = _validate_partition(partition)
    tid = _validate_trip_id(trip_id)
    map_file = EVAL_MAPS_DIR / f"trip_{part}_{tid}_predicted.html"
    if not map_file.exists():
        server_trip_dir = BASE_DIR / part / tid
        if not server_trip_dir.exists():
            raise HTTPException(status_code=404, detail=f"Map not generated for {part}/{tid}.")
        clf, hmm_config = _ensure_artifacts_loaded()
        evaluate_single_trip(server_trip_dir, part, tid, clf, hmm_config, generate_map=True)
    return HTMLResponse(content=map_file.read_text(encoding="utf-8"))


def main() -> None:
    """CLI entrypoint for launching the FastAPI server or running batch holdout evaluation."""
    parser = argparse.ArgumentParser(description="PathSnap Laptop 1 FastAPI Model & Evaluation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--evaluate-only", action="store_true", help="Run blind evaluation on evaluator_test & student_test and plot figures")
    args = parser.parse_args()

    if args.evaluate_only:
        if (BASE_DIR / "student_test").exists():
            run_partition_evaluation("student_test", save_outputs=True, generate_maps=True)
        if (BASE_DIR / "evaluator_test").exists():
            run_partition_evaluation("evaluator_test", save_outputs=True, generate_maps=True)
        if (EVAL_REPORTS_DIR / "evaluator_test_per_trip.csv").exists() and (EVAL_REPORTS_DIR / "student_test_per_trip.csv").exists():
            out_fig = plot_evaluator_test_benchmark()
            print(f"Saved evaluation figure: {out_fig}", flush=True)
        return

    uvicorn.run("src.api.server:app", host=args.host, port=args.port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
