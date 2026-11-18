"""Hidden Markov Model (HMM) + Viterbi sequential route reconstruction for PathSnap."""

from __future__ import annotations

import heapq
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eda.folium_visualisations import create_trip_folium_map
from src.eda.inventory_and_quality import BASE_DIR, load_trip_raw
from src.eda.trajectory_analysis import project_coords_to_utm
from src.features.feature_engineering import FEATURES_CACHE_DIR, FEATURE_NAMES, extract_trip_features
from src.models.train_and_compare import MODELS_ARTIFACTS_DIR

HMM_REPORTS_DIR = BASE_DIR / "reports" / "hmm"
HMM_FIGURES_DIR = HMM_REPORTS_DIR / "figures"
HMM_MAPS_DIR = HMM_REPORTS_DIR / "maps"
HMM_ARTIFACTS_DIR = BASE_DIR / "artifacts" / "hmm"


def build_directed_graph(
    nodes_xy: np.ndarray,
    arcs: np.ndarray,
) -> tuple[list[list[tuple[int, int, float]]], np.ndarray]:
    """Build directed adjacency list [(to_node, arc_id, length_m), ...] and arc length array."""
    n_nodes = len(nodes_xy)
    u = arcs[:, 0]
    v = arcs[:, 1]
    seg_ab = nodes_xy[v] - nodes_xy[u]
    arc_lens = np.maximum(np.hypot(seg_ab[:, 0], seg_ab[:, 1]), 0.1)

    adj: list[list[tuple[int, int, float]]] = [[] for _ in range(n_nodes)]
    for aid, (u_i, v_i, w_i) in enumerate(zip(u.tolist(), v.tolist(), arc_lens.tolist())):
        adj[u_i].append((v_i, aid, w_i))
    return adj, arc_lens


def bounded_dijkstra(
    adj: list[list[tuple[int, int, float]]],
    start_node: int,
    targets: set[int],
    max_dist_m: float,
    max_pops: int = 3000,
) -> tuple[dict[int, float], bool]:
    """Find shortest directed road distances from start_node to target nodes within bounds."""
    found: dict[int, float] = {}
    if start_node in targets:
        found[start_node] = 0.0
        if len(found) == len(targets):
            return found, False

    dist: dict[int, float] = {start_node: 0.0}
    pq: list[tuple[float, int]] = [(0.0, start_node)]
    pops = 0
    truncated = False

    while pq:
        if pops >= max_pops:
            truncated = True
            break
        d_u, u = heapq.heappop(pq)
        if d_u > dist.get(u, float("inf")):
            continue
        pops += 1
        if u in targets and u not in found:
            found[u] = d_u
            if len(found) == len(targets):
                break
        if d_u > max_dist_m:
            continue
        for v, _, w in adj[u]:
            nd = d_u + w
            if nd <= max_dist_m and nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return found, truncated


def shortest_arc_path(
    adj: list[list[tuple[int, int, float]]],
    start_node: int,
    target_node: int,
    max_dist_m: float = 1500.0,
    max_pops: int = 3000,
) -> list[int] | None:
    """Recover the list of directed arc_ids along the shortest path from start_node to target_node."""
    if start_node == target_node:
        return []

    dist: dict[int, float] = {start_node: 0.0}
    parent: dict[int, tuple[int, int]] = {}
    pq: list[tuple[float, int]] = [(0.0, start_node)]
    pops = 0

    while pq and pops < max_pops:
        d_u, u = heapq.heappop(pq)
        if d_u > dist.get(u, float("inf")):
            continue
        if u == target_node:
            path: list[int] = []
            cur = target_node
            while cur != start_node:
                prev_u, aid = parent[cur]
                path.append(aid)
                cur = prev_u
            path.reverse()
            return path
        pops += 1
        if d_u > max_dist_m:
            continue
        for v, aid, w in adj[u]:
            nd = d_u + w
            if nd <= max_dist_m and nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = (u, aid)
                heapq.heappush(pq, (nd, v))

    return None


def viterbi_decode_trip(
    probs_2d: np.ndarray,
    cand_arcs: np.ndarray,
    proj_frac: np.ndarray,
    step_dist: np.ndarray,
    dt_prev: np.ndarray,
    arcs: np.ndarray,
    adj: list[list[tuple[int, int, float]]],
    arc_lens: np.ndarray,
    alpha_emit: float = 1.0,
    beta_trans: float = 0.15,
    switch_cost: float = 0.5,
    uturn_cost: float = 2.0,
    disconnect_penalty: float = 30.0,
    max_conn_penalty: float = 12.0,
) -> tuple[np.ndarray, int]:
    """Decode the most probable road-arc sequence using Viterbi in log-probability space."""
    n_obs, k = cand_arcs.shape
    emit_log = alpha_emit * np.log(np.clip(probs_2d, 1e-6, 1.0 - 1e-6))

    dp = np.full((n_obs, k), -np.inf, dtype=np.float64)
    back = np.zeros((n_obs, k), dtype=np.int32)
    dp[0] = emit_log[0]

    sp_cache: dict[tuple[int, int, float], float] = {}
    n_dijkstra_trunc = 0

    for t in range(1, n_obs):
        d_gps = float(step_dist[t])
        dt = max(1.0, float(dt_prev[t]))
        gap_scale = np.sqrt(dt)
        raw_limit = max(200.0, min(55.0 * dt, 3500.0))
        bucket_limit = float(np.ceil(raw_limit / 10.0) * 10.0)

        prev_arcs = cand_arcs[t - 1]
        cur_arcs = cand_arcs[t]
        u_cur_set = {int(arcs[b, 0]) for b in cur_arcs}
        unique_v_prev = {int(arcs[a, 1]) for a in prev_arcs}

        v_to_u_dists: dict[int, dict[int, float]] = {}
        for vp in unique_v_prev:
            missing_targets = set()
            cached_map: dict[int, float] = {}
            for uc in u_cur_set:
                key = (vp, uc, bucket_limit)
                if key in sp_cache:
                    cached_map[uc] = sp_cache[key]
                else:
                    missing_targets.add(uc)
            if missing_targets:
                res, trunc = bounded_dijkstra(
                    adj, vp, missing_targets, max_dist_m=bucket_limit, max_pops=3000
                )
                if trunc:
                    n_dijkstra_trunc += 1
                for uc in missing_targets:
                    val = res.get(uc, float("inf"))
                    sp_cache[(vp, uc, bucket_limit)] = val
                    cached_map[uc] = val
            v_to_u_dists[vp] = cached_map

        # Keep disconnect penalty strictly above capped connected penalty even across gaps
        gap_disc_pen = (
            disconnect_penalty
            if dt <= 5.0
            else max(max_conn_penalty + 4.0, disconnect_penalty / np.sqrt(dt / 5.0))
        )

        for j in range(k):
            b = int(cur_arcs[j])
            ub = int(arcs[b, 0])
            vb = int(arcs[b, 1])
            len_b = float(arc_lens[b])
            pt_cur = float(proj_frac[t, j])

            best_score = -np.inf
            best_prev = 0
            for i in range(k):
                a = int(prev_arcs[i])
                ua = int(arcs[a, 0])
                va = int(arcs[a, 1])
                pt_prev = float(proj_frac[t - 1, i])

                if a == b:
                    d_road = abs(pt_cur - pt_prev) * float(arc_lens[a])
                    backward_pen = 1.5 if pt_cur < pt_prev - 0.15 else 0.0
                    conn_pen = min(max_conn_penalty, beta_trans * abs(d_road - d_gps) / gap_scale)
                    trans_log = -conn_pen - backward_pen
                else:
                    sp_dist = v_to_u_dists[va].get(ub, float("inf"))
                    if np.isfinite(sp_dist):
                        d_road = (1.0 - pt_prev) * float(arc_lens[a]) + sp_dist + pt_cur * len_b
                        conn_pen = min(max_conn_penalty, beta_trans * abs(d_road - d_gps) / gap_scale)
                        is_uturn = 1.0 if (ub == va and vb == ua) else 0.0
                        trans_log = -conn_pen - switch_cost - uturn_cost * is_uturn
                    else:
                        trans_log = -gap_disc_pen

                cand_score = dp[t - 1, i] + trans_log
                if cand_score > best_score:
                    best_score = cand_score
                    best_prev = i

            dp[t, j] = best_score + emit_log[t, j]
            back[t, j] = best_prev

    best_idx = np.zeros(n_obs, dtype=np.int32)
    best_idx[-1] = int(np.argmax(dp[-1]))
    for t in range(n_obs - 1, 0, -1):
        best_idx[t - 1] = back[t, best_idx[t]]

    return cand_arcs[np.arange(n_obs), best_idx], n_dijkstra_trunc


def reconstruct_continuous_route(
    pred_obs_arcs: np.ndarray,
    arcs: np.ndarray,
    adj: list[list[tuple[int, int, float]]],
    max_bridge_m: float = 1500.0,
) -> list[int]:
    """Collapse consecutive duplicate arcs and bridge multi-hop gaps via shortest directed paths."""
    if len(pred_obs_arcs) == 0:
        return []

    collapsed = [int(pred_obs_arcs[0])]
    for a in pred_obs_arcs[1:]:
        ai = int(a)
        if ai != collapsed[-1]:
            collapsed.append(ai)

    reconstructed = [collapsed[0]]
    for next_arc in collapsed[1:]:
        prev_arc = reconstructed[-1]
        v_prev = int(arcs[prev_arc, 1])
        u_next = int(arcs[next_arc, 0])
        if v_prev != u_next:
            bridge = shortest_arc_path(adj, v_prev, u_next, max_dist_m=max_bridge_m, max_pops=3000)
            if bridge:
                for b_arc in bridge:
                    if b_arc != reconstructed[-1]:
                        reconstructed.append(b_arc)
        if next_arc != reconstructed[-1]:
            reconstructed.append(next_arc)

    return reconstructed


def compute_sequence_metrics(
    pred_obs_arcs: np.ndarray,
    reconstructed_route: list[int],
    gt_route: np.ndarray,
    arcs: np.ndarray,
) -> dict[str, float]:
    """Compute observation continuity, reconstructed route continuity, and arc set F1/Jaccard vs .route."""
    n = len(pred_obs_arcs)
    if n > 1:
        same = pred_obs_arcs[1:] == pred_obs_arcs[:-1]
        adj_conn = arcs[pred_obs_arcs[:-1], 1] == arcs[pred_obs_arcs[1:], 0]
        obs_cont = 100.0 * float(np.mean(same | adj_conn))
    else:
        obs_cont = 100.0

    if len(reconstructed_route) > 1:
        r_arr = np.asarray(reconstructed_route, dtype=np.int64)
        seq_cont = 100.0 * float(np.mean(arcs[r_arr[:-1], 1] == arcs[r_arr[1:], 0]))
    else:
        seq_cont = 100.0

    pred_set = set(reconstructed_route)
    gt_set = set(int(x) for x in gt_route.tolist())
    inter = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    jaccard = 100.0 * inter / union if union else 0.0
    prec = 100.0 * inter / len(pred_set) if pred_set else 0.0
    rec = 100.0 * inter / len(gt_set) if gt_set else 0.0
    seq_f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "obs_continuity_pct": obs_cont,
        "seq_continuity_pct": seq_cont,
        "route_arc_precision_pct": prec,
        "route_arc_recall_pct": rec,
        "route_arc_f1_pct": seq_f1,
        "route_arc_jaccard_pct": jaccard,
        "n_reconstructed_arcs": len(reconstructed_route),
    }


def predict_trip_route(
    trip_dir: str | Path,
    clf: Any,
    hmm_config: dict[str, float],
) -> dict[str, Any]:
    """Inference-only entry point for predicting a trip's observation arcs and reconstructed route."""
    feats = extract_trip_features(trip_dir, k=8, include_labels=False)
    data = load_trip_raw(trip_dir)
    nodes, arcs = data["nodes"], data["arcs"]
    _, _, epsg = project_coords_to_utm(data["track"][:, 0], data["track"][:, 1])
    nx, ny, _ = project_coords_to_utm(nodes[:, 0], nodes[:, 1], epsg=epsg)
    adj, arc_lens = build_directed_graph(np.column_stack([nx, ny]), arcs)

    X_3d = feats["X"]
    n_obs, k, n_feat = X_3d.shape
    probs_2d = clf.predict_proba(X_3d.reshape(-1, n_feat))[:, 1].reshape(n_obs, k)
    proj_idx = FEATURE_NAMES.index("proj_fraction")

    hmm_obs_arcs, _ = viterbi_decode_trip(
        probs_2d,
        feats["cand_arcs"],
        X_3d[:, :, proj_idx],
        feats["step_dist"],
        feats["dt_prev"],
        arcs,
        adj,
        arc_lens,
        **hmm_config,
    )
    route_arcs = reconstruct_continuous_route(hmm_obs_arcs, arcs, adj)
    return {
        "trip_id": feats["trip_id"],
        "obs_arcs": hmm_obs_arcs,
        "reconstructed_route": route_arcs,
    }


def tune_hmm_on_train(
    clf: Any,
    tr_cache: Any,
    sample_trip_ids: tuple[str, ...] = ("00000001", "00000003", "00000004", "00000006", "00000008", "00000010"),
) -> tuple[dict[str, float], pd.DataFrame]:
    """Tune Viterbi decoder parameters strictly on representative train trips."""
    trip_names = [str(x) for x in tr_cache["trip_names"]]
    offsets = tr_cache["obs_offsets"]
    X_3d_all = tr_cache["X"]
    cand_all = tr_cache["cand_arcs"]
    sd_all = tr_cache["step_dist"]
    dt_all = tr_cache["dt_prev"]
    proj_idx = FEATURE_NAMES.index("proj_fraction")

    preloaded = []
    for tid in sample_trip_ids:
        idx = trip_names.index(tid)
        s, e = int(offsets[idx]), int(offsets[idx + 1])
        data = load_trip_raw(BASE_DIR / "train" / tid)
        nodes, arcs, route = data["nodes"], data["arcs"], data["route"]
        _, _, epsg = project_coords_to_utm(data["track"][:, 0], data["track"][:, 1])
        nx, ny, _ = project_coords_to_utm(nodes[:, 0], nodes[:, 1], epsg=epsg)
        adj, arc_lens = build_directed_graph(np.column_stack([nx, ny]), arcs)
        X_sub = X_3d_all[s:e]
        probs = clf.predict_proba(X_sub.reshape(-1, X_sub.shape[-1]))[:, 1].reshape(e - s, X_sub.shape[1])
        preloaded.append((tid, probs, cand_all[s:e], X_sub[:, :, proj_idx], sd_all[s:e], dt_all[s:e], arcs, adj, arc_lens, route))

    grid = [
        {"alpha_emit": 1.0, "beta_trans": 0.00, "switch_cost": 0.00, "uturn_cost": 0.0, "disconnect_penalty": 25.0, "max_conn_penalty": 12.0},
        {"alpha_emit": 1.0, "beta_trans": 0.10, "switch_cost": 0.10, "uturn_cost": 1.5, "disconnect_penalty": 25.0, "max_conn_penalty": 12.0},
        {"alpha_emit": 1.0, "beta_trans": 0.15, "switch_cost": 0.15, "uturn_cost": 1.5, "disconnect_penalty": 25.0, "max_conn_penalty": 12.0},
        {"alpha_emit": 1.0, "beta_trans": 0.25, "switch_cost": 0.20, "uturn_cost": 2.0, "disconnect_penalty": 25.0, "max_conn_penalty": 12.0},
        {"alpha_emit": 1.5, "beta_trans": 0.15, "switch_cost": 0.15, "uturn_cost": 1.5, "disconnect_penalty": 25.0, "max_conn_penalty": 12.0},
        {"alpha_emit": 1.0, "beta_trans": 0.15, "switch_cost": 0.50, "uturn_cost": 2.0, "disconnect_penalty": 30.0, "max_conn_penalty": 12.0},
    ]

    records = []
    best_cfg = grid[2]
    best_score = -1.0
    for cfg_id, cfg in enumerate(grid):
        f1s, conts = [], []
        for _, probs, cands, proj_f, sd, dt, arcs, adj, arc_lens, route in preloaded:
            obs_arcs, _ = viterbi_decode_trip(probs, cands, proj_f, sd, dt, arcs, adj, arc_lens, **cfg)
            rec_route = reconstruct_continuous_route(obs_arcs, arcs, adj)
            m = compute_sequence_metrics(obs_arcs, rec_route, route, arcs)
            f1s.append(m["route_arc_f1_pct"])
            conts.append(m["seq_continuity_pct"])
        mean_f1 = float(np.mean(f1s))
        mean_cont = float(np.mean(conts))
        records.append({"config_id": cfg_id, **cfg, "train_subset_route_f1_pct": round(mean_f1, 4), "train_subset_seq_continuity_pct": round(mean_cont, 4)})
        if mean_f1 > best_score:
            best_score = mean_f1
            best_cfg = cfg

    return best_cfg, pd.DataFrame(records)


def evaluate_hmm_vs_classifier() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Tune decoder on train subset and evaluate Classifier-Only vs. Classifier + HMM on student_test."""
    clf = joblib.load(MODELS_ARTIFACTS_DIR / "best_lgbm_tuned.joblib")
    tr_cache = np.load(FEATURES_CACHE_DIR / "train_k8.npz")
    te = np.load(FEATURES_CACHE_DIR / "student_test_k8.npz")

    best_cfg, tune_df = tune_hmm_on_train(clf, tr_cache)
    with open(HMM_ARTIFACTS_DIR / "hmm_config.json", "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2)

    X_te_3d = te["X"]
    cand_te = te["cand_arcs"]
    gt_te = te["gt_arcs"]
    dt_te = te["dt_prev"]
    sd_te = te["step_dist"]
    trip_names = te["trip_names"]
    offsets = te["obs_offsets"]

    n_te_obs, k, n_feat = X_te_3d.shape
    proj_idx = FEATURE_NAMES.index("proj_fraction")

    t_clf0 = time.perf_counter()
    probs_all = clf.predict_proba(X_te_3d.reshape(-1, n_feat))[:, 1].reshape(n_te_obs, k)
    clf_time_s = time.perf_counter() - t_clf0

    per_trip_rows: list[dict[str, Any]] = []
    sample_preds: dict[str, Any] = {}
    t_hmm_total = 0.0
    total_trunc = 0

    for idx, tid_b in enumerate(trip_names):
        tid = str(tid_b)
        s, e = int(offsets[idx]), int(offsets[idx + 1])

        data = load_trip_raw(BASE_DIR / "student_test" / tid)
        nodes, arcs, route = data["nodes"], data["arcs"], data["route"]
        _, _, epsg = project_coords_to_utm(data["track"][:, 0], data["track"][:, 1])
        nx, ny, _ = project_coords_to_utm(nodes[:, 0], nodes[:, 1], epsg=epsg)
        adj, arc_lens = build_directed_graph(np.column_stack([nx, ny]), arcs)

        probs_trip = probs_all[s:e]
        cands_trip = cand_te[s:e]
        gt_trip = gt_te[s:e]
        proj_trip = X_te_3d[s:e, :, proj_idx]
        sd_trip = sd_te[s:e]
        dt_trip = dt_te[s:e]

        clf_pred = cands_trip[np.arange(e - s), np.argmax(probs_trip, axis=1)]
        clf_route = reconstruct_continuous_route(clf_pred, arcs, adj)
        clf_top1 = 100.0 * float(np.mean(clf_pred == gt_trip))
        clf_seq = compute_sequence_metrics(clf_pred, clf_route, route, arcs)

        t_h0 = time.perf_counter()
        hmm_pred, n_trunc = viterbi_decode_trip(
            probs_trip, cands_trip, proj_trip, sd_trip, dt_trip, arcs, adj, arc_lens, **best_cfg
        )
        hmm_route = reconstruct_continuous_route(hmm_pred, arcs, adj)
        t_h1 = time.perf_counter() - t_h0
        t_hmm_total += t_h1
        total_trunc += n_trunc

        hmm_top1 = 100.0 * float(np.mean(hmm_pred == gt_trip))
        hmm_seq = compute_sequence_metrics(hmm_pred, hmm_route, route, arcs)

        if tid == "00000000":
            sample_preds["clf_00000000"] = clf_route
            sample_preds["hmm_00000000"] = hmm_route

        per_trip_rows.append(
            {
                "trip_id": tid,
                "n_obs": e - s,
                "n_route": len(route),
                "clf_top1_acc_pct": round(clf_top1, 4),
                "hmm_top1_acc_pct": round(hmm_top1, 4),
                "delta_top1_pp": round(hmm_top1 - clf_top1, 4),
                "clf_obs_continuity_pct": round(clf_seq["obs_continuity_pct"], 4),
                "hmm_obs_continuity_pct": round(hmm_seq["obs_continuity_pct"], 4),
                "clf_seq_continuity_pct": round(clf_seq["seq_continuity_pct"], 4),
                "hmm_seq_continuity_pct": round(hmm_seq["seq_continuity_pct"], 4),
                "clf_route_precision_pct": round(clf_seq["route_arc_precision_pct"], 4),
                "hmm_route_precision_pct": round(hmm_seq["route_arc_precision_pct"], 4),
                "clf_route_recall_pct": round(clf_seq["route_arc_recall_pct"], 4),
                "hmm_route_recall_pct": round(hmm_seq["route_arc_recall_pct"], 4),
                "clf_route_f1_pct": round(clf_seq["route_arc_f1_pct"], 4),
                "hmm_route_f1_pct": round(hmm_seq["route_arc_f1_pct"], 4),
                "clf_route_jaccard_pct": round(clf_seq["route_arc_jaccard_pct"], 4),
                "hmm_route_jaccard_pct": round(hmm_seq["route_arc_jaccard_pct"], 4),
                "dijkstra_truncations": n_trunc,
                "hmm_decode_ms": round(1000.0 * t_h1, 2),
            }
        )

    trip_df = pd.DataFrame(per_trip_rows)
    weights = trip_df["n_obs"].to_numpy(dtype=np.float64)
    summary_rows = [
        {
            "method": "Classifier-Only (Tuned LightGBM)",
            "weighted_top1_acc_pct": round(float(np.average(trip_df["clf_top1_acc_pct"], weights=weights)), 4),
            "mean_obs_continuity_pct": round(float(trip_df["clf_obs_continuity_pct"].mean()), 4),
            "mean_seq_continuity_pct": round(float(trip_df["clf_seq_continuity_pct"].mean()), 4),
            "mean_route_precision_pct": round(float(trip_df["clf_route_precision_pct"].mean()), 4),
            "mean_route_recall_pct": round(float(trip_df["clf_route_recall_pct"].mean()), 4),
            "mean_route_f1_pct": round(float(trip_df["clf_route_f1_pct"].mean()), 4),
            "mean_route_jaccard_pct": round(float(trip_df["clf_route_jaccard_pct"].mean()), 4),
            "total_dijkstra_truncations": 0,
            "total_inference_s": round(clf_time_s, 4),
        },
        {
            "method": "Classifier + HMM/Viterbi",
            "weighted_top1_acc_pct": round(float(np.average(trip_df["hmm_top1_acc_pct"], weights=weights)), 4),
            "mean_obs_continuity_pct": round(float(trip_df["hmm_obs_continuity_pct"].mean()), 4),
            "mean_seq_continuity_pct": round(float(trip_df["hmm_seq_continuity_pct"].mean()), 4),
            "mean_route_precision_pct": round(float(trip_df["hmm_route_precision_pct"].mean()), 4),
            "mean_route_recall_pct": round(float(trip_df["hmm_route_recall_pct"].mean()), 4),
            "mean_route_f1_pct": round(float(trip_df["hmm_route_f1_pct"].mean()), 4),
            "mean_route_jaccard_pct": round(float(trip_df["hmm_route_jaccard_pct"].mean()), 4),
            "total_dijkstra_truncations": total_trunc,
            "total_inference_s": round(clf_time_s + t_hmm_total, 4),
        },
    ]
    return pd.DataFrame(summary_rows), trip_df, tune_df, sample_preds


def plot_hmm_results(summary_df: pd.DataFrame, trip_df: pd.DataFrame, sample_preds: dict[str, Any]) -> None:
    """Create HMM vs Classifier comparison plot and visual predicted route map."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    metrics = [
        ("weighted_top1_acc_pct", "Obs Top-1 Acc (%)"),
        ("mean_obs_continuity_pct", "Obs Continuity (%)"),
        ("mean_seq_continuity_pct", "Reconstructed Seq Cont (%)"),
        ("mean_route_f1_pct", "Route Arc F1 vs .route (%)"),
    ]
    x = np.arange(len(metrics))
    w = 0.35
    v_clf = [summary_df.loc[0, m] for m, _ in metrics]
    v_hmm = [summary_df.loc[1, m] for m, _ in metrics]

    b1 = axes[0].bar(x - w / 2, v_clf, width=w, color="#64748b", label="Classifier-Only")
    b2 = axes[0].bar(x + w / 2, v_hmm, width=w, color="#16a34a", label="Classifier + HMM/Viterbi")
    for bars in (b1, b2):
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.7,
                f"{bar.get_height():.2f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([lbl for _, lbl in metrics], rotation=12, ha="right", fontsize=8.5)
    axes[0].set_ylim(55, 108)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("Student-Test: Classifier-Only vs. Classifier + HMM/Viterbi")
    axes[0].legend()

    x2 = np.arange(len(trip_df))
    axes[1].bar(x2 - w / 2, trip_df["clf_route_f1_pct"], width=w, color="#64748b", label="Classifier-Only Route F1")
    axes[1].bar(x2 + w / 2, trip_df["hmm_route_f1_pct"], width=w, color="#16a34a", label="HMM/Viterbi Route F1")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(trip_df["trip_id"], rotation=30, ha="right", fontsize=8)
    axes[1].set_ylim(50, 105)
    axes[1].set_ylabel("Route Arc F1 vs. .route (%)")
    axes[1].set_title("Per-Trip Route Arc F1 Across 10 Student-Test Trips")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(HMM_FIGURES_DIR / "fig1_classifier_vs_hmm_metrics.png", dpi=180)
    plt.close(fig)

    d0 = load_trip_raw(BASE_DIR / "student_test" / "00000000")
    tx, ty, epsg = project_coords_to_utm(d0["track"][:, 0], d0["track"][:, 1])
    nx, ny, _ = project_coords_to_utm(d0["nodes"][:, 0], d0["nodes"][:, 1], epsg=epsg)
    cx0, cy0 = tx[0], ty[0]
    tx_km = (tx - cx0) / 1000.0
    ty_km = (ty - cy0) / 1000.0
    nxy_km = np.column_stack([(nx - cx0) / 1000.0, (ny - cy0) / 1000.0])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, key, title, col in [
        (axes[0], "clf_00000000", "student_test/00000000: Classifier-Only Route", "#dc2626"),
        (axes[1], "hmm_00000000", "student_test/00000000: Classifier + HMM/Viterbi Route", "#ea580c"),
    ]:
        route = d0["route"]
        arcs = d0["arcs"]
        pred_arcs = np.asarray(sample_preds[key], dtype=np.int64)
        ur, vr = nxy_km[arcs[route, 0]], nxy_km[arcs[route, 1]]
        rxs = np.column_stack([ur[:, 0], vr[:, 0], np.full(len(route), np.nan)]).ravel()
        rys = np.column_stack([ur[:, 1], vr[:, 1], np.full(len(route), np.nan)]).ravel()
        ax.plot(rxs, rys, color="#16a34a", linewidth=3.4, alpha=0.7, label="Reference Route (.route)")

        up, vp = nxy_km[arcs[pred_arcs, 0]], nxy_km[arcs[pred_arcs, 1]]
        pxs = np.column_stack([up[:, 0], vp[:, 0], np.full(len(pred_arcs), np.nan)]).ravel()
        pys = np.column_stack([up[:, 1], vp[:, 1], np.full(len(pred_arcs), np.nan)]).ravel()
        ax.plot(pxs, pys, color=col, linewidth=2.0, linestyle="--", alpha=0.95, label="Predicted Route")

        ax.scatter(tx_km, ty_km, c="#2563eb", s=4, alpha=0.4, label="Raw GPS (.track)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("Relative Easting (km)")
        ax.set_ylabel("Relative Northing (km)")
        ax.set_title(title, fontsize=10.5)
        ax.legend(loc="best", fontsize=8.5)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(HMM_FIGURES_DIR / "fig2_predicted_vs_reference_route.png", dpi=180)
    plt.close(fig)

    create_trip_folium_map(
        BASE_DIR / "student_test" / "00000000",
        HMM_MAPS_DIR / "trip_student_test_00000000_predicted.html",
        predicted_arcs=sample_preds["hmm_00000000"],
        buffer_m=150.0,
    )


def main() -> None:
    HMM_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    HMM_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    HMM_MAPS_DIR.mkdir(parents=True, exist_ok=True)
    HMM_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_df, trip_df, tune_df, sample_preds = evaluate_hmm_vs_classifier()
    summary_df.to_csv(HMM_REPORTS_DIR / "hmm_vs_classifier_comparison.csv", index=False)
    trip_df.to_csv(HMM_REPORTS_DIR / "per_trip_hmm_results.csv", index=False)
    tune_df.to_csv(HMM_REPORTS_DIR / "hmm_tuning_trials.csv", index=False)
    plot_hmm_results(summary_df, trip_df, sample_preds)

    print("\n=== HMM Tuning Grid on Train Subset ===")
    print(tune_df.to_string(index=False))
    print("\n=== Classifier-Only vs. Classifier + HMM/Viterbi (student_test) ===")
    print(summary_df.to_string(index=False))
    print("\n=== Per-Trip Summary ===")
    print(
        trip_df[
            [
                "trip_id",
                "clf_top1_acc_pct",
                "hmm_top1_acc_pct",
                "clf_seq_continuity_pct",
                "hmm_seq_continuity_pct",
                "clf_route_f1_pct",
                "hmm_route_f1_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
