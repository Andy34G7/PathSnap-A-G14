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
from src.features.candidate_generation import (
    align_route_to_observations,
    compute_smoothed_velocities,
)
from src.features.feature_engineering import FEATURES_CACHE_DIR, FEATURE_NAMES, extract_trip_features
from src.models.train_and_compare import MODELS_ARTIFACTS_DIR, MODELS_REPORTS_DIR

HMM_REPORTS_DIR = BASE_DIR / "reports" / "hmm"
HMM_FIGURES_DIR = HMM_REPORTS_DIR / "figures"
HMM_MAPS_DIR = HMM_REPORTS_DIR / "maps"
HMM_ARTIFACTS_DIR = BASE_DIR / "artifacts" / "hmm"

# Frozen Iteration 1 baseline metrics from commit ba4d85e on student_test (19,160 obs, 10 trips)
ITERATION1_BASELINE_ROWS: list[dict[str, Any]] = [
    {
        "stage_id": "Stage 0",
        "stage_name": "Iter 1 Baseline Classifier-Only (commit ba4d85e)",
        "weighted_top1_acc_pct": 88.2777,
        "other9_top1_acc_pct": 91.7437,
        "trip77_top1_acc_pct": 74.7635,
        "mean_obs_continuity_pct": 93.8226,
        "mean_seq_continuity_pct": 98.7554,
        "mean_route_precision_pct": 67.1247,
        "mean_route_recall_pct": 99.3877,
        "mean_route_f1_pct": 79.2929,
        "mean_route_jaccard_pct": 66.7195,
        "cand_miss_errors": 78,
        "opp_twin_errors": None,
        "adj_boundary_errors": None,
        "other_detour_errors": None,
        "total_errors": 2246,
        "binary_macro_f1": 0.9308,
    },
    {
        "stage_id": "Stage 1",
        "stage_name": "Iter 1 Baseline HMM/Viterbi (commit ba4d85e)",
        "weighted_top1_acc_pct": 93.9405,
        "other9_top1_acc_pct": 95.7243,
        "trip77_top1_acc_pct": 86.9854,
        "mean_obs_continuity_pct": 97.6465,
        "mean_seq_continuity_pct": 99.8129,
        "mean_route_precision_pct": 92.8040,
        "mean_route_recall_pct": 99.0583,
        "mean_route_f1_pct": 95.7142,
        "mean_route_jaccard_pct": 91.9485,
        "cand_miss_errors": 78,
        "opp_twin_errors": None,
        "adj_boundary_errors": None,
        "other_detour_errors": None,
        "total_errors": 1161,
        "binary_macro_f1": 0.9308,
    },
]


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
    """Recover directed arc_ids along the shortest path from start_node to target_node."""
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
    cand_dists: np.ndarray | None = None,
    cos_sm: np.ndarray | None = None,
    alpha_emit: float = 1.0,
    geom_weight: float = 0.5,
    beta_trans: float = 0.35,
    switch_cost: float = 1.0,
    sp_weight: float = 0.1,
    uturn_cost: float = 8.0,
    backtrack_cost: float = 4.0,
    disconnect_penalty: float = 80.0,
    max_conn_penalty: float = 30.0,
) -> tuple[np.ndarray, int]:
    """Decode the optimal road-arc sequence using hybrid classifier + geometric Viterbi DP."""
    n_obs, k = cand_arcs.shape
    emit_log = alpha_emit * np.log(np.clip(probs_2d, 1e-6, 1.0 - 1e-6))
    if geom_weight > 0.0 and cand_dists is not None and cos_sm is not None:
        emit_log = emit_log - geom_weight * (cand_dists + 1.5 * (1.0 - cos_sm))

    dp = np.full((n_obs, k), -np.inf, dtype=np.float64)
    back = np.zeros((n_obs, k), dtype=np.int32)
    dp[0] = emit_log[0]

    sp_cache: dict[tuple[int, int, float], float] = {}
    n_dijkstra_trunc = 0
    sp_cap = 25.0

    for t in range(1, n_obs):
        d_gps = float(step_dist[t])
        dt = max(1.0, float(dt_prev[t]))
        gap_scale = np.sqrt(dt)
        sp_gap_scale = 1.0 if dt <= 5.0 else np.sqrt(dt / 5.0)
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

        # Keep disconnect penalty strictly above capped connected penalty + sp_cap across gaps
        gap_disc_pen = (
            disconnect_penalty
            if dt <= 5.0
            else max(max_conn_penalty + sp_cap + 10.0, disconnect_penalty / np.sqrt(dt / 5.0))
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
                    backward_pen = backtrack_cost if pt_cur < pt_prev - 0.12 else 0.0
                    conn_pen = min(max_conn_penalty, beta_trans * abs(d_road - d_gps) / gap_scale)
                    trans_log = -conn_pen - backward_pen
                else:
                    sp_dist = v_to_u_dists[va].get(ub, float("inf"))
                    if np.isfinite(sp_dist):
                        d_road = (1.0 - pt_prev) * float(arc_lens[a]) + sp_dist + pt_cur * len_b
                        conn_pen = min(max_conn_penalty, beta_trans * abs(d_road - d_gps) / gap_scale)
                        sp_pen = min(sp_cap, sp_weight * sp_dist / sp_gap_scale)
                        is_uturn = 1.0 if (ub == va and vb == ua) else 0.0
                        trans_log = -conn_pen - switch_cost - sp_pen - uturn_cost * is_uturn
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


def realign_route_boundaries(
    raw_hmm_obs_arcs: np.ndarray,
    reconstructed_route: list[int],
    pts_xy: np.ndarray,
    nodes_xy: np.ndarray,
    arcs: np.ndarray,
    v_sm_u: np.ndarray,
    cand_arcs: np.ndarray | None = None,
    snap_to_candidates: bool = False,
) -> np.ndarray:
    """Eliminate Viterbi transition lag by monotonically aligning GPS pings along reconstructed_route."""
    if len(raw_hmm_obs_arcs) == 0 or len(reconstructed_route) == 0:
        return raw_hmm_obs_arcs

    route_arr = np.asarray(reconstructed_route, dtype=np.int64)
    realigned_arcs, _ = align_route_to_observations(pts_xy, nodes_xy, arcs, route_arr, v_sm_u)
    if snap_to_candidates and cand_arcs is not None:
        in_cands = np.any(cand_arcs == realigned_arcs[:, None], axis=1)
        return np.where(in_cands, realigned_arcs, raw_hmm_obs_arcs).astype(np.int64)
    return realigned_arcs.astype(np.int64)


def decode_and_realign_trip(
    probs_2d: np.ndarray,
    cand_arcs: np.ndarray,
    proj_frac: np.ndarray,
    cand_dists: np.ndarray,
    cos_sm: np.ndarray,
    step_dist: np.ndarray,
    dt_prev: np.ndarray,
    pts_xy: np.ndarray,
    nodes_xy: np.ndarray,
    arcs: np.ndarray,
    adj: list[list[tuple[int, int, float]]],
    arc_lens: np.ndarray,
    v_sm_u: np.ndarray,
    hmm_config: dict[str, float],
    snap_to_candidates: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[int], int]:
    """Run Stage 3 Viterbi decoding, route bridging, and Stage 4 boundary realignment."""
    raw_obs_arcs, n_trunc = viterbi_decode_trip(
        probs_2d,
        cand_arcs,
        proj_frac,
        step_dist,
        dt_prev,
        arcs,
        adj,
        arc_lens,
        cand_dists=cand_dists,
        cos_sm=cos_sm,
        **hmm_config,
    )
    route_arcs = reconstruct_continuous_route(raw_obs_arcs, arcs, adj)
    final_obs_arcs = realign_route_boundaries(
        raw_obs_arcs,
        route_arcs,
        pts_xy,
        nodes_xy,
        arcs,
        v_sm_u,
        cand_arcs=cand_arcs,
        snap_to_candidates=snap_to_candidates,
    )
    return raw_obs_arcs, final_obs_arcs, route_arcs, n_trunc


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


def classify_error_buckets(
    pred_obs_arcs: np.ndarray,
    gt_obs_arcs: np.ndarray,
    cand_arcs: np.ndarray,
    arcs: np.ndarray,
) -> dict[str, int]:
    """Decompose observation mismatches into candidate-miss, opp-twin, boundary-lag, and detour buckets."""
    mis_idx = np.where(pred_obs_arcs != gt_obs_arcs)[0]
    cand_miss = 0
    opp_twin = 0
    adj_boundary = 0
    other_detour = 0

    for t in mis_idx.tolist():
        p = int(pred_obs_arcs[t])
        g = int(gt_obs_arcs[t])
        up, vp = int(arcs[p, 0]), int(arcs[p, 1])
        ug, vg = int(arcs[g, 0]), int(arcs[g, 1])
        if g not in cand_arcs[t]:
            cand_miss += 1
        elif up == vg and vp == ug:
            opp_twin += 1
        elif vp == ug or vg == up:
            adj_boundary += 1
        else:
            other_detour += 1

    return {
        "cand_miss_errors": cand_miss,
        "opp_twin_errors": opp_twin,
        "adj_boundary_errors": adj_boundary,
        "other_detour_errors": other_detour,
        "total_errors": int(len(mis_idx)),
    }


def predict_trip_route(
    trip_dir: str | Path,
    clf: Any,
    hmm_config: dict[str, float],
) -> dict[str, Any]:
    """Inference-only entry point returning Stage 3 HMM route and Stage 4 realigned observation arcs."""
    feats = extract_trip_features(trip_dir, k=8, include_labels=False)
    data = load_trip_raw(trip_dir)
    track, nodes, arcs = data["track"], data["nodes"], data["arcs"]

    tx, ty, epsg = project_coords_to_utm(track[:, 0], track[:, 1])
    nx, ny, _ = project_coords_to_utm(nodes[:, 0], nodes[:, 1], epsg=epsg)
    pts_xy = np.column_stack([tx, ty])
    nodes_xy = np.column_stack([nx, ny])
    adj, arc_lens = build_directed_graph(nodes_xy, arcs)
    _, _, _, _, v_sm_u = compute_smoothed_velocities(pts_xy, track[:, 2])

    X_3d = feats["X"]
    n_obs, k, n_feat = X_3d.shape
    probs_2d = clf.predict_proba(X_3d.reshape(-1, n_feat))[:, 1].reshape(n_obs, k)
    proj_idx = FEATURE_NAMES.index("proj_fraction")
    dist_idx = FEATURE_NAMES.index("dist_to_arc_m")
    cos_idx = FEATURE_NAMES.index("cos_heading_diff_smooth")

    raw_obs_arcs, final_obs_arcs, route_arcs, n_trunc = decode_and_realign_trip(
        probs_2d,
        feats["cand_arcs"],
        X_3d[:, :, proj_idx],
        X_3d[:, :, dist_idx],
        X_3d[:, :, cos_idx],
        feats["step_dist"],
        feats["dt_prev"],
        pts_xy,
        nodes_xy,
        arcs,
        adj,
        arc_lens,
        v_sm_u,
        hmm_config,
        snap_to_candidates=False,
    )
    return {
        "trip_id": feats["trip_id"],
        "obs_arcs": final_obs_arcs,
        "raw_hmm_obs_arcs": raw_obs_arcs,
        "reconstructed_route": route_arcs,
        "dijkstra_truncations": n_trunc,
    }


def tune_hmm_on_train(
    clf: Any,
    tr_cache: Any,
    sample_trip_ids: tuple[str, ...] = ("00000001", "00000004", "00000013", "00000018", "00000022", "00000035"),
) -> tuple[dict[str, float], pd.DataFrame]:
    """Tune hybrid Viterbi parameters strictly on 6 diverse stratified train trips."""
    trip_names = [str(x) for x in tr_cache["trip_names"]]
    offsets = tr_cache["obs_offsets"]
    X_3d_all = tr_cache["X"]
    cand_all = tr_cache["cand_arcs"]
    gt_all = tr_cache["gt_arcs"]
    sd_all = tr_cache["step_dist"]
    dt_all = tr_cache["dt_prev"]
    proj_idx = FEATURE_NAMES.index("proj_fraction")
    dist_idx = FEATURE_NAMES.index("dist_to_arc_m")
    cos_idx = FEATURE_NAMES.index("cos_heading_diff_smooth")

    preloaded = []
    for tid in sample_trip_ids:
        idx = trip_names.index(tid)
        s, e = int(offsets[idx]), int(offsets[idx + 1])
        data = load_trip_raw(BASE_DIR / "train" / tid)
        track, nodes, arcs, route = data["track"], data["nodes"], data["arcs"], data["route"]
        tx, ty, epsg = project_coords_to_utm(track[:, 0], track[:, 1])
        nx, ny, _ = project_coords_to_utm(nodes[:, 0], nodes[:, 1], epsg=epsg)
        pts_xy = np.column_stack([tx, ty])
        nodes_xy = np.column_stack([nx, ny])
        adj, arc_lens = build_directed_graph(nodes_xy, arcs)
        _, _, _, _, v_sm_u = compute_smoothed_velocities(pts_xy, track[:, 2])
        X_sub = X_3d_all[s:e]
        probs = clf.predict_proba(X_sub.reshape(-1, X_sub.shape[-1]))[:, 1].reshape(e - s, X_sub.shape[1])
        preloaded.append(
            (
                tid,
                probs,
                cand_all[s:e],
                gt_all[s:e],
                X_sub[:, :, proj_idx],
                X_sub[:, :, dist_idx],
                X_sub[:, :, cos_idx],
                sd_all[s:e],
                dt_all[s:e],
                pts_xy,
                nodes_xy,
                arcs,
                adj,
                arc_lens,
                v_sm_u,
                route,
            )
        )

    grid = [
        {
            "alpha_emit": 1.0, "geom_weight": 0.0, "beta_trans": 0.15, "switch_cost": 0.50,
            "sp_weight": 0.00, "uturn_cost": 2.0, "backtrack_cost": 1.5,
            "disconnect_penalty": 30.0, "max_conn_penalty": 12.0,
        },
        {
            "alpha_emit": 1.0, "geom_weight": 0.50, "beta_trans": 0.35, "switch_cost": 1.00,
            "sp_weight": 0.10, "uturn_cost": 8.0, "backtrack_cost": 4.0,
            "disconnect_penalty": 80.0, "max_conn_penalty": 30.0,
        },
        {
            "alpha_emit": 1.0, "geom_weight": 1.00, "beta_trans": 0.55, "switch_cost": 1.60,
            "sp_weight": 0.20, "uturn_cost": 12.0, "backtrack_cost": 6.0,
            "disconnect_penalty": 100.0, "max_conn_penalty": 45.0,
        },
        {
            "alpha_emit": 1.5, "geom_weight": 1.00, "beta_trans": 0.60, "switch_cost": 1.80,
            "sp_weight": 0.25, "uturn_cost": 12.0, "backtrack_cost": 6.0,
            "disconnect_penalty": 110.0, "max_conn_penalty": 45.0,
        },
        {
            "alpha_emit": 2.5, "geom_weight": 1.00, "beta_trans": 0.55, "switch_cost": 1.60,
            "sp_weight": 0.20, "uturn_cost": 12.0, "backtrack_cost": 6.0,
            "disconnect_penalty": 100.0, "max_conn_penalty": 45.0,
        },
        {
            "alpha_emit": 0.5, "geom_weight": 1.00, "beta_trans": 0.75, "switch_cost": 2.00,
            "sp_weight": 0.25, "uturn_cost": 12.0, "backtrack_cost": 6.0,
            "disconnect_penalty": 120.0, "max_conn_penalty": 60.0,
        },
    ]

    records = []
    best_cfg = grid[2]
    best_score = -1.0
    for cfg_id, cfg in enumerate(grid):
        f1s, conts, raw_accs, final_accs = [], [], [], []
        for (
            _, probs, cands, gt_obs, proj_f, cdists, cos_s, sd, dt,
            pts_xy, nodes_xy, arcs, adj, arc_lens, v_sm_u, route,
        ) in preloaded:
            raw_obs, final_obs, rec_route, _ = decode_and_realign_trip(
                probs, cands, proj_f, cdists, cos_s, sd, dt,
                pts_xy, nodes_xy, arcs, adj, arc_lens, v_sm_u, cfg,
            )
            m = compute_sequence_metrics(final_obs, rec_route, route, arcs)
            f1s.append(m["route_arc_f1_pct"])
            conts.append(m["seq_continuity_pct"])
            raw_accs.append(100.0 * float(np.mean(raw_obs == gt_obs)))
            final_accs.append(100.0 * float(np.mean(final_obs == gt_obs)))

        mean_f1 = float(np.mean(f1s))
        mean_cont = float(np.mean(conts))
        mean_raw_top1 = float(np.mean(raw_accs))
        mean_final_top1 = float(np.mean(final_accs))
        combined_score = 0.7 * mean_f1 + 0.3 * mean_final_top1
        records.append(
            {
                "config_id": cfg_id,
                **cfg,
                "train_subset_route_f1_pct": round(mean_f1, 4),
                "train_subset_raw_top1_pct": round(mean_raw_top1, 4),
                "train_subset_realigned_top1_pct": round(mean_final_top1, 4),
                "train_subset_seq_continuity_pct": round(mean_cont, 4),
            }
        )
        if combined_score > best_score:
            best_score = combined_score
            best_cfg = cfg

    return best_cfg, pd.DataFrame(records)


def _sync_experiment_config(best_cfg: dict[str, float]) -> None:
    """Persist hmm_config, dijkstra_max_pops, and realignment flag into experiment_config.json."""
    HMM_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HMM_ARTIFACTS_DIR / "hmm_config.json", "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2)

    exp_cfg_path = MODELS_REPORTS_DIR / "experiment_config.json"
    exp_cfg = json.loads(exp_cfg_path.read_text(encoding="utf-8")) if exp_cfg_path.exists() else {}
    exp_cfg["dijkstra_max_pops"] = 3000
    exp_cfg["stage4_continuous_realignment"] = True
    exp_cfg["hmm_tune_train_trips"] = ["00000001", "00000004", "00000013", "00000018", "00000022", "00000035"]
    exp_cfg["hmm_config"] = best_cfg
    exp_cfg_path.write_text(json.dumps(exp_cfg, indent=2) + "\n", encoding="utf-8")


def _aggregate_stage_row(
    stage_id: str,
    stage_name: str,
    top1_col: str,
    obs_cont_col: str,
    seq_cont_col: str,
    prec_col: str,
    rec_col: str,
    f1_col: str,
    jacc_col: str,
    trip_df: pd.DataFrame,
    bucket_totals: dict[str, int],
    macro_f1: float,
) -> dict[str, Any]:
    """Aggregate weighted and per-trip metrics plus error buckets for one ablation stage."""
    weights = trip_df["n_obs"].to_numpy(dtype=np.float64)
    w_top1 = float(np.average(trip_df[top1_col], weights=weights))
    non77 = trip_df[trip_df["trip_id"] != "00000077"]
    other9_top1 = float(np.average(non77[top1_col], weights=non77["n_obs"].to_numpy(dtype=np.float64)))
    t77_top1 = float(trip_df.loc[trip_df["trip_id"] == "00000077", top1_col].iloc[0])

    return {
        "stage_id": stage_id,
        "stage_name": stage_name,
        "weighted_top1_acc_pct": round(w_top1, 4),
        "other9_top1_acc_pct": round(other9_top1, 4),
        "trip77_top1_acc_pct": round(t77_top1, 4),
        "mean_obs_continuity_pct": round(float(trip_df[obs_cont_col].mean()), 4),
        "mean_seq_continuity_pct": round(float(trip_df[seq_cont_col].mean()), 4),
        "mean_route_precision_pct": round(float(trip_df[prec_col].mean()), 4),
        "mean_route_recall_pct": round(float(trip_df[rec_col].mean()), 4),
        "mean_route_f1_pct": round(float(trip_df[f1_col].mean()), 4),
        "mean_route_jaccard_pct": round(float(trip_df[jacc_col].mean()), 4),
        "cand_miss_errors": bucket_totals["cand_miss_errors"],
        "opp_twin_errors": bucket_totals["opp_twin_errors"],
        "adj_boundary_errors": bucket_totals["adj_boundary_errors"],
        "other_detour_errors": bucket_totals["other_detour_errors"],
        "total_errors": bucket_totals["total_errors"],
        "binary_macro_f1": round(macro_f1, 4),
    }


def evaluate_hmm_vs_classifier() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Tune decoder on stratified train subset and evaluate full 5-stage ablation on student_test."""
    clf = joblib.load(MODELS_ARTIFACTS_DIR / "best_lgbm_tuned.joblib")
    tr_cache = np.load(FEATURES_CACHE_DIR / "train_k8.npz")
    te = np.load(FEATURES_CACHE_DIR / "student_test_k8.npz")

    best_cfg, tune_df = tune_hmm_on_train(clf, tr_cache)
    _sync_experiment_config(best_cfg)

    comp_path = MODELS_REPORTS_DIR / "untuned_vs_tuned_comparison.csv"
    tuned_macro_f1 = 0.0
    if comp_path.exists():
        comp_df = pd.read_csv(comp_path)
        tuned_macro_f1 = float(comp_df.iloc[-1]["student_test_macro_f1"])

    X_te_3d = te["X"]
    cand_te = te["cand_arcs"]
    gt_te = te["gt_arcs"]
    dt_te = te["dt_prev"]
    sd_te = te["step_dist"]
    trip_names = te["trip_names"]
    offsets = te["obs_offsets"]

    n_te_obs, k, n_feat = X_te_3d.shape
    proj_idx = FEATURE_NAMES.index("proj_fraction")
    dist_idx = FEATURE_NAMES.index("dist_to_arc_m")
    cos_idx = FEATURE_NAMES.index("cos_heading_diff_smooth")

    t_clf0 = time.perf_counter()
    probs_all = clf.predict_proba(X_te_3d.reshape(-1, n_feat))[:, 1].reshape(n_te_obs, k)
    clf_time_s = time.perf_counter() - t_clf0

    per_trip_rows: list[dict[str, Any]] = []
    sample_preds: dict[str, Any] = {}
    t_hmm_total = 0.0
    total_trunc = 0

    bucket_sums = {
        stage: {
            "cand_miss_errors": 0,
            "opp_twin_errors": 0,
            "adj_boundary_errors": 0,
            "other_detour_errors": 0,
            "total_errors": 0,
        }
        for stage in ("clf", "raw_hmm", "final_hmm")
    }

    for idx, tid_b in enumerate(trip_names):
        tid = str(tid_b)
        s, e = int(offsets[idx]), int(offsets[idx + 1])

        data = load_trip_raw(BASE_DIR / "student_test" / tid)
        track, nodes, arcs, route = data["track"], data["nodes"], data["arcs"], data["route"]
        tx, ty, epsg = project_coords_to_utm(track[:, 0], track[:, 1])
        nx, ny, _ = project_coords_to_utm(nodes[:, 0], nodes[:, 1], epsg=epsg)
        pts_xy = np.column_stack([tx, ty])
        nodes_xy = np.column_stack([nx, ny])
        adj, arc_lens = build_directed_graph(nodes_xy, arcs)
        _, _, _, _, v_sm_u = compute_smoothed_velocities(pts_xy, track[:, 2])

        probs_trip = probs_all[s:e]
        cands_trip = cand_te[s:e]
        gt_trip = gt_te[s:e]
        proj_trip = X_te_3d[s:e, :, proj_idx]
        dist_trip = X_te_3d[s:e, :, dist_idx]
        cos_trip = X_te_3d[s:e, :, cos_idx]
        sd_trip = sd_te[s:e]
        dt_trip = dt_te[s:e]

        clf_pred = cands_trip[np.arange(e - s), np.argmax(probs_trip, axis=1)]
        clf_route = reconstruct_continuous_route(clf_pred, arcs, adj)
        clf_top1 = 100.0 * float(np.mean(clf_pred == gt_trip))
        clf_seq = compute_sequence_metrics(clf_pred, clf_route, route, arcs)
        clf_b = classify_error_buckets(clf_pred, gt_trip, cands_trip, arcs)

        t_h0 = time.perf_counter()
        raw_pred, hmm_pred, hmm_route, n_trunc = decode_and_realign_trip(
            probs_trip, cands_trip, proj_trip, dist_trip, cos_trip, sd_trip, dt_trip,
            pts_xy, nodes_xy, arcs, adj, arc_lens, v_sm_u, best_cfg,
        )
        t_h1 = time.perf_counter() - t_h0
        t_hmm_total += t_h1
        total_trunc += n_trunc

        raw_top1 = 100.0 * float(np.mean(raw_pred == gt_trip))
        raw_seq = compute_sequence_metrics(raw_pred, hmm_route, route, arcs)
        raw_b = classify_error_buckets(raw_pred, gt_trip, cands_trip, arcs)

        hmm_top1 = 100.0 * float(np.mean(hmm_pred == gt_trip))
        hmm_seq = compute_sequence_metrics(hmm_pred, hmm_route, route, arcs)
        hmm_b = classify_error_buckets(hmm_pred, gt_trip, cands_trip, arcs)

        flips_mask = hmm_pred != raw_pred
        flips_good = int(np.sum(flips_mask & (hmm_pred == gt_trip) & (raw_pred != gt_trip)))
        flips_bad = int(np.sum(flips_mask & (hmm_pred != gt_trip) & (raw_pred == gt_trip)))

        for st_key, b_dict in (("clf", clf_b), ("raw_hmm", raw_b), ("final_hmm", hmm_b)):
            for bk, bv in b_dict.items():
                bucket_sums[st_key][bk] += bv

        if tid == "00000000":
            sample_preds["clf_00000000"] = clf_route
            sample_preds["hmm_00000000"] = hmm_route

        per_trip_rows.append(
            {
                "trip_id": tid,
                "n_obs": e - s,
                "n_route": len(route),
                "clf_top1_acc_pct": round(clf_top1, 4),
                "hmm_raw_top1_acc_pct": round(raw_top1, 4),
                "hmm_top1_acc_pct": round(hmm_top1, 4),
                "delta_top1_pp": round(hmm_top1 - clf_top1, 4),
                "stage4_flips_total": int(np.sum(flips_mask)),
                "stage4_flips_good": flips_good,
                "stage4_flips_bad": flips_bad,
                "clf_obs_continuity_pct": round(clf_seq["obs_continuity_pct"], 4),
                "hmm_raw_obs_continuity_pct": round(raw_seq["obs_continuity_pct"], 4),
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
                "opp_twin_errors": hmm_b["opp_twin_errors"],
                "adj_boundary_errors": hmm_b["adj_boundary_errors"],
                "other_errors": hmm_b["cand_miss_errors"] + hmm_b["other_detour_errors"],
                "dijkstra_truncations": n_trunc,
                "hmm_decode_ms": round(1000.0 * t_h1, 2),
            }
        )

    trip_df = pd.DataFrame(per_trip_rows)
    weights = trip_df["n_obs"].to_numpy(dtype=np.float64)
    summary_rows = [
        {
            "method": "Classifier-Only (Upgraded Features + Tuned LightGBM)",
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
            "method": "Hybrid HMM/Viterbi (Pre-Realignment)",
            "weighted_top1_acc_pct": round(float(np.average(trip_df["hmm_raw_top1_acc_pct"], weights=weights)), 4),
            "mean_obs_continuity_pct": round(float(trip_df["hmm_raw_obs_continuity_pct"].mean()), 4),
            "mean_seq_continuity_pct": round(float(trip_df["hmm_seq_continuity_pct"].mean()), 4),
            "mean_route_precision_pct": round(float(trip_df["hmm_route_precision_pct"].mean()), 4),
            "mean_route_recall_pct": round(float(trip_df["hmm_route_recall_pct"].mean()), 4),
            "mean_route_f1_pct": round(float(trip_df["hmm_route_f1_pct"].mean()), 4),
            "mean_route_jaccard_pct": round(float(trip_df["hmm_route_jaccard_pct"].mean()), 4),
            "total_dijkstra_truncations": total_trunc,
            "total_inference_s": round(clf_time_s + t_hmm_total, 4),
        },
        {
            "method": "Hybrid HMM/Viterbi + Route Boundary Realignment",
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

    ablation_rows = list(ITERATION1_BASELINE_ROWS) + [
        _aggregate_stage_row(
            "Stage 2",
            "Upgraded Spatial-Chord & Window Features (Classifier-Only)",
            "clf_top1_acc_pct", "clf_obs_continuity_pct", "clf_seq_continuity_pct",
            "clf_route_precision_pct", "clf_route_recall_pct", "clf_route_f1_pct", "clf_route_jaccard_pct",
            trip_df, bucket_sums["clf"], tuned_macro_f1,
        ),
        _aggregate_stage_row(
            "Stage 3",
            "Hybrid Probabilistic-Geometric HMM/Viterbi (Pre-Realignment)",
            "hmm_raw_top1_acc_pct", "hmm_raw_obs_continuity_pct", "hmm_seq_continuity_pct",
            "hmm_route_precision_pct", "hmm_route_recall_pct", "hmm_route_f1_pct", "hmm_route_jaccard_pct",
            trip_df, bucket_sums["raw_hmm"], tuned_macro_f1,
        ),
        _aggregate_stage_row(
            "Stage 4",
            "Continuous Route Boundary Realignment (Final)",
            "hmm_top1_acc_pct", "hmm_obs_continuity_pct", "hmm_seq_continuity_pct",
            "hmm_route_precision_pct", "hmm_route_recall_pct", "hmm_route_f1_pct", "hmm_route_jaccard_pct",
            trip_df, bucket_sums["final_hmm"], tuned_macro_f1,
        ),
    ]
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df["delta_top1_pp"] = ablation_df["weighted_top1_acc_pct"].diff().fillna(0.0).round(4)

    return pd.DataFrame(summary_rows), trip_df, tune_df, ablation_df, sample_preds


def plot_hmm_results(
    summary_df: pd.DataFrame,
    trip_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    sample_preds: dict[str, Any],
) -> None:
    """Create Classifier vs HMM comparison, incremental gains chart, and visual route maps."""
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
    v_hmm = [summary_df.loc[len(summary_df) - 1, m] for m, _ in metrics]

    b1 = axes[0].bar(x - w / 2, v_clf, width=w, color="#64748b", label="Upgraded Classifier-Only")
    b2 = axes[0].bar(x + w / 2, v_hmm, width=w, color="#16a34a", label="Final HMM + Realignment")
    for bars in (b1, b2):
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.6,
                f"{bar.get_height():.2f}%",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([lbl for _, lbl in metrics], rotation=12, ha="right", fontsize=8.5)
    axes[0].set_ylim(60, 108)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("Student-Test: Classifier-Only vs. Final HMM Pipeline")
    axes[0].legend()

    x2 = np.arange(len(trip_df))
    axes[1].bar(x2 - w / 2, trip_df["clf_route_f1_pct"], width=w, color="#64748b", label="Classifier-Only Route F1")
    axes[1].bar(x2 + w / 2, trip_df["hmm_route_f1_pct"], width=w, color="#16a34a", label="Final HMM Route F1")
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
        (axes[1], "hmm_00000000", "student_test/00000000: Final HMM + Realignment Route", "#ea580c"),
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

    # Fig 3: Incremental Accuracy Gains & 4-Bucket Error Reduction (Stages 0 to 4)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    stages = ablation_df["stage_id"].tolist()
    xs = np.arange(len(stages))
    top1_vals = ablation_df["weighted_top1_acc_pct"].tolist()
    f1_vals = ablation_df["mean_route_f1_pct"].tolist()

    axes[0].plot(xs, top1_vals, "o-", color="#2563eb", linewidth=2.5, markersize=8, label="Obs Top-1 Accuracy (%)")
    axes[0].plot(xs, f1_vals, "s--", color="#16a34a", linewidth=2.2, markersize=7, label="Mean Route Arc F1 (%)")
    axes[0].axhspan(96.0, 98.0, color="#16a34a", alpha=0.08, label="Target 96-98% Band")
    for idx_pt, (xv, tv, fv) in enumerate(zip(xs, top1_vals, f1_vals)):
        t_off = -14 if idx_pt == 1 else 8
        axes[0].annotate(f"{tv:.2f}%", (xv, tv), textcoords="offset points", xytext=(0, t_off), ha="center", fontsize=8, color="#1d4ed8", fontweight="bold")
        if idx_pt in (0, 1, 4):
            f_off = 8 if idx_pt == 1 else -14
            axes[0].annotate(f"{fv:.2f}%", (xv, fv), textcoords="offset points", xytext=(0, f_off), ha="center", fontsize=7.5, color="#15803d", fontweight="bold")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(["Stage 0\nIter-1 Clf", "Stage 1\nIter-1 HMM", "Stage 2\nv2 Features", "Stage 3\nHybrid HMM", "Stage 4\nRealigned"], fontsize=8.5)
    axes[0].set_ylim(75, 102)
    axes[0].set_ylabel("Accuracy / Route F1 (%)")
    axes[0].set_title("Progressive Accuracy Climb (Stages 0 to 4, student_test)")
    axes[0].legend(loc="lower right", fontsize=8)

    err_df = ablation_df[ablation_df["stage_id"].isin(["Stage 2", "Stage 3", "Stage 4"])].copy()
    ex = np.arange(len(err_df))
    c_miss = err_df["cand_miss_errors"].to_numpy(dtype=float)
    opp_t = err_df["opp_twin_errors"].to_numpy(dtype=float)
    adj_b = err_df["adj_boundary_errors"].to_numpy(dtype=float)
    oth_d = err_df["other_detour_errors"].to_numpy(dtype=float)

    axes[1].bar(ex, opp_t, width=0.52, color="#dc2626", label="Opposite-Direction Twin")
    axes[1].bar(ex, adj_b, bottom=opp_t, width=0.52, color="#f59e0b", label="Intersection Boundary Lag")
    axes[1].bar(ex, oth_d, bottom=opp_t + adj_b, width=0.52, color="#64748b", label="Parallel / Detour Error")
    axes[1].bar(ex, c_miss, bottom=opp_t + adj_b + oth_d, width=0.52, color="#7c3aed", label="Candidate Truncation Miss")
    totals = opp_t + adj_b + oth_d + c_miss
    for xv, tot in zip(ex, totals):
        axes[1].text(xv, tot + 25, f"{int(tot):,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axes[1].set_xticks(ex)
    axes[1].set_xticklabels(err_df["stage_id"].tolist(), fontsize=9)
    axes[1].set_ylim(0, max(totals) * 1.15)
    axes[1].set_ylabel("Incorrect Observations (out of 19,160)")
    axes[1].set_title("Error Taxonomy Reduction Across Pipeline Stages")
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(HMM_FIGURES_DIR / "fig3_incremental_accuracy_gains.png", dpi=180)
    plt.close(fig)

    create_trip_folium_map(
        BASE_DIR / "train" / "00000001",
        BASE_DIR / "reports" / "eda" / "maps" / "trip_train_00000001.html",
        buffer_m=150.0,
    )
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

    summary_df, trip_df, tune_df, ablation_df, sample_preds = evaluate_hmm_vs_classifier()
    summary_df.to_csv(HMM_REPORTS_DIR / "hmm_vs_classifier_comparison.csv", index=False)
    trip_df.to_csv(HMM_REPORTS_DIR / "per_trip_hmm_results.csv", index=False)
    tune_df.to_csv(HMM_REPORTS_DIR / "hmm_tuning_trials.csv", index=False)
    ablation_df.to_csv(HMM_REPORTS_DIR / "incremental_accuracy_ablation.csv", index=False)
    plot_hmm_results(summary_df, trip_df, ablation_df, sample_preds)

    print("\n=== HMM Tuning Grid on Stratified Train Subset ===")
    print(tune_df.to_string(index=False))
    print("\n=== Classifier-Only vs. Hybrid HMM vs. Final Realigned (student_test) ===")
    print(summary_df.to_string(index=False))
    print("\n=== 5-Stage Incremental Accuracy Ablation (student_test) ===")
    print(
        ablation_df[
            [
                "stage_id",
                "weighted_top1_acc_pct",
                "delta_top1_pp",
                "other9_top1_acc_pct",
                "trip77_top1_acc_pct",
                "mean_route_f1_pct",
                "opp_twin_errors",
                "adj_boundary_errors",
                "total_errors",
            ]
        ].to_string(index=False)
    )
    print("\n=== Per-Trip Summary ===")
    print(
        trip_df[
            [
                "trip_id",
                "clf_top1_acc_pct",
                "hmm_raw_top1_acc_pct",
                "hmm_top1_acc_pct",
                "hmm_seq_continuity_pct",
                "hmm_route_f1_pct",
                "stage4_flips_good",
                "stage4_flips_bad",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
