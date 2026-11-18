"""Candidate generation and monotonic GPS-to-route ground-truth alignment for PathSnap."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from src.eda.inventory_and_quality import BASE_DIR, load_trip_raw
from src.eda.trajectory_analysis import project_coords_to_utm

FEATURES_REPORTS_DIR = BASE_DIR / "reports" / "features"


def compute_smoothed_velocities(
    pts_xy: np.ndarray,
    timestamps: np.ndarray,
    chord_r_m: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute step distances, time deltas, speeds, local derivative headings, and turnaround-aware spatial chords."""
    n = len(pts_xy)
    step_dist = np.zeros(n, dtype=np.float64)
    dt_prev = np.ones(n, dtype=np.float64)
    if n > 1:
        step_dist[1:] = np.hypot(np.diff(pts_xy[:, 0]), np.diff(pts_xy[:, 1]))
        dt_prev[1:] = np.maximum(np.diff(timestamps), 1e-6)

    gps_speed = np.clip(step_dist / dt_prev, 0.0, 60.0)
    if n > 1:
        gps_speed[0] = gps_speed[1]

    if n <= 1:
        unit_default = np.tile(np.array([1.0, 0.0], dtype=np.float64), (n, 1))
        return step_dist, dt_prev, gps_speed, unit_default, unit_default.copy()

    gx = np.gradient(gaussian_filter1d(pts_xy[:, 0], sigma=1.2, mode="nearest"))
    gy = np.gradient(gaussian_filter1d(pts_xy[:, 1], sigma=1.2, mode="nearest"))
    gn = np.hypot(gx, gy)
    v_inst_u = np.where(
        (gn > 1e-6)[:, None],
        np.column_stack([gx, gy]) / np.maximum(gn[:, None], 1e-6),
        np.array([1.0, 0.0]),
    )

    sm_x = gaussian_filter1d(pts_xy[:, 0], sigma=2.5, mode="nearest").tolist()
    sm_y = gaussian_filter1d(pts_xy[:, 1], sigma=2.5, mode="nearest").tolist()
    r2 = chord_r_m * chord_r_m

    v_sm_u = np.empty((n, 2), dtype=np.float64)
    for i in range(n):
        xi, yi = sm_x[i], sm_y[i]
        ib = i
        while ib > 0:
            dx = xi - sm_x[ib]
            dy = yi - sm_y[ib]
            if dx * dx + dy * dy >= r2:
                break
            ib -= 1
        if_ = i
        while if_ < n - 1:
            dx = sm_x[if_] - xi
            dy = sm_y[if_] - yi
            if dx * dx + dy * dy >= r2:
                break
            if_ += 1

        vbx, vby = xi - sm_x[ib], yi - sm_y[ib]
        vfx, vfy = sm_x[if_] - xi, sm_y[if_] - yi
        nb = (vbx * vbx + vby * vby) ** 0.5
        nf = (vfx * vfx + vfy * vfy) ** 0.5
        ubx = vbx / nb if nb > 1e-6 else float(v_inst_u[i, 0])
        uby = vby / nb if nb > 1e-6 else float(v_inst_u[i, 1])
        ufx = vfx / nf if nf > 1e-6 else ubx
        ufy = vfy / nf if nf > 1e-6 else uby

        if ubx * ufx + uby * ufy < -0.2:
            vx, vy = (ubx, uby) if (i - ib) <= (if_ - i) else (ufx, ufy)
        else:
            vx, vy = ubx + ufx, uby + ufy
        nv = (vx * vx + vy * vy) ** 0.5
        if nv > 1e-6:
            v_sm_u[i, 0] = vx / nv
            v_sm_u[i, 1] = vy / nv
        else:
            v_sm_u[i, 0] = ubx
            v_sm_u[i, 1] = uby

    return step_dist, dt_prev, gps_speed, v_inst_u, v_sm_u


def align_route_to_observations(
    pts_xy: np.ndarray,
    nodes_xy: np.ndarray,
    arcs: np.ndarray,
    route: np.ndarray,
    v_sm_u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align GPS observations monotonically onto the ordered .route arc sequence via DP."""
    n = len(pts_xy)
    m = len(route)
    r_a = nodes_xy[arcs[route, 0]]
    r_b = nodes_xy[arcs[route, 1]]
    r_ab = r_b - r_a
    r_len = np.maximum(np.hypot(r_ab[:, 0], r_ab[:, 1]), 1e-6)
    r_unit = r_ab / r_len[:, None]
    r_len2 = r_len * r_len

    # Emission cost matrix D_r (N, M) in float32 to bound memory on large trips
    d_mat = np.empty((n, m), dtype=np.float32)
    for start in range(0, n, 500):
        end = min(start + 500, n)
        ap = pts_xy[start:end, None, :] - r_a[None, :, :]
        proj_t = np.clip(np.sum(ap * r_ab[None, :, :], axis=2) / r_len2[None, :], 0.0, 1.0)
        proj = r_a[None, :, :] + proj_t[:, :, None] * r_ab[None, :, :]
        dist = np.sqrt(np.sum((pts_xy[start:end, None, :] - proj) ** 2, axis=2))
        cos_h = np.sum(v_sm_u[start:end, None, :] * r_unit[None, :, :], axis=2)
        d_mat[start:end] = (dist + 1.5 * (1.0 - cos_h)).astype(np.float32)

    dp = np.full((n, m), np.inf, dtype=np.float64)
    back = np.zeros((n, m), dtype=np.int32)
    start_win = min(m, 25)
    dp[0, :start_win] = d_mat[0, :start_win]
    j_indices = np.arange(m, dtype=np.int32)

    for i in range(1, n):
        prev_row = dp[i - 1]
        cur_row = dp[i]
        cur_back = back[i]

        # Local monotonic steps 0..15
        cur_row[:] = prev_row
        cur_back[:] = j_indices
        for step in range(1, min(16, m)):
            pen = 0.5 * max(0, step - 2)
            cand = prev_row[:-step] + pen
            target = cur_row[step:]
            mask = cand < target
            target[mask] = cand[mask]
            cur_back[step:][mask] = j_indices[:-step][mask]

        # Safety fallback for long GPS gaps that skip more than 15 route arcs
        cum_min = np.minimum.accumulate(prev_row)
        jump_cand = cum_min + 50.0
        jump_mask = jump_cand < cur_row
        if np.any(jump_mask):
            # Track running argmin index for backpointers
            running_arg = np.zeros(m, dtype=np.int32)
            best_val = prev_row[0]
            best_idx = 0
            for j in range(1, m):
                if prev_row[j] < best_val:
                    best_val = prev_row[j]
                    best_idx = j
                running_arg[j] = best_idx
            cur_row[jump_mask] = jump_cand[jump_mask]
            cur_back[jump_mask] = running_arg[jump_mask]

        cur_row += d_mat[i]

    end_start = max(0, m - 25)
    gt_r_idx = np.zeros(n, dtype=np.int32)
    gt_r_idx[-1] = int(np.argmin(dp[-1, end_start:])) + end_start
    for i in range(n - 1, 0, -1):
        gt_r_idx[i - 1] = back[i, gt_r_idx[i]]

    gt_arcs = route[gt_r_idx].astype(np.int64)
    return gt_arcs, gt_r_idx


def generate_candidates(
    pts_xy: np.ndarray,
    nodes_xy: np.ndarray,
    arcs: np.ndarray,
    v_sm_u: np.ndarray,
    k: int = 8,
    prefilter_m: int = 72,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve top-K unique candidate road arcs per GPS observation via 3-point cKDTree."""
    n = len(pts_xy)
    e = len(arcs)
    seg_a = nodes_xy[arcs[:, 0]]
    seg_b = nodes_xy[arcs[:, 1]]
    seg_ab = seg_b - seg_a
    seg_len = np.maximum(np.hypot(seg_ab[:, 0], seg_ab[:, 1]), 1e-6)
    seg_unit = seg_ab / seg_len[:, None]
    seg_len2 = seg_len * seg_len

    # Index 3 points along every road segment so long arcs are covered
    tree_pts = np.vstack([seg_a + 0.15 * seg_ab, seg_a + 0.50 * seg_ab, seg_a + 0.85 * seg_ab])
    tree = cKDTree(tree_pts)

    n_query = min(max(prefilter_m, 6 * k), 3 * e)
    _, raw_idx = tree.query(pts_xy, k=n_query)
    if n_query == 1:
        raw_idx = raw_idx[:, None]
    pre_idx = raw_idx % e

    ca = seg_a[pre_idx]
    cab = seg_ab[pre_idx]
    clen2 = seg_len2[pre_idx]
    cap = pts_xy[:, None, :] - ca
    ct = np.clip(np.sum(cap * cab, axis=2) / clen2, 0.0, 1.0)
    cproj = ca + ct[:, :, None] * cab
    cdist = np.sqrt(np.sum((pts_xy[:, None, :] - cproj) ** 2, axis=2))
    ccos = np.sum(v_sm_u[:, None, :] * seg_unit[pre_idx], axis=2)

    score = cdist + 0.5 * (1.0 - ccos)
    order = np.argsort(score, axis=1)
    sorted_idx = np.take_along_axis(pre_idx, order, axis=1)
    sorted_dist = np.take_along_axis(cdist, order, axis=1)
    sorted_t = np.take_along_axis(ct, order, axis=1)

    cand_arcs = np.empty((n, k), dtype=np.int64)
    cand_dists = np.empty((n, k), dtype=np.float64)
    cand_proj_t = np.empty((n, k), dtype=np.float64)

    for r in range(n):
        row = sorted_idx[r]
        _, first_pos = np.unique(row, return_index=True)
        keep = np.sort(first_pos)[:k]
        m_found = len(keep)
        cand_arcs[r, :m_found] = row[keep]
        cand_dists[r, :m_found] = sorted_dist[r, keep]
        cand_proj_t[r, :m_found] = sorted_t[r, keep]
        if m_found < k:
            cand_arcs[r, m_found:] = row[keep[-1]]
            cand_dists[r, m_found:] = sorted_dist[r, keep[-1]]
            cand_proj_t[r, m_found:] = sorted_t[r, keep[-1]]

    return cand_arcs, cand_dists, cand_proj_t


def evaluate_candidate_miss_rates(k_values: tuple[int, ...] = (1, 3, 5, 8, 10, 12)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure Candidate Miss Rate across all train and student_test trips for varying K."""
    max_k = max(k_values)
    per_trip_rows: list[dict[str, Any]] = []

    for partition in ("train", "student_test"):
        trip_dirs = sorted(p for p in (BASE_DIR / partition).iterdir() if p.is_dir())
        for trip_dir in trip_dirs:
            t0 = time.perf_counter()
            data = load_trip_raw(trip_dir)
            track, nodes, arcs, route = data["track"], data["nodes"], data["arcs"], data["route"]

            tx, ty, epsg = project_coords_to_utm(track[:, 0], track[:, 1])
            nx, ny, _ = project_coords_to_utm(nodes[:, 0], nodes[:, 1], epsg=epsg)
            pts_xy = np.column_stack([tx, ty])
            nodes_xy = np.column_stack([nx, ny])

            _, _, _, _, v_sm_u = compute_smoothed_velocities(pts_xy, track[:, 2])
            gt_arcs, _ = align_route_to_observations(pts_xy, nodes_xy, arcs, route, v_sm_u)
            cand_arcs, cand_dists, _ = generate_candidates(
                pts_xy, nodes_xy, arcs, v_sm_u, k=max_k, prefilter_m=72
            )

            row: dict[str, Any] = {
                "partition": partition,
                "trip_id": data["trip_id"],
                "n_obs": len(track),
                "n_route": len(route),
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "nearest_cand_median_m": float(np.median(cand_dists[:, 0])),
                "nearest_cand_p95_m": float(np.percentile(cand_dists[:, 0], 95)),
            }
            for k in k_values:
                hits = np.any(cand_arcs[:, :k] == gt_arcs[:, None], axis=1)
                miss_cnt = int(np.sum(~hits))
                row[f"miss_count_k{k}"] = miss_cnt
                row[f"miss_rate_pct_k{k}"] = 100.0 * miss_cnt / len(track)
            per_trip_rows.append(row)

    trip_df = pd.DataFrame(per_trip_rows)

    sens_rows = []
    for partition, grp in trip_df.groupby("partition", sort=False):
        total_obs = int(grp["n_obs"].sum())
        for k in k_values:
            miss_total = int(grp[f"miss_count_k{k}"].sum())
            sens_rows.append(
                {
                    "partition": partition,
                    "k": k,
                    "total_obs": total_obs,
                    "miss_count": miss_total,
                    "hit_count": total_obs - miss_total,
                    "candidate_miss_rate_pct": round(100.0 * miss_total / total_obs, 4),
                    "candidate_recall_pct": round(100.0 * (total_obs - miss_total) / total_obs, 4),
                    "median_trip_miss_rate_pct": round(float(grp[f"miss_rate_pct_k{k}"].median()), 4),
                    "max_trip_miss_rate_pct": round(float(grp[f"miss_rate_pct_k{k}"].max()), 4),
                }
            )
    sens_df = pd.DataFrame(sens_rows)
    return sens_df, trip_df


def main() -> None:
    FEATURES_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    sens_df, trip_df = evaluate_candidate_miss_rates()
    elapsed = time.perf_counter() - t0

    sens_df.to_csv(FEATURES_REPORTS_DIR / "candidate_miss_rate_sensitivity.csv", index=False)
    trip_df.to_csv(FEATURES_REPORTS_DIR / "per_trip_candidate_stats.csv", index=False)

    print(f"Evaluated 90 trips in {elapsed:.2f}s")
    print("\n=== Candidate Miss Rate Sensitivity ===")
    print(sens_df.to_string(index=False))

    print("\n=== Worst 5 Train Trips at K=8 ===")
    worst_train = trip_df[trip_df["partition"] == "train"].sort_values("miss_rate_pct_k8", ascending=False).head(5)
    print(worst_train[["trip_id", "n_obs", "n_route", "miss_count_k8", "miss_rate_pct_k8"]].to_string(index=False))


if __name__ == "__main__":
    main()
