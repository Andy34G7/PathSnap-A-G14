"""24-feature extraction pipeline, cached NPZ builders, and separability analysis for PathSnap."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eda.inventory_and_quality import BASE_DIR, load_trip_raw
from src.eda.trajectory_analysis import project_coords_to_utm
from src.features.candidate_generation import (
    FEATURES_REPORTS_DIR,
    align_route_to_observations,
    compute_smoothed_velocities,
    generate_candidates,
)

FEATURES_CACHE_DIR = BASE_DIR / "artifacts" / "features"
FEATURES_FIGURES_DIR = FEATURES_REPORTS_DIR / "figures"

FEATURE_NAMES: list[str] = [
    # Spatial Proximity (6)
    "dist_to_arc_m",
    "dist_rank",
    "dist_diff_from_best_m",
    "dist_ratio_to_best",
    "dist_zscore_in_cands",
    "gaussian_emission_prior",
    # Segment Projection & Geometry (4)
    "proj_fraction",
    "dist_to_from_node_m",
    "dist_to_to_node_m",
    "arc_length_m",
    # Heading & Directionality (5)
    "cos_heading_diff_inst",
    "sin_heading_diff_inst",
    "cos_heading_diff_smooth",
    "heading_diff_deg_smooth",
    "is_forward_aligned",
    # Trajectory Kinematics & Local Window (5)
    "gps_speed_mps",
    "dt_prev_s",
    "dist_prev_obs_to_arc_m",
    "dist_next_obs_to_arc_m",
    "mean_window3_dist_to_arc_m",
    # Graph Topology & Continuity (4)
    "from_node_in_deg",
    "to_node_out_deg",
    "is_bidirectional_arc",
    "prev_cand_connected",
]


def _point_to_cand_segments_dist(
    pts_xy: np.ndarray,
    seg_a: np.ndarray,
    seg_ab: np.ndarray,
    seg_len2: np.ndarray,
) -> np.ndarray:
    """Compute point-to-segment distance from pts_xy (N, 2) to candidate segments (N, K, 2)."""
    ap = pts_xy[:, None, :] - seg_a
    t = np.clip(np.sum(ap * seg_ab, axis=2) / np.maximum(seg_len2, 1e-12), 0.0, 1.0)
    proj = seg_a + t[:, :, None] * seg_ab
    return np.sqrt(np.sum((pts_xy[:, None, :] - proj) ** 2, axis=2))


def extract_trip_features(
    trip_dir: str | Path,
    k: int = 8,
    include_labels: bool = True,
) -> dict[str, Any]:
    """Extract (N, K, 24) feature tensor and candidate arrays for a single trip."""
    data = load_trip_raw(trip_dir)
    track = data["track"]
    nodes = data["nodes"]
    arcs = data["arcs"]
    route = data["route"]

    tx, ty, epsg = project_coords_to_utm(track[:, 0], track[:, 1])
    nx, ny, _ = project_coords_to_utm(nodes[:, 0], nodes[:, 1], epsg=epsg)
    pts_xy = np.column_stack([tx, ty])
    nodes_xy = np.column_stack([nx, ny])

    n = len(pts_xy)
    n_nodes = len(nodes_xy)

    step_dist, dt_prev, gps_speed, v_inst_u, v_sm_u = compute_smoothed_velocities(pts_xy, track[:, 2])
    cand_arcs, cand_dists, cand_proj_t = generate_candidates(
        pts_xy, nodes_xy, arcs, v_sm_u, k=k, prefilter_m=72
    )

    # Precompute arc geometry & topology across full road graph
    u_all = arcs[:, 0]
    v_all = arcs[:, 1]
    seg_a_all = nodes_xy[u_all]
    seg_b_all = nodes_xy[v_all]
    seg_ab_all = seg_b_all - seg_a_all
    seg_len_all = np.maximum(np.hypot(seg_ab_all[:, 0], seg_ab_all[:, 1]), 1e-6)
    seg_unit_all = seg_ab_all / seg_len_all[:, None]
    seg_len2_all = seg_len_all * seg_len_all

    out_deg = np.bincount(u_all, minlength=n_nodes)
    in_deg = np.bincount(v_all, minlength=n_nodes)
    edge_keys = u_all.astype(np.int64) * n_nodes + v_all.astype(np.int64)
    rev_keys = v_all.astype(np.int64) * n_nodes + u_all.astype(np.int64)
    is_bidir_all = np.isin(rev_keys, edge_keys).astype(np.float64)

    # Gather candidate segment properties (N, K, ...)
    cand_u = u_all[cand_arcs]
    cand_v = v_all[cand_arcs]
    ca = seg_a_all[cand_arcs]
    cb = seg_b_all[cand_arcs]
    cab = seg_ab_all[cand_arcs]
    clen = seg_len_all[cand_arcs]
    clen2 = seg_len2_all[cand_arcs]
    cunit = seg_unit_all[cand_arcs]

    # Group 1: Spatial Proximity (6)
    dist_rank = np.argsort(np.argsort(cand_dists, axis=1), axis=1).astype(np.float64)
    best_dist = np.min(cand_dists, axis=1, keepdims=True)
    dist_diff = cand_dists - best_dist
    dist_ratio = (cand_dists + 1.0) / (best_dist + 1.0)
    cand_mean = np.mean(cand_dists, axis=1, keepdims=True)
    cand_std = np.std(cand_dists, axis=1, keepdims=True)
    dist_z = (cand_dists - cand_mean) / (cand_std + 1.0)
    gauss_prior = np.exp(-0.5 * (cand_dists / 10.0) ** 2)

    # Group 2: Segment Projection & Geometry (4)
    dist_from = np.sqrt(np.sum((pts_xy[:, None, :] - ca) ** 2, axis=2))
    dist_to = np.sqrt(np.sum((pts_xy[:, None, :] - cb) ** 2, axis=2))

    # Group 3: Heading & Directionality (5)
    cos_inst = np.sum(v_inst_u[:, None, :] * cunit, axis=2)
    sin_inst = v_inst_u[:, None, 0] * cunit[:, :, 1] - v_inst_u[:, None, 1] * cunit[:, :, 0]
    cos_sm = np.clip(np.sum(v_sm_u[:, None, :] * cunit, axis=2), -1.0, 1.0)
    deg_sm = np.degrees(np.arccos(cos_sm))
    fwd_aligned = (cos_sm > 0.0).astype(np.float64)

    # Group 4: Trajectory Kinematics & Local Window (5)
    speed_feat = np.broadcast_to(gps_speed[:, None], (n, k))
    dt_feat = np.broadcast_to(np.clip(dt_prev, 0.0, 600.0)[:, None], (n, k))
    pts_prev = np.vstack([pts_xy[:1], pts_xy[:-1]]) if n > 1 else pts_xy
    pts_next = np.vstack([pts_xy[1:], pts_xy[-1:]]) if n > 1 else pts_xy
    dist_prev_obs = _point_to_cand_segments_dist(pts_prev, ca, cab, clen2)
    dist_next_obs = _point_to_cand_segments_dist(pts_next, ca, cab, clen2)
    mean_win3 = (dist_prev_obs + cand_dists + dist_next_obs) / 3.0

    # Group 5: Graph Topology & Continuity (4)
    from_in = in_deg[cand_u].astype(np.float64)
    to_out = out_deg[cand_v].astype(np.float64)
    bidir = is_bidir_all[cand_arcs]

    # Top-3 nearest candidates by distance at t-1 for connectivity check
    dist_order_prev = np.argsort(cand_dists, axis=1)[:, : min(3, k)]
    top3_arcs_prev = np.take_along_axis(cand_arcs, dist_order_prev, axis=1)
    top3_v_prev = v_all[top3_arcs_prev]

    prev_conn = np.zeros((n, k), dtype=np.float64)
    if n > 1:
        same_arc = np.any(cand_arcs[1:, :, None] == top3_arcs_prev[:-1, None, :], axis=2)
        adj_arc = np.any(cand_u[1:, :, None] == top3_v_prev[:-1, None, :], axis=2)
        prev_conn[1:] = (same_arc | adj_arc).astype(np.float64)

    X = np.stack(
        [
            cand_dists,
            dist_rank,
            dist_diff,
            dist_ratio,
            dist_z,
            gauss_prior,
            cand_proj_t,
            dist_from,
            dist_to,
            clen,
            cos_inst,
            sin_inst,
            cos_sm,
            deg_sm,
            fwd_aligned,
            speed_feat,
            dt_feat,
            dist_prev_obs,
            dist_next_obs,
            mean_win3,
            from_in,
            to_out,
            bidir,
            prev_conn,
        ],
        axis=-1,
    ).astype(np.float32)

    out: dict[str, Any] = {
        "trip_id": data["trip_id"],
        "n_obs": n,
        "epsg": epsg,
        "X": X,
        "cand_arcs": cand_arcs.astype(np.int32),
        "dt_prev": dt_prev.astype(np.float32),
        "step_dist": step_dist.astype(np.float32),
    }

    if include_labels and route is not None:
        gt_arcs, _ = align_route_to_observations(pts_xy, nodes_xy, arcs, route, v_sm_u)
        y = (cand_arcs == gt_arcs[:, None]).astype(np.uint8)
        out["gt_arcs"] = gt_arcs.astype(np.int32)
        out["y"] = y

    return out


def build_partition_cache(partition: str, k: int = 8) -> Path:
    """Extract and save compressed NPZ feature cache for all trips in a partition."""
    trip_dirs = sorted(p for p in (BASE_DIR / partition).iterdir() if p.is_dir())
    xs, ys, cands, gts, dt_list, sd_list = [], [], [], [], [], []
    trip_ids_obs, trip_names, obs_offsets = [], [], [0]

    for tid_idx, trip_dir in enumerate(trip_dirs):
        res = extract_trip_features(trip_dir, k=k, include_labels=True)
        n = res["n_obs"]
        xs.append(res["X"])
        ys.append(res["y"])
        cands.append(res["cand_arcs"])
        gts.append(res["gt_arcs"])
        dt_list.append(res["dt_prev"])
        sd_list.append(res["step_dist"])
        trip_ids_obs.append(np.full(n, tid_idx, dtype=np.int32))
        trip_names.append(res["trip_id"])
        obs_offsets.append(obs_offsets[-1] + n)

    cache_path = FEATURES_CACHE_DIR / f"{partition}_k{k}.npz"
    np.savez_compressed(
        cache_path,
        X=np.concatenate(xs, axis=0),
        y=np.concatenate(ys, axis=0),
        cand_arcs=np.concatenate(cands, axis=0),
        gt_arcs=np.concatenate(gts, axis=0),
        dt_prev=np.concatenate(dt_list, axis=0),
        step_dist=np.concatenate(sd_list, axis=0),
        trip_idx=np.concatenate(trip_ids_obs, axis=0),
        trip_names=np.array(trip_names),
        obs_offsets=np.array(obs_offsets, dtype=np.int64),
        feature_names=np.array(FEATURE_NAMES),
    )
    return cache_path


def compute_feature_statistics(train_cache: Path) -> pd.DataFrame:
    """Compute positive vs negative class statistics and correlations on train cache."""
    data = np.load(train_cache)
    X_flat = data["X"].reshape(-1, len(FEATURE_NAMES))
    y_flat = data["y"].reshape(-1)
    pos_mask = y_flat == 1
    neg_mask = y_flat == 0

    rows = []
    y_float = y_flat.astype(np.float64)
    y_centered = y_float - y_float.mean()
    y_norm = np.sqrt(np.sum(y_centered * y_centered))

    for idx, fname in enumerate(FEATURE_NAMES):
        col = X_flat[:, idx].astype(np.float64)
        col_c = col - col.mean()
        denom = np.sqrt(np.sum(col_c * col_c)) * y_norm
        corr = float(np.sum(col_c * y_centered) / denom) if denom > 1e-12 else 0.0
        rows.append(
            {
                "feature": fname,
                "pos_mean": round(float(np.mean(col[pos_mask])), 4),
                "pos_std": round(float(np.std(col[pos_mask])), 4),
                "pos_median": round(float(np.median(col[pos_mask])), 4),
                "neg_mean": round(float(np.mean(col[neg_mask])), 4),
                "neg_std": round(float(np.std(col[neg_mask])), 4),
                "neg_median": round(float(np.median(col[neg_mask])), 4),
                "corr_with_y": round(corr, 4),
            }
        )
    return pd.DataFrame(rows)


def plot_miss_rate_sensitivity(sens_csv: Path) -> None:
    """Plot Candidate Miss Rate vs K for train and student_test."""
    df = pd.read_csv(sens_csv)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    palette = {"train": "#2563eb", "student_test": "#ea580c"}

    for part, marker in (("train", "o"), ("student_test", "^")):
        sub = df[df["partition"] == part]
        ax.plot(
            sub["k"],
            sub["candidate_miss_rate_pct"],
            marker=marker,
            linewidth=2.2,
            markersize=7,
            color=palette[part],
            label=f"{part} (K=8 miss={sub.loc[sub['k']==8, 'candidate_miss_rate_pct'].values[0]:.2f}%)",
        )
        for k_val, miss_val in zip(sub["k"], sub["candidate_miss_rate_pct"]):
            ax.annotate(f"{miss_val:.2f}%", (k_val, miss_val), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    ax.axvline(8, color="#16a34a", linestyle="--", linewidth=1.4, label="Selected K=8")
    ax.set_xticks(sorted(df["k"].unique()))
    ax.set_xlabel("Candidate Set Size K per GPS Observation")
    ax.set_ylabel("Candidate Miss Rate (%)")
    ax.set_title("Candidate Miss Rate vs. Candidate Set Size K")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FEATURES_FIGURES_DIR / "fig1_candidate_miss_rate_vs_k.png", dpi=180)
    plt.close(fig)


def plot_feature_separability(train_cache: Path) -> None:
    """Plot 4-panel distribution comparison for matched (y=1) vs unmatched (y=0) candidates."""
    data = np.load(train_cache)
    X_flat = data["X"].reshape(-1, len(FEATURE_NAMES))
    y_flat = data["y"].reshape(-1)

    # Sample up to 150,000 candidates deterministically for smooth plotting
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(y_flat), size=min(150000, len(y_flat)), replace=False)
    Xs = X_flat[sample_idx]
    ys = y_flat[sample_idx]

    panels = [
        ("dist_to_arc_m", "Point-to-Arc Distance (m, clipped 60 m)", (0.0, 60.0)),
        ("dist_diff_from_best_m", "Distance Above Best Candidate (m, clipped 50 m)", (0.0, 50.0)),
        ("cos_heading_diff_smooth", "Cosine of Smoothed Heading Difference", (-1.0, 1.0)),
        ("prev_cand_connected", "Connected to Top-3 Candidates at t-1", None),
    ]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    for ax, (fname, title, clip_range) in zip(axes.ravel(), panels):
        fidx = FEATURE_NAMES.index(fname)
        vals_pos = Xs[ys == 1, fidx]
        vals_neg = Xs[ys == 0, fidx]
        if clip_range is not None:
            vals_pos = np.clip(vals_pos, clip_range[0], clip_range[1])
            vals_neg = np.clip(vals_neg, clip_range[0], clip_range[1])
            ax.hist(vals_neg, bins=40, density=True, alpha=0.5, color="#64748b", label="Unmatched (y=0)")
            ax.hist(vals_pos, bins=40, density=True, alpha=0.6, color="#16a34a", label="Matched (y=1)")
            ax.set_ylabel("Density")
        else:
            pos_rate = 100.0 * np.mean(vals_pos > 0.5)
            neg_rate = 100.0 * np.mean(vals_neg > 0.5)
            bars = ax.bar(["Unmatched (y=0)", "Matched (y=1)"], [neg_rate, pos_rate], color=["#64748b", "#16a34a"], width=0.5)
            for b, val in zip(bars, [neg_rate, pos_rate]):
                ax.text(b.get_x() + b.get_width() / 2, val + 1.5, f"{val:.1f}%", ha="center", fontsize=9)
            ax.set_ylim(0, 108)
            ax.set_ylabel("Rate of Connected Candidates (%)")
        ax.set_title(title, fontsize=10.5)
        ax.legend(loc="best", fontsize=8.5) if clip_range is not None else None

    plt.tight_layout()
    fig.savefig(FEATURES_FIGURES_DIR / "fig2_feature_distributions_pos_vs_neg.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FEATURES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    train_cache = build_partition_cache("train", k=8)
    test_cache = build_partition_cache("student_test", k=8)
    print(f"Built feature caches in {time.perf_counter() - t0:.2f}s:")
    print(f"  train_k8.npz: {train_cache.stat().st_size / (1024*1024):.2f} MB")
    print(f"  student_test_k8.npz: {test_cache.stat().st_size / (1024*1024):.2f} MB")

    stats_df = compute_feature_statistics(train_cache)
    stats_df.to_csv(FEATURES_REPORTS_DIR / "feature_statistics.csv", index=False)
    print("\n=== Top Feature Correlations with Ground-Truth Label y (train) ===")
    print(stats_df.reindex(stats_df["corr_with_y"].abs().sort_values(ascending=False).index).head(10).to_string(index=False))

    sens_csv = FEATURES_REPORTS_DIR / "candidate_miss_rate_sensitivity.csv"
    if sens_csv.exists():
        plot_miss_rate_sensitivity(sens_csv)
    plot_feature_separability(train_cache)
    print("\nSaved feature_statistics.csv and fig1-fig2 in reports/features/figures/.")


if __name__ == "__main__":
    main()
