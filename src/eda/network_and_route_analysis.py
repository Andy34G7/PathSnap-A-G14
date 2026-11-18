"""Road-network topology, connectivity, and reference-route GPS noise analysis for PathSnap."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from src.eda.inventory_and_quality import BASE_DIR, EDA_REPORTS_DIR, load_trip_raw
from src.eda.trajectory_analysis import FIGURES_DIR, project_coords_to_utm


def point_to_segments_min_dist(
    pts_xy: np.ndarray,
    seg_a_xy: np.ndarray,
    seg_b_xy: np.ndarray,
    batch_size: int = 500,
) -> np.ndarray:
    """Compute exact Euclidean distance (meters) from each point to the nearest road segment."""
    n_pts = len(pts_xy)
    min_dists = np.empty(n_pts, dtype=np.float64)

    ab = seg_b_xy - seg_a_xy  # (M, 2)
    ab_len2 = np.sum(ab * ab, axis=1)  # (M,)
    safe_len2 = np.where(ab_len2 > 1e-12, ab_len2, 1.0)

    for start in range(0, n_pts, batch_size):
        end = min(start + batch_size, n_pts)
        p = pts_xy[start:end, None, :]  # (B, 1, 2)
        ap = p - seg_a_xy[None, :, :]  # (B, M, 2)
        dot = np.sum(ap * ab[None, :, :], axis=2)  # (B, M)
        t = np.clip(dot / safe_len2[None, :], 0.0, 1.0)
        proj = seg_a_xy[None, :, :] + t[:, :, None] * ab[None, :, :]
        diff = p - proj
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        min_dists[start:end] = np.min(dists, axis=1)

    return min_dists


def analyze_networks_and_routes() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute road-network topology and GPS-to-reference-route noise stats across all trips."""
    records = []
    all_seg_lengths = []
    all_gps_noise_dists = []

    for partition in ("train", "student_test"):
        trip_dirs = sorted(p for p in (BASE_DIR / partition).iterdir() if p.is_dir())
        for trip_dir in trip_dirs:
            data = load_trip_raw(trip_dir)
            track = data["track"]
            nodes = data["nodes"]
            arcs = data["arcs"]
            route = data["route"]

            n_nodes = len(nodes)
            n_arcs = len(arcs)

            # Project both track and nodes using the track's UTM zone
            tx, ty, epsg = project_coords_to_utm(track[:, 0], track[:, 1])
            nx, ny, _ = project_coords_to_utm(nodes[:, 0], nodes[:, 1], epsg=epsg)

            pts_xy = np.column_stack([tx, ty])
            nodes_xy = np.column_stack([nx, ny])

            u = arcs[:, 0]
            v = arcs[:, 1]
            seg_a = nodes_xy[u]
            seg_b = nodes_xy[v]
            seg_len = np.hypot(seg_b[:, 0] - seg_a[:, 0], seg_b[:, 1] - seg_a[:, 1])

            # Sample segment lengths for global histogram to keep memory reasonable
            step = max(1, len(seg_len) // 2000)
            all_seg_lengths.append(seg_len[::step])

            # Node degrees and weakly connected components
            out_deg = np.bincount(u, minlength=n_nodes)
            in_deg = np.bincount(v, minlength=n_nodes)
            total_deg = out_deg + in_deg
            n_isolated = int(np.sum(total_deg == 0))
            n_sink_deadend = int(np.sum((in_deg > 0) & (out_deg == 0)))
            n_source_deadend = int(np.sum((in_deg == 0) & (out_deg > 0)))

            adj = sparse.csr_matrix(
                (np.ones(n_arcs, dtype=np.int8), (u, v)),
                shape=(n_nodes, n_nodes),
            )
            n_components, comp_labels = connected_components(adj, directed=False)
            comp_sizes = np.bincount(comp_labels)
            largest_comp_size = int(np.max(comp_sizes))
            largest_comp_pct = float(100.0 * largest_comp_size / n_nodes)

            # Bidirectional arc check
            edge_keys = u.astype(np.int64) * n_nodes + v.astype(np.int64)
            rev_keys = v.astype(np.int64) * n_nodes + u.astype(np.int64)
            n_bidir = int(np.sum(np.isin(rev_keys, edge_keys)))
            bidir_pct = float(100.0 * n_bidir / n_arcs) if n_arcs else 0.0

            # Reference route structure and GPS noise
            n_route = len(route) if route is not None else 0
            if route is not None and n_route > 0:
                route_len_km = float(np.sum(seg_len[route]) / 1000.0)
                n_unique_route_arcs = int(len(np.unique(route)))
                n_repeated_arcs = int(n_route - n_unique_route_arcs)
                n_consec_dups = int(np.sum(route[1:] == route[:-1])) if n_route > 1 else 0
                n_disconn = (
                    int(np.sum((arcs[route[:-1], 1] != arcs[route[1:], 0]) & (route[:-1] != route[1:])))
                    if n_route > 1
                    else 0
                )

                route_seg_a = seg_a[route]
                route_seg_b = seg_b[route]
                gps_noise = point_to_segments_min_dist(pts_xy, route_seg_a, route_seg_b)
                all_gps_noise_dists.append(gps_noise)

                gps_med = float(np.median(gps_noise))
                gps_mean = float(np.mean(gps_noise))
                gps_p90 = float(np.percentile(gps_noise, 90))
                gps_p95 = float(np.percentile(gps_noise, 95))
                gps_p99 = float(np.percentile(gps_noise, 99))
                gps_max = float(np.max(gps_noise))
            else:
                route_len_km = 0.0
                n_unique_route_arcs = 0
                n_repeated_arcs = 0
                n_consec_dups = 0
                n_disconn = 0
                gps_med = gps_mean = gps_p90 = gps_p95 = gps_p99 = gps_max = np.nan

            records.append(
                {
                    "partition": partition,
                    "trip_id": data["trip_id"],
                    "utm_epsg": epsg,
                    "n_nodes": n_nodes,
                    "n_arcs": n_arcs,
                    "mean_seg_len_m": float(np.mean(seg_len)),
                    "median_seg_len_m": float(np.median(seg_len)),
                    "p95_seg_len_m": float(np.percentile(seg_len, 95)),
                    "max_seg_len_m": float(np.max(seg_len)),
                    "n_isolated_nodes": n_isolated,
                    "n_sink_deadends": n_sink_deadend,
                    "n_source_deadends": n_source_deadend,
                    "n_weak_components": int(n_components),
                    "largest_comp_pct": largest_comp_pct,
                    "bidir_arc_pct": bidir_pct,
                    "n_track": len(track),
                    "n_route": n_route,
                    "track_to_route_ratio": float(len(track) / n_route) if n_route else np.nan,
                    "route_length_km": route_len_km,
                    "n_unique_route_arcs": n_unique_route_arcs,
                    "n_repeated_route_arcs": n_repeated_arcs,
                    "n_consec_dup_route_arcs": n_consec_dups,
                    "n_disconn_route_steps": n_disconn,
                    "gps_dist_median_m": gps_med,
                    "gps_dist_mean_m": gps_mean,
                    "gps_dist_p90_m": gps_p90,
                    "gps_dist_p95_m": gps_p95,
                    "gps_dist_p99_m": gps_p99,
                    "gps_dist_max_m": gps_max,
                }
            )

    return pd.DataFrame(records), np.concatenate(all_seg_lengths), np.concatenate(all_gps_noise_dists)


def plot_network_topology(topo_df: pd.DataFrame, seg_lengths: np.ndarray) -> None:
    """Create Fig 4: road-network scale, segment lengths, and connectivity."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    palette = {"train": "#2563eb", "student_test": "#ea580c"}

    # Panel 1: Nodes vs Arcs
    for part, marker in (("train", "o"), ("student_test", "^")):
        sub = topo_df[topo_df["partition"] == part]
        axes[0].scatter(
            sub["n_nodes"],
            sub["n_arcs"],
            c=palette[part],
            marker=marker,
            alpha=0.82,
            s=50,
            label=part,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Number of Nodes (log)")
    axes[0].set_ylabel("Number of Arcs (log)")
    axes[0].set_title("Road-Network Graph Size (Nodes vs. Arcs)")
    axes[0].legend()

    # Panel 2: Road segment length distribution
    clipped_seg = np.clip(seg_lengths, 0.0, 500.0)
    axes[1].hist(clipped_seg, bins=50, color="#0f766e", alpha=0.8, edgecolor="white")
    axes[1].axvline(np.median(seg_lengths), color="#dc2626", linestyle="--", label=f"Median={np.median(seg_lengths):.1f} m")
    axes[1].set_xlabel("Arc Segment Length (m, clipped at 500 m)")
    axes[1].set_ylabel("Sampled Arc Count")
    axes[1].set_title("Road Segment Length Distribution")
    axes[1].legend()

    # Panel 3: Largest weakly connected component % vs isolated nodes
    for part, marker in (("train", "o"), ("student_test", "^")):
        sub = topo_df[topo_df["partition"] == part]
        axes[2].scatter(
            sub["n_weak_components"],
            sub["largest_comp_pct"],
            c=palette[part],
            marker=marker,
            alpha=0.82,
            s=50,
            label=part,
        )
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Weakly Connected Components per Graph (log)")
    axes[2].set_ylabel("% Nodes in Largest Component")
    axes[2].set_title("Graph Fragmentation & Largest Component Coverage")
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_network_size_and_connectivity.png", dpi=180)
    plt.close(fig)


def plot_gps_noise_and_ratio(topo_df: pd.DataFrame, gps_noise: np.ndarray) -> None:
    """Create Fig 5: GPS-to-reference-route noise CDF and track-to-route cardinality ratio."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    # Left: Empirical noise histogram + CDF up to 100 m
    clipped_noise = np.clip(gps_noise, 0.0, 100.0)
    axes[0].hist(clipped_noise, bins=50, density=True, color="#3b82f6", alpha=0.45, label="Density")
    ax_cdf = axes[0].twinx()
    sorted_noise = np.sort(clipped_noise)
    cdf = np.arange(1, len(sorted_noise) + 1) / len(sorted_noise)
    ax_cdf.plot(sorted_noise, cdf, color="#1d4ed8", linewidth=2.0, label="Empirical CDF")
    ax_cdf.set_ylabel("Cumulative Fraction of GPS Points")
    ax_cdf.set_ylim(0, 1.05)

    p50 = float(np.percentile(gps_noise, 50))
    p90 = float(np.percentile(gps_noise, 90))
    p95 = float(np.percentile(gps_noise, 95))
    p99 = float(np.percentile(gps_noise, 99))
    for val, lbl, col in [
        (p50, f"50% ({p50:.1f}m)", "#16a34a"),
        (p90, f"90% ({p90:.1f}m)", "#ca8a04"),
        (p95, f"95% ({p95:.1f}m)", "#ea580c"),
        (p99, f"99% ({p99:.1f}m)", "#dc2626"),
    ]:
        axes[0].axvline(min(val, 100.0), color=col, linestyle="--", linewidth=1.4, label=lbl)

    axes[0].set_xlabel("Shortest Distance to Reference .route Segment (m, clipped 100 m)")
    axes[0].set_ylabel("Probability Density")
    axes[0].set_title(f"GPS Noise Distribution Relative to .route (n={len(gps_noise):,} obs)")
    axes[0].legend(loc="center right", fontsize=8.5)

    # Right: Track observations vs Route traversed arcs
    palette = {"train": "#2563eb", "student_test": "#ea580c"}
    for part, marker in (("train", "o"), ("student_test", "^")):
        sub = topo_df[topo_df["partition"] == part]
        axes[1].scatter(
            sub["n_route"],
            sub["n_track"],
            c=palette[part],
            marker=marker,
            alpha=0.82,
            s=55,
            label=f"{part} (median ratio={sub['track_to_route_ratio'].median():.2f}x)",
        )
    max_val = max(topo_df["n_route"].max(), topo_df["n_track"].max())
    axes[1].plot([50, max_val], [50, max_val], color="#64748b", linestyle=":", label="1:1 line")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Reference Route Arcs (.route length, log)")
    axes[1].set_ylabel("GPS Observations (.track length, log)")
    axes[1].set_title("Cardinality Mismatch: .track Observations vs. .route Arcs")
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_gps_noise_to_reference_route.png", dpi=180)
    plt.close(fig)


def main() -> None:
    EDA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    topo_df, seg_lengths, gps_noise = analyze_networks_and_routes()
    topo_df.to_csv(EDA_REPORTS_DIR / "network_topology_stats.csv", index=False)

    print("=== Road-Network Topology & Route Summary ===")
    print(
        topo_df.groupby("partition")[
            [
                "n_nodes",
                "n_arcs",
                "n_isolated_nodes",
                "largest_comp_pct",
                "bidir_arc_pct",
                "track_to_route_ratio",
                "n_disconn_route_steps",
                "gps_dist_median_m",
                "gps_dist_p95_m",
                "gps_dist_p99_m",
            ]
        ]
        .median()
        .round(3)
        .to_string()
    )
    print(
        f"\nGlobal GPS-to-.route distance percentiles (n={len(gps_noise):,}): "
        f"p50={np.percentile(gps_noise, 50):.2f}m, "
        f"p90={np.percentile(gps_noise, 90):.2f}m, "
        f"p95={np.percentile(gps_noise, 95):.2f}m, "
        f"p99={np.percentile(gps_noise, 99):.2f}m, "
        f"max={np.max(gps_noise):.2f}m"
    )

    plot_network_topology(topo_df, seg_lengths)
    plot_gps_noise_and_ratio(topo_df, gps_noise)
    print("Saved network_topology_stats.csv and fig4-fig5.")


if __name__ == "__main__":
    main()
