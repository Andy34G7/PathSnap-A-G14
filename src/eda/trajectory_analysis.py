"""Spatial projection comparison and GPS trajectory dynamics analysis for PathSnap."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import seaborn as sns

from src.eda.inventory_and_quality import BASE_DIR, EDA_REPORTS_DIR, get_utm_epsg, load_trip_raw

FIGURES_DIR = EDA_REPORTS_DIR / "figures"


def project_coords_to_utm(
    lon: np.ndarray,
    lat: np.ndarray,
    epsg: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Project WGS84 (lon, lat) arrays to local UTM (x, y) coordinates in meters."""
    if epsg is None:
        epsg = get_utm_epsg(float(np.mean(lon)), float(np.mean(lat)))
    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), epsg


def compute_crs_distortion_comparison(sample_trip_dir: Path) -> pd.DataFrame:
    """Compare naive degree Euclidean scaling vs. local UTM vs. WGS84 geodesic on a sample trip."""
    data = load_trip_raw(sample_trip_dir)
    track = data["track"]
    lon, lat = track[:, 0], track[:, 1]
    mean_lon, mean_lat = float(np.mean(lon)), float(np.mean(lat))

    x_utm, y_utm, epsg = project_coords_to_utm(lon, lat)
    utm_steps = np.hypot(np.diff(x_utm), np.diff(y_utm))
    utm_total_m = float(np.sum(utm_steps))

    geod = pyproj.Geod(ellps="WGS84")
    _, _, geod_steps = geod.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    geod_steps = np.asarray(geod_steps, dtype=np.float64)
    geod_total_m = float(np.sum(geod_steps))

    # Naive isotropic degree-to-meter scaling (~111,319.9 m/deg) ignoring cos(latitude)
    deg_steps = np.hypot(np.diff(lon), np.diff(lat))
    naive_total_m = float(np.sum(deg_steps) * 111319.9)

    # Unit degree distortion at trip centroid
    _, _, dlon_1deg_m = geod.inv(mean_lon - 0.5, mean_lat, mean_lon + 0.5, mean_lat)
    _, _, dlat_1deg_m = geod.inv(mean_lon, mean_lat - 0.5, mean_lon, mean_lat + 0.5)
    lon_vs_lat_distortion_pct = 100.0 * (1.0 - dlon_1deg_m / dlat_1deg_m)

    utm_rel_err_pct = 100.0 * abs(utm_total_m - geod_total_m) / geod_total_m
    naive_rel_err_pct = 100.0 * abs(naive_total_m - geod_total_m) / geod_total_m
    max_step_utm_geod_diff_m = float(np.max(np.abs(utm_steps - geod_steps)))

    rows = [
        {
            "trip_id": data["trip_id"],
            "mean_lon": round(mean_lon, 6),
            "mean_lat": round(mean_lat, 6),
            "utm_epsg": epsg,
            "method": "WGS84 Geodesic (Reference)",
            "total_length_m": round(geod_total_m, 3),
            "relative_error_vs_geodesic_pct": 0.0,
            "max_step_diff_vs_geodesic_m": 0.0,
            "one_deg_lon_m": round(dlon_1deg_m, 2),
            "one_deg_lat_m": round(dlat_1deg_m, 2),
            "lon_lat_anisotropy_pct": round(lon_vs_lat_distortion_pct, 2),
        },
        {
            "trip_id": data["trip_id"],
            "mean_lon": round(mean_lon, 6),
            "mean_lat": round(mean_lat, 6),
            "utm_epsg": epsg,
            "method": f"Local UTM (EPSG:{epsg})",
            "total_length_m": round(utm_total_m, 3),
            "relative_error_vs_geodesic_pct": round(utm_rel_err_pct, 5),
            "max_step_diff_vs_geodesic_m": round(max_step_utm_geod_diff_m, 5),
            "one_deg_lon_m": round(dlon_1deg_m, 2),
            "one_deg_lat_m": round(dlat_1deg_m, 2),
            "lon_lat_anisotropy_pct": round(lon_vs_lat_distortion_pct, 2),
        },
        {
            "trip_id": data["trip_id"],
            "mean_lon": round(mean_lon, 6),
            "mean_lat": round(mean_lat, 6),
            "utm_epsg": epsg,
            "method": "Naive Isotropic Degree Scaling (111.32 km/deg)",
            "total_length_m": round(naive_total_m, 3),
            "relative_error_vs_geodesic_pct": round(naive_rel_err_pct, 5),
            "max_step_diff_vs_geodesic_m": round(float(np.max(np.abs(deg_steps * 111319.9 - geod_steps))), 5),
            "one_deg_lon_m": round(dlon_1deg_m, 2),
            "one_deg_lat_m": round(dlat_1deg_m, 2),
            "lon_lat_anisotropy_pct": round(lon_vs_lat_distortion_pct, 2),
        },
    ]
    return pd.DataFrame(rows)


def analyze_all_trajectories() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute physical trajectory dynamics for all train and student_test trips."""
    records = []
    dt_collect: dict[str, list[np.ndarray]] = {"train": [], "student_test": []}
    speed_collect: dict[str, list[np.ndarray]] = {"train": [], "student_test": []}

    for partition in ("train", "student_test"):
        trip_dirs = sorted(p for p in (BASE_DIR / partition).iterdir() if p.is_dir())
        for trip_dir in trip_dirs:
            data = load_trip_raw(trip_dir)
            track = data["track"]
            lon, lat, ts = track[:, 0], track[:, 1], track[:, 2]

            x, y, epsg = project_coords_to_utm(lon, lat)
            step_d = np.hypot(np.diff(x), np.diff(y))
            dt = np.diff(ts)

            valid_dt = dt > 0.0
            speeds = np.where(valid_dt, step_d / np.where(valid_dt, dt, 1.0), 0.0)
            valid_speeds = speeds[valid_dt]

            dt_collect[partition].append(dt)
            speed_collect[partition].append(valid_speeds)

            width_km = float((np.max(x) - np.min(x)) / 1000.0)
            height_km = float((np.max(y) - np.min(y)) / 1000.0)
            total_length_km = float(np.sum(step_d) / 1000.0)
            n_spikes = int(np.sum(valid_speeds > 50.0))

            records.append(
                {
                    "partition": partition,
                    "trip_id": data["trip_id"],
                    "n_obs": len(track),
                    "mean_lon": float(np.mean(lon)),
                    "mean_lat": float(np.mean(lat)),
                    "utm_epsg": epsg,
                    "total_length_km": total_length_km,
                    "bbox_width_km": width_km,
                    "bbox_height_km": height_km,
                    "bbox_area_km2": width_km * height_km,
                    "mean_step_m": float(np.mean(step_d)),
                    "median_step_m": float(np.median(step_d)),
                    "max_step_m": float(np.max(step_d)),
                    "mean_speed_mps": float(np.mean(valid_speeds)) if len(valid_speeds) else 0.0,
                    "median_speed_mps": float(np.median(valid_speeds)) if len(valid_speeds) else 0.0,
                    "p95_speed_mps": float(np.percentile(valid_speeds, 95)) if len(valid_speeds) else 0.0,
                    "max_speed_mps": float(np.max(valid_speeds)) if len(valid_speeds) else 0.0,
                    "mean_speed_kmh": float(np.mean(valid_speeds) * 3.6) if len(valid_speeds) else 0.0,
                    "n_speed_gt_50mps": n_spikes,
                    "pct_speed_gt_50mps": (100.0 * n_spikes / len(valid_speeds)) if len(valid_speeds) else 0.0,
                }
            )

    dt_arrays = {k: np.concatenate(v) for k, v in dt_collect.items()}
    speed_arrays = {k: np.concatenate(v) for k, v in speed_collect.items()}
    return pd.DataFrame(records), dt_arrays, speed_arrays


def plot_sampling_and_speeds(dt_arrays: dict[str, np.ndarray], speed_arrays: dict[str, np.ndarray]) -> None:
    """Create Fig 1: sampling interval breakdown and step speed distribution."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: sampling interval buckets (log scale % so irregular gaps are visible)
    buckets = ["dt < 1s", "dt == 1s", "1s < dt <= 5s", "5s < dt <= 60s", "dt > 60s"]
    x_pos = np.arange(len(buckets))
    width = 0.36
    colors = {"train": "#2563eb", "student_test": "#ea580c"}

    for idx, part in enumerate(("train", "student_test")):
        dt = dt_arrays[part]
        n = len(dt)
        counts = [
            np.sum(dt < 0.999),
            np.sum(np.isclose(dt, 1.0, atol=1e-3)),
            np.sum((dt > 1.001) & (dt <= 5.0)),
            np.sum((dt > 5.0) & (dt <= 60.0)),
            np.sum(dt > 60.0),
        ]
        pcts = [100.0 * c / n for c in counts]
        offset = -width / 2 if idx == 0 else width / 2
        bars = axes[0].bar(
            x_pos + offset,
            pcts,
            width=width,
            label=f"{part} (n={n:,} steps)",
            color=colors[part],
            alpha=0.88,
        )
        for bar, pct, cnt in zip(bars, pcts, counts):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                max(bar.get_height(), 0.02) * 1.15,
                f"{pct:.2f}%\n({cnt})",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    axes[0].set_yscale("log")
    axes[0].set_ylim(0.005, 300)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(buckets, rotation=15, fontsize=9)
    axes[0].set_ylabel("Percentage of GPS Steps (log scale)")
    axes[0].set_title("GPS Sampling Intervals & Gap Breakdown")
    axes[0].legend()

    # Right: instantaneous step speed distribution (clipped at 50 m/s)
    for part in ("train", "student_test"):
        sp = np.clip(speed_arrays[part], 0.0, 50.0)
        axes[1].hist(
            sp,
            bins=50,
            density=True,
            alpha=0.45,
            color=colors[part],
            label=f"{part} (median={np.median(speed_arrays[part]):.1f} m/s)",
        )
    axes[1].axvline(35.0, color="#dc2626", linestyle="--", linewidth=1.2, label="Highway ref (~35 m/s / 126 km/h)")
    axes[1].set_xlabel("Instantaneous Step Speed (m/s, clipped at 50 m/s)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Step Speed Distribution Across Partitions")
    axes[1].legend(fontsize=8.5)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_sampling_intervals_and_gaps.png", dpi=180)
    plt.close(fig)


def plot_lengths_and_obs(dyn_df: pd.DataFrame) -> None:
    """Create Fig 2: GPS observation counts and physical trajectory length distributions."""
    np.random.seed(0)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = {"train": "#2563eb", "student_test": "#ea580c"}

    sns.boxplot(data=dyn_df, x="partition", y="n_obs", hue="partition", palette=palette, legend=False, ax=axes[0])
    sns.stripplot(data=dyn_df, x="partition", y="n_obs", color="#1e293b", alpha=0.5, size=4, ax=axes[0])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Partition")
    axes[0].set_ylabel("GPS Observations per Trip (log scale)")
    axes[0].set_title("GPS Observation Count per Trip")

    sns.boxplot(
        data=dyn_df, x="partition", y="total_length_km", hue="partition", palette=palette, legend=False, ax=axes[1]
    )
    sns.stripplot(data=dyn_df, x="partition", y="total_length_km", color="#1e293b", alpha=0.5, size=4, ax=axes[1])
    axes[1].set_xlabel("Partition")
    axes[1].set_ylabel("Total Trajectory Length (km)")
    axes[1].set_title("Physical Trajectory Length (UTM Projected)")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_trajectory_lengths_and_obs.png", dpi=180)
    plt.close(fig)


def plot_global_trip_locations(dyn_df: pd.DataFrame) -> None:
    """Create Fig 3: global distribution of trip centroids and UTM zones."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13, 6))
    palette = {"train": "#2563eb", "student_test": "#ea580c"}

    # Draw UTM longitudinal zone gridlines every 6 degrees
    for lon_line in range(-180, 181, 6):
        ax.axvline(lon_line, color="#e2e8f0", linewidth=0.5, zorder=0)
    ax.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=0.9, label="Equator (UTM North/South split)")

    for part, marker in (("train", "o"), ("student_test", "^")):
        sub = dyn_df[dyn_df["partition"] == part]
        ax.scatter(
            sub["mean_lon"],
            sub["mean_lat"],
            c=palette[part],
            marker=marker,
            s=75 if part == "student_test" else 55,
            alpha=0.85,
            edgecolors="white",
            linewidth=0.6,
            label=f"{part} ({len(sub)} trips, {sub['utm_epsg'].nunique()} UTM zones)",
            zorder=3,
        )

    # Annotate distinct geographic clusters with UTM EPSG codes
    cluster_labels = []
    for epsg, grp in dyn_df.groupby("utm_epsg"):
        cluster_labels.append((grp["mean_lon"].mean(), grp["mean_lat"].mean(), f"EPSG:{epsg}\n(n={len(grp)})"))
    for cx, cy, txt in cluster_labels:
        ax.annotate(
            txt,
            (cx, cy),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7.5,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cbd5e1", alpha=0.8),
        )

    ax.set_xlim(-120, 155)
    ax.set_ylim(-45, 70)
    ax.set_xlabel("Centroid Longitude (degrees)")
    ax.set_ylabel("Centroid Latitude (degrees)")
    ax.set_title("Global Trip Centroids & Dynamic UTM Zone Mapping (90 Trips)")
    ax.legend(loc="lower left")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_global_trip_locations.png", dpi=180)
    plt.close(fig)


def main() -> None:
    EDA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    crs_df = compute_crs_distortion_comparison(BASE_DIR / "train" / "00000001")
    crs_df.to_csv(EDA_REPORTS_DIR / "crs_distortion_comparison.csv", index=False)
    print("=== CRS Distortion Comparison (train/00000001) ===")
    print(crs_df.to_string(index=False))

    dyn_df, dt_arrays, speed_arrays = analyze_all_trajectories()
    dyn_df.to_csv(EDA_REPORTS_DIR / "trajectory_dynamics.csv", index=False)

    print("\n=== Trajectory Dynamics Summary by Partition ===")
    print(
        dyn_df.groupby("partition")[
            ["total_length_km", "mean_speed_mps", "median_speed_mps", "max_speed_mps", "n_speed_gt_50mps"]
        ]
        .agg(["min", "median", "max"])
        .round(3)
        .to_string()
    )

    plot_sampling_and_speeds(dt_arrays, speed_arrays)
    plot_lengths_and_obs(dyn_df)
    plot_global_trip_locations(dyn_df)
    print("\nSaved trajectory_dynamics.csv, crs_distortion_comparison.csv, and fig1-fig3.")


if __name__ == "__main__":
    main()
