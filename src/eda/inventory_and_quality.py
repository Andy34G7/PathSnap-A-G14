"""Dataset inventory, sample inspection, and data-quality audit for PathSnap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
EDA_REPORTS_DIR = BASE_DIR / "reports" / "eda"


def get_utm_epsg(mean_lon: float, mean_lat: float) -> int:
    """Resolve the local WGS84 UTM EPSG code from geographic centroid coordinates."""
    zone = int((mean_lon + 180.0) // 6.0) + 1
    zone = max(1, min(60, zone))
    base = 32600 if mean_lat >= 0.0 else 32700
    return base + zone


def load_trip_raw(trip_dir: str | Path) -> dict[str, Any]:
    """Load raw .track, .nodes, .arcs, and .route files for a single trip directory."""
    trip_path = Path(trip_dir)
    trip_id = trip_path.name

    track_file = trip_path / f"{trip_id}.track"
    nodes_file = trip_path / f"{trip_id}.nodes"
    arcs_file = trip_path / f"{trip_id}.arcs"
    route_file = trip_path / f"{trip_id}.route"

    file_sizes = {
        "track_bytes": track_file.stat().st_size if track_file.exists() else 0,
        "nodes_bytes": nodes_file.stat().st_size if nodes_file.exists() else 0,
        "arcs_bytes": arcs_file.stat().st_size if arcs_file.exists() else 0,
        "route_bytes": route_file.stat().st_size if route_file.exists() else 0,
    }

    track_df = pd.read_csv(track_file, sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
    nodes_df = pd.read_csv(nodes_file, sep=r"\s+", header=None, names=["lon", "lat"])
    arcs_df = pd.read_csv(arcs_file, sep=r"\s+", header=None, names=["from_node", "to_node"])
    route_df = (
        pd.read_csv(route_file, sep=r"\s+", header=None, names=["arc_id"])
        if route_file.exists()
        else None
    )

    return {
        "trip_id": trip_id,
        "track": track_df.to_numpy(dtype=np.float64),
        "nodes": nodes_df.to_numpy(dtype=np.float64),
        "arcs": arcs_df.to_numpy(dtype=np.int64),
        "route": route_df["arc_id"].to_numpy(dtype=np.int64) if route_df is not None else None,
        "file_sizes": file_sizes,
    }


def inspect_sample_trip(trip_dir: Path) -> None:
    """Print detailed inspection output for a sample trip."""
    data = load_trip_raw(trip_dir)
    track = data["track"]
    nodes = data["nodes"]
    arcs = data["arcs"]
    route = data["route"]

    print(f"\n=== Sample Inspection: {trip_dir.parent.name}/{data['trip_id']} ===")
    print(
        f"Shapes -> track: {track.shape}, nodes: {nodes.shape}, "
        f"arcs: {arcs.shape}, route: {None if route is None else route.shape}"
    )
    print(f"File sizes (bytes): {data['file_sizes']}")
    print("Track first 5 rows (lon, lat, timestamp):")
    print(track[:5])
    print("Nodes first 5 rows (lon, lat):")
    print(nodes[:5])
    print("Arcs first 5 rows (from_node, to_node):")
    print(arcs[:5])
    if route is not None:
        print(f"Route first 5 arc_ids: {route[:5]}")

    dt = np.diff(track[:, 2])
    print(
        f"Track lon bounds: [{track[:, 0].min():.6f}, {track[:, 0].max():.6f}], "
        f"lat bounds: [{track[:, 1].min():.6f}, {track[:, 1].max():.6f}]"
    )
    print(
        f"Timestamps: start={track[0, 2]:.2f}s, end={track[-1, 2]:.2f}s, "
        f"dt min={dt.min():.3f}s, median={np.median(dt):.3f}s, max={dt.max():.3f}s"
    )
    print(
        f"Arc node index bounds: min={arcs.min()}, max={arcs.max()} "
        f"(valid range [0, {len(nodes) - 1}])"
    )
    if route is not None:
        print(
            f"Route arc index bounds: min={route.min()}, max={route.max()} "
            f"(valid range [0, {len(arcs) - 1}])"
        )


def analyze_trip(partition: str, trip_dir: Path) -> dict[str, Any]:
    """Compute inventory, schema, coordinate, timestamp, and reference-route quality metrics."""
    data = load_trip_raw(trip_dir)
    track = data["track"]
    nodes = data["nodes"]
    arcs = data["arcs"]
    route = data["route"]
    sizes = data["file_sizes"]

    n_track = len(track)
    n_nodes = len(nodes)
    n_arcs = len(arcs)
    n_route = len(route) if route is not None else 0

    n_missing_nan_inf = int(
        np.sum(~np.isfinite(track))
        + np.sum(~np.isfinite(nodes))
        + np.sum(~np.isfinite(arcs))
        + (np.sum(~np.isfinite(route)) if route is not None else 0)
    )

    invalid_track_coords = np.sum(
        (np.abs(track[:, 0]) > 180.0) | (np.abs(track[:, 1]) > 90.0)
    )
    invalid_node_coords = np.sum(
        (np.abs(nodes[:, 0]) > 180.0) | (np.abs(nodes[:, 1]) > 90.0)
    )
    n_invalid_coords = int(invalid_track_coords + invalid_node_coords)

    n_invalid_arc_nodes = int(np.sum((arcs < 0) | (arcs >= n_nodes)))
    n_invalid_route_arcs = (
        int(np.sum((route < 0) | (route >= n_arcs))) if route is not None else 0
    )

    timestamps = track[:, 2]
    dt = np.diff(timestamps) if n_track > 1 else np.array([], dtype=np.float64)
    n_steps = len(dt)
    dt_min = float(np.min(dt)) if n_steps > 0 else 0.0
    dt_max = float(np.max(dt)) if n_steps > 0 else 0.0
    dt_median = float(np.median(dt)) if n_steps > 0 else 0.0
    n_dt_exactly_1s = int(np.sum(np.isclose(dt, 1.0, atol=1e-3))) if n_steps > 0 else 0
    pct_dt_1s = float(100.0 * n_dt_exactly_1s / n_steps) if n_steps > 0 else 0.0
    n_negative_dt = int(np.sum(dt < 0.0)) if n_steps > 0 else 0
    n_zero_dt = int(np.sum(np.isclose(dt, 0.0, atol=1e-6))) if n_steps > 0 else 0
    n_gaps_gt_5s = int(np.sum(dt > 5.0)) if n_steps > 0 else 0
    n_gaps_gt_60s = int(np.sum(dt > 60.0)) if n_steps > 0 else 0

    n_disconnected_route_steps = 0
    if route is not None and len(route) > 1 and n_invalid_route_arcs == 0:
        from_nodes = arcs[route[1:], 0]
        to_nodes = arcs[route[:-1], 1]
        same_arc = route[1:] == route[:-1]
        n_disconnected_route_steps = int(np.sum((to_nodes != from_nodes) & (~same_arc)))

    lon_min = float(np.min(track[:, 0]))
    lon_max = float(np.max(track[:, 0]))
    lat_min = float(np.min(track[:, 1]))
    lat_max = float(np.max(track[:, 1]))
    mean_lon = float(np.mean(track[:, 0]))
    mean_lat = float(np.mean(track[:, 1]))
    utm_epsg = get_utm_epsg(mean_lon, mean_lat)

    return {
        "partition": partition,
        "trip_id": data["trip_id"],
        "n_track": n_track,
        "n_nodes": n_nodes,
        "n_arcs": n_arcs,
        "n_route": n_route,
        **sizes,
        "n_missing_nan_inf": n_missing_nan_inf,
        "n_invalid_coords": n_invalid_coords,
        "n_invalid_arc_nodes": n_invalid_arc_nodes,
        "n_invalid_route_arcs": n_invalid_route_arcs,
        "t_start": float(timestamps[0]),
        "t_end": float(timestamps[-1]),
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "dt_min": dt_min,
        "dt_max": dt_max,
        "dt_median": dt_median,
        "n_steps": n_steps,
        "n_dt_exactly_1s": n_dt_exactly_1s,
        "pct_dt_1s": pct_dt_1s,
        "n_negative_dt": n_negative_dt,
        "n_zero_dt": n_zero_dt,
        "n_gaps_gt_5s": n_gaps_gt_5s,
        "n_gaps_gt_60s": n_gaps_gt_60s,
        "n_disconnected_route_steps": n_disconnected_route_steps,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "mean_lon": mean_lon,
        "mean_lat": mean_lat,
        "utm_epsg": utm_epsg,
    }


def build_partition_summary(inv_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate dataset inventory statistics by partition."""
    rows = []
    for partition, grp in inv_df.groupby("partition", sort=False):
        total_steps = int(grp["n_steps"].sum())
        total_1s = int(grp["n_dt_exactly_1s"].sum())
        rows.append(
            {
                "partition": partition,
                "n_trips": len(grp),
                "total_track_obs": int(grp["n_track"].sum()),
                "min_track_obs": int(grp["n_track"].min()),
                "median_track_obs": float(grp["n_track"].median()),
                "max_track_obs": int(grp["n_track"].max()),
                "total_nodes": int(grp["n_nodes"].sum()),
                "median_nodes": float(grp["n_nodes"].median()),
                "max_nodes": int(grp["n_nodes"].max()),
                "total_arcs": int(grp["n_arcs"].sum()),
                "median_arcs": float(grp["n_arcs"].median()),
                "max_arcs": int(grp["n_arcs"].max()),
                "total_route_arcs": int(grp["n_route"].sum()),
                "median_route_arcs": float(grp["n_route"].median()),
                "max_route_arcs": int(grp["n_route"].max()),
                "total_missing_nan_inf": int(grp["n_missing_nan_inf"].sum()),
                "total_invalid_coords": int(grp["n_invalid_coords"].sum()),
                "total_invalid_arc_nodes": int(grp["n_invalid_arc_nodes"].sum()),
                "total_invalid_route_arcs": int(grp["n_invalid_route_arcs"].sum()),
                "pct_steps_exactly_1s": round(100.0 * total_1s / total_steps, 4) if total_steps else 0.0,
                "total_negative_dt": int(grp["n_negative_dt"].sum()),
                "total_zero_dt": int(grp["n_zero_dt"].sum()),
                "total_gaps_gt_5s": int(grp["n_gaps_gt_5s"].sum()),
                "total_gaps_gt_60s": int(grp["n_gaps_gt_60s"].sum()),
                "max_dt_s": float(grp["dt_max"].max()),
                "total_disconnected_route_steps": int(grp["n_disconnected_route_steps"].sum()),
                "n_distinct_utm_zones": int(grp["utm_epsg"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    EDA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    inspect_sample_trip(BASE_DIR / "train" / "00000001")
    inspect_sample_trip(BASE_DIR / "student_test" / "00000000")

    records = []
    for partition in ("train", "student_test"):
        part_dir = BASE_DIR / partition
        trip_dirs = sorted(p for p in part_dir.iterdir() if p.is_dir())
        for trip_dir in trip_dirs:
            records.append(analyze_trip(partition, trip_dir))

    inv_df = pd.DataFrame(records)
    summary_df = build_partition_summary(inv_df)

    anomaly_mask = (
        (inv_df["dt_max"] > 10.0)
        | (inv_df["n_dt_exactly_1s"] < inv_df["n_steps"])
        | (inv_df["n_disconnected_route_steps"] > 0)
        | (inv_df["n_missing_nan_inf"] > 0)
        | (inv_df["n_invalid_coords"] > 0)
        | (inv_df["n_negative_dt"] > 0)
        | (inv_df["n_zero_dt"] > 0)
    )
    anomalies_df = inv_df.loc[
        anomaly_mask,
        [
            "partition",
            "trip_id",
            "n_track",
            "n_route",
            "dt_min",
            "dt_max",
            "pct_dt_1s",
            "n_negative_dt",
            "n_zero_dt",
            "n_gaps_gt_5s",
            "n_gaps_gt_60s",
            "n_disconnected_route_steps",
            "n_missing_nan_inf",
            "n_invalid_coords",
        ],
    ].sort_values(by=["n_gaps_gt_60s", "dt_max", "n_disconnected_route_steps"], ascending=False)

    inv_df.to_csv(EDA_REPORTS_DIR / "trip_inventory.csv", index=False)
    summary_df.to_csv(EDA_REPORTS_DIR / "partition_summary.csv", index=False)
    anomalies_df.to_csv(EDA_REPORTS_DIR / "anomalies_and_gaps.csv", index=False)

    print("\n=== Partition Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved {len(inv_df)} trips to trip_inventory.csv; {len(anomalies_df)} flagged in anomalies_and_gaps.csv.")


if __name__ == "__main__":
    main()
