"""Interactive Folium geographical maps and static trajectory plots for PathSnap EDA."""

from __future__ import annotations

import json
from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from src.eda.inventory_and_quality import BASE_DIR, EDA_REPORTS_DIR, load_trip_raw
from src.eda.network_and_route_analysis import point_to_segments_min_dist
from src.eda.trajectory_analysis import FIGURES_DIR, project_coords_to_utm

MAPS_DIR = EDA_REPORTS_DIR / "maps"


def _stitch_arc_polylines(arcs: np.ndarray, arc_ids: np.ndarray, nodes_lonlat: np.ndarray) -> list[list[list[float]]]:
    """Stitch consecutive connected arcs into continuous [lat, lon] polylines for Folium."""
    if len(arc_ids) == 0:
        return []

    polylines: list[list[list[float]]] = []
    first_u, first_v = int(arcs[arc_ids[0], 0]), int(arcs[arc_ids[0], 1])
    current_line = [
        [float(nodes_lonlat[first_u, 1]), float(nodes_lonlat[first_u, 0])],
        [float(nodes_lonlat[first_v, 1]), float(nodes_lonlat[first_v, 0])],
    ]
    prev_v = first_v

    for aid in arc_ids[1:]:
        u, v = int(arcs[aid, 0]), int(arcs[aid, 1])
        if u == prev_v:
            current_line.append([float(nodes_lonlat[v, 1]), float(nodes_lonlat[v, 0])])
        else:
            polylines.append(current_line)
            current_line = [
                [float(nodes_lonlat[u, 1]), float(nodes_lonlat[u, 0])],
                [float(nodes_lonlat[v, 1]), float(nodes_lonlat[v, 0])],
            ]
        prev_v = v

    polylines.append(current_line)
    return polylines


def _filter_corridor_arcs(
    track_xy: np.ndarray,
    nodes_xy: np.ndarray,
    arcs: np.ndarray,
    buffer_m: float,
    must_include_arcs: np.ndarray | None = None,
) -> np.ndarray:
    """Return indices of road arcs whose sampled points fall within buffer_m of the GPS track."""
    tree = cKDTree(track_xy)
    u_xy = nodes_xy[arcs[:, 0]]
    v_xy = nodes_xy[arcs[:, 1]]
    mask = np.zeros(len(arcs), dtype=bool)
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        sample_xy = (1.0 - t) * u_xy + t * v_xy
        dist, _ = tree.query(sample_xy, distance_upper_bound=buffer_m)
        mask |= np.isfinite(dist)

    if must_include_arcs is not None and len(must_include_arcs) > 0:
        mask[must_include_arcs] = True

    return np.flatnonzero(mask)


def create_trip_folium_map(
    trip_dir: str | Path,
    output_html_path: str | Path,
    predicted_arcs: np.ndarray | list[int] | None = None,
    buffer_m: float = 150.0,
) -> Path:
    """Create an interactive Folium map showing road network, GPS track, reference route, and optional predicted route."""
    trip_path = Path(trip_dir)
    out_path = Path(output_html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_trip_raw(trip_path)
    track = data["track"]
    nodes = data["nodes"]
    arcs = data["arcs"]
    route = data["route"]

    tx, ty, epsg = project_coords_to_utm(track[:, 0], track[:, 1])
    nx, ny, _ = project_coords_to_utm(nodes[:, 0], nodes[:, 1], epsg=epsg)
    track_xy = np.column_stack([tx, ty])
    nodes_xy = np.column_stack([nx, ny])

    pred_arr = np.asarray(predicted_arcs, dtype=np.int64) if predicted_arcs is not None else None
    must_inc_parts = [arr for arr in (route, pred_arr) if arr is not None and len(arr) > 0]
    must_inc = np.concatenate(must_inc_parts) if must_inc_parts else None
    corridor_arc_ids = _filter_corridor_arcs(track_xy, nodes_xy, arcs, buffer_m=buffer_m, must_include_arcs=must_inc)

    mean_lat = float(np.mean(track[:, 1]))
    mean_lon = float(np.mean(track[:, 0]))
    fmap = folium.Map(location=[mean_lat, mean_lon], zoom_start=13, tiles="OpenStreetMap")

    # 1. Relevant Road Network layer (GeoJSON MultiLineString for fast rendering)
    road_group = folium.FeatureGroup(name=f"Road Network (within {int(buffer_m)}m: {len(corridor_arc_ids):,} arcs)", show=True)
    if len(corridor_arc_ids) > 0:
        u_coords = nodes[arcs[corridor_arc_ids, 0]]
        v_coords = nodes[arcs[corridor_arc_ids, 1]]
        lines_geojson = {
            "type": "Feature",
            "geometry": {
                "type": "MultiLineString",
                "coordinates": [
                    [[float(u[0]), float(u[1])], [float(v[0]), float(v[1])]]
                    for u, v in zip(u_coords, v_coords)
                ],
            },
        }
        folium.GeoJson(
            data=json.dumps(lines_geojson),
            style_function=lambda _: {"color": "#64748b", "weight": 1.6, "opacity": 0.55},
            tooltip=f"Road Network Corridor ({len(corridor_arc_ids):,} segments)",
        ).add_to(road_group)
    road_group.add_to(fmap)

    # 2. Raw GPS Observations (.track) underneath route overlays so routes remain legible
    gps_group = folium.FeatureGroup(name=f"Raw GPS Observations (.track: {len(track):,} pts)", show=True)
    track_latlon = [[float(lat), float(lon)] for lon, lat in track[:, :2]]
    folium.PolyLine(
        locations=track_latlon,
        color="#3b82f6",
        weight=1.8,
        opacity=0.45,
        dash_array="4, 4",
        tooltip="Raw GPS Trajectory",
    ).add_to(gps_group)

    marker_stride = 1 if len(track) <= 3000 else max(1, len(track) // 2000)
    for obs_idx in range(0, len(track), marker_stride):
        lon_v, lat_v, ts_v = track[obs_idx]
        folium.CircleMarker(
            location=[float(lat_v), float(lon_v)],
            radius=1.8,
            color="#1d4ed8",
            fill=True,
            fill_color="#60a5fa",
            fill_opacity=0.45,
            opacity=0.45,
            weight=0,
            popup=f"Obs #{obs_idx} | t={ts_v:.1f}s | ({lat_v:.6f}, {lon_v:.6f})",
        ).add_to(gps_group)
    gps_group.add_to(fmap)

    # 3. Reference Ground-Truth Route (.route)
    if route is not None and len(route) > 0:
        route_group = folium.FeatureGroup(name=f"Reference Route (.route: {len(route)} arcs)", show=True)
        for idx, poly in enumerate(_stitch_arc_polylines(arcs, route, nodes)):
            folium.PolyLine(
                locations=poly,
                color="#16a34a",
                weight=6.0,
                opacity=0.85,
                tooltip=f"Reference Route (segment block {idx + 1})",
            ).add_to(route_group)
        start_node = int(arcs[route[0], 0])
        end_node = int(arcs[route[-1], 1])
        folium.Marker(
            location=[float(nodes[start_node, 1]), float(nodes[start_node, 0])],
            popup=f"Route Start (arc {route[0]})",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(route_group)
        folium.Marker(
            location=[float(nodes[end_node, 1]), float(nodes[end_node, 0])],
            popup=f"Route End (arc {route[-1]})",
            icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
        ).add_to(route_group)
        route_group.add_to(fmap)

    # 4. Predicted Route on top (optional, reused in Phase 2 & UI)
    if pred_arr is not None and len(pred_arr) > 0:
        pred_group = folium.FeatureGroup(name=f"Predicted Route ({len(pred_arr)} arcs)", show=True)
        for idx, poly in enumerate(_stitch_arc_polylines(arcs, pred_arr, nodes)):
            folium.PolyLine(
                locations=poly,
                color="#ea580c",
                weight=3.2,
                opacity=0.95,
                dash_array="8, 6",
                tooltip=f"Predicted Route (segment block {idx + 1})",
            ).add_to(pred_group)
        pred_group.add_to(fmap)

    # Custom legend box
    pred_legend_item = (
        '<div><span style="display:inline-block;width:16px;height:3px;border-top:3px dashed #ea580c;margin-right:6px;"></span>Predicted Route</div>'
        if pred_arr is not None and len(pred_arr) > 0
        else ""
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999;
                background-color: rgba(255,255,255,0.94); border: 1px solid #cbd5e1;
                border-radius: 8px; padding: 10px 14px; font-family: sans-serif;
                font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
      <div style="font-weight: 700; margin-bottom: 6px; color: #0f172a;">
        PathSnap: {trip_path.parent.name}/{data['trip_id']} (EPSG:{epsg})
      </div>
      <div><span style="display:inline-block;width:16px;height:3px;background:#64748b;margin-right:6px;vertical-align:middle;"></span>Road Network (.arcs)</div>
      <div><span style="display:inline-block;width:16px;height:4px;background:#16a34a;margin-right:6px;vertical-align:middle;"></span>Reference Route (.route)</div>
      {pred_legend_item}
      <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#2563eb;margin-right:9px;margin-left:3px;vertical-align:middle;"></span>Raw GPS Points (.track)</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(fmap)

    fmap.fit_bounds([[float(track[:, 1].min()), float(track[:, 0].min())], [float(track[:, 1].max()), float(track[:, 0].max())]])
    fmap.save(str(out_path))
    return out_path


def _plot_trip_panel(
    ax: plt.Axes,
    track_xy: np.ndarray,
    nodes_xy: np.ndarray,
    arcs: np.ndarray,
    route: np.ndarray,
    corridor_ids: np.ndarray,
    title: str,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Helper to draw road network, ground-truth route, and GPS points in relative km/m."""
    cx0, cy0 = track_xy[0, 0], track_xy[0, 1]
    tx = (track_xy[:, 0] - cx0) / 1000.0
    ty = (track_xy[:, 1] - cy0) / 1000.0
    nxy = np.column_stack([(nodes_xy[:, 0] - cx0) / 1000.0, (nodes_xy[:, 1] - cy0) / 1000.0])

    # Plot corridor road segments
    u_c = nxy[arcs[corridor_ids, 0]]
    v_c = nxy[arcs[corridor_ids, 1]]
    xs = np.column_stack([u_c[:, 0], v_c[:, 0], np.full(len(corridor_ids), np.nan)]).ravel()
    ys = np.column_stack([u_c[:, 1], v_c[:, 1], np.full(len(corridor_ids), np.nan)]).ravel()
    ax.plot(xs, ys, color="#94a3b8", linewidth=0.7, alpha=0.6, label="Road Network (.arcs)")

    # Plot reference route
    u_r = nxy[arcs[route, 0]]
    v_r = nxy[arcs[route, 1]]
    rxs = np.column_stack([u_r[:, 0], v_r[:, 0], np.full(len(route), np.nan)]).ravel()
    rys = np.column_stack([u_r[:, 1], v_r[:, 1], np.full(len(route), np.nan)]).ravel()
    ax.plot(rxs, rys, color="#16a34a", linewidth=2.6, alpha=0.9, label="Reference Route (.route)")

    # Plot GPS observations
    ax.plot(tx, ty, color="#2563eb", linewidth=1.0, linestyle="--", alpha=0.65)
    ax.scatter(tx, ty, c="#1d4ed8", s=8, alpha=0.8, zorder=4, label="Raw GPS (.track)")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    adjustable = "box" if (xlim is not None or ylim is not None) else "datalim"
    ax.set_aspect("equal", adjustable=adjustable)
    ax.set_xlabel("Relative Easting (km)")
    ax.set_ylabel("Relative Northing (km)")
    ax.set_title(title, fontsize=10.5)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)


def plot_sample_trajectories() -> None:
    """Create Fig 6: static trajectory maps + zoomed urban noise inset for PDF report."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    # Trip 1: train/00000001
    t1_dir = BASE_DIR / "train" / "00000001"
    d1 = load_trip_raw(t1_dir)
    tx1, ty1, epsg1 = project_coords_to_utm(d1["track"][:, 0], d1["track"][:, 1])
    nx1, ny1, _ = project_coords_to_utm(d1["nodes"][:, 0], d1["nodes"][:, 1], epsg=epsg1)
    track_xy1 = np.column_stack([tx1, ty1])
    nodes_xy1 = np.column_stack([nx1, ny1])
    corr1 = _filter_corridor_arcs(track_xy1, nodes_xy1, d1["arcs"], buffer_m=250.0, must_include_arcs=d1["route"])

    _plot_trip_panel(
        axes[0],
        track_xy1,
        nodes_xy1,
        d1["arcs"],
        d1["route"],
        corr1,
        f"train/00000001 Full Trip (EPSG:{epsg1})",
    )

    # Zoomed-in urban GPS offset inset around max noise observation (or dense urban area)
    seg_a1 = nodes_xy1[d1["arcs"][d1["route"], 0]]
    seg_b1 = nodes_xy1[d1["arcs"][d1["route"], 1]]
    dists1 = point_to_segments_min_dist(track_xy1, seg_a1, seg_b1)
    # Pick 95th percentile noisy point to show offset near a motorway interchange
    p95_idx = int(np.argsort(dists1)[int(0.95 * len(dists1))])
    zx = (track_xy1[p95_idx, 0] - track_xy1[0, 0]) / 1000.0
    zy = (track_xy1[p95_idx, 1] - track_xy1[0, 1]) / 1000.0
    _plot_trip_panel(
        axes[1],
        track_xy1,
        nodes_xy1,
        d1["arcs"],
        d1["route"],
        corr1,
        "train/00000001 95th-Percentile Offset Zoom (400m box)",
        xlim=(zx - 0.2, zx + 0.2),
        ylim=(zy - 0.2, zy + 0.2),
    )

    # Trip 2: student_test/00000000
    t2_dir = BASE_DIR / "student_test" / "00000000"
    d2 = load_trip_raw(t2_dir)
    tx2, ty2, epsg2 = project_coords_to_utm(d2["track"][:, 0], d2["track"][:, 1])
    nx2, ny2, _ = project_coords_to_utm(d2["nodes"][:, 0], d2["nodes"][:, 1], epsg=epsg2)
    track_xy2 = np.column_stack([tx2, ty2])
    nodes_xy2 = np.column_stack([nx2, ny2])
    corr2 = _filter_corridor_arcs(track_xy2, nodes_xy2, d2["arcs"], buffer_m=250.0, must_include_arcs=d2["route"])

    _plot_trip_panel(
        axes[2],
        track_xy2,
        nodes_xy2,
        d2["arcs"],
        d2["route"],
        corr2,
        f"student_test/00000000 Full Trip (EPSG:{epsg2})",
    )

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_sample_trip_trajectories.png", dpi=180)
    plt.close(fig)


def main() -> None:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    targets = [
        (BASE_DIR / "train" / "00000001", MAPS_DIR / "trip_train_00000001.html"),
        (BASE_DIR / "student_test" / "00000000", MAPS_DIR / "trip_student_test_00000000.html"),
        (BASE_DIR / "student_test" / "00000007", MAPS_DIR / "trip_student_test_00000007.html"),
    ]
    for trip_dir, out_html in targets:
        saved = create_trip_folium_map(trip_dir, out_html, buffer_m=150.0)
        size_kb = saved.stat().st_size / 1024.0
        print(f"Generated Folium map: {saved.name} ({size_kb:.1f} KB)")

    plot_sample_trajectories()
    print("Generated fig6_sample_trip_trajectories.png")


if __name__ == "__main__":
    main()
