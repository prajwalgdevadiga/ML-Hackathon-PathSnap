"""Reproduce the training-only Stage 2 audit from the original student ZIP."""
import argparse
import csv
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from branca.element import Element
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def load(archive, trip):
    prefix = f"Student folder/train/{trip}/{trip}."
    arrays = []
    for suffix in ("track", "nodes", "arcs", "route"):
        with archive.open(prefix + suffix) as stream:
            arrays.append(np.loadtxt(stream, dtype=int if suffix in ("arcs", "route") else float,
                                     ndmin=1 if suffix == "route" else 2))
    return arrays


def local_metres(points, origin):
    # Local tangent-plane approximation using the WGS84 radii at the trip's mean latitude.
    lat = np.deg2rad(origin[1])
    factor = 1 - 0.00669437999014 * np.sin(lat)**2
    scales = [6378137 * np.cos(lat) / np.sqrt(factor),
              6378137 * (1 - 0.00669437999014) / factor**1.5]
    return np.deg2rad(points - origin) * scales


def route_distances(gps, segments):
    start = segments[:, 0]
    delta = segments[:, 1] - start
    squared = (delta**2).sum(axis=1)
    distances = np.empty((len(gps), len(segments)), dtype=np.float32)
    for i in range(0, len(gps), 64):
        offset = gps[i:i+64, None] - start
        t = (offset * delta).sum(axis=2) / np.where(squared > 0, squared, 1)
        residual = offset - np.clip(t, 0, 1)[:, :, None] * delta
        distances[i:i+64] = np.sqrt((residual**2).sum(axis=2))
    return distances


def ordered_alignment(distance):
    # A state is an occurrence in the reference list, so repeated arcs remain distinguishable.
    # A free start/end and forward skips let unsampled arcs stay unassigned.
    width = distance.shape[1]
    positions = np.arange(width)
    back = np.zeros(distance.shape, np.int32)
    cost = np.minimum(distance[0], 60).astype(float)
    for i in range(1, len(distance)):
        best = np.minimum.accumulate(cost)
        improved = np.r_[True, best[1:] < best[:-1]]
        back[i] = np.maximum.accumulate(np.where(improved, positions, 0))
        cost = best + np.minimum(distance[i], 60)
    path = np.empty(len(distance), dtype=np.int32)
    path[-1] = np.argmin(cost)
    for i in range(len(path)-1, 0, -1):
        path[i-1] = back[i, path[i]]
    assert np.all(np.diff(path) >= 0)
    return path


def inspect(trip, data):
    track, nodes, arcs, route = data
    origin = track[:, :2].mean(axis=0)
    gps = local_metres(track[:, :2], origin)
    edges = arcs[route]
    d = route_distances(gps, local_metres(nodes, origin)[edges])
    path = ordered_alignment(d)
    label = route[path]
    residual = d[np.arange(len(path)), path].copy()
    nearest = route[d.argmin(axis=1)]
    # Compare with every DIFFERENT reference arc, including other visits and opposing arcs.
    margin = np.empty(len(path))
    for i in range(0, len(path), 128):
        s = slice(i, i+128)
        alt = np.where(route[None, :] == label[s, None], np.inf, d[s]).min(axis=1)
        margin[s] = alt - residual[s]
    gaps = np.diff(track[:, 2])
    edge_set = set(map(tuple, edges))
    large = gaps > 30
    route_breaks = edges[:-1, 1] != edges[1:, 0]
    cumulative = np.r_[0, np.cumsum(route_breaks)]
    crosses_break = np.diff(cumulative[path]) > 0
    boundary = large | crosses_break
    near_boundary = np.r_[False, boundary] | np.r_[boundary, False]
    usable = (residual <= 25) & (margin >= 3) & ~near_boundary
    graph = csr_matrix((np.ones(len(arcs), dtype=bool), arcs.T), shape=(len(nodes), len(nodes)))
    components, group = connected_components(graph, directed=True, connection="weak")
    strong = connected_components(graph, directed=True, connection="strong", return_labels=False)
    row = dict(trip=trip, gps=len(track), nodes=len(nodes), arcs=len(arcs), route_entries=len(route),
               lon_min=float(track[:,0].min()), lat_min=float(track[:,1].min()),
               lon_max=float(track[:,0].max()), lat_max=float(track[:,1].max()),
               weak_components=int(components), strong_components=int(strong),
               largest_weak_nodes=int(np.bincount(group).max()),
               isolated_nodes=len(nodes)-len(np.unique(arcs)),
               gap_over_30s=int(large.sum()), max_gap_s=float(gaps.max()),
               route_breaks=int(route_breaks.sum()), crossed_break_steps=int(crosses_break.sum()),
               reverse_also_in_reference=sum((b,a) in edge_set for a,b in edges),
               far_from_all_reference_arcs=int((d.min(axis=1) > 25).sum()),
               residual_over_25m=int((residual > 25).sum()), margin_under_3m=int((margin < 3).sum()),
               boundary_points=int(near_boundary.sum()), retained=int(usable.sum()),
               nearest_disagrees=int((nearest != label).sum()),
               missing_numeric_cells=int(np.isnan(track).sum()+np.isnan(nodes).sum()),
               duplicate_timestamps=int(np.sum(gaps == 0)))
    for limit, separation in [(15,5), (40,1)]:
        row[f"retained_{limit}m_{separation}m"] = int(((residual <= limit) & (margin >= separation) & ~near_boundary).sum())
    return row, (path, residual, margin, near_boundary, usable, gaps)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_map(out, trip, data, details):
    track, nodes, arcs, route = data
    path, residual, margin, boundary, usable, gaps = details
    points = track[:, 1::-1]
    m = folium.Map(location=points.mean(axis=0), tiles=None, prefer_canvas=True, zoom_start=13)
    # Only native Leaflet paths/tooltips are used, so the maps can run without tile servers.
    m.default_js = []
    m.default_css = []
    assets = Path(__file__).parent / "assets"
    for name, tag in [("leaflet.js","script"), ("leaflet.css","style")]:
        m.get_root().header.add_child(Element(f"<{tag}>" + (assets/name).read_text(encoding="utf8") + f"</{tag}>"))
    m.get_root().header.add_child(Element("<style>.leaflet-container{background:#fafaf5}.leaflet-control-layers{font:14px Arial}</style>"))
    network = nodes[arcs][:,:,::-1]
    pad = np.array([0.003,0.005])
    active = ((network.max(axis=1) >= points.min(axis=0)-pad) & (network.min(axis=1) <= points.max(axis=0)+pad)).all(axis=1)
    roads = folium.FeatureGroup("Road network (grey)").add_to(m)
    folium.PolyLine(network[active].tolist(), color="#aeb8bb", weight=1, opacity=0.8).add_to(roads)
    ref = folium.FeatureGroup("Supplied reference route (green)").add_to(m)
    folium.PolyLine(network[route].tolist(), color="#00876c", weight=5, opacity=0.8).add_to(ref)
    groups = [folium.FeatureGroup(name).add_to(m) for name in ("Retained GPS (blue)", "Uncertain GPS (orange)")]
    for i, point in enumerate(points):
        status = "retained" if usable[i] else "uncertain"
        tip = (f"GPS row {i} | t={track[i,2]:g} s | arc {route[path[i]]} | {status} | "
               f"offset {residual[i]:.1f} m | margin {margin[i]:.1f} m"
               + (" | gap or reference break" if boundary[i] else ""))
        color = "#185ea8" if usable[i] else "#d45514"
        folium.CircleMarker(point.tolist(), radius=3, color=color, weight=1, fill=True,
                            fill_color=color, fill_opacity=0.85, tooltip=tip).add_to(groups[int(not usable[i])])
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([points.min(axis=0).tolist(),points.max(axis=0).tolist()])
    m.get_root().html.add_child(Element(f'<div style="position:fixed;z-index:900;bottom:12px;left:12px;background:white;padding:9px 13px;font:14px Arial">Training trip <b>{trip}</b> · Hover over GPS points for evidence</div>'))
    m.save(out/f"map-{trip}.html")
    examples = []
    for i in range(len(points)):
        examples.append(dict(gps_row=i, time_s=float(track[i,2]), route_position=int(path[i]),
                             arc_id=int(route[path[i]]), residual_m=round(float(residual[i]),3),
                             other_arc_margin_m=round(float(margin[i]),3),
                             near_gap_or_reference_break=bool(boundary[i]), retained=bool(usable[i])))
    write_csv(out/f"alignment-{trip}.csv", examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    rows, residuals, intervals = [], [], []
    with ZipFile(args.archive) as archive:
        trips = sorted(PurePosixPath(n).stem for n in archive.namelist() if n.startswith("Student folder/train/") and n.endswith(".track"))
        assert len(trips) == len(set(trips)) == 80
        for trip in trips:
            row, diagnostics = inspect(trip, load(archive, trip))
            rows.append(row)
            residuals.append(diagnostics[1])
            intervals.append(diagnostics[-1])
        typical_size = np.median([r["gps"] for r in rows])
        typical = min(rows, key=lambda r: (abs(r["gps"]-typical_size),r["trip"]))
        gap_case = max(rows, key=lambda r:r["max_gap_s"])
        for record in (typical, gap_case):
            trip = record["trip"]
            data = load(archive, trip)
            _, details = inspect(trip, data)
            make_map(args.out, trip, data, details)
    write_csv(args.out/"training-trip-audit.csv", rows)
    dist, gaps = np.concatenate(residuals), np.concatenate(intervals)
    fig, axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax, array, xlabel, title in [(axes[0],gaps,"Time between GPS points (s)","Sampling intervals"),(axes[1],dist,"Distance to assigned arc (m)","Reference alignment")]:
        x = np.sort(array)
        ax.plot(x, np.arange(1,len(x)+1)/len(x)*100,color="#175e78",lw=2)
        ax.set_xscale("symlog",linthresh=1)
        ax.set(xlabel=xlabel, ylabel="Cumulative share (%)", title=title, ylim=(0,103))
        ax.grid(alpha=.2)
    fig.suptitle("PathSnap · 80 training trips",fontweight="bold")
    fig.savefig(args.out/"training-distributions.png",dpi=160)
    plt.close(fig)
    summary = dict(typical_trip=typical["trip"],gap_trip=gap_case["trip"],
                   median_residual=float(np.median(dist)),p95_residual=float(np.percentile(dist,95)),
                   max_residual=float(dist.max()), totals={key:sum(r[key] for r in rows)
                   for key in ("gps","retained","residual_over_25m","margin_under_3m","boundary_points","gap_over_30s","route_breaks","nearest_disagrees","retained_15m_5m","retained_40m_1m","isolated_nodes")},
                   disconnected_graphs=sum(r["weak_components"]>1 for r in rows),
                   trips_with_gaps=sum(r["gap_over_30s"]>0 for r in rows),
                   trips_with_route_breaks=sum(r["route_breaks"]>0 for r in rows))
    print(json.dumps(summary,indent=2))
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")


if __name__ == "__main__":
    main()
