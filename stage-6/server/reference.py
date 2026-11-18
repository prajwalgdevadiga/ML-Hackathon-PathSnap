"""Scoring only. Alignment functions retained from the approved Stage 2 audit."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from geometry import local_metres


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
