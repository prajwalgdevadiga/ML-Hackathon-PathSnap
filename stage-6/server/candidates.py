"""Reference-free candidate generation for a complete GPS trip."""
import numpy as np
from scipy.spatial import cKDTree

from geometry import local_metres

RADIUS_M = 50.0
CAP = 8
SAMPLE_MAX_STEP_M = 40.0
FEATURES = ("distance_m", "distance_above_best_m", "arc_length_m", "directed_fraction",
            "heading_cosine", "heading_available", "speed_mps", "speed_available")


def motion(gps, times):
    index = np.arange(len(gps))
    # No pair crosses a 30 s gap. A heading needs displacement of at least 3 m.
    start = np.searchsorted(times, times - 5.0, side="left")
    elapsed = times - times[start]
    movement = gps - gps[start]
    length = np.linalg.norm(movement, axis=1)
    speed_ok = (index > start) & (elapsed > 0)
    heading_ok = speed_ok & (length >= 3)
    heading = np.divide(movement, length[:, None], out=np.zeros_like(movement),
                        where=heading_ok[:, None])
    speed = np.divide(length, elapsed, out=np.zeros_like(length), where=speed_ok)
    return heading, heading_ok, speed, speed_ok


def generate(track, nodes, arcs):
    """Return ordered arc IDs, matching features and a per-observation fallback flag."""
    if not len(track) or not len(arcs):
        raise ValueError("A trip needs at least one observation and one road arc.")
    if np.any(np.diff(track[:, 2]) <= 0):
        raise ValueError("Expected strictly increasing observation times.")
    origin = track[:, :2].mean(axis=0)
    gps = local_metres(track[:, :2], origin)
    xy = local_metres(nodes, origin)
    # Identical reverse arcs share the same physical calculation, even at endpoints.
    start = xy[arcs.min(axis=1)]
    delta = xy[arcs.max(axis=1)] - start
    forward = arcs[:, 0] <= arcs[:, 1]
    squared = np.einsum("ij,ij->i", delta, delta)
    length = np.sqrt(squared)
    unit = delta / np.where(length > 0, length, 1)[:, None]
    unit[~forward] *= -1
    heading, heading_ok, speed, speed_ok = motion(gps, track[:, 2])

    # Midpoints of equal-length pieces put an indexed sample within 20 m of any
    # point on the segment, including an endpoint or a zero-length segment.
    pieces = np.maximum(1, np.ceil(length / SAMPLE_MAX_STEP_M).astype(int))
    owner = np.repeat(np.arange(len(arcs)), pieces)
    ordinal = np.arange(len(owner)) - np.repeat(np.cumsum(pieces) - pieces, pieces)
    samples = start[owner] + delta[owner] * ((ordinal + 0.5) / pieces[owner])[:, None]
    tree = cKDTree(samples)
    ids = np.full((len(gps), CAP), -1, dtype=np.int32)
    features = np.zeros((len(gps), CAP, len(FEATURES)), dtype=np.float32)
    fallback = np.zeros(len(gps), dtype=bool)
    for begin in range(0, len(gps), 256):
        batch = gps[begin:begin+256]
        upper = tree.query(batch)[0]
        # If the 50 m disk is empty, the closest sample bounds the global nearest
        # segment's distance. Adding 20 m covers that segment's indexed sample.
        radius = np.maximum(RADIUS_M, upper) + SAMPLE_MAX_STEP_M / 2 + 1e-7
        neighbours = tree.query_ball_point(batch, radius)
        owners = [np.unique(owner[hits]) for hits in neighbours]
        sizes = np.array([len(items) for items in owners])
        p = np.repeat(np.arange(len(batch)), sizes)
        a = np.concatenate(owners)
        off = batch[p] - start[a]
        t = np.clip(np.einsum("ij,ij->i", off, delta[a]) / np.where(squared[a] > 0, squared[a], 1), 0, 1)
        dist = np.linalg.norm(off - t[:,None] * delta[a], axis=1)
        order = np.lexsort((a, dist, p))
        p, a, t, dist = p[order], a[order], t[order], dist[order]
        first = np.cumsum(sizes) - sizes
        rank = np.arange(len(p)) - first[p]
        keep = (rank < CAP) & ((dist <= RADIUS_M) | (rank == 0))
        fallback[begin:begin+len(batch)] = dist[first] > RADIUS_M
        best = dist[first]
        p, a, t, dist, rank = (v[keep] for v in (p, a, t, dist, rank))
        obs = begin + p
        direction_ok = heading_ok[obs] & (length[a] > 0)
        cosine = np.einsum("ij,ij->i", heading[obs], unit[a])
        ids[obs,rank] = a
        features[obs,rank] = np.column_stack((dist, dist-best[p], length[a],
            np.where(forward[a], t, 1-t), np.where(direction_ok, np.clip(cosine,-1,1), 0),
            direction_ok, speed[obs], speed_ok[obs]))
    return ids, features, fallback
