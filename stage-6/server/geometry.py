"""Metric projection shared with the approved Stage 2 audit."""
import numpy as np


def local_metres(points, origin):
    # Local tangent-plane approximation using the WGS84 radii at the trip's mean latitude.
    lat = np.deg2rad(origin[1])
    factor = 1 - 0.00669437999014 * np.sin(lat)**2
    scales = [6378137 * np.cos(lat) / np.sqrt(factor),
              6378137 * (1 - 0.00669437999014) / factor**1.5]
    return np.deg2rad(points - origin) * scales
