"""Interactive RealSense RGB point cloud for GROWBOT v5.

Requires only NumPy, pandas and pydeck (pandas/pydeck are dependencies of
Streamlit 1.33). Does not communicate with the cameras or gantry.
"""
from __future__ import annotations

import numpy as np


def pointcloud_data(rgb, depth, intrinsics, depth_scale_m, point_limit=15000,
                    depth_range_m=(0.15, 2.0), mask=None):
    """Return a compact DataFrame of RGB-coloured points in metres."""
    import pandas as pd

    if intrinsics is None:
        raise ValueError("RealSense colour intrinsics unavailable; reconnect camera")
    h, w = depth.shape
    if rgb.shape[:2] != (h, w) or rgb.shape[2] != 3:
        raise ValueError("RGB and depth must have the same spatial shape")
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["ppx"]), float(intrinsics["ppy"])
    if min(fx, fy) <= 0:
        raise ValueError("Invalid RealSense camera intrinsics")
    if mask is not None and mask.shape != (h, w):
        raise ValueError("Point-cloud mask shape does not match depth")

    step = max(1, int(np.ceil(np.sqrt(h * w / float(point_limit)))))
    y, x = np.mgrid[0:h:step, 0:w:step]
    z = depth[::step, ::step].astype(np.float32) * float(depth_scale_m)
    valid = (z >= depth_range_m[0]) & (z <= depth_range_m[1]) & np.isfinite(z)
    if mask is not None:
        valid &= mask[::step, ::step] != 0
    if not np.any(valid):
        raise ValueError("No valid depth points within the selected range and mask")
    x = x[valid].astype(np.float32)
    y = y[valid].astype(np.float32)
    z = z[valid]
    colors = rgb[::step, ::step][valid]

    # RealSense camera convention: Z points away from sensor, Y down.
    # Orbit viewer is centred on the scene, Y flipped to point upwards.
    px = (x - cx) * z / fx
    py = -(y - cy) * z / fy
    pz = -z
    center = np.median(np.stack((px, py, pz), axis=1), axis=0)
    coords = np.stack((px, py, pz), axis=1) - center

    if len(coords) > point_limit:
        take = np.linspace(0, len(coords) - 1, point_limit, dtype=np.int64)
        coords, colors = coords[take], colors[take]

    return pd.DataFrame({
        "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2],
        "r": colors[:, 0], "g": colors[:, 1], "b": colors[:, 2],
    })


def pointcloud_deck(points):
    """Return a PyDeck OrbitView; the browser renders a rotatable 3D view."""
    import pydeck as pdk

    layer = pdk.Layer(
        "PointCloudLayer", data=points,
        get_position="[x, y, z]", get_color="[r, g, b]",
        get_normal=[0, 0, 1], point_size=3,
        pickable=False,
    )
    view = pdk.View(type="OrbitView", controller=True)
    state = pdk.ViewState(target=[0, 0, 0], rotation_x=15,
                          rotation_orbit=20, zoom=6)
    return pdk.Deck(layers=[layer], views=[view], initial_view_state=state,
                    tooltip=False)
