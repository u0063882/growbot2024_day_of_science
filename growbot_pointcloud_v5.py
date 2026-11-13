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


def pointcloud_deck(points, point_size=5):
    """Make a camera-facing, auto-fitted PyDeck OrbitView of metric XYZ data.

    In OrbitView 1 metre occupies 2**zoom screen pixels; the former zoom=6
    rendered a 40-cm plant only ~26 pixels wide. Auto-fit fixes that.
    """
    import pydeck as pdk

    if points.empty:
        raise ValueError("No valid point-cloud data to display")
    coords = points[["x", "y", "z"]].to_numpy(dtype=np.float64)
    if not np.isfinite(coords).all():
        raise ValueError("Point cloud contains invalid XYZ coordinates")
    span = np.percentile(coords, 97.5, axis=0) - np.percentile(
        coords, 2.5, axis=0
    )
    visible_extent_m = max(float(np.max(span[:2])), float(span[2]) * 1.5, 0.05)
    zoom = float(np.clip(np.log2(430.0 / visible_extent_m), 6.5, 12.0))

    # XYZ are Cartesian metres relative to the camera, not WGS84 coordinates.
    # Draw RGB values without a material/lighting tint on the measured pixels.
    layer = pdk.Layer(
        "PointCloudLayer",
        data=points,
        get_position=["x", "y", "z"],
        get_color=["r", "g", "b"],
        get_normal=[0, 0, 1],
        point_size=int(point_size),
        coordinate_system="cartesian",
        material=False,
        pickable=False,
    )
    view = pdk.View(type="OrbitView", controller=True)
    state = pdk.ViewState(
        target=[0, 0, 0],
        rotation_x=0,
        rotation_orbit=0,
        zoom=zoom,
    )
    return pdk.Deck(
        layers=[layer], views=[view], initial_view_state=state,
        map_style=None, tooltip=False
    )


def estimate_scene_height_m(depth, depth_scale_m, foreground_mask=None):
    """Estimate height above the surrounding scene surface using depth percentiles.

    This is a simple camera-facing relative-height proxy, not a calibrated
    plant-height measurement. Assumes the border contains visible tabletop/
    background at approximately the same camera-to-surface distance.

    Returns None if there is insufficient valid depth. No SDK or motion calls.
    """
    if depth.ndim != 2 or depth_scale_m <= 0:
        return None
    h, w = depth.shape
    if h < 30 or w < 30:
        return None
    d = depth.astype(np.float32) * float(depth_scale_m)
    valid = np.isfinite(d) & (d >= 0.1) & (d <= 5.0)

    border = np.zeros((h, w), dtype=bool)
    bh = max(4, int(round(0.12 * h)))
    bw = max(4, int(round(0.12 * w)))
    border[:bh, :] = True
    border[-bh:, :] = True
    border[:, :bw] = True
    border[:, -bw:] = True

    center = np.zeros((h, w), dtype=bool)
    center[int(0.18*h):int(0.82*h), int(0.25*w):int(0.75*w)] = True
    subject = valid & center
    if foreground_mask is not None and foreground_mask.shape == depth.shape:
        masked = subject & (foreground_mask != 0)
        if np.count_nonzero(masked) >= 60:
            subject = masked

    bg = d[valid & border]
    fg = d[subject]
    if bg.size < 200 or fg.size < 60:
        return None
    background_depth = float(np.percentile(bg, 85))
    foreground_depth = float(np.percentile(fg, 15))
    return max(0.0, background_depth - foreground_depth)
