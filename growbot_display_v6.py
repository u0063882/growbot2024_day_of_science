"""Display-only helpers for GROWBOT v6 (no device or motor access).

The raw RGB/depth/thermal arrays, camera intrinsics, segmentation and
motor coordinates remain in their original acquisition orientations.
"""
from __future__ import annotations

import numpy as np
import cv2

DISPLAY_ROTATIONS = (-90, 0, 90, 180)
DEPTH_MODES = ("Fixed range", "Auto (5–95% valid depth)", "Original 0–2 m")


def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate a displayed image CCW by degrees; never modify the source."""
    if degrees not in DISPLAY_ROTATIONS:
        raise ValueError("Display rotation must be -90, 0, +90 or 180 degrees.")
    if image.ndim not in (2, 3):
        raise ValueError("Expected a 2D or 3D image array.")
    return np.ascontiguousarray(np.rot90(image, k=degrees // 90))


def rotate_pointcloud_xy(points, degrees: int):
    """Roll the 3D viewer's initial orientation around camera viewing axis.

    Works on a pandas DataFrame returned by pointcloud_data; does not modify
    the original dataframe or depth back-projection. Rotations follow the
    same CCW convention as rotate_image and preserve depth (z).
    """
    if degrees not in DISPLAY_ROTATIONS:
        raise ValueError("Unsupported 3D display rotation.")
    out = points.copy()
    x = points["x"].to_numpy(copy=True)
    y = points["y"].to_numpy(copy=True)
    if degrees == 90:
        out["x"], out["y"] = -y, x
    elif degrees == -90:
        out["x"], out["y"] = y, -x
    elif degrees == 180:
        out["x"], out["y"] = -x, -y
    return out


def depth_limits_m(depth: np.ndarray, depth_scale_m: float,
                   mode: str = "Fixed range",
                   fixed_range_m: tuple[float, float] = (0.35, 1.25)
                   ) -> tuple[float, float]:
    """Choose display-only near/far colour limits, without modifying depth."""
    if depth.ndim != 2 or depth_scale_m <= 0 or not np.isfinite(depth_scale_m):
        raise ValueError("Expected 2D RealSense depth with positive scale.")
    if mode not in DEPTH_MODES:
        raise ValueError(f"Unknown depth display mode: {mode}")
    if mode == "Original 0–2 m":
        return 0.0, 2.0
    if mode == "Fixed range":
        lo, hi = [float(x) for x in fixed_range_m]
        if not (0 <= lo < hi <= 5.0):
            raise ValueError("Invalid fixed depth colour range.")
        return lo, hi

    metres = depth.astype(np.float32) * float(depth_scale_m)
    valid = metres[(depth > 0) & np.isfinite(metres)
                   & (metres >= 0.12) & (metres <= 3.0)]
    if valid.size < 200:
        return float(fixed_range_m[0]), float(fixed_range_m[1])
    lo, hi = np.percentile(valid, (5, 95))
    if hi - lo < 0.10:
        centre = float(np.median(valid))
        lo, hi = max(0.12, centre - 0.05), centre + 0.05
    return float(lo), float(hi)


def depth_rgb_display(depth: np.ndarray, depth_scale_m: float,
                      near_m: float, far_m: float) -> np.ndarray:
    """RGB JET palette: near=red, far=blue, missing depth=black.

    Unlike the original hard-coded 0–2m colour window, this function supports
    a custom window. Physical distances and depth storage stay unchanged.
    """
    if depth.ndim != 2 or not (0 <= near_m < far_m <= 5.0):
        raise ValueError("Invalid near/far depth limits.")
    metres = depth.astype(np.float32) * float(depth_scale_m)
    fraction = np.clip((metres - near_m) / (far_m - near_m), 0, 1)
    encoded = (255 - np.rint(fraction * 255)).astype(np.uint8)
    bgr = cv2.applyColorMap(encoded, cv2.COLORMAP_JET)
    bad = (depth == 0) | ~np.isfinite(metres)
    bgr[bad] = 0
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
