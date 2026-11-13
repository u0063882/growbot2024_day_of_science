"""Conservative, hardware-free Optris Z standoff planner for GROWBOT.

Inputs are RealSense aligned depth + an explicitly segmented plant mask.
Returns a proposal; NEVER operates any motors. GRBL bounds and commands
are checked again by the controller layer before Z is moved.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ZStandoffPlan:
    usable: bool
    lowering_mm: float
    median_plant_distance_mm: float | None
    near_plant_distance_mm: float | None
    estimated_optris_distance_mm: float | None
    usable_pixels: int
    status: str


def plan_z_standoff(
    depth: np.ndarray,
    depth_scale_m: float,
    plant_mask: np.ndarray,
    *,
    target_optris_distance_mm: float = 450.0,
    optris_minus_realsense_mm: float = 0.0,
    max_lowering_mm: float = 100.0,
    min_nearest_clearance_mm: float = 300.0,
) -> ZStandoffPlan:
    """Estimate safe *downward travel* from the top-homed Z (millimetres).

    target_optris_distance_mm: desired distance from Optris to mean plant.
    optris_minus_realsense_mm: signed vertical offset between optical centres.
      Negative when Optris is LOWER/closer to the plant than RealSense.
    min_nearest_clearance_mm: conservatively cap Z drop using the 10th
      percentile of foreground depth (near leaves). This is NOT a substitute
      for physically checking camera housing and all obstacles.

    An invalid/missing plant mask never produces a positive Z movement.
    """
    def refused(reason: str, count: int = 0) -> ZStandoffPlan:
        return ZStandoffPlan(False, 0.0, None, None, None, count, reason)

    if depth.ndim != 2 or plant_mask.shape != depth.shape:
        return refused("Plant mask and aligned depth must have the same 2D shape.")
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        return refused("Invalid RealSense depth scale.")
    settings = (
        target_optris_distance_mm, optris_minus_realsense_mm,
        max_lowering_mm, min_nearest_clearance_mm,
    )
    if not all(math.isfinite(float(v)) for v in settings):
        return refused("Non-finite Z standoff setting.")
    if not 250 <= target_optris_distance_mm <= 1200:
        return refused("Target distance outside 250–1200 mm.")
    if not -200 <= optris_minus_realsense_mm <= 200:
        return refused("Unverified camera height offset outside ±200 mm.")
    if not 0 <= max_lowering_mm <= 100:
        return refused("Z lowering exceeds the 100 mm experimental safety cap.")
    if not 250 <= min_nearest_clearance_mm <= 900:
        return refused("Nearest-leaf clearance must be at least 250 mm.")

    foreground = plant_mask.astype(bool)
    mask_count = int(np.count_nonzero(foreground))
    if mask_count < 150:
        return refused("Insufficient segmented plant pixels (need at least 150).", mask_count)
    if mask_count > depth.size * 0.65:
        return refused("Segmentation covers too much of the scene; not a reliable plant.", mask_count)

    d_mm = depth.astype(np.float32) * float(depth_scale_m) * 1000
    valid = foreground & np.isfinite(d_mm) & (d_mm >= 200) & (d_mm <= 2500)
    n = int(np.count_nonzero(valid))
    if n < 150 or n < mask_count * 0.55:
        return refused("Too few valid foreground depth pixels for Z adjustment.", n)

    samples = d_mm[valid]
    d10, d50, d90 = (float(v) for v in np.percentile(samples, (10, 50, 90)))
    if d90 - d10 > 750:
        return refused("Plant depth distribution too broad for safe automatic Z.", n)

    # Current Optris range is estimated from the RealSense depth at Z-top.
    current = d50 + optris_minus_realsense_mm
    near_current = d10 + optris_minus_realsense_mm

    desired_lower = max(0.0, current - target_optris_distance_mm)
    clearance_cap = max(0.0, near_current - min_nearest_clearance_mm)
    applied = min(desired_lower, max_lowering_mm, clearance_cap)
    # Never round UP a movement or violate clearance by rounding.
    applied = math.floor(max(0.0, applied) * 2.0) / 2.0
    if applied < 0.5:
        applied = 0.0

    reason = ("No Z lowering required; remain at top." if applied == 0 else
              "Z proposal calculated. Inspect physical clearances before enabling motion.")
    return ZStandoffPlan(
        usable=True,
        lowering_mm=applied,
        median_plant_distance_mm=d50,
        near_plant_distance_mm=d10,
        estimated_optris_distance_mm=current - applied,
        usable_pixels=n,
        status=reason,
    )
