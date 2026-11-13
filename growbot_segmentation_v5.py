"""Experimental GROWBOT v5 plant segmentation (OpenCV/NumPy only).

Improves over the 2024 green-only threshold by using green seeds, local
colour modelling (GrabCut), optional aligned depth and nearby pale petals.
Not a calibrated biomass model. The original 2024 model remains unchanged.
"""
from __future__ import annotations

import cv2
import numpy as np


def enhanced_plant_mask(
    bgr: np.ndarray,
    depth: np.ndarray | None = None,
    depth_scale_m: float = 0.001,
    pale_min_value: int = 150,
) -> tuple[np.ndarray, dict]:
    """Return uint8 mask (0/255) and diagnostics; no image/file side effects."""
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("Expected uint8 BGR image (height, width, 3)")
    h, w = bgr.shape[:2]
    if h < 20 or w < 20:
        raise ValueError("Image too small")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    B, G, R = (bgr[:, :, i].astype(np.int16) for i in range(3))
    exg = 2 * G - R - B
    green = (
        (hsv[:, :, 0] >= 26) & (hsv[:, :, 0] <= 100)
        & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 30)
        & (exg > 12)
    ).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k3)
    n_seed = int(green.sum())
    if n_seed < 40:
        return np.zeros((h, w), np.uint8), {
            "status": "Insufficient green seed; use original segmentation",
            "green_seed_pixels": n_seed,
            "enhanced_pixels": 0,
        }

    # Distance to nearest trusted green pixel. The vicinity assumption reduces
    # false positives from the surrounding black/brown box/background.
    distance = cv2.distanceTransform((1 - green).astype(np.uint8), cv2.DIST_L2, 3)
    near = distance <= 95.0
    probable = distance <= 55.0
    grabcut_mask = np.full((h, w), cv2.GC_BGD, np.uint8)
    grabcut_mask[near] = cv2.GC_PR_BGD
    grabcut_mask[probable] = cv2.GC_PR_FGD

    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    hue = hsv[:, :, 0]
    dark_bg = v < 48
    brown_bg = (hue >= 4) & (hue <= 23) & (s >= 65) & (distance > 25)
    grabcut_mask[(dark_bg | brown_bg) & (distance > 25)] = cv2.GC_BGD
    grabcut_mask[:5, :] = cv2.GC_BGD
    grabcut_mask[-5:, :] = cv2.GC_BGD
    grabcut_mask[:, :5] = cv2.GC_BGD
    grabcut_mask[:, -5:] = cv2.GC_BGD

    # Optional depth cue: plant pixels closer than a robust border background
    # are plausible, not guaranteed, foreground. Keep this a weak prior only.
    depth_cue = np.zeros((h, w), dtype=bool)
    depth_used = False
    if depth is not None and depth.shape == (h, w) and depth_scale_m > 0:
        d = depth.astype(np.float32) * float(depth_scale_m)
        border = np.zeros((h, w), bool)
        border[: max(6, h // 20)] = True
        border[-max(6, h // 20):] = True
        border[:, :max(6, w // 20)] = True
        border[:, -max(6, w // 20):] = True
        bg = d[border & np.isfinite(d) & (d > 0.12) & (d < 5.0)]
        if bg.size >= 200:
            bg_depth = float(np.median(bg))
            depth_cue = (d > 0.12) & (d < bg_depth - 0.035)
            depth_used = True
            weak = probable & depth_cue & ~dark_bg
            grabcut_mask[weak] = cv2.GC_PR_FGD

    trusted = cv2.erode(green, k3).astype(bool)
    trusted[:5, :] = False
    trusted[-5:, :] = False
    trusted[:, :5] = False
    trusted[:, -5:] = False
    grabcut_mask[trusted] = cv2.GC_FGD
    if np.count_nonzero(grabcut_mask == cv2.GC_FGD) == 0:
        return (green * 255).astype(np.uint8), {
            "status": "Only green seed retained (at image edge)",
            "green_seed_pixels": n_seed,
            "enhanced_pixels": n_seed,
        }

    models = (np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64))
    try:
        cv2.grabCut(bgr, grabcut_mask, None, *models, 3, cv2.GC_INIT_WITH_MASK)
        foreground = ((grabcut_mask == cv2.GC_FGD) |
                      (grabcut_mask == cv2.GC_PR_FGD))
        status = "GrabCut + pale-flower + optional depth"
    except cv2.error:
        foreground = green.astype(bool)
        status = "Green/pale fallback (GrabCut unavailable)"

    # Pale flowers may be missing from the learned green colour distribution.
    # Add bright, low-saturation adjacent parts; make the reach larger only
    # where the aligned depth supports a physically protruding object.
    pale = (v >= pale_min_value) & (s <= 150) & ~dark_bg
    petals = pale & ((distance <= 24) | ((distance <= 65) & depth_cue))
    foreground |= petals
    foreground |= green.astype(bool)
    foreground &= near

    mask = foreground.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    retained = np.zeros((h, w), np.uint8)
    for label in range(1, n):
        component = labels == label
        if stats[label, cv2.CC_STAT_AREA] >= 20 and np.count_nonzero(component & green.astype(bool)):
            retained[component] = 255

    return retained, {
        "status": status,
        "green_seed_pixels": n_seed,
        "enhanced_pixels": int(np.count_nonzero(retained)),
        "depth_prior_used": depth_used,
    }


def segmented_rgb(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return RGB preview with white background; original BGR is unmodified."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = np.full_like(rgb, 255)
    out[mask > 0] = rgb[mask > 0]
    return out
