# GROWBOT v5 experiment: segmentation and interactive RGB point cloud

**Branch:** `feature/segmentation-pointcloud-v5` (not merged into `main`).

The **verified v4 motion, capture and shutdown code** remains accessible via `Start_GROWBOT.bat` and `growbot_gantry_realsense_optris_v4.py`. The feature app is a **separate** `growbot_gantry_realsense_optris_v5.py` with a `Start_GROWBOT_v5.bat` launcher. It uses the same original BLACKBOX, RealSense D435i and Optris SDK, not new drivers.

## Run

On the existing Windows GROWBOT computer, in Git Bash:

```bash
git status
git fetch origin
git switch feature/segmentation-pointcloud-v5
git pull --ff-only
```

Keep local original SDK binaries, camera-specific calibration, XML files and icons in the project folder. These are deliberately not distributed in GitHub. Close OpenBuilds CONTROL and PIX Connect so the devices are free.

Double-click **`Start_GROWBOT_v5.bat`** to launch the experiment. To return to the stable version without switching branches, double-click **`Start_GROWBOT.bat`**.

## What changed?

### Shared with `main` (v4 and v5)

- **Save captures to disk** is **off** on a fresh Streamlit session. You may turn it on under *Capture settings*. Results still appear onscreen when saving is off.
- Optris display colour limits are **fixed at 18–23 °C by default on experimental v5** (10–45 °C on stable v4). Under *Thermal display*, optionally turn on **Auto-scale (2nd–98th percentile)**, or adjust the fixed range.
- These controls affect **display only**, not the raw thermal temperatures, GRBL settings, acquisition hardware or segmentation model.

### Experimental v5 only

- **Enhanced segmentation** uses green seeds, colour-modelled GrabCut, nearby bright/pale floral regions and a *weak optional* aligned-depth cue. It aims to retain more pale flowers/foliage while suppressing black/brown background.
- It displays both the **original 2024 mask** and **experimental v5 mask**. **Biomass is always calculated with the unchanged 2024 mask**; the new mask has not been quantitatively validated.
- An optional **interactive 3D RGB-coloured point cloud**, using the camera's RGB intrinsics and aligned depth. Point cloud choices: full scene / original green mask / enhanced mask. It is downsampled to up to 20,000 points for responsiveness; point cloud computation is **off by default**, and starts only when *Render point cloud* is checked.
- Uses **pydeck** and **pandas**, which are dependencies of Streamlit 1.33 (no extra packages intended).

### Limitations

- The colour/depth segmentation uses heuristics, not a trained species-independent model. Pale petals far from any green seed can still be excluded; wood/brown plant material may be rejected. Real acquisitions should be visually inspected before relying on area or biomass estimates.
- Depth gaps, reflections and missing-depth pixels cannot be reconstructed by the point-cloud viewer.
- The Optris and RealSense cameras are **not spatially calibrated/registered** to one another.
- Neither segmentation nor the point-cloud preview changes motor control; all original travel limits, homing, one-click plant capture and safe disconnect stay in v4.

## Offline regression tests

From the cloned project folder:

```bat
conda activate growbot2024
python -m unittest discover -s tests -v
```

The tests use synthetic images and fake aligned depth, so they do **not** connect a camera, open COM4 or move the gantry.

Tested offline: a pale flower touching a green plant on brown/black background, an empty scene fallback, RGB/depth back-projection shape and invalid intrinsic handling. Hardware-side accuracy and browser rendering still need validation.


## Update: row-wise plant numbering and dual-position imaging

The **displayed 7 × 5 grid now numbers plants row by row**:

```text
  1   2   3   4   5   6   7
  8   9  10  11  12  13  14
 15  16  17  18  19  20  21
 22  23  24  25  26  27  28
 29  30  31  32  33  34  35
```

The numeric labels are rendered into copies of the existing plant icon in memory; the original PNGs are untouched. **The physical 35 XY destinations have not been redefined.** The row-major labels are mirrored left-to-right relative to the original physical column ordering: **plant 1 is the top-left physical station**, and plant 7 is top-right. When disk saving is enabled, metadata includes both numbering schemes.

### Adjustable Optris XY offset

Under **Capture settings → Optris XY camera offset**, enter X and Y displacements in **millimetres** (**defaults X = 0, Y = +5 mm**; allowed inputs −400 to +400). These are signed **machine-coordinate** offsets, not screen pixels. Use a small offset first and check the path. Editing either value automatically **disarms one-click movement**, requiring you to acknowledge the new path again.

For a plant click, the sequence is:

1. Validate both the plant position **and** proposed Optris position against the configured XY axis limits; never start if either endpoint is invalid
2. Move to the original physical plant location and settle
3. Capture RealSense RGB and aligned depth at that location
4. If Optris is connected and X/Y offset is nonzero, move to the **plant position + offset**, then settle
5. Capture Optris thermal frames; display all results together

**0,0 means both cameras capture at the same gantry position**, without a second motion. If Optris is disconnected and not required, no thermal offset movement is performed. **Manual "Capture at current position" never moves**, regardless of the offset settings. The gantry **remains at the Optris position after capture**, it does not automatically move back to the RealSense position. Neither homing nor Z movement was altered.

The returned measurement records the two distinct machine positions and the applied XY offset. This allows checking image provenance when the thermal and RGB views are taken from different locations. It does **not** optically align the thermal and RGB images.

### Variegated ornamental plants

An **Experimental plant mask** selector now offers **Variegated / pale-edged** and **Green foliage + pale flowers**. The variegated approach uses the **full RealSense image** rather than the old 2024 crop, with central foreground modelling for teal/dark leaf interiors and yellow/cream margins. The original green-only mask and biomass calculation are unchanged, and the former "Enhanced / original mask area" metric has been removed.

The variegated approach was visually inspected on the provided snake-plant example. It recovers the main dark/cream leaves, but still includes some neighbouring plant pixels, so it should be treated as **experimental**, not a species-independent calibrated segmentation.

### Safety and testing

- The stable `main` branch and `Start_GROWBOT.bat` retain the working v4 hardware behavior (the previously requested saving and thermal-display defaults were applied separately to both branches)
- `Start_GROWBOT_v5.bat` runs only the experimental v5 application
- `growbot_mapping_v5.py` isolates 2024 destination mapping, the new row-major numbering and Optris target validation for hardware-free testing
- `tests/test_v5_offline.py` includes grid position invariance, offset limits, variegated leaves and capture-order checks
- GitHub Actions runs syntax and offline tests on feature-branch pushes; this does not verify hardware movement or the optical offset calibration

**First hardware test:** Verify the top-left plant target with the gantry path clear and inspect the configured X = 0, Y = +5 mm thermal move. If needed, temporarily set the offset to 0,0 to confirm a same-position capture before enabling an offset.


## Dashboard and point-cloud adjustments (November 2026)

- Only the **Experimental v5: Variegated / pale-edged** segmentation preview is shown alongside the RGB, depth and thermal frames. The original HSV mask is still used internally for the historical biomass proxy (and remains available for the optional masked point-cloud view); it is not separately shown on the main dashboard.
- Three **main metrics**: historical biomass proxy (g), **scene height estimate** (cm) relative to surrounding surface, and median scene thermal temperature (°C). Scene height uses robust percentiles of background and visible foreground depth; **it is not a calibrated plant-height measurement** and can be biased by neighbouring pots, occlusion and invalid depth.
- Thermal fixed colour range now defaults to **18–23 °C**; auto-scale remains optional. The default Optris translation is **X = 0 mm, Y = +5 mm**, only during plant-click acquisition. Manual capture at current location never moves.
- XY speed slider now permits requested G-code feeds **200–3600 mm/min**, initially **2500 mm/min**. The confirmed GRBL `$110=$111=2500 mm/min` firmware maxima remain unchanged. Commands over 2500 mm/min **do not automatically increase axis motor speeds past those limits**. No GRBL firmware parameters were altered.
- 3D RGB point cloud now uses an **auto-fitted Cartesian OrbitView**. Previously, zoom=6 showed a 40-cm plant only ~26 pixels wide; the auto-fit uses the cloud's metric extent to choose an appropriate initial zoom. Adjustable 3D point size and maximum depth are available. Camera orientation starts approximately frontal. Rendering remains experimental and may require a GPU/WebGL-capable browser..
