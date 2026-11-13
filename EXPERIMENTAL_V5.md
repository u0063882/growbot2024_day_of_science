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
- Optris display colour limits are **fixed at 10–45 °C by default** for a wider range. Under *Thermal display*, optionally turn on **Auto-scale (2nd–98th percentile)**, or adjust the fixed range.
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
