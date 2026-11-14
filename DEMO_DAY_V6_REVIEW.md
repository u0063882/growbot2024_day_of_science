# GROWBOT demo-day readiness (v6 display review)

This is a review/quality branch, **not an on-hardware validated release**. It
does not change `main`, the earlier v5 branch, or the original v6 branch.

## Review summary

- **Existing camera SDK and GRBL motion code are preserved.** New image
  orientation, colour contrast and optional 3D orientation are display only.
  Camera acquisition, plant addressing, depth units, thermal temperature
  calculations, file saving, motion coordinates and biomass proxy are unchanged.
- **Display orientation** defaults to **+90° (counterclockwise)**. Under
  *Image display*, choose -90°, 0°, +90° or 180°. This applies to RGB, false-colour
  depth, thermal, displayed segmentation, thermal comparisons, and the optional
  interactive 3D point-cloud orientation. Raw saved images are not rotated.
- **Depth colour range** defaults to **Fixed range: 0.35–1.25 m**. This
  compresses the camera-to-object range relative to the old 0–2 m palette,
  increasing contrast among near plants. If your scene falls outside it,
  switch to **Auto (5–95% valid depth)** or **Original 0–2 m**. The palette
  convention remains near=red, far=blue, missing=black. The numeric 16-bit
  RealSense depth and Z planner still use *unmodified* depth.
- **Thermal autoscale** still only changes the false-colour image, not the
  32-bit temperature array. Previous fixed colour range settings remain
  available and are independent of the depth colour range.
- **Auto-Z** is disabled by default. Its Z target calculation and GRBL
  direction were not verified on the physical machine. The 100 mm bound and
  automatic return-to-top are safety software checks, **not collision avoidance**.
  The first-run max drop remains limited to 10 mm unless explicitly changed.
- **Z retraction on thermal image failure** is attempted and explicitly
  checked. If the controller state is uncertain or Z cannot be returned,
  the code refuses subsequent XY movement. A retraction fault still needs
  local operator intervention; automatic recovery cannot be guaranteed.
- **Output metadata** distinguishes proposed lowering from executed lowering
  and does not display an affirmative retract message when Z never descended.

## Day-of-use sequence (few-minute test)

1. Confirm the original working v4 folder or Git branch is available as a
   fallback. Do **not** uninstall or upgrade camera drivers or Python packages.
2. In Git Bash in the local clone:
   ```bash
   git status
   git fetch origin
   git switch feature/v6-demo-hardening
   git pull --ff-only
   ```
   Do not discard uncommitted local changes. The Optris SDK, XML, calibration
   and plant icons must remain beside the Python scripts.
3. Close PIX Connect, OpenBuilds CONTROL and any existing GROWBOT process,
   then double-click **Start_GROWBOT_v6.bat**.
4. Check the sidebar: *Optris Z standoff* = **Off**, *Save captures to disk* =
   **off** unless desired, *Display rotation* = **+90°**,
   *Depth contrast* = **Fixed 0.35–1.25 m**.
5. Connect BLACKBOX, perform the same verified homing as v4, connect RealSense
   and Optris.
6. **Capture at current position (no movement)**. Inspect the three images
   and colour ranges. If the depth map is saturated, choose *Auto 5–95%*.
7. Only if there is adequate time and clear hardware travel, arm one-click
   acquisition and capture **one plant with auto-Z Off**.
8. If anything is unexpected: stop Streamlit, and use **Start_GROWBOT.bat**
   to launch original v4 (present in the cloned branch). Do not experiment
   with automatic Z on the public demo floor.

## Separate supervised Z commissioning (not part of fast demo check)

1. Verify GRBL \$132, homed Z coordinate, and which *physical direction*
   corresponds to a negative machine-Z command.
2. Measure the physical RealSense–Optris optical-centre height offset in mm.
3. Inspect cable slack, both sensor housings, brackets, overhead obstacles,
   and collision margins for every XY plant position.
4. In Preview mode, compare depth-derived Z proposals with a ruler measurement.
5. Test a small Z displacement with physical emergency stop accessible and
   a clear work area; verify the return-to-top position on every cycle.
6. Only then enable Automatic. If unverified, leave this feature **Off**.

## Offline testing

The GitHub workflow checks Python 3.10 syntax and includes hardware-free
unit tests for motor bounds, XY interlock at lowered Z, automatic retraction
after successful or failing thermal acquisition, and 2D/3D rotation and
depth colour palette. This does not simulate USB timing or validate the real
GRBL controller.

## Plastic versus real plants (no automatic classifier implemented)

A single radiometric thermal image has numeric pixel temperatures independent
of its false-colour scale. When a warm background is visibly present, the
lower tail or a contiguous cooler region can indicate transpiring leaves.
However, when plant material fills the whole Optris field of view, no
background temperature can be estimated reliably from that single image.
Height differences change the amount of background; apparent temperature
also depends on emissivity, reflections, illumination, air movement and water
status. A shutter/lens-cap image does not provide a room-temperature reference.

For a future, honest **ranking** across the 35 plants, use thermal-only image
statistics (e.g. lower 10–25% temperature quantiles) compared with neighbouring
captures in time, and label them tentative. This avoids RGB/thermal registration
and hand-drawing masks. It can fail when the thermal FOV has cold background,
a real plant is not transpiring, or the time/ambient temperature changes.
Do not silently label plants as definitely real/plastic based only on these
statistics. A reference target visible in every thermal frame or an ambient
sensor would improve this markedly, but requires hardware configuration.

Given the very limited commissioning window, **do not introduce an unvalidated
plastic classifier on the demo branch**.
