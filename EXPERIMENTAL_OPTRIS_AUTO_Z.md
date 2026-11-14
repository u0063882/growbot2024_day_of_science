# Experimental GROWBOT v6: depth-guided Optris Z standoff

This feature is isolated in `feature/optris-auto-z-standoff` and a **draft pull request** against the experimental v5 branch. Neither `main` (tested v4) nor the earlier v5 application is modified.

## Why

The Optris camera should remain at a comparable distance from the plant canopy. The RealSense depth frame is captured first at the **top** of the Z axis, and a segmented foreground depth median is used to calculate a conservative Z lowering proposal. RealSense and Optris optical centres need not share the same Z position, so a signed relative height offset must be calibrated.

## Procedure

1. On the cloned repository, `git fetch origin && git switch feature/optris-auto-z-standoff`, then double-click `Start_GROWBOT_v6.bat`.
2. Home the gantry as normal. Check that Z really is at the **top** of its stroke.
3. Under *Optris Z standoff (experimental)*, select **Preview only**.
4. Choose a plant. The gantry moves XY, captures RealSense, moves by the existing Optris XY offset, captures thermal **without Z movement**, and displays the proposed lowering in *More details*.
5. Independently measure the difference between the Optris and RealSense optical-centre heights. Set the signed `Optris minus RealSense` offset (negative if Optris is closer to the plant).
6. Verify available vertical clearance, motor direction, cable slack and physical housing interference. Do a **supervised small Z movement** outside this experimental automatic workflow before enabling the automatic option.
7. Select **Automatic (experimental)**, acknowledge clearance and arm one-click acquisition.
8. On a plant click, the sequence becomes: **XY plant position → RealSense RGB/depth at top Z → Optris XY offset → optional Z descent → thermal image → verified Z retract to top**.
9. No subsequent XY travel is allowed unless the controller reports Z at the top position.

If the depth mask is missing or untrustworthy, the system does **not lower Z**. It still captures thermal at the original top height; the reason is included in the capture result.

### Numerical safety constraints

- Original controller reports `$132=110 mm`, *not* 150 mm. The existing confirmed Z home was `MPos Z=-1 mm`.
- Experiment restricts automatic Z to **0–100 mm down from home**, so Z targets are between **-101 and -1 mm machine coordinates**.
- Z speed fixed at **600 mm/min**; no GRBL configuration changes.
- Adjustable Optris desired leaf standoff (default **450 mm**), measured optical-centre relative height (default **0 mm**; must be calibrated), max drop (**default 10 mm for the first supervised test**, adjustable up to 100 mm), and nearest-leaf clearance (default **300 mm**).
- Foreground depth must have enough valid pixels, reasonable spread, and a plausible real plant mask. Invalid input produces a **0 mm movement**.
- The 10th percentile of segmented canopy distance limits descent to preserve a conservative estimated near-leaf clearance.
- Automatic mode is off by default and requires explicit operator acknowledgement. Changing standoff settings disarms one-click acquisition.
- Z is retracted in a `finally` block even when the Optris frame capture fails, provided the controller's status and position remain trustworthy. If retraction cannot be verified, the app blocks further XY commands and requires operator recovery. Do not bypass with `$X`.

**The planner is only a rough camera-to-canopy standoff estimate.** It is not obstacle avoidance, certified collision protection, mechanical clearance measurement or absolute plant height measurement. The actual position of Optris glass/housing, mounting brackets and neighbouring plants must still be checked.

## Files

- `growbot_gantry_realsense_optris_v6.py`: new experimental GUI and motor workflow.
- `growbot_optris_auto_z.py`: pure depth/mask standoff calculator, no motion.
- `Start_GROWBOT_v6.bat`: independent launcher, same existing `growbot2024` Conda environment.
- `tests/test_auto_z.py`: synthetic, hardware-free Z-planning tests.
- `.github/workflows/experimental-z-offline.yml`: source compilation and unit checks.

Run tests locally (safe, no camera or gantry commands):

```bat
conda activate growbot2024
python -m unittest discover -s tests -v
python -m py_compile growbot_gantry_realsense_optris_v6.py growbot_optris_auto_z.py
```

## Real versus artificial plants

The physical discriminant should be a **numerical apparent temperature contrast**, not the auto-scaled thermal palette. Compare a calibrated plant ROI mean/median with **simultaneous room/background reference temperature**: `ΔT = T_plant - T_reference`. Real, transpiring plants may have negative ΔT; plastic leaves may be closer to zero. Camera emissivity/reflections, illumination, airflow and water stress affect this, so do not label plastic solely from image colours. This is a **future experimental idea**, not implemented as a classifier in v6.
