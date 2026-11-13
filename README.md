# GROWBOT 2024 – Day of Science

**Working v4 baseline** for the original Nerdland GROWBOT with OpenBuilds BLACKBOX XY gantry, RealSense D435i RGB/depth and Optris thermal camera. The tested Python v4 application and Windows launcher are included unchanged. Future feature development (improved plant segmentation, 3D point-cloud view) should be done on separate branches.

## Running on the existing Windows PC

1. Keep the original GROWBOT hardware drivers, firmware and SDK installation. The bundled Python package versions are in `requirements.txt` (Python 3.10, Conda environment `growbot2024`).
2. Copy locally held Optris SDK and calibration files into the working folder (see below). They are **not in this public repository**.
3. Close OpenBuilds CONTROL and PIX Connect so they release COM4 and the camera.
4. Double-click `Start_GROWBOT.bat`. Alternatively, use:

   ```bat
   conda activate growbot2024
   streamlit run growbot_gantry_realsense_optris_v4.py
   ```

5. In the app: connect BLACKBOX, home, start RealSense, connect Optris, then capture from the plant grid. On exit, use **Shutdown / Disconnect All**, then Ctrl+C in the launcher.

To create a Python environment on a fresh Windows PC, install Miniconda and run:

```bat
conda create -n growbot2024 python=3.10 -y
conda activate growbot2024
python -m pip install -r requirements.txt
```

The original camera SDK/drivers and Visual C++ 2013 Redistributable (x64) must be installed separately.

## Required local Optris files (not committed)

Copy from the original GROWBOT project or a properly licensed Optris SDK installation:

- `generic.xml`
- `Formats.def`
- `irDirectSDK/sdk/x64/libirimager.dll` plus any companion SDK files
- `Califiles SN18082084/` (calibration for the original camera)

Camera-specific files, DLLs and installers are excluded from version control through `.gitignore`. The Python source files are committed, but the original **35 plant icons and UI logos still need to be uploaded** from the working local project. For now, copy `plant icons/`, `mebios.png`, `growbot_logo.png`, `logo KUL.png`, and `logo_600jr.png` into the project folder before launching the UI. They are ordinary UI assets and can later be committed to the public repository.

## Files

- `growbot_gantry_realsense_optris_v4.py`: original working v4 Streamlit app (RGB, depth, thermal, biomass proxy and gantry control).
- `Start_GROWBOT.bat`: Windows double-click launcher for `growbot2024`.
- `utils.py`: original Nerdland HSV green-mask segmentation helper (unchanged).
- `growbot_optris_backend.py`: optional local Optris SDK adapter.
- `plant icons/`: 35 plant selection images (currently local only; upload from working PC).
- `requirements.txt`: pinned 2024 Python dependencies.

## Safety / compatibility

The machine configuration is for the **original Nerdland gantry** (GRBL 1.1g custom, COM4, work envelope and homing coordinates coded into v4). Before moving, ensure the gantry is clear, verify physical emergency-stop access and homing operation. Never reuse these coordinates on another machine without validation. Keep the v4 baseline stable when testing experimental algorithms.

No real/plastic classification is included in v4. RGB and depth are aligned to each other, but thermal images are not spatially registered to the RealSense.
