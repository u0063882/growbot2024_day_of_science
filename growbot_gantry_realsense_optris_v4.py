"""GROWBOT 2024 | BLACKBOX XY + RealSense D435i + Optris thermal.

Version 4: retains the verified v2 motion controls, original 35-plant
layout and 2024 RGB green-pixel biomass estimate, plus the v3 Optris
acquisition. The dashboard presents RGB, depth, thermal and green-mask
results together, with an optional time-limited live camera preview.
No plastic/artificial-plant classification. No JAI.

Place next to the original GROWBOT assets and utils.py, then run:
    conda activate growbot2024
    streamlit run growbot_gantry_realsense_optris_v4.py

Hardware/configuration confirmed 2026-11-09:
  GRBL 1.1g CUSTOM, COM4 at 115200, homing $H
  machine home MPos (-1499, -1, -1) mm
  X/Y configured machine limits: [-1499, -1] mm
  RealSense D435i, 640x480 at 30 fps (original SDK 2.54.2)

SAFETY:
  - Does not move or start cameras on startup / ordinary Streamlit reruns.
  - Optris uses the original 2024 SDK/XML and calibration files unmodified.
  - Close PIX Connect before connecting Optris here.
  - A plant icon click triggers exactly one move+capture ONLY when armed.
  - Requires home within the same serial connection before enabling motion.
  - Does not move Z; keep camera/gantry clear at the homed Z height.
  - 35 original plant positions were recovered from the 2024 code.
    An operator acknowledgement is required once per controller connection.
  - GRBL G53 machine coordinates are used to avoid dependence on G54 offsets.
  - Physical emergency stop and mechanical limit switches remain essential.
"""

from __future__ import annotations

import base64
import ctypes as ct
import os
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import serial
import streamlit as st
from st_clickable_images import clickable_images

from utils import create_mask  # Existing 2024 function; no modifications required.


# -----------------------------------------------------------------------------
# USER CONFIGURATION -- retained from the confirmed 2024 project / 2026 tests.
# -----------------------------------------------------------------------------
PORT = "COM4"
BAUDRATE = 115200
HOME_MPOS = (-1499.0, -1.0, -1.0)  # Confirmed after a successful $H.
HOME_TOLERANCE_MM = 2.0
AXIS_LIMITS = {"X": (-1499.0, -1.0), "Y": (-1499.0, -1.0)}
POSITION_TOLERANCE_MM = 0.5
DEFAULT_FEED_MM_MIN = 1800  # User-adjustable up to 2400; GRBL $110/111 = 2500.
MIN_FEED_MM_MIN = 200
MAX_FEED_MM_MIN = 2400
MAX_MANUAL_JOG_MM = 5.0
CAPTURE_DIR = Path(__file__).resolve().parent / "captures"
EXPECTED_OPTRIS_SERIAL = "18082084"
PROJECT_DIR = Path(__file__).resolve().parent

# From growbot.py: x=940-(clicked//5)*120; y=-240-(clicked%5)*220.
# Index 0..34 corresponds to plant number 1..35.
PLANTS = 35
GRID_COLUMNS = 7
GRID_ROWS = 5
# Display row-major, while the original plant numbering is column-major.
DISPLAY_IDS = [1 + row + column * GRID_ROWS
               for row in range(GRID_ROWS) for column in range(GRID_COLUMNS)]
FIRST_X_MM = 940.0
FIRST_Y_MM = -240.0
COLUMN_PITCH_MM = -120.0
ROW_PITCH_MM = -220.0

# From growbot.py (an optional legacy image-analysis region, not full frame).
LEGACY_RGB_ROI = (40, 370, 170, 480)  # top, bottom, left, right
LEGACY_BIOMASS_FACTOR_G_PER_PIXEL = 0.00015


def plant_offset_mm(plant_number: int) -> tuple[float, float]:
    """Original GRBL G54 XY targets relative to the homed platform origin."""
    if not 1 <= plant_number <= PLANTS:
        raise ValueError("Plant number must be between 1 and 35")
    column, row = divmod(plant_number - 1, GRID_ROWS)
    return (FIRST_X_MM + column * COLUMN_PITCH_MM,
            FIRST_Y_MM + row * ROW_PITCH_MM)


def plant_machine_xy(plant_number: int) -> tuple[float, float]:
    """Map original 2024 work-position XY to GRBL machine-coordinate XY.

    At the confirmed home, the old G54 work origin was (0,0), so the 2024
    coordinates are the same as displacements from HOME_MPOS. Using G53
    bypasses any changed G54/G92 work offset on the new PC.
    """
    x, y = plant_offset_mm(plant_number)
    return HOME_MPOS[0] + x, HOME_MPOS[1] + y


def check_xy_limits(x: float, y: float) -> None:
    for axis, target in (("X", x), ("Y", y)):
        low, high = AXIS_LIMITS[axis]
        if not low <= target <= high:
            raise ValueError(
                f"{axis} machine-coordinate target {target:.1f} mm is outside "
                f"[{low:.1f}, {high:.1f}] mm"
            )


class BlackBox:
    """Persistent serial session, explicit homing and verified GRBL motion."""

    def __init__(self, port: str = PORT):
        self.serial = serial.Serial(port, BAUDRATE, timeout=0.4, write_timeout=2)
        # GRBL boards often reset when USB serial opens (including this one).
        time.sleep(2.2)
        self.serial.reset_input_buffer()
        self.homed_in_this_connection = False

    def close(self) -> None:
        if self.serial.is_open:
            self.serial.close()
        self.homed_in_this_connection = False

    def _line(self) -> str:
        return self.serial.readline().decode("ascii", errors="replace").strip()

    def _command(self, command: str, timeout: float = 5.0) -> None:
        self.serial.write((command + "\n").encode("ascii"))
        self.serial.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._line()
            if line == "ok":
                return  # Acknowledged by GRBL; motion may still be running.
            if line.startswith(("error:", "ALARM:")):
                self.homed_in_this_connection = False
                raise RuntimeError(f"GRBL rejected {command}: {line}")
        raise TimeoutError(f"No GRBL acknowledgment for {command}")

    def status(self, timeout: float = 4.0) -> tuple[str, tuple[float, float, float], str]:
        self.serial.write(b"?")
        self.serial.flush()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._line()
            if line.startswith("<") and line.endswith(">"):
                m = re.search(
                    r"(?:^|\|)MPos:([+-]?[\d.]+),([+-]?[\d.]+),([+-]?[\d.]+)",
                    line,
                )
                if not m:
                    raise RuntimeError("Missing MPos in GRBL status: " + line)
                state = line[1:].split("|", 1)[0]
                if state == "Alarm":
                    self.homed_in_this_connection = False
                return state, tuple(float(v) for v in m.groups()), line
            if line.startswith(("error:", "ALARM:")):
                self.homed_in_this_connection = False
                raise RuntimeError("GRBL: " + line)
        raise TimeoutError("No realtime GRBL status received")

    def home(self) -> str:
        self.homed_in_this_connection = False
        self._command("$H", timeout=240)
        state, pos, raw = self.status(timeout=5)
        if state != "Idle":
            raise RuntimeError("Homing did not end in Idle: " + raw)
        if any(abs(a - b) > HOME_TOLERANCE_MM for a, b in zip(pos, HOME_MPOS)):
            raise RuntimeError(f"Unexpected home MPos {pos}; expected {HOME_MPOS}")
        self.homed_in_this_connection = True
        return raw

    def _must_be_homed_and_idle(self) -> tuple[float, float, float]:
        if not self.homed_in_this_connection:
            raise RuntimeError("Home using this app before commanding motion")
        state, pos, raw = self.status()
        if state != "Idle":
            raise RuntimeError("GRBL must be Idle before moving: " + raw)
        return pos

    def _wait_for_position(
        self, target: tuple[float, float, float], timeout: float
    ) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state, pos, raw = self.status()
            if state in ("Alarm", "Hold", "Door", "Sleep"):
                self.homed_in_this_connection = False
                raise RuntimeError("GRBL interrupted the move: " + raw)
            if state == "Idle" and all(
                abs(a - b) <= POSITION_TOLERANCE_MM for a, b in zip(pos, target)
            ):
                return raw
            time.sleep(0.2)
        # No assumption of a completed move; do not automatically try again.
        self.homed_in_this_connection = False
        raise TimeoutError(
            "Move completion could not be verified. Stop/check the machine "
            "and reconnect/home before further motion."
        )

    def move_machine_xy(self, x_mm: float, y_mm: float, feed: int) -> str:
        """Absolute XY travel at current Z, verified by position/status.

        G53 is machine positioning; G90/G21 are explicitly set, and the
        command includes no Z word. GRBL still enforces physical limits.
        """
        before = self._must_be_homed_and_idle()
        check_xy_limits(x_mm, y_mm)
        if not MIN_FEED_MM_MIN <= feed <= MAX_FEED_MM_MIN:
            raise ValueError(f"Feed must be between {MIN_FEED_MM_MIN} and {MAX_FEED_MM_MIN} mm/min")
        self._command("G90 G21")
        # Do not move Z. Motion at the current Z must be physically clear.
        self._command(f"G53 G1 X{x_mm:.3f} Y{y_mm:.3f} F{feed}", timeout=6)
        distance = math.hypot(x_mm - before[0], y_mm - before[1])
        timeout = max(30.0, distance / feed * 60.0 * 3.0 + 30.0)
        return self._wait_for_position((x_mm, y_mm, before[2]), timeout)

    def jog(self, axis: str, delta_mm: float, feed: int) -> str:
        before = self._must_be_homed_and_idle()
        if not MIN_FEED_MM_MIN <= feed <= MAX_FEED_MM_MIN:
            raise ValueError("Jog speed is outside configured limits")
        if axis not in AXIS_LIMITS:
            raise ValueError("Only X and Y jogging is allowed")
        if not 0 < abs(delta_mm) <= MAX_MANUAL_JOG_MM:
            raise ValueError("Only jog distances of 1 or 5 mm are enabled")
        idx = "XYZ".index(axis)
        target = before[idx] + delta_mm
        low, high = AXIS_LIMITS[axis]
        if not low <= target <= high:
            raise ValueError(f"Jog would exceed {axis} machine limits")
        self._command(f"$J=G91 G21 {axis}{delta_mm:+.3f} F{feed}")
        expected = list(before)
        expected[idx] = target
        return self._wait_for_position(tuple(expected), timeout=20)



class OptrisUSB:
    """Original Optris irDirectSDK USB integration (persistent, direct ctypes).

    Uses the exact bundled libirimager.dll, generic.xml, Formats.def and
    Califiles SN18082084 directory without rewriting them. No USB access
    occurs before the operator clicks Connect Optris.
    """

    def __init__(self, project_dir: Path):
        self.root = Path(project_dir).resolve()
        self.lib = None
        self.connected = False
        self.width = 0
        self.height = 0
        self.serial_number = None
        self._dll_dir = None

    def connect(self) -> None:
        if self.connected:
            return
        if os.name != "nt" or ct.sizeof(ct.c_void_p) != 8:
            raise RuntimeError("The original Optris SDK requires 64-bit Windows Python")
        dll = self.root / "irDirectSDK" / "sdk" / "x64" / "libirimager.dll"
        for needed in (dll, self.root / "generic.xml", self.root / "Formats.def"):
            if not needed.is_file():
                raise FileNotFoundError(f"Missing original Optris file: {needed}")
        califiles = self.root / f"Califiles SN{EXPECTED_OPTRIS_SERIAL}"
        if not califiles.is_dir():
            raise FileNotFoundError(f"Missing Optris calibration directory: {califiles}")
        if hasattr(os, "add_dll_directory"):
            self._dll_dir = os.add_dll_directory(str(dll.parent))

        try:
            self.lib = ct.CDLL(str(dll))
            lib = self.lib
            lib.evo_irimager_usb_init.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_char_p]
            lib.evo_irimager_usb_init.restype = ct.c_int
            lib.evo_irimager_terminate.argtypes = []
            lib.evo_irimager_terminate.restype = ct.c_int
            lib.evo_irimager_get_thermal_image_size.argtypes = [
                ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)
            ]
            lib.evo_irimager_get_thermal_image_size.restype = ct.c_int
            lib.evo_irimager_get_thermal_image.argtypes = [
                ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.POINTER(ct.c_ushort)
            ]
            lib.evo_irimager_get_thermal_image.restype = ct.c_int
            lib.evo_irimager_get_serial.argtypes = [ct.POINTER(ct.c_ulong)]
            lib.evo_irimager_get_serial.restype = ct.c_int

            # Match the original Nerdland GROWBOT camera initialization:
            # usb_init('generic.xml'). The XML contains relative SDK paths,
            # so set the process working directory to the original project.
            os.chdir(self.root)
            ret = lib.evo_irimager_usb_init(b"generic.xml", None, None)
            if ret != 0:
                raise RuntimeError(
                    f"Optris USB initialization returned {ret}. "
                    "Close PIX Connect and verify the camera USB connection."
                )
            self.connected = True
            w, h = ct.c_int(0), ct.c_int(0)
            ret = lib.evo_irimager_get_thermal_image_size(ct.byref(w), ct.byref(h))
            if ret != 0 or not (1 <= w.value <= 4096 and 1 <= h.value <= 4096):
                raise RuntimeError(f"Cannot read Optris thermal frame size (return {ret})")
            self.width, self.height = w.value, h.value
            serial_number = ct.c_ulong(0)
            if lib.evo_irimager_get_serial(ct.byref(serial_number)) == 0:
                self.serial_number = str(serial_number.value)
        except Exception:
            self.close()
            raise

    def capture(self, frames: int = 3, timeout_s: float = 15.0) -> np.ndarray:
        """Return per-pixel median temperature image in degrees Celsius.

        The SDK returns uint16 thermal data with T_C = raw/10 - 100.
        Failed frame calls and all-zero frames are skipped with a timeout.
        No palette colours are used for temperatures.
        """
        if not self.connected or self.lib is None:
            raise RuntimeError("Connect Optris before acquisition")
        data = np.empty((self.height, self.width), dtype=np.uint16)
        ptr = data.ctypes.data_as(ct.POINTER(ct.c_ushort))
        stack = []
        deadline = time.monotonic() + timeout_s
        while len(stack) < frames and time.monotonic() < deadline:
            w, h = ct.c_int(self.width), ct.c_int(self.height)
            ret = self.lib.evo_irimager_get_thermal_image(ct.byref(w), ct.byref(h), ptr)
            if ret == 0:
                if w.value != self.width or h.value != self.height:
                    raise RuntimeError(f"Unexpected thermal size {w.value} x {h.value}")
                if np.any(data):
                    temp = data.astype(np.float32) / 10.0 - 100.0
                    if np.isfinite(temp).all() and -50.0 <= float(np.median(temp)) <= 300.0:
                        stack.append(temp)
            if len(stack) < frames:
                time.sleep(0.15)
        if len(stack) < frames:
            raise TimeoutError(
                f"Only {len(stack)}/{frames} thermal frames acquired in {timeout_s:.0f} s"
            )
        return np.median(np.stack(stack), axis=0).astype(np.float32)

    def close(self) -> None:
        try:
            if self.connected and self.lib is not None:
                self.lib.evo_irimager_terminate()
        finally:
            self.connected = False
            self.lib = None
            if self._dll_dir is not None:
                self._dll_dir.close()
                self._dll_dir = None


def thermal_preview(temp_c: np.ndarray, lower_c: float | None = None,
                    upper_c: float | None = None) -> np.ndarray:
    """Make an RGB false-colour preview only (the temperature array remains raw)."""
    if lower_c is None or upper_c is None:
        lo, hi = np.percentile(temp_c, (2, 98))
    else:
        lo, hi = float(lower_c), float(upper_c)
    if hi <= lo:
        hi = lo + 0.5
    image = np.uint8(np.clip((temp_c - lo) / (hi - lo) * 255, 0, 255))
    return cv2.cvtColor(cv2.applyColorMap(image, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def make_optris_camera() -> OptrisUSB:
    """Prefer the standalone Optris backend if one is already available.

    This lets users retain a locally corrected thermal backend. The complete
    original-SDK implementation above is used if no separate backend exists.
    Both expose connect()/capture(frames=3)/close() and width/height/serial.
    """
    if st.session_state.get("reuse_existing_optris_backend", False):
        from growbot_optris_backend import OptrisUSB as ExistingOptrisUSB
        return ExistingOptrisUSB(PROJECT_DIR)
    return OptrisUSB(PROJECT_DIR)


class RealSense:
    """Original 640x480 30fps streams, aligned depth, bounded capture retries."""

    def __init__(self):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.align = rs.align(rs.stream.color)
        self.active = False
        self.serial_number = ""
        self.device_name = ""
        self.depth_scale = 0.001

    def start(self) -> None:
        if self.active:
            return
        rs = self.rs
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense detected. Connect the D435i via USB 3.x.")
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            profile = self.pipeline.start(cfg)
            self.active = True
            device = profile.get_device()
            self.serial_number = device.get_info(rs.camera_info.serial_number)
            self.device_name = device.get_info(rs.camera_info.name)
            self.depth_scale = device.first_depth_sensor().get_depth_scale()
            # Settle automatic exposure on startup.
            for _ in range(8):
                self.pipeline.wait_for_frames(timeout_ms=5000)
        except Exception:
            if self.active:
                self.stop()
            raise

    def stop(self) -> None:
        if self.active:
            self.pipeline.stop()
            self.active = False

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (BGR color uint8, aligned depth raw uint16)."""
        if not self.active:
            raise RuntimeError("Start RealSense before capturing")
        for attempt in range(3):
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                aligned = self.align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    raise RuntimeError("Missing depth/color frame")
                bgr = np.asanyarray(color_frame.get_data()).copy()
                depth = np.asanyarray(depth_frame.get_data()).copy()
                if bgr.shape[:2] != depth.shape[:2]:
                    raise RuntimeError("Color and depth frame sizes do not match")
                return bgr, depth
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"RealSense capture failed: {exc}") from exc
                time.sleep(0.3)
        raise RuntimeError("Unable to capture RealSense frames")


def depth_preview(depth: np.ndarray, scale: float, maximum_m: float = 2.0) -> np.ndarray:
    distance_m = depth.astype(np.float32) * scale
    visual = np.uint8(255 * np.clip(distance_m / maximum_m, 0, 1))
    colored = cv2.applyColorMap(255 - visual, cv2.COLORMAP_JET)
    colored[depth == 0] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def make_result(
    bgr: np.ndarray,
    depth: np.ndarray,
    camera: RealSense,
    thermal_c: np.ndarray | None,
    thermal_camera: OptrisUSB | None,
    plant_number: int | None,
    mpos: tuple[float, float, float] | None,
    save_images: bool,
    use_legacy_roi: bool,
) -> dict:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    top, bottom, left, right = LEGACY_RGB_ROI
    roi_bgr = bgr[top:bottom, left:right] if use_legacy_roi else bgr
    # utils.create_mask expects BGR and returns an RGB segmentation.
    segmentation, mask = create_mask(roi_bgr)
    biomass_proxy = round(float(np.count_nonzero(mask)) * LEGACY_BIOMASS_FACTOR_G_PER_PIXEL, 2)
    now = datetime.now()
    result = {
        "rgb": rgb,
        "depth": depth,
        "depth_preview": depth_preview(depth, camera.depth_scale),
        "depth_scale_m_per_unit": float(camera.depth_scale),
        "segmentation": segmentation,
        "mask": mask,
        "roi_rgb": cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB),
        "biomass_proxy_g": biomass_proxy,
        "plant_number": plant_number,
        "mpos": mpos,
        "timestamp": now.isoformat(timespec="seconds"),
        "saved_dir": None,
        "thermal_c": thermal_c,
        "thermal_preview": thermal_preview(thermal_c) if thermal_c is not None else None,
        "thermal_median_c": float(np.median(thermal_c)) if thermal_c is not None else None,
        "thermal_camera_serial": thermal_camera.serial_number if thermal_camera is not None else None,
    }
    if save_images:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        part = f"plant_{plant_number:02d}" if plant_number is not None else "manual"
        folder = CAPTURE_DIR / f"{now.strftime('%Y%m%d_%H%M%S_%f')}_{part}"
        folder.mkdir(parents=True, exist_ok=False)
        # Raw depth PNG preserves 16-bit SDK depth values. Multiply by
        # depth_scale to obtain metres (documented in metadata).
        for fname, image in (
            ("rgb.png", bgr),
            ("depth_raw_16bit.png", depth),
            ("segmentation.png", cv2.cvtColor(segmentation, cv2.COLOR_RGB2BGR)),
            ("mask.png", mask),
        ):
            if not cv2.imwrite(str(folder / fname), image):
                raise RuntimeError(f"Failed to save {fname}")
        if thermal_c is not None:
            np.save(folder / "thermal_temperature_C.npy", thermal_c)
            # Original Optris encoding: T_C = uint16 / 10 - 100.
            thermal_raw = np.clip(
                np.rint((thermal_c + 100.0) * 10.0), 0, 65535
            ).astype(np.uint16)
            if not cv2.imwrite(str(folder / "thermal_raw_16bit.png"), thermal_raw):
                raise RuntimeError("Failed to save thermal_raw_16bit.png")
            if not cv2.imwrite(
                str(folder / "thermal_preview.png"),
                cv2.cvtColor(result["thermal_preview"], cv2.COLOR_RGB2BGR),
            ):
                raise RuntimeError("Failed to save thermal_preview.png")
        meta = {
            "timestamp_local": result["timestamp"],
            "plant_number": plant_number,
            "machine_position_mm": list(mpos) if mpos is not None else None,
            "original_2024_xy_mm": list(plant_offset_mm(plant_number)) if plant_number else None,
            "realsense_name": camera.device_name,
            "realsense_serial": camera.serial_number,
            "depth_scale_m_per_unit": camera.depth_scale,
            "rgb_format": "PNG, RGB representation of SDK BGR8",
            "depth_format": "PNG uint16, depth aligned to color",
            "legacy_roi": use_legacy_roi,
            "legacy_biomass_factor_g_per_pixel": LEGACY_BIOMASS_FACTOR_G_PER_PIXEL,
            "biomass_proxy_g_uncalibrated": biomass_proxy,
            "optris_present": thermal_c is not None,
            "optris_serial": result["thermal_camera_serial"],
            "thermal_dimensions_px": [int(thermal_c.shape[1]), int(thermal_c.shape[0])]
                if thermal_c is not None else None,
            "thermal_encoding": "original Optris uint16 = (T_C + 100) * 10",
            "thermal_temperature_units": "degree Celsius, SDK radiometric output",
            "thermal_median_c": result["thermal_median_c"],
        }
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        result["saved_dir"] = str(folder)
    return result


def perform_capture(plant_number: int | None) -> None:
    """Capture RealSense + optionally Optris at the current stationary position."""
    camera = st.session_state.realsense
    if camera is None or not camera.active:
        raise RuntimeError("RealSense must be connected for RGB/depth and biomass model")
    thermal_camera = st.session_state.optris
    if st.session_state.require_both_cameras and (
        thermal_camera is None or not thermal_camera.connected
    ):
        raise RuntimeError("Both cameras are required. Connect Optris before acquisition")
    controller = st.session_state.blackbox
    mpos = None
    if controller and controller.serial.is_open:
        state, pos, _ = controller.status()
        if state != "Idle":
            raise RuntimeError("Cannot capture while gantry is not Idle")
        mpos = pos

    bgr, depth = camera.capture()
    thermal_c = None
    if thermal_camera is not None and thermal_camera.connected:
        thermal_c = thermal_camera.capture(frames=3)

    previous = st.session_state.last_result
    previous_thermal = None
    previous_plant = None
    if previous is not None and previous.get("thermal_c") is not None:
        previous_thermal = previous["thermal_c"]
        previous_plant = previous.get("plant_number")

    result = make_result(
        bgr, depth, camera, thermal_c, thermal_camera, plant_number, mpos,
        st.session_state.save_images, st.session_state.use_legacy_roi,
    )
    result["previous_thermal_c"] = previous_thermal
    result["previous_plant_number"] = previous_plant
    st.session_state.last_result = result


def read_icons() -> list[str]:
    folder = PROJECT_DIR / "plant icons"
    icons = {p.stem.lower(): p for p in folder.glob("*.png")}
    result = []
    for number in DISPLAY_IDS:
        icon = icons.get(f"plant{number}")
        if not icon:
            raise FileNotFoundError(f"Missing original plant icon: plant{number}.png")
        encoded = base64.b64encode(icon.read_bytes()).decode("ascii")
        result.append("data:image/png;base64," + encoded)
    return result


def disconnect_all() -> None:
    """Stop RealSense and close serial safely; never send motion or home commands.

    Refuse disconnect when movement/hold is in progress or GRBL status is
    unknown, so a running move is never abandoned with no software control.
    """
    box = st.session_state.blackbox
    camera = st.session_state.realsense
    thermal_camera = st.session_state.optris

    if box is not None and box.serial.is_open:
        state, _, raw = box.status()
        if state not in ("Idle", "Alarm"):
            raise RuntimeError(
                f"Cannot disconnect while the controller is {state}. "
                f"Wait for Idle or use the physical stop if needed: {raw}"
            )

    errors = []
    if camera is not None:
        try:
            camera.stop()
            st.session_state.realsense = None
        except Exception as exc:
            errors.append(f"RealSense: {exc}")

    if thermal_camera is not None:
        try:
            thermal_camera.close()
            st.session_state.optris = None
        except Exception as exc:
            errors.append(f"Optris: {exc}")

    if box is not None:
        try:
            box.close()
            st.session_state.blackbox = None
        except Exception as exc:
            errors.append(f"BLACKBOX: {exc}")

    st.session_state.enable_grid = False
    st.session_state.selected_plant = None
    st.session_state.plant_click_epoch += 1
    # Keep last_result available to inspect or save after disconnecting.
    if errors:
        st.session_state.notice = ("error", "Disconnect incomplete: " + "; ".join(errors))
    else:
        st.session_state.notice = (
            "success", "All connected devices disconnected. "
            "The gantry has not been moved; stop Streamlit using Ctrl+C when done."
        )


# -----------------------------------------------------------------------------
# Streamlit UI -- events only. No hardware actions during initial rendering.
# -----------------------------------------------------------------------------
st.set_page_config(page_title="GROWBOT | Plant imaging", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 0.9rem; padding-bottom: 1.5rem; max-width: 1800px; }
[data-testid="stImage"] img { border-radius: 8px; }
[data-testid="stMetric"] { padding: 0.15rem 0; }
</style>
""", unsafe_allow_html=True)

for name, value in (
    ("blackbox", None),
    ("realsense", None),
    ("optris", None),
    ("reuse_existing_optris_backend", (PROJECT_DIR / "growbot_optris_backend.py").exists()),
    ("require_both_cameras", True),
    ("selected_plant", None),
    ("last_result", None),
    ("feedback", ""),
    ("enable_grid", False),
    ("save_images", True),
    ("use_legacy_roi", True),
    ("travel_feed_mm_min", DEFAULT_FEED_MM_MIN),
    ("plant_click_epoch", 0),
    ("notice", None),
    ("clear_arm_next_rerun", False),
):
    if name not in st.session_state:
        st.session_state[name] = value

if st.session_state.clear_arm_next_rerun:
    st.session_state.enable_grid = False
    st.session_state.clear_arm_next_rerun = False

header_left, header_right = st.columns([5, 1])
with header_left:
    st.title("GROWBOT · Plant imaging")    st.caption("Nerdland 2024 · 35 plants · BLACKBOX + RealSense D435i + Optris")
with header_right:
    if (PROJECT_DIR / "mebios.png").exists():
        st.image(str(PROJECT_DIR / "mebios.png"), width=135)

if st.session_state.feedback:
    st.success(st.session_state.feedback)
    st.session_state.feedback = ""

if st.session_state.notice is not None:
    kind, message = st.session_state.notice
    if kind == "error":
        st.error(message)
    elif kind == "warning":
        st.warning(message)
    else:
        st.success(message)
    st.session_state.notice = None

# Sidebar hardware and acquisition controls.
with st.sidebar:
    for file in ("logo KUL.png", "logo_600jr.png", "growbot_logo.png"):
        if (PROJECT_DIR / file).exists():
            st.image(str(PROJECT_DIR / file), width=175)
    st.header("Hardware")

    box = st.session_state.blackbox
    connected = box is not None and box.serial.is_open
    if st.button("Connect BLACKBOX (COM4)", disabled=connected, use_container_width=True):
        try:
            st.session_state.blackbox = BlackBox()
            st.session_state.enable_grid = False
            st.session_state.feedback = "BLACKBOX connected. Home before movement."
            st.rerun()
        except Exception as exc:
            st.error(f"BLACKBOX connection failed: {exc}")
    if st.button("Disconnect BLACKBOX", disabled=not connected, use_container_width=True):
        try:
            state, _, _ = box.status()
            if state != "Idle" and state != "Alarm":
                raise RuntimeError("Controller must be idle before disconnecting")
            box.close()
            st.session_state.blackbox = None
            st.session_state.enable_grid = False
            st.session_state.plant_click_epoch += 1
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    camera = st.session_state.realsense
    camera_ready = camera is not None and camera.active
    if st.button("Start RealSense D435i", disabled=camera_ready, use_container_width=True):
        try:
            new_camera = RealSense()
            new_camera.start()
            st.session_state.realsense = new_camera
            st.session_state.feedback = f"RealSense started: {new_camera.device_name} ({new_camera.serial_number})"
            st.rerun()
        except Exception as exc:
            st.error(f"RealSense startup failed: {exc}")
    if st.button("Stop RealSense", disabled=not camera_ready, use_container_width=True):
        try:
            camera.stop()
            st.session_state.realsense = None
            st.session_state.plant_click_epoch += 1
            st.rerun()
        except Exception as exc:
            st.error(f"RealSense stop failed: {exc}")

    if camera_ready:
        st.caption(f"Camera: {camera.device_name} · SN {camera.serial_number}")
    else:
        st.caption("Gantry manual controls work without a RealSense camera")

    st.divider()
    st.header("Optris thermal")
    thermal_camera = st.session_state.optris
    thermal_ready = thermal_camera is not None and thermal_camera.connected
    if (PROJECT_DIR / "growbot_optris_backend.py").is_file():
        st.checkbox(
            "Reuse existing standalone Optris backend",
            key="reuse_existing_optris_backend",
            disabled=thermal_ready,
            help="Recommended if you have already fixed and tested the standalone thermal camera app.",
        )
    if st.button("Connect Optris USB", disabled=thermal_ready, use_container_width=True):
        try:
            new_thermal = make_optris_camera()
            new_thermal.connect()
            st.session_state.optris = new_thermal
            st.session_state.feedback = (
                f"Optris connected: {new_thermal.width}×{new_thermal.height} · "
                f"SN {new_thermal.serial_number or 'unknown'}"
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Optris connection failed: {exc}")
    if st.button("Disconnect Optris", disabled=not thermal_ready, use_container_width=True):
        try:
            thermal_camera.close()
            st.session_state.optris = None
            st.session_state.plant_click_epoch += 1
            st.rerun()
        except Exception as exc:
            st.error(f"Optris disconnect failed: {exc}")
    if thermal_ready:
        st.caption(
            f"Thermal: {thermal_camera.width}×{thermal_camera.height} · "
            f"SN {thermal_camera.serial_number or 'unknown'}"
        )
        if (thermal_camera.serial_number and
                thermal_camera.serial_number != EXPECTED_OPTRIS_SERIAL):
            st.error("Optris serial differs from the bundled calibration folder. "
                     "Check calibration before interpreting temperatures.")
    else:
        st.caption("Close PIX Connect before connecting the Optris SDK.")

    if st.button(
        "Shutdown / Disconnect All",
        disabled=not (connected or camera_ready or thermal_ready),
        use_container_width=True,
        help="Stops both cameras and closes COM4 after confirming the gantry is idle; "
             "does not move the gantry or exit Streamlit.",
    ):
        try:
            disconnect_all()
            st.rerun()
        except Exception as exc:
            st.error(f"Shutdown cancelled: {exc}")
    st.caption("Does not move the gantry or switch off the power.")

    st.divider()
    st.header("Movement")
    st.slider(
        "XY speed (mm/min)",
        min_value=MIN_FEED_MM_MIN,
        max_value=MAX_FEED_MM_MIN,
        step=100,
        key="travel_feed_mm_min",
        help="Used for plant moves, return-to-home, and manual jogging. "
             "Your controller is set to 2500 mm/min maximum per XY axis.",
    )
    st.caption("Speed setting is applied to manual jogs and all XY travel.")

    st.divider()
    st.header("Capture settings")
    st.checkbox("Save captures to disk", key="save_images")
    st.checkbox("Use original 2024 biomass crop", key="use_legacy_roi")
    st.checkbox("Require both RealSense + Optris for capture", key="require_both_cameras")
    st.caption("Biomass remains the 2024 green-pixel demo estimate, not a calibrated measurement.")

# -----------------------------------------------------------------------------
# One-screen imaging workspace: compact plant selector + imaging dashboard.
# -----------------------------------------------------------------------------
box = st.session_state.blackbox
connected = box is not None and box.serial.is_open
camera = st.session_state.realsense
camera_ready = camera is not None and camera.active
thermal_camera = st.session_state.optris
thermal_ready = thermal_camera is not None and thermal_camera.connected
acquisition_ready = (camera_ready and
                     (thermal_ready or not st.session_state.require_both_cameras))
state = "Disconnected"
pos = None
raw = ""
if connected:
    try:
        state, pos, raw = box.status()
    except Exception as exc:
        state = "Unknown"
        st.error(f"BLACKBOX status error: {exc}")

st.caption(
    f"Motion: {'COM4 · ' + state if connected else 'disconnected'}"
    f"  |  RealSense: {'connected' if camera_ready else 'off'}"
    f"  |  Optris: {'connected' if thermal_ready else 'off'}"
    f"  |  XY speed: {st.session_state.travel_feed_mm_min} mm/min"
)

selector_col, results_col = st.columns([1.05, 2.35], gap="large")

with selector_col:
    st.subheader("Select a plant")
    st.caption("Click = move → settle → capture → show results")
    if st.button("Home gantry ($H)",
                 disabled=not connected or state not in ("Idle", "Alarm"),
                 use_container_width=True,
                 help="Homes all configured axes; watch the full movement and keep emergency stop accessible."):
        try:
            with st.spinner("Homing all configured axes; monitor gantry"):
                response = box.home()
            st.session_state.feedback = "Homing completed: " + response
            st.rerun()
        except Exception as exc:
            st.error(f"Homing failed: {exc}")
    st.checkbox(
        "Enable one-click acquisition — XY path clear",
        key="enable_grid",
        help="Arm only after confirming all plant targets and the gantry travel path are clear."
    )
    if not connected or not box.homed_in_this_connection or state != "Idle":
        st.caption("Connect and home the BLACKBOX first.")
    elif not camera_ready:
        st.caption("Start RealSense before selecting a plant.")
    elif st.session_state.require_both_cameras and not thermal_ready:
        st.caption("Connect Optris, or disable the 'require both' setting.")
    elif not st.session_state.enable_grid:
        st.caption("Arm one-click acquisition above to enable movement.")
    else:
        st.caption("Ready to move and acquire.")

    selected_from_click = None
    try:
        icons = read_icons()
        clicked_index = clickable_images(
            icons,
            titles=[f"Plant {i}" for i in DISPLAY_IDS],
            div_style={
                "display": "grid",
                "grid-template-columns": "repeat(7, minmax(0, 1fr))",
                "gap": "3px",
                "margin": "0",
                "width": "100%",
            },
            img_style={
                "width": "100%", "height": "auto", "max-height": "62px",
                "object-fit": "contain", "cursor": "pointer",
            },
            key=f"plant_selector_{st.session_state.plant_click_epoch}",
        )
        if clicked_index is not None and 0 <= clicked_index < len(DISPLAY_IDS):
            selected_from_click = DISPLAY_IDS[clicked_index]
    except FileNotFoundError as exc:
        st.warning(f"Original plant icons unavailable: {exc}")
        manual_selection = st.number_input("Plant number", 1, PLANTS, 1)
        if st.button("Move + capture selected plant", use_container_width=True):
            selected_from_click = int(manual_selection)

    # Keep the exact verified v3 movement + click-consumption procedure.
    if selected_from_click is not None:
        st.session_state.selected_plant = selected_from_click
        try:
            if not connected:
                raise RuntimeError("Connect the BLACKBOX first. No movement was sent.")
            if not box.homed_in_this_connection or state != "Idle":
                raise RuntimeError("Home the gantry and verify it is Idle first. No movement was sent.")
            if not camera_ready:
                raise RuntimeError("Start the RealSense camera first. No movement was sent.")
            if st.session_state.require_both_cameras and not thermal_ready:
                raise RuntimeError("Connect Optris first. No movement was sent.")
            if not st.session_state.enable_grid:
                raise RuntimeError("Enable one-click acquisition after checking the full travel path. No movement was sent.")

            target_x, target_y = plant_machine_xy(selected_from_click)
            check_xy_limits(target_x, target_y)
            feed = st.session_state.travel_feed_mm_min
            with st.spinner(f"Plant {selected_from_click}: moving → capturing RealSense and thermal"):
                box.move_machine_xy(target_x, target_y, feed)
                time.sleep(0.8)  # Wait for motion vibration to settle.
                perform_capture(selected_from_click)
            st.session_state.notice = (
                "success", f"Plant {selected_from_click}: arrived, captured and "
                + ("saved images." if st.session_state.save_images else "processed images (saving disabled)."),
            )
        except Exception as exc:
            st.session_state.notice = ("error", f"Plant {selected_from_click}: {exc}")
            if connected and not box.homed_in_this_connection:
                st.session_state.clear_arm_next_rerun = True
        finally:
            # Consume every event to prevent repeated motion on reruns.
            st.session_state.plant_click_epoch += 1
            st.rerun()

    plant = st.session_state.selected_plant
    if plant is not None:
        wx, wy = plant_offset_mm(plant)
        st.caption(f"Selected plant {plant} · G54 X={wx:.0f}, Y={wy:.0f} mm")

    if st.button("Capture at current position (no movement)",
                 disabled=not acquisition_ready, use_container_width=True):
        try:
            with st.spinner("Acquiring RealSense and Optris"):
                perform_capture(None)
            st.session_state.notice = ("success", "Manual camera capture complete")
            st.rerun()
        except Exception as exc:
            st.error(f"Camera capture failed: {exc}")

    can_return = connected and box.homed_in_this_connection and state == "Idle"
    if st.button("Return XY to home (no homing cycle)", disabled=not can_return,
                 use_container_width=True):
        try:
            with st.spinner("Returning XY to homed coordinates"):
                response = box.move_machine_xy(
                    HOME_MPOS[0], HOME_MPOS[1], st.session_state.travel_feed_mm_min
                )
            st.session_state.feedback = "Returned XY home: " + response
            st.rerun()
        except Exception as exc:
            st.error(f"Return-to-home failed: {exc}")

with results_col:
    head, actions = st.columns([2.4, 2.8])
    with head:
        st.subheader("Imaging dashboard")
    with actions:
        live_actions = st.columns([1.2, 1, 1])
        with live_actions[0]:
            start_live = st.button(
                "▶ Live cameras", disabled=not (camera_ready or thermal_ready),
                use_container_width=True,
                help="Read-only, time-limited live RGB, depth and thermal previews; no saving or motion."
            )
        with live_actions[1]:
            st.button("■ Stop live", use_container_width=True,
                      help="Interrupt live preview and return to the last captured result.")
        with live_actions[2]:
            preview_duration_s = st.selectbox(
                "Preview limit", [15, 30, 60, 120], index=2,
                format_func=lambda s: f"{s} s",
                label_visibility="collapsed",
                help="Maximum live preview duration per start. Can be stopped earlier."
            )

    result = st.session_state.last_result
    if start_live:
        # Streamlit 1.33 has no st.fragment or native timer. Instead we render
        # into three placeholders during one bounded server run. Clicking Stop
        # (or any other widget) triggers a Streamlit rerun, interrupting this
        # preview and leaving all existing camera/serial connections intact.
        st.caption("LIVE · preview only · no movement, segmentation or files saved · cameras not synchronized")
        cam_cols = st.columns(3, gap="small")
        preview_slots = [col.empty() for col in cam_cols]
        live_info = st.empty()
        start_time = time.monotonic()
        count = 0
        while time.monotonic() - start_time < preview_duration_s:
            # At least one image is expected. Individual camera failures are
            # clearly reported; stop rather than looping indefinitely.
            try:
                if camera_ready:
                    bgr, depth = camera.capture()
                    preview_slots[0].image(
                        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                        caption="LIVE · RealSense RGB", use_column_width=True,
                    )
                    preview_slots[1].image(
                        depth_preview(depth, camera.depth_scale),
                        caption="LIVE · aligned depth (0–2 m)", use_column_width=True,
                    )
                else:
                    preview_slots[0].info("RealSense not connected")
                    preview_slots[1].info("Depth not available")
                if thermal_ready:
                    temp = thermal_camera.capture(frames=1, timeout_s=5.0)
                    lo, hi = np.percentile(temp, (2, 98))
                    preview_slots[2].image(
                        thermal_preview(temp, float(lo), float(hi)),
                        caption=f"LIVE · Optris · {np.median(temp):.1f} °C median",
                        use_column_width=True,
                    )
                    live_info.caption(f"Thermal preview auto-scale: {lo:.1f}–{hi:.1f} °C · "
                                      "apparent radiometric temperature")
                else:
                    preview_slots[2].info("Optris not connected")
                count += 1
                time.sleep(0.3)  # About 2–3 previews/second, not 30-fps video.
            except Exception as exc:
                live_info.error(f"Live preview stopped: {exc}")
                break
        if count > 0:
            st.caption("Live preview ended. Click ▶ Live cameras to restart, "
                       "or select a plant to move and capture.")
    elif result is None:
        st.info("No capture yet. Select a plant or capture at the current position. "
                "You can also start the live cameras without moving the gantry.")
        empty = st.columns(3)
        for col, label in zip(empty, ("RGB", "Aligned depth", "Optris thermal")):
            with col:
                st.caption(label)
                st.markdown("*Awaiting acquisition*")
    else:
        caption = f"Plant {result['plant_number']:02d}" if result['plant_number'] else "Manual capture"
        st.caption(f"**{caption}** · {result['timestamp']} · RealSense and Optris captured sequentially")
        image_cols = st.columns(3, gap="small")
        with image_cols[0]:
            st.image(result["rgb"], caption="RGB · RealSense", use_column_width=True)
        with image_cols[1]:
            st.image(result["depth_preview"], caption="Depth · aligned to RGB · 0–2 m", use_column_width=True)
        with image_cols[2]:
            tc = result.get("thermal_c")
            if tc is not None:
                # Use current frame's robust 2–98% scaling and show numeric
                # limits. Do not imply geometric registration to RGB/depth.
                lo, hi = np.percentile(tc, (2, 98))
                st.image(
                    thermal_preview(tc, float(lo), float(hi)),
                    caption=f"Thermal · Optris · {lo:.1f}–{hi:.1f} °C",
                    use_column_width=True,
                )
            else:
                st.info("Thermal image not captured")

        model_col, summary_col = st.columns([1.2, 1.8], gap="medium")
        with model_col:
            st.image(result["segmentation"],
                     caption="2024 green-pixel segmentation (ROI)",
                     use_column_width=True)
        with summary_col:
            # Scene depth is a simple valid-pixel median, not segmented plant height.
            valid_depth = result["depth"][result["depth"] > 0]
            scene_depth_m = (float(np.median(valid_depth)) * result["depth_scale_m_per_unit"]
                             if len(valid_depth) else None)
            st.metric("Biomass proxy · original model", f"{result['biomass_proxy_g']:.2f} g")
            st.caption("Green-pixel estimate only; not calibrated biomass.")
            metric_cols = st.columns(2)
            with metric_cols[0]:
                st.metric("Median scene depth", f"{scene_depth_m:.2f} m" if scene_depth_m is not None else "—")
            with metric_cols[1]:
                st.metric("Median thermal", f"{result['thermal_median_c']:.1f} °C"
                          if result.get("thermal_median_c") is not None else "—")
            if result.get("saved_dir"):
                st.caption("Saved: " + str(result["saved_dir"]))

        with st.expander("More results · original ROI, previous thermal and files", expanded=False):
            st.image(result["roi_rgb"], caption="2024 biomass ROI", width=440)
            current_t = result.get("thermal_c")
            previous_t = result.get("previous_thermal_c")
            if current_t is not None and previous_t is not None:
                lo = min(float(np.percentile(current_t, 2)), float(np.percentile(previous_t, 2)))
                hi = max(float(np.percentile(current_t, 98)), float(np.percentile(previous_t, 98)))
                label = (f"Plant {result['previous_plant_number']}" if
                         result.get("previous_plant_number") else "Previous capture")
                c1, c2 = st.columns(2)
                with c1:
                    st.image(thermal_preview(current_t, lo, hi),
                             caption=f"Current · {lo:.1f}–{hi:.1f} °C", use_column_width=True)
                with c2:
                    st.image(thermal_preview(previous_t, lo, hi),
                             caption=f"{label} · same thermal scale", use_column_width=True)
            st.caption("Thermal pixels are not registered to RealSense pixels. "
                       "Capture times differ; the displayed temperature depends on emissivity and reflections.")

st.divider()
with st.expander("Machine control, homing and manual jog", expanded=False):
    st.write(f"**Controller:** {'COM4 connected' if connected else 'Disconnected'} · "
             f"**State:** `{state}` · "
             f"**Homed in this connection:** {box.homed_in_this_connection if connected else False}")
    if pos is not None:
        pcols = st.columns(3)
        for col, axis, val in zip(pcols, "XYZ", pos):
            col.metric(f"Machine {axis}", f"{val:.1f} mm")
        st.caption(raw)
    if connected:
        can_jog = box.homed_in_this_connection and state == "Idle"
        st.subheader("Manual XY movement")
        step = st.radio("Jog distance", [1.0, 5.0], horizontal=True,
                        format_func=lambda n: f"{n:g} mm")
        jog_cols = st.columns(4)
        for col, label, axis, delta in zip(
            jog_cols, ("X +", "X −", "Y +", "Y −"),
            ("X", "X", "Y", "Y"), (step, -step, step, -step),
        ):
            with col:
                if st.button(label, disabled=not can_jog, use_container_width=True):
                    try:
                        with st.spinner(f"Moving {axis} {delta:+g} mm"):
                            response = box.jog(axis, delta, st.session_state.travel_feed_mm_min)
                        st.session_state.feedback = "Jog complete: " + response
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Jog failed: {exc}")

st.caption("Safety: plant clicks command automatic XY motion when armed. Z is not moved during plant travel. "
           "Live preview never commands motion. Use Shutdown / Disconnect All before closing the server.")
