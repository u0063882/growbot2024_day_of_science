"""Optris Direct SDK 8.8.1 USB adapter, isolated from the working GROWBOT app.

Uses the same libirimager.dll, generic.xml and calibration files as the 2024
project. Hardware access is explicit; importing this module does not open USB.
This adapter is prepared from the original SDK examples but not yet tested with
an attached Optris camera.
"""
from __future__ import annotations

import ctypes as ct
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


class EvoIRFrameMetadata(ct.Structure):
    _fields_ = [
        ("counter", ct.c_uint),
        ("counterHW", ct.c_uint),
        ("timestamp", ct.c_longlong),
        ("timestampMedia", ct.c_longlong),
        ("flagState", ct.c_int),
        ("tempChip", ct.c_float),
        ("tempFlag", ct.c_float),
        ("tempBox", ct.c_float),
    ]


class OptrisUSB:
    def __init__(self, project_dir: Path):
        self.root = Path(project_dir).resolve()
        self.lib = None
        self.connected = False
        self.width = 0
        self.height = 0
        self.serial_number = None
        self._dll_dir = None

    def _config_copy(self) -> Path:
        """Use project files unmodified; make a copy with absolute resource paths."""
        original = self.root / "generic.xml"
        if not original.is_file():
            raise FileNotFoundError(f"Original Optris config missing: {original}")
        if not (self.root / "Formats.def").is_file():
            raise FileNotFoundError("Original Formats.def missing")
        tree = ET.parse(original)
        root = tree.getroot()
        cali = root.find("calipath")
        if cali is not None and cali.text and cali.text.strip():
            cali_path = (self.root / cali.text.strip()).resolve()
            if not cali_path.is_dir():
                raise FileNotFoundError(f"Optris calibration folder missing: {cali_path}")
            cali.text = str(cali_path)
        formats = root.find("formatspath")
        if formats is not None:
            formats.text = str(self.root) + os.sep
        config_dir = self.root / "thermal_experiments"
        config_dir.mkdir(exist_ok=True)
        copy_path = config_dir / "optris_config_absolute_paths.xml"
        tree.write(copy_path, encoding="utf-8", xml_declaration=True)
        return copy_path

    def connect(self) -> None:
        if self.connected:
            return
        if os.name != "nt":
            raise RuntimeError("The original Optris DLL is for Windows x64")
        dll = self.root / "irDirectSDK" / "sdk" / "x64" / "libirimager.dll"
        if not dll.is_file():
            raise FileNotFoundError(f"Original Optris DLL missing: {dll}")
        config = self._config_copy()
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
                ct.POINTER(ct.c_int), ct.POINTER(ct.c_int)]
            lib.evo_irimager_get_thermal_image_size.restype = ct.c_int
            lib.evo_irimager_get_serial.argtypes = [ct.POINTER(ct.c_ulong)]
            lib.evo_irimager_get_serial.restype = ct.c_int
            self._with_metadata = hasattr(lib, "evo_irimager_get_thermal_image_metadata")
            if self._with_metadata:
                func = lib.evo_irimager_get_thermal_image_metadata
                func.argtypes = [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int),
                                 ct.POINTER(ct.c_ushort),
                                 ct.POINTER(EvoIRFrameMetadata)]
                func.restype = ct.c_int
            else:
                func = lib.evo_irimager_get_thermal_image
                func.argtypes = [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int),
                                 ct.POINTER(ct.c_ushort)]
                func.restype = ct.c_int
            # Using the exact original SDK, XML and calibration files.
            ret = lib.evo_irimager_usb_init(str(config).encode("utf-8"), None, None)
            if ret != 0:
                raise RuntimeError(f"Optris USB initialization returned {ret}")
            self.connected = True
            width, height = ct.c_int(), ct.c_int()
            ret = lib.evo_irimager_get_thermal_image_size(
                ct.byref(width), ct.byref(height))
            if ret != 0 or not (1 <= width.value <= 4096 and 1 <= height.value <= 4096):
                raise RuntimeError(f"Cannot read thermal image size (SDK return {ret})")
            self.width, self.height = width.value, height.value
            serial = ct.c_ulong()
            if lib.evo_irimager_get_serial(ct.byref(serial)) == 0:
                self.serial_number = str(serial.value)
        except Exception:
            self.close()
            raise

    def _single_frame(self) -> np.ndarray | None:
        w, h = ct.c_int(self.width), ct.c_int(self.height)
        data = np.empty((self.height, self.width), dtype=np.uint16)
        ptr = data.ctypes.data_as(ct.POINTER(ct.c_ushort))
        if self._with_metadata:
            meta = EvoIRFrameMetadata()
            ret = self.lib.evo_irimager_get_thermal_image_metadata(
                ct.byref(w), ct.byref(h), ptr, ct.byref(meta))
            if ret != 0 or meta.flagState != 0:
                return None  # Not ready, shutter closed, or camera calibrating.
        else:
            ret = self.lib.evo_irimager_get_thermal_image(ct.byref(w), ct.byref(h), ptr)
            if ret != 0:
                return None
        if w.value != self.width or h.value != self.height:
            raise RuntimeError("Optris frame dimensions changed unexpectedly")
        # SDK documented encoding: T(C) = uint16 / 10 - 100.
        temperature = data.astype(np.float32) / 10.0 - 100.0
        if not np.isfinite(temperature).all() or np.median(temperature) < -50:
            return None
        return temperature

    def capture(self, frames: int = 3, timeout_s: float = 12.0) -> np.ndarray:
        if not self.connected or self.lib is None:
            raise RuntimeError("Connect the Optris camera first")
        acquired = []
        deadline = time.monotonic() + timeout_s
        while len(acquired) < frames and time.monotonic() < deadline:
            result = self._single_frame()
            if result is not None:
                acquired.append(result)
            time.sleep(0.12)
        if len(acquired) < frames:
            raise TimeoutError("Could not obtain valid Optris thermal frames")
        return np.median(np.stack(acquired), axis=0).astype(np.float32)

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
