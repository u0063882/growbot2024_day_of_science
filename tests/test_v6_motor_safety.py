"""No-hardware tests of v6 GRBL guardrails and thermal capture Z retraction.

These tests parse two functions from the shipped app and execute them with fake
in-memory motors/cameras. Neither the Streamlit app nor PySerial is imported.
"""
from __future__ import annotations

import ast
import math
import re
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


APP = Path(__file__).resolve().parents[1] / "growbot_gantry_realsense_optris_v6.py"
SOURCE = APP.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, str(APP))


def extracted(name, env):
    """Compile one original function/class definition, no app side effects."""
    nodes = [n for n in TREE.body if isinstance(n, (ast.ClassDef,
             ast.FunctionDef)) and n.name == name]
    assert len(nodes) == 1, (name, len(nodes))
    future = ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations")], level=0)
    fragment = ast.fix_missing_locations(ast.Module(body=[future, nodes[0]],
                                                    type_ignores=[]))
    exec(compile(fragment, str(APP), "exec"), env)
    return env[name]


class MotorGuardrailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = dict(
            serial=None, time=time, re=re, math=math,
            PORT="COM4", BAUDRATE=115200,
            Z_HOME_MPOS_MM=-1.0, Z_MIN_MPOS_MM=-101.0,
            Z_FEED_MM_MIN=600, Z_POSITION_TOLERANCE_MM=0.5,
            MIN_FEED_MM_MIN=200, MAX_FEED_MM_MIN=3600,
            MAX_MANUAL_JOG_MM=5.0,
            POSITION_TOLERANCE_MM=0.5,
            HOME_MPOS=(-1499, -1, -1), HOME_TOLERANCE_MM=2,
            AXIS_LIMITS={"X": (-1499, -1), "Y": (-1499, -1)},
            check_xy_limits=lambda x, y: None,
        )
        cls.BlackBox = extracted("BlackBox", cls.env)

    def box(self, z=-1.0):
        obj = object.__new__(self.BlackBox)
        obj.homed_in_this_connection = True
        commands = []
        obj._must_be_homed_and_idle = lambda: (-1100.0, -500.0, z)
        obj._command = lambda cmd, timeout=5.0: commands.append(cmd)
        obj._wait_for_position = lambda target, timeout: str(target)
        obj.status = lambda: ("Idle", (-1100.0, -500.0, z), "ok")
        return obj, commands

    def test_out_of_bounds_z_refused_before_any_command(self):
        box, cmds = self.box()
        with self.assertRaises(ValueError):
            box.move_machine_z(-105.0)
        self.assertEqual(cmds, [])

    def test_z_command_does_not_include_xy(self):
        box, cmds = self.box()
        box.move_machine_z(-11.0)
        self.assertEqual(cmds, ["G90 G21", "G53 G1 Z-11.000 F600"])

    def test_xy_blocked_while_z_lowered(self):
        box, cmds = self.box(z=-11.0)
        with self.assertRaisesRegex(RuntimeError, "XY movement blocked"):
            box.move_machine_xy(-1000.0, -500.0, 2000)
        self.assertEqual(cmds, [])

    def test_xy_jog_blocked_while_z_lowered(self):
        box, cmds = self.box(z=-11.0)
        with self.assertRaisesRegex(RuntimeError, "XY jog blocked"):
            box.jog("X", 1.0, 2000)
        self.assertEqual(cmds, [])


class CaptureZRetractionTests(unittest.TestCase):
    class FakeBox:
        def __init__(self, fail_retract=False):
            self.z = -1.0
            self.homed_in_this_connection = True
            self.serial = SimpleNamespace(is_open=True)
            self.moves = []
            self.fail_retract = fail_retract

        def status(self):
            return "Idle", (-1100.0, -500.0, self.z), "ok"

        def move_machine_z(self, target_z_mm):
            self.moves.append(target_z_mm)
            if self.fail_retract and target_z_mm == -1.0:
                raise RuntimeError("simulated retraction failure")
            self.z = target_z_mm

    def make_env(self, thermal_error=False, fail_retract=False, mode="Automatic (experimental)"):
        box = self.FakeBox(fail_retract=fail_retract)
        camera = SimpleNamespace(
            active=True, depth_scale=0.001,
            capture=lambda: (
                np.zeros((100, 100, 3), np.uint8),
                np.full((100, 100), 800, np.uint16),
            ),
        )

        def thermal_capture(frames=3):
            if thermal_error:
                raise RuntimeError("simulated Optris USB timeout")
            return np.full((20, 30), 22.0, np.float32)

        cam = SimpleNamespace(connected=True, capture=thermal_capture)
        session = SimpleNamespace(
            realsense=camera, optris=cam, require_both_cameras=True,
            blackbox=box, optris_offset_x_mm=0.0, optris_offset_y_mm=0.0,
            optris_z_mode=mode, enhanced_segmentation_mode="Variegated / pale-edged",
            z_target_distance_mm=450, z_camera_height_offset_mm=0,
            z_max_lowering_mm=10, z_min_clearance_mm=300,
            z_clearance_confirmed=True, save_images=False,
            use_legacy_roi=True, thermal_auto_scale=False,
            thermal_fixed_range_c=(18.0, 23.0),
            last_result=None, enable_grid=True,
        )
        st = SimpleNamespace(session_state=session)
        z_plan = SimpleNamespace(usable=True, lowering_mm=10.0,
                                 median_plant_distance_mm=800.0,
                                 estimated_optris_distance_mm=790.0,
                                 status="proposal")
        env = dict(
            st=st, np=np, time=SimpleNamespace(sleep=lambda s: None),
            Z_HOME_MPOS_MM=-1.0, Z_POSITION_TOLERANCE_MM=0.5,
            enhanced_plant_mask=lambda *a, **kw: (
                np.full((100, 100), 255, np.uint8), {"status": "ok"}
            ),
            plan_z_standoff=lambda *a, **kw: z_plan,
            mapping=SimpleNamespace(optris_machine_xy=lambda pos, off:
                                    (pos[0]+off[0], pos[1]+off[1])),
            make_result=lambda *a, **kw: {
                "plant_number": a[5], "thermal_c": a[3],
                "optris_z_plan": kw["thermal_z_plan"],
                "z_retraction_confirmed": kw["z_retraction_confirmed"],
            },
        )
        return extracted("perform_capture", env), box, session

    def test_successful_thermal_capture_retracts_before_result(self):
        fn, box, ss = self.make_env()
        fn(plant_number=1, apply_optris_offset=True)
        self.assertEqual(box.moves, [-11.0, -1.0])
        self.assertEqual(box.z, -1.0)
        self.assertTrue(ss.last_result["z_retraction_confirmed"])

    def test_usb_capture_failure_also_retracts(self):
        fn, box, ss = self.make_env(thermal_error=True)
        with self.assertRaisesRegex(RuntimeError, "simulated Optris USB timeout"):
            fn(plant_number=1, apply_optris_offset=True)
        self.assertEqual(box.moves, [-11.0, -1.0])
        self.assertEqual(box.z, -1.0)

    def test_failure_to_retract_disarms_and_reports_stop(self):
        fn, box, ss = self.make_env(fail_retract=True)
        with self.assertRaisesRegex(RuntimeError, "Z RETRACTION NOT VERIFIED"):
            fn(plant_number=1, apply_optris_offset=True)
        self.assertFalse(ss.enable_grid)
        self.assertFalse(ss.z_clearance_confirmed)
        self.assertEqual(box.z, -11.0)

    def test_manual_capture_does_not_move_z(self):
        fn, box, ss = self.make_env()
        fn(plant_number=None, apply_optris_offset=False)
        self.assertEqual(box.moves, [])
        self.assertEqual(box.z, -1.0)


if __name__ == "__main__":
    unittest.main()
