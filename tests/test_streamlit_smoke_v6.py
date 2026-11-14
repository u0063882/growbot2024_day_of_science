"""Startup smoke test under the actual pinned Streamlit version.

Runs the complete v6 UI without pressing hardware connection buttons.
No serial, Optris SDK or RealSense device is used.
"""
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "growbot_gantry_realsense_optris_v6.py"


class V6StartupTests(unittest.TestCase):
    def test_full_dashboard_starts_without_hardware(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        problems = [str(err.message) for err in app.exception]
        self.assertEqual(problems, [], "\n".join(problems))
        self.assertEqual(app.session_state["display_rotation_deg"], 90)
        self.assertEqual(app.session_state["depth_colour_mode"], "Fixed range")
        self.assertEqual(tuple(app.session_state["depth_colour_range_m"]), (0.35, 1.25))
        self.assertEqual(app.session_state["optris_z_mode"], "Off")
        self.assertFalse(app.session_state["save_images"])


if __name__ == "__main__":
    unittest.main()
