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


    def test_dashboard_renders_synthetic_capture_without_hardware(self):
        """Exercise the after-capture layout (not covered by empty startup)."""
        import numpy as np

        app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[:, :, 1] = 110
        depth = np.full((480, 640), 800, dtype=np.uint16)
        thermal = np.full((80, 96), 22.0, dtype=np.float32)
        mask = np.full((330, 310), 255, dtype=np.uint8)
        result = {
            "plant_number": 1, "timestamp": "2026-11-14T10:00:00",
            "rgb": rgb, "depth": depth,
            "depth_scale_m_per_unit": 0.001,
            "segmentation": rgb[40:370, 170:480],
            "mask": mask, "roi_bounds": (40, 370, 170, 480),
            "enhanced_mask": np.full((480, 640), 255, np.uint8),
            "enhanced_segmentation": rgb.copy(),
            "enhanced_info": {"status": "Synthetic UI smoke test"},
            "enhanced_roi_bounds": None,
            "color_intrinsics": {
                "fx": 600, "fy": 600, "ppx": 320, "ppy": 240,
            },
            "biomass_proxy_g": 12.3,
            "scene_height_m": 0.12,
            "thermal_c": thermal, "thermal_median_c": 22.0,
            "mpos": (-1000.0, -500.0, -1.0),
            "thermal_mpos": (-1000.0, -495.0, -1.0),
            "previous_thermal_c": thermal + 0.2,
            "previous_plant_number": 2,
            "optris_z_plan": {
                "mode": "Preview only", "usable": True,
                "lowering_mm": 10.0,
                "applied_lowering_mm": 0.0,
                "median_plant_distance_mm": 800.0,
                "estimated_optris_distance_mm": 790.0,
                "status": "Synthetic Z proposal",
            },
            "saved_dir": None,
        }
        app.session_state["last_result"] = result
        app.session_state["show_3d_view"] = True
        app.run()
        problems = [str(err.message) for err in app.exception]
        self.assertEqual(problems, [], "\\n".join(problems))


if __name__ == "__main__":
    unittest.main()
