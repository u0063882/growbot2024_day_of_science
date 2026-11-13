"""Run offline using: python -m unittest discover -s tests -v

No camera, GRBL connection, Streamlit server or hardware movement required.
"""
import unittest
import ast
from pathlib import Path

import cv2
import numpy as np

from growbot_segmentation_v5 import enhanced_plant_mask
from growbot_pointcloud_v5 import pointcloud_data
from growbot_segmentation_v5 import variegated_plant_mask
from growbot_mapping_v5 import (
    GRID_COLUMNS, GRID_ROWS, legacy_plant_number, plant_offset_mm,
    plant_machine_xy, optris_machine_xy,
)


class EnhancedSegmentationTests(unittest.TestCase):
    def test_pale_flower_is_kept_and_brown_background_rejected(self):
        bgr = np.full((240, 320, 3), (30, 55, 90), dtype=np.uint8)
        bgr[:, :30] = (12, 12, 12)
        cv2.ellipse(bgr, (160, 130), (32, 80), 10, 0, 360,
                    (40, 160, 35), thickness=-1)
        cv2.circle(bgr, (160, 44), 21, (220, 220, 224), thickness=-1)
        mask, status = enhanced_plant_mask(bgr)
        self.assertGreater(np.mean(mask[120:150, 145:175] > 0), 0.7)
        self.assertGreater(np.mean(mask[30:58, 152:168] > 0), 0.3)
        self.assertLess(np.mean(mask[20:60, 270:310] > 0), 0.05)
        self.assertGreater(status['enhanced_pixels'], 0)

    def test_no_green_seed_returns_empty_mask(self):
        bgr = np.full((80, 80, 3), (30, 55, 90), dtype=np.uint8)
        mask, info = enhanced_plant_mask(bgr)
        self.assertEqual(np.count_nonzero(mask), 0)
        self.assertIn('Insufficient green seed', info['status'])


class PointCloudTests(unittest.TestCase):
    def test_aligned_rgb_depth_to_cloud(self):
        rgb = np.zeros((60, 80, 3), dtype=np.uint8)
        rgb[:, :, 1] = 120
        depth = np.full((60, 80), 700, dtype=np.uint16)
        intr = {'fx': 100, 'fy': 100, 'ppx': 40, 'ppy': 30}
        cloud = pointcloud_data(rgb, depth, intr, 0.001, point_limit=1200)
        self.assertGreater(len(cloud), 0)
        self.assertLessEqual(len(cloud), 1200)
        self.assertTrue((cloud['g'] == 120).all())
        self.assertLess(abs(float(cloud['z'].mean())), 0.01)

    def test_invalid_intrinsics_fails_safely(self):
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        depth = np.ones((20, 20), dtype=np.uint16)
        with self.assertRaises(ValueError):
            pointcloud_data(rgb, depth, None, 0.001)



class RowMajorAndOffsetTests(unittest.TestCase):
    def test_row_major_labels_preserve_every_2024_destination(self):
        self.assertEqual(GRID_COLUMNS, 7)
        self.assertEqual(GRID_ROWS, 5)
        # Prior code used original 1..35 IDs in column-major order.
        old_positions = {}
        for old_id in range(1, 36):
            column, row = divmod(old_id - 1, 5)
            old_positions[old_id] = (940.0 - 120.0 * column,
                                      -240.0 - 220.0 * row)
        for new_id in range(1, 36):
            old_id = legacy_plant_number(new_id)
            self.assertEqual(plant_offset_mm(new_id), old_positions[old_id])
        self.assertEqual(len(set(plant_offset_mm(i) for i in range(1, 36))), 35)

    def test_labels_are_1_to_7_then_8_to_14(self):
        self.assertEqual(legacy_plant_number(1), 1)
        self.assertEqual(legacy_plant_number(2), 6)
        self.assertEqual(legacy_plant_number(7), 31)
        self.assertEqual(legacy_plant_number(8), 2)
        self.assertEqual(plant_offset_mm(7), (220.0, -240.0))
        self.assertEqual(plant_offset_mm(8), (940.0, -460.0))
        self.assertEqual(plant_offset_mm(35), (220.0, -1120.0))

    def test_zero_and_small_offsets(self):
        plant = plant_machine_xy(1)
        self.assertEqual(optris_machine_xy(plant, (0, 0)), plant)
        self.assertEqual(optris_machine_xy(plant, (50, -40)),
                         (plant[0] + 50, plant[1] - 40))

    def test_invalid_thermal_target_rejected_before_travel(self):
        plant = plant_machine_xy(1)
        with self.assertRaises(ValueError):
            optris_machine_xy(plant, (-1000, 0))
        with self.assertRaises(ValueError):
            optris_machine_xy(plant, (0, 500))


class VariegatedSegmentationTests(unittest.TestCase):
    def test_teal_centres_and_pale_leaf_margins(self):
        # Brown background and a centred dark blue-green leaf with pale rims.
        bgr = np.full((480, 640, 3), (140, 170, 185), dtype=np.uint8)
        cv2.ellipse(bgr, (320, 220), (150, 130), 20, 0, 360,
                    (80, 195, 225), thickness=-1)
        cv2.ellipse(bgr, (320, 220), (138, 116), 20, 0, 360,
                    (105, 95, 62), thickness=-1)
        mask, data = variegated_plant_mask(bgr)
        self.assertGreater(mask[220, 320], 0)  # dark teal centre
        self.assertGreater(mask[220, 470], 0)  # yellow margin
        self.assertEqual(mask[20, 20], 0)  # brown background
        self.assertGreater(data['enhanced_pixels'], 0)



class CaptureSequenceSourceTests(unittest.TestCase):
    """Inspect the application without importing Streamlit or touching hardware."""

    def test_realsense_is_captured_before_optris_offset_move(self):
        path = Path(__file__).resolve().parents[1] / "growbot_gantry_realsense_optris_v5.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        fn = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "perform_capture")
        statements = ast.get_source_segment(path.read_text(encoding="utf-8"), fn)
        rgb_at = statements.index("bgr, depth = camera.capture()")
        offset_move_at = statements.index("controller.move_machine_xy(")
        optris_at = statements.index("thermal_camera.capture(frames=3)")
        self.assertLess(rgb_at, offset_move_at)
        self.assertLess(offset_move_at, optris_at)
        self.assertIn("if thermal_connected and apply_optris_offset:", statements)
        self.assertIn("optris_target = mapping.optris_machine_xy(", statements)

    def test_plant_click_validates_thermal_target_before_motor_motion(self):
        path = Path(__file__).resolve().parents[1] / "growbot_gantry_realsense_optris_v5.py"
        code = path.read_text(encoding="utf-8")
        call = code.index("mapping.optris_machine_xy(", code.index("selected_from_click = None"))
        motor = code.index("box.move_machine_xy(target_x, target_y, feed)", call)
        self.assertLess(call, motor)
        self.assertIn("apply_optris_offset=True", code)


if __name__ == '__main__':
    unittest.main()
