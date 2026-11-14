"""Hardware-free GROWBOT v6 display tests.

Run: python -m unittest discover -s tests -v
No serial port, cameras, Streamlit server, or motor commands are used.
"""
import unittest
import numpy as np
from growbot_display_v6 import (
    rotate_image, rotate_pointcloud_xy, depth_limits_m, depth_rgb_display,
)


class DisplayRotationTests(unittest.TestCase):
    def test_image_90_counterclockwise_and_inverse(self):
        a = np.arange(12, dtype=np.uint8).reshape(3, 4)
        rotated = rotate_image(a, 90)
        np.testing.assert_array_equal(rotated, np.rot90(a))
        self.assertEqual(rotated.shape, (4, 3))
        np.testing.assert_array_equal(rotate_image(rotated, -90), a)
        np.testing.assert_array_equal(rotate_image(rotate_image(a, 180), 180), a)
        np.testing.assert_array_equal(rotate_image(a, 0), a)
        np.testing.assert_array_equal(a, np.arange(12, dtype=np.uint8).reshape(3, 4))

    def test_rgb_channels_preserved(self):
        image = np.zeros((2, 3, 3), np.uint8)
        image[0, 0] = [240, 10, 20]
        result = rotate_image(image, 90)
        self.assertEqual(int(np.sum(result[:, :, 0])), 240)
        self.assertEqual(int(np.sum(result[:, :, 1])), 10)

    def test_rotating_3d_coordinates_does_not_change_depth(self):
        import pandas as pd
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, -4.0],
                           "z": [0.4, 0.6], "r": [255, 0]})
        turned = rotate_pointcloud_xy(df, 90)
        np.testing.assert_array_equal(turned["x"], [-3.0, 4.0])
        np.testing.assert_array_equal(turned["y"], [1.0, 2.0])
        np.testing.assert_array_equal(turned["z"], df["z"])
        np.testing.assert_array_equal(df["x"], [1.0, 2.0])
        return_to_start = rotate_pointcloud_xy(turned, -90)
        np.testing.assert_allclose(return_to_start[["x", "y"]], df[["x", "y"]])

    def test_unsupported_angle_rejected(self):
        with self.assertRaises(ValueError):
            rotate_image(np.zeros((2, 2), np.uint8), 45)


class DepthPaletteTests(unittest.TestCase):
    def test_fixed_depth_range_is_not_full_2_metres(self):
        depth = np.full((40, 50), 700, dtype=np.uint16)
        self.assertEqual(depth_limits_m(depth, 0.001, "Fixed range", (0.35, 1.25)),
                         (0.35, 1.25))
        a = depth_rgb_display(depth, 0.001, 0.35, 1.25)
        b = depth_rgb_display(depth, 0.001, 0.0, 2.0)
        self.assertEqual(a.shape, (40, 50, 3))
        self.assertFalse(np.array_equal(a, b))

    def test_auto_robust_bounds_exclude_missing_depth(self):
        depth = np.full((60, 80), 850, dtype=np.uint16)
        depth[:3] = 0
        depth[-3:] = 1200
        near, far = depth_limits_m(depth, 0.001, "Auto (5–95% valid depth)")
        self.assertGreater(near, 0.5)
        self.assertGreater(far, near)
        self.assertLessEqual(far, 1.3)

    def test_invalid_pixels_black_and_raw_unchanged(self):
        depth = np.array([[0, 400, 850], [1400, 0, 300]], dtype=np.uint16)
        original = depth.copy()
        rgb = depth_rgb_display(depth, 0.001, 0.35, 1.25)
        self.assertTrue((rgb[0, 0] == 0).all())
        self.assertTrue((rgb[1, 1] == 0).all())
        np.testing.assert_array_equal(depth, original)


if __name__ == "__main__":
    unittest.main()
