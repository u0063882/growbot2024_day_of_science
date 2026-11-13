"""Run offline using: python -m unittest discover -s tests -v

No camera, GRBL connection, Streamlit server or hardware movement required.
"""
import unittest

import cv2
import numpy as np

from growbot_segmentation_v5 import enhanced_plant_mask
from growbot_pointcloud_v5 import pointcloud_data


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


if __name__ == '__main__':
    unittest.main()
