"""Unit tests for monocular face distance and horizontal angle estimation."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from face_distance import (
    FaceEstimate,
    estimate_faces_from_cv_boxes,
    estimate_from_box,
    load_camera_params,
    load_focal_length,
)


class TestFaceDistanceEstimation(unittest.TestCase):
    def test_prompt_derivation_example(self):
        """Verify calculation against the exact numerical example from the task specification."""
        # Setup example parameters:
        # Frame: 640x480, c_x = 320, f = 640, W = 0.15
        # Detector box: xmin=0.32, ymin=0.25, width=0.18, height=0.24
        est = estimate_from_box(
            index=1,
            rel_x=0.32,
            rel_y=0.25,
            rel_w=0.18,
            rel_h=0.24,
            frame_width=640,
            frame_height=480,
            focal_length_px=640.0,
            principal_x_px=320.0,
            real_face_width_m=0.15,
        )

        # Expected pixel conversions:
        # x_px = 0.32 * 640 = 204.8
        # w_px = 0.18 * 640 = 115.2
        # center_x_px = 204.8 + 115.2 / 2 = 262.4
        self.assertAlmostEqual(est.center_x_px, 262.4, places=4)

        # Expected depth: Z = (640 * 0.15) / 115.2 = 96 / 115.2 = 0.8333... m
        self.assertAlmostEqual(est.distance_m, 0.8333333333333334, places=4)
        self.assertEqual(round(est.distance_m, 2), 0.83)

        # Expected angle: theta = atan((262.4 - 320) / 640) = atan(-57.6 / 640) = atan(-0.09) = -5.1427... deg
        expected_theta = math.degrees(math.atan((262.4 - 320.0) / 640.0))
        self.assertAlmostEqual(est.angle_deg, expected_theta, places=4)
        self.assertEqual(round(est.angle_deg, 1), -5.1)

    def test_centered_face(self):
        """Face centred on optical axis should yield theta = 0 degrees."""
        # rel_x = 0.41, rel_w = 0.18 -> center_x = (0.41 + 0.09) * 640 = 320 = c_x
        est = estimate_from_box(
            index=1,
            rel_x=0.41,
            rel_y=0.25,
            rel_w=0.18,
            rel_h=0.24,
            frame_width=640,
            frame_height=480,
            focal_length_px=640.0,
            principal_x_px=320.0,
        )
        self.assertAlmostEqual(est.angle_deg, 0.0, places=5)

    def test_right_side_face(self):
        """Face to the right of optical axis should yield positive theta."""
        est = estimate_from_box(
            index=1,
            rel_x=0.60,
            rel_y=0.25,
            rel_w=0.18,
            rel_h=0.24,
            frame_width=640,
            frame_height=480,
            focal_length_px=640.0,
            principal_x_px=320.0,
        )
        self.assertGreater(est.angle_deg, 0.0)

    def test_farther_face_larger_depth(self):
        """A smaller bounding box width w_px should correspond to a larger depth Z."""
        near_est = estimate_from_box(
            index=1,
            rel_x=0.30,
            rel_y=0.20,
            rel_w=0.20,
            rel_h=0.20,
            frame_width=640,
            frame_height=480,
            focal_length_px=640.0,
        )
        far_est = estimate_from_box(
            index=2,
            rel_x=0.30,
            rel_y=0.20,
            rel_w=0.10,
            rel_h=0.10,
            frame_width=640,
            frame_height=480,
            focal_length_px=640.0,
        )
        self.assertGreater(far_est.distance_m, near_est.distance_m)

    def test_invalid_arguments_raise_exceptions(self):
        """Invalid inputs (non-positive width, focal length, real face width) must raise ValueError."""
        with self.assertRaises(ValueError):
            estimate_from_box(
                index=1,
                rel_x=0.1,
                rel_y=0.1,
                rel_w=0.0,
                rel_h=0.1,
                frame_width=640,
                frame_height=480,
                focal_length_px=640.0,
            )

        with self.assertRaises(ValueError):
            estimate_from_box(
                index=1,
                rel_x=0.1,
                rel_y=0.1,
                rel_w=0.1,
                rel_h=0.1,
                frame_width=640,
                frame_height=480,
                focal_length_px=-100.0,
            )

        with self.assertRaises(ValueError):
            estimate_from_box(
                index=1,
                rel_x=0.1,
                rel_y=0.1,
                rel_w=0.1,
                rel_h=0.1,
                frame_width=640,
                frame_height=480,
                focal_length_px=640.0,
                real_face_width_m=0.0,
            )

    def test_camera_params_loading(self):
        """Verify priority: CLI override > camera.json > default fallback."""
        # 1. CLI override
        focal, cx = load_camera_params(Path("nonexistent.json"), 640, 480, cli_focal=800.0)
        self.assertEqual(focal, 800.0)
        self.assertEqual(cx, 320.0)

        # 2. Config file
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
            json.dump({"f_x": 750.0, "c_x": 310.0}, tmp)
            tmp_path = Path(tmp.name)

        try:
            focal, cx = load_camera_params(tmp_path, 640, 480, cli_focal=None)
            self.assertEqual(focal, 750.0)
            self.assertEqual(cx, 310.0)

            # Test backwards compatibility helper
            legacy_focal = load_focal_length(tmp_path, 640, cli_focal=None)
            self.assertEqual(legacy_focal, 750.0)
        finally:
            tmp_path.unlink(missing_ok=True)

        # 3. Missing config file defaults
        focal, cx = load_camera_params(Path("missing_file.json"), 1280, 720, cli_focal=None)
        self.assertEqual(focal, 1280.0)
        self.assertEqual(cx, 640.0)

    def test_multi_face_cv_boxes(self):
        """Multiple bounding boxes should return independent estimates indexed from 1."""
        boxes = [(100, 100, 120, 120), (300, 150, 80, 80)]
        estimates = estimate_faces_from_cv_boxes(
            boxes=boxes,
            frame_width=640,
            frame_height=480,
            focal_length_px=640.0,
        )
        self.assertEqual(len(estimates), 2)
        self.assertEqual(estimates[0].index, 1)
        self.assertEqual(estimates[1].index, 2)
        self.assertGreater(estimates[1].distance_m, estimates[0].distance_m)


if __name__ == "__main__":
    unittest.main()
