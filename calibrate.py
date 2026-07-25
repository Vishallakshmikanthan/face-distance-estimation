"""Calibrate camera focal length with a checkerboard and save camera.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect_checkerboard_frames(args: argparse.Namespace):
    import cv2
    import numpy as np

    pattern_size = (args.columns, args.rows)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    objp = np.zeros((args.rows * args.columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0 : args.columns, 0 : args.rows].T.reshape(-1, 2)
    objp *= args.square_size

    object_points = []
    image_points = []

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    print("Press SPACE when the checkerboard is detected. Press q or Esc to finish.")
    print(f"Target samples: {args.samples}")

    image_size = None
    try:
        while len(object_points) < args.samples:
            ok, frame = cap.read()
            if not ok:
                print("No frame received from camera; stopping.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            preview = frame.copy()
            if found:
                refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(preview, pattern_size, refined, found)
                status = f"Detected - samples {len(object_points)}/{args.samples}"
            else:
                refined = None
                status = f"Not detected - samples {len(object_points)}/{args.samples}"

            cv2.putText(
                preview,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if found else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Camera Calibration", preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == 32 and found and refined is not None:
                object_points.append(objp.copy())
                image_points.append(refined)
                print(f"Captured sample {len(object_points)}/{args.samples}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if image_size is None:
        raise SystemExit("No camera frames were captured.")
    if len(object_points) < args.min_samples:
        raise SystemExit(
            f"Need at least {args.min_samples} valid samples; captured {len(object_points)}."
        )

    return object_points, image_points, image_size


def calibrate(args: argparse.Namespace) -> dict:
    import cv2

    object_points, image_points, image_size = collect_checkerboard_frames(args)
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    f_x = float(camera_matrix[0, 0])
    f_y = float(camera_matrix[1, 1])
    c_x = float(camera_matrix[0, 2])
    c_y = float(camera_matrix[1, 2])

    return {
        "focal_length_px": f_x,
        "f_x": f_x,
        "f_y": f_y,
        "c_x": c_x,
        "c_y": c_y,
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "rms_reprojection_error": float(rms),
        "distortion_coefficients": dist_coeffs.ravel().astype(float).tolist(),
        "checkerboard_inner_corners": [args.columns, args.rows],
        "square_size_m": args.square_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a camera from a 9x6 inner-corner checkerboard."
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--columns", type=int, default=9, help="Checkerboard inner corners across.")
    parser.add_argument("--rows", type=int, default=6, help="Checkerboard inner corners down.")
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.024,
        help="Checkerboard square size in metres.",
    )
    parser.add_argument("--samples", type=int, default=15, help="Number of frames to capture.")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=8,
        help="Minimum usable frames required for calibration.",
    )
    parser.add_argument("--output", type=Path, default=Path("camera.json"), help="Output JSON file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.columns <= 0 or args.rows <= 0:
        raise SystemExit("Checkerboard rows and columns must be positive.")
    if args.square_size <= 0:
        raise SystemExit("--square-size must be positive.")
    if args.samples < args.min_samples:
        raise SystemExit("--samples must be greater than or equal to --min-samples.")

    data = calibrate(args)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Saved calibration to {args.output}")
    print(f"f_x = {data['f_x']:.2f}px, f_y = {data['f_y']:.2f}px")
    print(f"c_x = {data['c_x']:.2f}px, c_y = {data['c_y']:.2f}px")
    print(f"RMS reprojection error = {data['rms_reprojection_error']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
