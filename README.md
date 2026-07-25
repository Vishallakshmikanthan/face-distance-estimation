# Monocular Face Distance Estimation

This project estimates a face's distance from a single 2D camera and its horizontal angular deviation from the camera optical axis.

## Files

- `face_distance.py`: real-time camera demo with face box, distance, and angle overlay.
- `calibrate.py`: optional checkerboard camera calibration script.
- `camera.json`: focal length storage used by the demo.
- `requirements.txt`: Python dependencies.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

MediaPipe currently has stricter Python wheel support than OpenCV. On newer Python versions, the app automatically falls back to OpenCV's Haar face detector so installation still succeeds. For the MediaPipe detector, use a Python version supported by MediaPipe and run the same install command.

## Run

```powershell
python face_distance.py
```

Press `q` or `Esc` to exit.

Useful options:

```powershell
python face_distance.py --camera 1
python face_distance.py --detector haar
python face_distance.py --detector mediapipe
python face_distance.py --focal-length 1280
python face_distance.py --face-width 0.15
```

If no calibrated focal length is supplied, the demo uses `camera.json`. If `camera.json` is missing or does not contain a valid focal length, it falls back to `f = frame_width`.

## Math

The pinhole camera model is:

```text
u = f * (X / Z) + c_x
v = f * (Y / Z) + c_y
```

For a face with real width `W` metres and detected bounding-box width `w_px` pixels:

```text
Z = (f * W) / w_px
```

This project uses `W = 0.15 m` by default.

For horizontal angle, using the face centre `x`:

```text
theta = atan((x - c_x) / f)
```

The displayed angle is converted to degrees. Positive means the face is to the right of image centre; negative means it is to the left.

## Calibration

Print a checkerboard with `9 x 6` inner corners. Measure the square size in metres, then run:

```powershell
python calibrate.py --square-size 0.024
```

Move the checkerboard through different positions and angles. Press `Space` when corners are detected. The script writes `camera.json` with `f_x`, `f_y`, principal point, distortion coefficients, and reprojection error.

## Validation Table

Fill this with measurements from your own camera setup.

| True Z (m) | Measured Z (m) | Error (m) | Theta at centre (deg) |
|---:|---:|---:|---:|
| 0.50 |  |  | approx 0 |
| 1.00 |  |  | approx 0 |
| 1.50 |  |  | approx 0 |
| 2.00 |  |  | approx 0 |
| 2.50 |  |  | approx 0 |

## Notes

- Accuracy depends mostly on focal length and the assumed real face width.
- Calibration usually improves distance accuracy significantly.
- Multiple faces are handled independently; each detected face gets its own distance and angle.
