"""Real-time monocular face distance and horizontal angle estimation.

Uses MediaPipe Face Detection for 2D face boxes and the pinhole camera model:
    Z = (f * W) / w_px
    theta = atan((face_cx - c_x) / f)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_FACE_WIDTH_M = 0.15


@dataclass(frozen=True)
class FaceEstimate:
    index: int
    x_px: int
    y_px: int
    w_px: int
    h_px: int
    center_x_px: float
    distance_m: float
    angle_deg: float


def load_focal_length(config_path: Path, frame_width: int, cli_focal: float | None) -> float:
    """Return focal length in pixels from CLI, camera.json, or frame-width fallback."""
    if cli_focal is not None:
        if cli_focal <= 0:
            raise ValueError("--focal-length must be positive")
        return cli_focal

    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("f_x", "fx", "focal_length_px"):
            value = data.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)

    return float(frame_width)


def estimate_from_box(
    *,
    index: int,
    rel_x: float,
    rel_y: float,
    rel_w: float,
    rel_h: float,
    frame_width: int,
    frame_height: int,
    focal_length_px: float,
    real_face_width_m: float = DEFAULT_FACE_WIDTH_M,
) -> FaceEstimate:
    """Convert a normalized MediaPipe bounding box to distance and angle."""
    if rel_w <= 0:
        raise ValueError("Face bounding box width must be positive")
    if focal_length_px <= 0:
        raise ValueError("Focal length must be positive")
    if real_face_width_m <= 0:
        raise ValueError("Real face width must be positive")

    x_px_float = rel_x * frame_width
    y_px_float = rel_y * frame_height
    w_px_float = rel_w * frame_width
    h_px_float = rel_h * frame_height

    center_x_px = x_px_float + (w_px_float / 2.0)
    principal_x = frame_width / 2.0
    distance_m = (focal_length_px * real_face_width_m) / w_px_float
    angle_deg = math.degrees(math.atan((center_x_px - principal_x) / focal_length_px))

    return FaceEstimate(
        index=index,
        x_px=max(0, int(round(x_px_float))),
        y_px=max(0, int(round(y_px_float))),
        w_px=max(1, int(round(w_px_float))),
        h_px=max(1, int(round(h_px_float))),
        center_x_px=center_x_px,
        distance_m=distance_m,
        angle_deg=angle_deg,
    )


def estimate_faces(
    detections: Iterable[object],
    frame_width: int,
    frame_height: int,
    focal_length_px: float,
    real_face_width_m: float,
) -> list[FaceEstimate]:
    estimates: list[FaceEstimate] = []
    for index, detection in enumerate(detections, start=1):
        rel_box = detection.location_data.relative_bounding_box
        estimates.append(
            estimate_from_box(
                index=index,
                rel_x=rel_box.xmin,
                rel_y=rel_box.ymin,
                rel_w=rel_box.width,
                rel_h=rel_box.height,
                frame_width=frame_width,
                frame_height=frame_height,
                focal_length_px=focal_length_px,
                real_face_width_m=real_face_width_m,
            )
        )
    return estimates


def estimate_faces_from_cv_boxes(
    boxes,
    frame_width: int,
    frame_height: int,
    focal_length_px: float,
    real_face_width_m: float,
) -> list[FaceEstimate]:
    estimates: list[FaceEstimate] = []
    for index, (x, y, w, h) in enumerate(boxes, start=1):
        estimates.append(
            estimate_from_box(
                index=index,
                rel_x=float(x) / frame_width,
                rel_y=float(y) / frame_height,
                rel_w=float(w) / frame_width,
                rel_h=float(h) / frame_height,
                frame_width=frame_width,
                frame_height=frame_height,
                focal_length_px=focal_length_px,
                real_face_width_m=real_face_width_m,
            )
        )
    return estimates


def draw_overlay(frame, estimates: list[FaceEstimate], focal_length_px: float) -> None:
    import cv2

    cv2.putText(
        frame,
        f"f = {focal_length_px:.1f}px",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for estimate in estimates:
        x, y, w, h = estimate.x_px, estimate.y_px, estimate.w_px, estimate.h_px
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)

        label = f"Face {estimate.index}: Z={estimate.distance_m:.2f}m, theta={estimate.angle_deg:+.1f}deg"
        label_y = max(24, y - 10)
        cv2.putText(
            frame,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )


def run(args: argparse.Namespace) -> int:
    try:
        import cv2
    except ImportError as exc:
        missing = exc.name or "required package"
        raise SystemExit(
            f"Missing dependency: {missing}. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    try:
        import mediapipe as mp
    except ImportError:
        mp = None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("Camera opened but did not return a frame")

    frame_height, frame_width = frame.shape[:2]
    focal_length_px = load_focal_length(args.config, frame_width, args.focal_length)

    face_detection = None
    haar_detector = None
    if mp is not None and args.detector in ("mediapipe", "auto"):
        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=args.model_selection,
            min_detection_confidence=args.min_confidence,
        )
        detector_name = "MediaPipe"
    elif args.detector == "mediapipe":
        cap.release()
        raise SystemExit(
            "MediaPipe is not installed for this Python. Use --detector haar or install a Python version supported by MediaPipe."
        )
    else:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        haar_detector = cv2.CascadeClassifier(str(cascade_path))
        if haar_detector.empty():
            cap.release()
            raise SystemExit(f"Could not load Haar cascade at {cascade_path}")
        detector_name = "OpenCV Haar"

    print(f"Using {detector_name} detector. Press q or Esc to exit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No frame received from camera; stopping.")
                break

            frame_height, frame_width = frame.shape[:2]
            if face_detection is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)
                detections = results.detections or []
                estimates = estimate_faces(
                    detections=detections,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    focal_length_px=focal_length_px,
                    real_face_width_m=args.face_width,
                )
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                boxes = haar_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(40, 40),
                )
                estimates = estimate_faces_from_cv_boxes(
                    boxes=boxes,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    focal_length_px=focal_length_px,
                    real_face_width_m=args.face_width,
                )
            draw_overlay(frame, estimates, focal_length_px)

            cv2.imshow("Monocular Face Distance Estimation", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        if face_detection is not None:
            face_detection.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate face distance and horizontal angle from a single camera."
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("camera.json"),
        help="Path to camera calibration JSON.",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=None,
        help="Override focal length in pixels. Defaults to camera.json or frame width.",
    )
    parser.add_argument(
        "--face-width",
        type=float,
        default=DEFAULT_FACE_WIDTH_M,
        help="Assumed real face width in metres.",
    )
    parser.add_argument(
        "--detector",
        choices=("auto", "mediapipe", "haar"),
        default="auto",
        help="Face detector backend. auto uses MediaPipe when available, otherwise OpenCV Haar.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="MediaPipe minimum detection confidence.",
    )
    parser.add_argument(
        "--model-selection",
        type=int,
        choices=(0, 1),
        default=0,
        help="MediaPipe model: 0 for near range, 1 for farther range.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
