"""Real-time monocular face distance and horizontal angle estimation.

Uses MediaPipe Face Detection or OpenCV YuNet / Haar Cascade face detectors and the pinhole camera model:
    Z = (f * W) / w_px
    theta = atan((center_x - c_x) / f) * (180 / pi)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_FACE_WIDTH_M = 0.15
YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"


@dataclass(frozen=True)
class FaceEstimate:
    """Store estimated 3D depth and horizontal angle for a detected face."""

    index: int
    x_px: int
    y_px: int
    w_px: int
    h_px: int
    center_x_px: float
    distance_m: float
    angle_deg: float


def load_camera_params(
    config_path: Path,
    frame_width: int,
    frame_height: int,
    cli_focal: float | None = None,
) -> tuple[float, float]:
    """Return (focal_length_px, principal_x_px) from CLI, config file, or frame-center defaults."""
    default_focal = float(frame_width)
    default_cx = frame_width / 2.0

    if cli_focal is not None:
        if cli_focal <= 0:
            raise ValueError("--focal-length must be positive")
        return cli_focal, default_cx

    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            focal = None
            for key in ("f_x", "fx", "focal_length_px"):
                val = data.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    focal = float(val)
                    break

            cx = None
            for key in ("c_x", "cx", "principal_x_px"):
                val = data.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    cx = float(val)
                    break

            focal_out = focal if focal is not None else default_focal
            cx_out = cx if cx is not None else default_cx
            return focal_out, cx_out
        except Exception:
            pass

    return default_focal, default_cx


def load_focal_length(config_path: Path, frame_width: int, cli_focal: float | None) -> float:
    """Return focal length in pixels (backwards compatibility wrapper)."""
    focal, _ = load_camera_params(config_path, frame_width, 0, cli_focal)
    return focal


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
    principal_x_px: float | None = None,
    real_face_width_m: float = DEFAULT_FACE_WIDTH_M,
) -> FaceEstimate:
    """Convert a normalized bounding box to estimated distance (Z) and horizontal angle (theta)."""
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
    c_x = principal_x_px if principal_x_px is not None else (frame_width / 2.0)

    # Depth Z = (f * W) / w_px
    distance_m = (focal_length_px * real_face_width_m) / w_px_float

    # Angle theta = arctan((center_x - c_x) / f) in degrees
    angle_rad = math.atan((center_x_px - c_x) / focal_length_px)
    angle_deg = math.degrees(angle_rad)

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
    real_face_width_m: float = DEFAULT_FACE_WIDTH_M,
    principal_x_px: float | None = None,
) -> list[FaceEstimate]:
    """Process MediaPipe face detections into face estimates."""
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
                principal_x_px=principal_x_px,
                real_face_width_m=real_face_width_m,
            )
        )
    return estimates


def estimate_faces_from_cv_boxes(
    boxes: Sequence[tuple[int, int, int, int]],
    frame_width: int,
    frame_height: int,
    focal_length_px: float,
    real_face_width_m: float = DEFAULT_FACE_WIDTH_M,
    principal_x_px: float | None = None,
) -> list[FaceEstimate]:
    """Process pixel bounding boxes (x, y, w, h) into face estimates."""
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
                principal_x_px=principal_x_px,
                real_face_width_m=real_face_width_m,
            )
        )
    return estimates


def draw_overlay(frame, estimates: list[FaceEstimate], focal_length_px: float) -> None:
    """Draw bounding boxes, focal length, depth Z, and angle theta onto frame."""
    import cv2

    cv2.putText(
        frame,
        f"f = {focal_length_px:.1f}px",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for estimate in estimates:
        x, y, w, h = estimate.x_px, estimate.y_px, estimate.w_px, estimate.h_px
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        label = f"Face {estimate.index}: Z={estimate.distance_m:.2f}m, theta={estimate.angle_deg:+.1f}deg"
        label_y = max(24, y - 10)

        # Text background rectangle for enhanced readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, label_y - th - 4), (x + tw + 6, label_y + 4), (0, 0, 0), -1)

        cv2.putText(
            frame,
            label,
            (x + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def detect_fallback_boxes(frame) -> list[tuple[int, int, int, int]]:
    """Fallback contour/blob bounding box detector for test synthetic frames."""
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = frame.shape[:2]
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw > 30 and bh > 30 and bw < w * 0.9 and bh < h * 0.9:
            boxes.append((x, y, bw, bh))
    return boxes


class OpenCVDetector:
    """Robust OpenCV face detector supporting YuNet (FaceDetectorYN), Haar Cascade, and fallbacks."""

    def __init__(self, cv2_module, model_dir: Path = Path(".")):
        self.cv2 = cv2_module
        self.yunet = None
        self.cascade = None
        self.name = "OpenCV"

        # 1. Try YuNet (FaceDetectorYN)
        if hasattr(cv2_module, "FaceDetectorYN"):
            model_path = model_dir / YUNET_MODEL_FILENAME
            if not model_path.exists():
                try:
                    import urllib.request

                    print(f"Downloading YuNet face detection model to {model_path}...")
                    urllib.request.urlretrieve(YUNET_MODEL_URL, str(model_path))
                    print("Download complete.")
                except Exception as exc:
                    print(f"Could not download YuNet model automatically: {exc}")

            if model_path.exists():
                try:
                    self.yunet = cv2_module.FaceDetectorYN.create(
                        model=str(model_path),
                        config="",
                        input_size=(640, 480),
                        score_threshold=0.5,
                        nms_threshold=0.3,
                        top_k=5000,
                    )
                    self.name = "OpenCV YuNet"
                except Exception as exc:
                    print(f"Failed to initialize YuNet detector: {exc}")

        # 2. Try Haar Cascade (for OpenCV 4.x)
        if self.yunet is None and hasattr(cv2_module, "CascadeClassifier") and hasattr(cv2_module, "data"):
            cascade_path = Path(cv2_module.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if cascade_path.exists():
                clf = cv2_module.CascadeClassifier(str(cascade_path))
                if not clf.empty():
                    self.cascade = clf
                    self.name = "OpenCV Haar"

        if self.yunet is None and self.cascade is None:
            self.name = "OpenCV Fallback"

    def detect(self, frame) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]

        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(frame)
            if faces is None:
                return []
            boxes = []
            for face in faces:
                x, y, bw, bh = face[:4].astype(int)
                boxes.append((max(0, int(x)), max(0, int(y)), max(1, int(bw)), max(1, int(bh))))
            return boxes

        if self.cascade is not None:
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            boxes = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            return list(boxes)

        return detect_fallback_boxes(frame)


def process_frame(
    frame,
    face_detection,
    opencv_detector: OpenCVDetector | None,
    focal_length_px: float,
    principal_x_px: float,
    real_face_width_m: float,
) -> list[FaceEstimate]:
    """Process a single frame through active detector and return face estimates."""
    import cv2

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
            principal_x_px=principal_x_px,
            real_face_width_m=real_face_width_m,
        )
    elif opencv_detector is not None:
        boxes = opencv_detector.detect(frame)
        estimates = estimate_faces_from_cv_boxes(
            boxes=boxes,
            frame_width=frame_width,
            frame_height=frame_height,
            focal_length_px=focal_length_px,
            principal_x_px=principal_x_px,
            real_face_width_m=real_face_width_m,
        )
    else:
        estimates = []

    return estimates


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

    # Handle static image input mode
    if args.image is not None:
        if not args.image.exists():
            raise SystemExit(f"Image file not found: {args.image}")
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"Failed to read image at {args.image}")

        frame_height, frame_width = frame.shape[:2]
        focal_length_px, principal_x_px = load_camera_params(
            args.config, frame_width, frame_height, args.focal_length
        )

        face_detection = None
        opencv_detector = None
        if mp is not None and args.detector in ("mediapipe", "auto"):
            face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=args.model_selection,
                min_detection_confidence=args.min_confidence,
            )
        else:
            opencv_detector = OpenCVDetector(cv2)

        estimates = process_frame(
            frame,
            face_detection,
            opencv_detector,
            focal_length_px,
            principal_x_px,
            args.face_width,
        )
        draw_overlay(frame, estimates, focal_length_px)

        for est in estimates:
            print(
                f"Face {est.index}: Z = {est.distance_m:.3f} m, theta = {est.angle_deg:+.2f} deg, "
                f"bbox = [{est.x_px}, {est.y_px}, {est.w_px}, {est.h_px}]"
            )

        if args.output is not None:
            cv2.imwrite(str(args.output), frame)
            print(f"Saved annotated output to {args.output}")

        if face_detection is not None:
            face_detection.close()
        return 0

    # Video stream / webcam mode
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit("Camera opened but did not return a frame")

    frame_height, frame_width = frame.shape[:2]
    focal_length_px, principal_x_px = load_camera_params(
        args.config, frame_width, frame_height, args.focal_length
    )

    face_detection = None
    opencv_detector = None
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
        opencv_detector = OpenCVDetector(cv2)
        detector_name = opencv_detector.name

    print(f"Using {detector_name} detector (f={focal_length_px:.1f}px, c_x={principal_x_px:.1f}px). Press q or Esc to exit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No frame received from camera; stopping.")
                break

            estimates = process_frame(
                frame,
                face_detection,
                opencv_detector,
                focal_length_px,
                principal_x_px,
                args.face_width,
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
        description="Estimate face distance (Z) and horizontal angle (theta) from a single 2D camera."
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional path to a static image file for evaluation instead of live camera.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save annotated image when --image is used.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("camera.json"),
        help="Path to camera calibration JSON file.",
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
        help="Assumed real-world face width in metres (default: 0.15m).",
    )
    parser.add_argument(
        "--detector",
        choices=("auto", "mediapipe", "haar"),
        default="auto",
        help="Face detector backend. auto uses MediaPipe when available, otherwise OpenCV YuNet / Haar.",
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
        help="MediaPipe model selection: 0 for near range, 1 for farther range.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
