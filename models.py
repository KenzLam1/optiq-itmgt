import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import supervision as sv

from detections import DetectionResult

logger = logging.getLogger(__name__)


def _select_device(explicit_device: Optional[str]) -> str:
    """Resolve the best runtime device, falling back gracefully."""
    if explicit_device:
        return explicit_device

    try:
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:  # noqa: BLE001
        pass

    try:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass

    return "cpu"


def _clip_sv_detections(
    detections: sv.Detections, frame_height: int, frame_width: int
) -> sv.Detections:
    """Clamp detection boxes to the frame bounds for older supervision builds."""
    if len(detections) == 0:
        return detections

    clipped = detections.xyxy.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, frame_width - 1)
    clipped[:, 2] = np.clip(clipped[:, 2], 0, frame_width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, frame_height - 1)
    clipped[:, 3] = np.clip(clipped[:, 3], 0, frame_height - 1)
    detections.xyxy = clipped
    return detections


class YOLOAgeGenderDetector:
    """Ultralytics YOLO model that predicts age/gender-specific detections."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        conf_threshold: float = 0.4,
        imgsz: int = 640,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "Ultralytics is not installed. Install dependencies with 'pip install -r requirements.txt'."
            ) from exc

        if not Path(model_path).exists():
            raise RuntimeError(
                f"Age/gender detector weights not found at '{model_path}'. "
                "Place the model file alongside main.py or provide an absolute path."
            )

        self.device = _select_device(device)
        self.model = YOLO(model_path)

        try:
            self.model.to(self.device)
        except AttributeError:
            self.model.model.to(self.device)

        logger.info("Age/Gender detector initialised on device '%s'", self.device)
        self.names = getattr(self.model.model, "names", getattr(self.model, "names", {}))
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    @staticmethod
    def _parse_label(label: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not label:
            return None, None

        cleaned = label.replace("AGE-", "").replace("AGE", "").strip()
        parts = cleaned.split()
        if not parts:
            return None, None

        gender = parts[-1].lower()
        age_range = " ".join(parts[:-1]).strip() or None
        return age_range, gender

    @staticmethod
    def _estimate_age(age_range: Optional[str]) -> Optional[float]:
        if not age_range:
            return None

        bounds = age_range.replace("+", "-").split("-")
        bounds = [b for b in bounds if b]
        try:
            if len(bounds) == 2:
                low = float(bounds[0])
                high = float(bounds[1])
                return (low + high) / 2.0
            if len(bounds) == 1:
                low = float(bounds[0])
                return low + 10.0  # assume a 10-year span for open-ended ranges
        except ValueError:
            return None

        return None

    def predict(self, frame: np.ndarray) -> List[DetectionResult]:
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[DetectionResult] = []
        if not results:
            return detections

        frame_height, frame_width = frame.shape[:2]
        result = results[0]
        sv_detections = sv.Detections.from_ultralytics(result)
        if len(sv_detections) == 0:
            return detections

        sv_detections = _clip_sv_detections(
            sv_detections, frame_height=frame_height, frame_width=frame_width
        )
        class_ids = (
            sv_detections.class_id.astype(int).tolist()
            if sv_detections.class_id is not None
            else [None] * len(sv_detections)
        )

        age_ranges: List[Optional[str]] = []
        genders: List[Optional[str]] = []
        age_estimates: List[Optional[float]] = []
        class_labels: List[Optional[str]] = []

        for xyxy, conf, cls_id in zip(
            sv_detections.xyxy, sv_detections.confidence, class_ids
        ):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            score = float(conf) if conf is not None else 0.0

            if isinstance(self.names, dict):
                raw_label = self.names.get(cls_id)
            else:
                raw_label = self.names[cls_id] if cls_id is not None and 0 <= cls_id < len(self.names) else None

            age_range, gender = self._parse_label(raw_label)
            age_estimate = self._estimate_age(age_range)
            capitalized_gender = gender.capitalize() if gender else None

            age_ranges.append(age_range)
            genders.append(capitalized_gender)
            age_estimates.append(age_estimate)
            class_labels.append(raw_label)

            detections.append(
                DetectionResult(
                    bbox=(x1, y1, w, h),
                    confidence=score,
                    age_range=age_range,
                    age_estimate=age_estimate,
                    gender_label=capitalized_gender,
                    class_label=raw_label,
                    source="age_gender",
                )
            )

        sv_detections.data["age_range"] = age_ranges
        sv_detections.data["gender_label"] = genders
        sv_detections.data["age_estimate"] = age_estimates
        sv_detections.data["class_label"] = class_labels

        return detections


class YOLOPersonDetector:
    """Separate YOLO model that focuses on generic person detections."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
        imgsz: int = 640,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "Ultralytics is not installed. Install dependencies with 'pip install -r requirements.txt'."
            ) from exc

        if not Path(model_path).exists():
            raise RuntimeError(
                f"Person detector weights not found at '{model_path}'. "
                "Download a YOLO model (for example yolov8n.pt) and provide its path."
            )

        self.device = _select_device(device)
        self.model = YOLO(model_path)

        try:
            self.model.to(self.device)
        except AttributeError:
            self.model.model.to(self.device)

        logger.info("Person detector initialised on device '%s'", self.device)
        self.names = getattr(self.model.model, "names", getattr(self.model, "names", {}))
        self.person_class_ids = self._resolve_person_class_ids()
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    def _resolve_person_class_ids(self) -> List[int]:
        ids: List[int] = []
        if isinstance(self.names, dict):
            for idx, label in self.names.items():
                if isinstance(label, str) and label.lower() == "person":
                    ids.append(int(idx))
        elif isinstance(self.names, list):
            for idx, label in enumerate(self.names):
                if isinstance(label, str) and label.lower() == "person":
                    ids.append(idx)
        return ids or [0]

    def predict(self, frame: np.ndarray) -> List[DetectionResult]:
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[DetectionResult] = []
        if not results:
            return detections

        frame_height, frame_width = frame.shape[:2]
        result = results[0]
        sv_detections = sv.Detections.from_ultralytics(result)
        if len(sv_detections) == 0:
            return detections

        sv_detections = _clip_sv_detections(
            sv_detections, frame_height=frame_height, frame_width=frame_width
        )
        if sv_detections.class_id is not None:
            mask = np.isin(sv_detections.class_id.astype(int), self.person_class_ids)
            sv_detections = sv_detections[mask]
        if len(sv_detections) == 0:
            return detections

        class_ids = (
            sv_detections.class_id.astype(int).tolist()
            if sv_detections.class_id is not None
            else [None] * len(sv_detections)
        )

        for xyxy, conf, cls_id in zip(
            sv_detections.xyxy, sv_detections.confidence, class_ids
        ):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            score = float(conf) if conf is not None else 0.0

            if isinstance(self.names, dict):
                raw_label = self.names.get(cls_id, "Person")
            else:
                raw_label = (
                    self.names[cls_id] if cls_id is not None and 0 <= cls_id < len(self.names) else "Person"
                )

            label_text = raw_label or "Person"
            if isinstance(label_text, str) and label_text.islower():
                label_text = label_text.capitalize()

            detections.append(
                DetectionResult(
                    bbox=(x1, y1, w, h),
                    confidence=score,
                    age_range=None,
                    age_estimate=None,
                    gender_label=None,
                    class_label=label_text,
                    source="person",
                )
            )

        return detections
