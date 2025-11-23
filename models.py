import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import supervision as sv

from detections import DetectionResult
from hardware import select_runtime_device

logger = logging.getLogger(__name__)


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


class BaseYOLODetector(ABC):
    """Shared runtime for Ultralytics detectors used in the dashboard."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str],
        conf_threshold: float,
        imgsz: int,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = select_runtime_device(device)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.model = self._load_model()
        self.names = getattr(self.model.model, "names", getattr(self.model, "names", {}))

    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "Ultralytics is not installed. Install dependencies with 'pip install -r requirements.txt'."
            ) from exc

        if not self.model_path.exists():
            raise RuntimeError(self._missing_weights_message())

        model = YOLO(str(self.model_path))
        try:
            model.to(self.device)
        except AttributeError:
            model.model.to(self.device)

        logger.info("%s initialised on device '%s'", self.__class__.__name__, self.device)
        return model

    def predict(self, frame: np.ndarray) -> List[DetectionResult]:
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        frame_height, frame_width = frame.shape[:2]
        result = results[0]
        sv_detections = sv.Detections.from_ultralytics(result)
        if len(sv_detections) == 0:
            return []

        sv_detections = _clip_sv_detections(
            sv_detections, frame_height=frame_height, frame_width=frame_width
        )
        return self._build_detections(sv_detections, frame_height, frame_width)

    def _iterate_raw_detections(
        self, sv_detections: sv.Detections
    ) -> Iterable[Tuple[np.ndarray, float, Optional[int]]]:
        class_ids = (
            sv_detections.class_id.astype(int).tolist()
            if sv_detections.class_id is not None
            else [None] * len(sv_detections)
        )
        for xyxy, conf, cls_id in zip(sv_detections.xyxy, sv_detections.confidence, class_ids):
            yield xyxy, float(conf) if conf is not None else 0.0, cls_id

    @abstractmethod
    def _missing_weights_message(self) -> str:
        """Return a user-friendly error if weights are missing."""

    @abstractmethod
    def _build_detections(
        self, sv_detections: sv.Detections, frame_height: int, frame_width: int
    ) -> List[DetectionResult]:
        """Convert supervision detections to dashboard DetectionResult objects."""


class YOLOAgeGenderDetector(BaseYOLODetector):
    """Ultralytics YOLO model that predicts age/gender-specific detections."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        conf_threshold: float = 0.4,
        imgsz: int = 640,
    ):
        super().__init__(model_path, device, conf_threshold, imgsz)

    def _missing_weights_message(self) -> str:
        return (
            f"Age/gender detector weights not found at '{self.model_path}'. "
            "Place the model file alongside main.py or provide an absolute path."
        )

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

    def _build_detections(
        self, sv_detections: sv.Detections, frame_height: int, frame_width: int
    ) -> List[DetectionResult]:
        detections: List[DetectionResult] = []
        age_ranges: List[Optional[str]] = []
        genders: List[Optional[str]] = []
        age_estimates: List[Optional[float]] = []
        class_labels: List[Optional[str]] = []

        for xyxy, score, cls_id in self._iterate_raw_detections(sv_detections):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

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


class YOLOPersonDetector(BaseYOLODetector):
    """Separate YOLO model that focuses on generic person detections."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
        imgsz: int = 640,
    ):
        super().__init__(model_path, device, conf_threshold, imgsz)
        self.person_class_ids = self._resolve_person_class_ids()

    def _missing_weights_message(self) -> str:
        return (
            f"Person detector weights not found at '{self.model_path}'. "
            "Download a YOLO model (for example yolov8n.pt) and provide its path."
        )

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

    def _build_detections(
        self, sv_detections: sv.Detections, frame_height: int, frame_width: int
    ) -> List[DetectionResult]:
        if sv_detections.class_id is not None:
            mask = np.isin(sv_detections.class_id.astype(int), self.person_class_ids)
            sv_detections = sv_detections[mask]
        if len(sv_detections) == 0:
            return []

        detections: List[DetectionResult] = []
        for xyxy, score, cls_id in self._iterate_raw_detections(sv_detections):
            x1, y1, x2, y2 = map(int, xyxy.tolist())
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

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
