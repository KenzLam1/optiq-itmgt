from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


@dataclass
class DetectionResult:
    """Container for a single detection and its age/gender estimate."""

    bbox: Tuple[int, int, int, int]
    confidence: float
    age_range: Optional[str]
    age_estimate: Optional[float]
    gender_label: Optional[str]
    class_label: Optional[str]
    source: str = "age_gender"


def detection_label_text(detection: DetectionResult) -> str:
    """Best-effort display label for UI and logs."""
    label_text = detection.gender_label or detection.class_label or "Person"
    if isinstance(label_text, str):
        return label_text.title()
    return "Person"


@dataclass
class DetectionSnapshot:
    frame: int
    source: str
    label: str
    age_range: str
    age_estimate: Optional[float]
    confidence: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int

    @classmethod
    def from_detection(cls, frame_idx: int, detection: DetectionResult) -> "DetectionSnapshot":
        return cls(
            frame=frame_idx,
            source="Age/Gender" if detection.source == "age_gender" else "Person",
            label=detection_label_text(detection),
            age_range=detection.age_range or "",
            age_estimate=detection.age_estimate,
            confidence=detection.confidence,
            bbox_x=detection.bbox[0],
            bbox_y=detection.bbox[1],
            bbox_w=detection.bbox[2],
            bbox_h=detection.bbox[3],
        )

    def to_row(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "source": self.source,
            "label": self.label,
            "age_range": self.age_range,
            "age_estimate": self.age_estimate,
            "confidence": self.confidence,
            "bbox_x": self.bbox_x,
            "bbox_y": self.bbox_y,
            "bbox_w": self.bbox_w,
            "bbox_h": self.bbox_h,
        }


@dataclass
class DetectionLogEntry:
    run_id: str
    logged_at: datetime
    frame_idx: int
    processed_frame: int
    source: str
    label: str
    gender: str
    age_range: str
    age_estimate: Optional[float]
    confidence: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    center_x: float
    center_y: float
    frame_width: int
    frame_height: int

    @classmethod
    def from_detection(
        cls,
        *,
        run_id: str,
        detection: DetectionResult,
        logged_at: datetime,
        frame_idx: int,
        processed_frame: int,
        normalized_center: Tuple[float, float],
        frame_dimensions: Tuple[int, int],
    ) -> "DetectionLogEntry":
        frame_width, frame_height = frame_dimensions
        bbox_x, bbox_y, bbox_w, bbox_h = detection.bbox
        return cls(
            run_id=run_id,
            logged_at=logged_at,
            frame_idx=frame_idx,
            processed_frame=processed_frame,
            source="Age/Gender" if detection.source == "age_gender" else "Person",
            label=detection_label_text(detection),
            gender=detection.gender_label.capitalize() if detection.gender_label else "Unknown",
            age_range=detection.age_range or "",
            age_estimate=detection.age_estimate,
            confidence=float(detection.confidence),
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            center_x=normalized_center[0],
            center_y=normalized_center[1],
            frame_width=frame_width,
            frame_height=frame_height,
        )

    def to_row(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "logged_at": self.logged_at,
            "frame_idx": self.frame_idx,
            "processed_frame": self.processed_frame,
            "source": self.source,
            "label": self.label,
            "gender": self.gender,
            "age_range": self.age_range,
            "age_estimate": self.age_estimate,
            "confidence": self.confidence,
            "bbox_x": self.bbox_x,
            "bbox_y": self.bbox_y,
            "bbox_w": self.bbox_w,
            "bbox_h": self.bbox_h,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }

