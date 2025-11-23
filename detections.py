from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

@dataclass
class DetectionResult:
    """Container for a single detection and its age/gender estimate. Raw output, no formatting."""

    bbox: Tuple[int, int, int, int]
    confidence: float
    age_range: Optional[str]
    age_estimate: Optional[float]
    gender_label: Optional[str]
    class_label: Optional[str]
    source: str = "age_gender"

"""Text label for a detection, preferring gender labels over class labels"""
def detection_label_text(detection: DetectionResult) -> str:
    label_text = detection.gender_label or detection.class_label or "Person"
    if isinstance(label_text, str):
        return label_text.title()
    return "Person"

@dataclass
class DetectionSnapshot:
    """Container for a snapshot of a detection at a specific frame. For UI preview and summary (no DB metadata)."""

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
        """A factory that returns a new detection snapshot from a detection result"""

        return cls(
            frame=frame_idx,
            source="Age/Gender" if detection.source == "age_gender" else "Person",
            label=detection_label_text(detection),
            age_range=detection.age_range or "", #if None, use empty string instead of none to avoid extra null handling
            age_estimate=detection.age_estimate,
            confidence=detection.confidence,
            bbox_x=detection.bbox[0], 
            bbox_y=detection.bbox[1],
            bbox_w=detection.bbox[2],
            bbox_h=detection.bbox[3],
        )

    def to_row(self) -> Dict[str, Any]:     #Turns the dataclass into a plain dictionary. Used for UI preview and summary display.
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
    """Container for a single detection log entry with all metadata for database storage."""

    run_id: str
    logged_at: datetime
    frame_idx: int
    processed_frame: int    # The index of the processed frame (skipped frames not counted). For future flexibility.
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
        *,  # Safety guard to force keyword arguments. Arguements after cant be passed positionally so they dont accidentaly swap.
        run_id: str,
        detection: DetectionResult,
        logged_at: datetime,
        frame_idx: int,
        processed_frame: int,
        normalized_center: Tuple[float, float],
        frame_dimensions: Tuple[int, int],
    ) -> "DetectionLogEntry":
        """A factory that returns a new detection log entry from a detection result and metadata"""

        frame_width, frame_height = frame_dimensions    
        bbox_x, bbox_y, bbox_w, bbox_h = detection.bbox 
        return cls(
            run_id=run_id,
            logged_at=logged_at,
            frame_idx=frame_idx,
            processed_frame=processed_frame,
            source="Age/Gender" if detection.source == "age_gender" else "Person",
            label=detection_label_text(detection),
            gender=detection.gender_label.capitalize() if detection.gender_label else "Unknown",   # Unkown if None or empty 
            age_range=detection.age_range or "",    #if None, use empty string instead of none to avoid extra null handling
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

    def to_row(self) -> Dict[str, Any]:     # Turns the dataclass into a plain dictionary. Used for database insertion.
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

