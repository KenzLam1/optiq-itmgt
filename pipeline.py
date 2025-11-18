from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import supervision as sv

from detections import DetectionResult
from models import YOLOAgeGenderDetector, YOLOPersonDetector


class VisionPipeline:
    """Runs YOLO detections on incoming frames."""

    def __init__(
        self,
        model_path: str,
        person_model_path: str,
        device: Optional[str] = None,
        enable_age_detector: bool = True,
        enable_person_detector: bool = True,
        age_conf_threshold: float = 0.4,
        person_conf_threshold: float = 0.35,
        imgsz: int = 640,
    ):
        self.age_detector: Optional[YOLOAgeGenderDetector] = None
        if enable_age_detector:
            self.age_detector = YOLOAgeGenderDetector(
                model_path,
                device=device,
                conf_threshold=age_conf_threshold,
                imgsz=imgsz,
            )
        self.person_detector: Optional[YOLOPersonDetector] = None
        if enable_person_detector:
            self.person_detector = YOLOPersonDetector(
                person_model_path,
                device=device,
                conf_threshold=person_conf_threshold,
                imgsz=imgsz,
            )
        self.frame_interval = 1
        self._box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        self._label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

    def set_frame_interval(self, interval: int) -> None:
        self.frame_interval = max(1, interval)

    def process(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, List[DetectionResult]]:
        if frame_idx % self.frame_interval != 0:
            return frame, []

        if self.age_detector is None and self.person_detector is None:
            return frame, []

        age_detections: List[DetectionResult] = []
        if self.age_detector is not None:
            age_detections = self.age_detector.predict(frame)
        person_detections: List[DetectionResult] = []
        if self.person_detector is not None:
            person_detections = self.person_detector.predict(frame)
        detections: List[DetectionResult] = []
        if age_detections:
            detections.extend(age_detections)

        filtered_person_detections = self._filter_overlap(
            person_detections, age_detections, iou_threshold=0.35
        )
        detections.extend(filtered_person_detections)

        annotated = frame.copy()
        if detections:
            sv_detections = self._results_to_sv_detections(detections)
            labels = [self._format_label(det) for det in detections]
            annotated = self._box_annotator.annotate(
                scene=annotated, detections=sv_detections
            )
            annotated = self._label_annotator.annotate(
                scene=annotated, detections=sv_detections, labels=labels
            )

        return annotated, detections

    def _filter_overlap(
        self,
        person_detections: List[DetectionResult],
        age_detections: List[DetectionResult],
        iou_threshold: float,
    ) -> List[DetectionResult]:
        if not person_detections:
            return []
        if not age_detections:
            return person_detections

        person_boxes = self._results_to_sv_detections(person_detections).xyxy
        age_boxes = self._results_to_sv_detections(age_detections).xyxy
        iou_matrix = self._pairwise_iou(person_boxes, age_boxes)
        suppress_mask = (iou_matrix.max(axis=1) <= iou_threshold) if iou_matrix.size else np.ones(len(person_detections), dtype=bool)

        return [
            det for det, keep in zip(person_detections, suppress_mask) if keep
        ]

    @staticmethod
    def _pairwise_iou(
        boxes_a: np.ndarray,
        boxes_b: np.ndarray,
    ) -> np.ndarray:
        if boxes_a.size == 0 or boxes_b.size == 0:
            return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

        ax1, ay1, ax2, ay2 = boxes_a[:, 0][:, None], boxes_a[:, 1][:, None], boxes_a[:, 2][:, None], boxes_a[:, 3][:, None]
        bx1, by1, bx2, by2 = boxes_b[:, 0][None, :], boxes_b[:, 1][None, :], boxes_b[:, 2][None, :], boxes_b[:, 3][None, :]

        inter_x1 = np.maximum(ax1, bx1)
        inter_y1 = np.maximum(ay1, by1)
        inter_x2 = np.minimum(ax2, bx2)
        inter_y2 = np.minimum(ay2, by2)

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = np.maximum(0.0, (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))[:, None]
        area_b = np.maximum(0.0, (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))[None, :]

        union = area_a + area_b - inter_area + 1e-9
        return inter_area / union

    @staticmethod
    def _results_to_sv_detections(results: List[DetectionResult]) -> sv.Detections:
        if not results:
            return sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
            )

        xyxy: List[List[float]] = []
        confidences: List[float] = []
        for det in results:
            x, y, w, h = det.bbox
            xyxy.append([x, y, x + w, y + h])
            confidences.append(det.confidence)

        return sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
        )

    @staticmethod
    def _format_label(detection: DetectionResult) -> str:
        primary_label: Optional[str] = None
        if detection.gender_label and detection.age_range:
            primary_label = f"{detection.gender_label} ({detection.age_range})"
        elif detection.class_label:
            label = detection.class_label
            if isinstance(label, str):
                primary_label = label.title()
        else:
            primary_label = "Person"

        if primary_label is None:
            primary_label = "Person"

        confidence_pct = f"{detection.confidence * 100:0.0f}%"
        return f"{primary_label} {confidence_pct}".strip()



def _resolve_device(choice: str) -> Optional[str]:
    choice = choice.lower()
    if choice == "auto":
        return None
    if choice == "cuda":
        return "cuda:0"
    return choice


@st.cache_resource(show_spinner=False)
def load_pipeline(
    age_model_path: str,
    person_model_path: str,
    device_choice: str,
    enable_age_detector: bool,
    enable_person_detector: bool,
    age_conf: float,
    person_conf: float,
    imgsz: int,
) -> VisionPipeline:
    if not enable_age_detector and not enable_person_detector:
        raise RuntimeError("Enable at least one detector to run analysis.")
    device = _resolve_device(device_choice)
    return VisionPipeline(
        model_path=age_model_path,
        person_model_path=person_model_path,
        device=device,
        enable_age_detector=enable_age_detector,
        enable_person_detector=enable_person_detector,
        age_conf_threshold=age_conf,
        person_conf_threshold=person_conf,
        imgsz=imgsz,
    )
