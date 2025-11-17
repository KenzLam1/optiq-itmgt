from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

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

        for person_det in person_detections:
            if age_detections and any(self._iou(person_det.bbox, age_det.bbox) > 0.35 for age_det in age_detections):
                continue
            detections.append(person_det)

        annotated = frame.copy()

        for det in detections:
            x, y, w, h = det.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (20, 170, 255), 2)

            label_lines: List[str] = []
            if det.gender_label and det.age_range:
                label_lines.append(f"{det.gender_label} ({det.age_range})")
            elif det.class_label:
                label = det.class_label
                if isinstance(label, str) and label.islower():
                    label = label.title()
                label_lines.append(label)
            else:
                label_lines.append("Person")

            label_lines.append(f"{det.confidence * 100:0.0f}%")

            self._draw_label_block(annotated, x, y, label_lines)

        return annotated, detections

    @staticmethod
    def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, aw, ah = box_a
        bx1, by1, bw, bh = box_b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = aw * ah
        area_b = bw * bh

        union = area_a + area_b - inter_area + 1e-9
        return inter_area / union

    @staticmethod
    def _draw_label_block(image: np.ndarray, x: int, y: int, lines: List[str]) -> None:
        """Render a semi-transparent label block above the bounding box."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 1
        margin = 4
        line_height = 18

        block_width = 0
        for line in lines:
            size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            block_width = max(block_width, size[0])

        block_height = line_height * len(lines) + margin * 2

        overlay = image.copy()
        top = max(0, y - block_height - 4)
        cv2.rectangle(
            overlay,
            (x, top),
            (x + block_width + margin * 2, top + block_height),
            (15, 20, 30),
            -1,
        )
        cv2.addWeighted(overlay, 0.65, image, 0.35, 0, dst=image)

        for idx, line in enumerate(lines):
            baseline = top + margin + (idx + 1) * line_height - 6
            cv2.putText(image, line, (x + margin, baseline), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)


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
