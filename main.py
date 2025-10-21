import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PyQt5 import QtCore, QtGui, QtWidgets


AGE_GENDER_MODEL_PATH = "age-gender_detector.pt"
PERSON_DETECTOR_MODEL_PATH = "person_detector.pt"


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

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)

        # Move the underlying torch.nn.Module to the target device.
        try:
            self.model.to(self.device)
        except AttributeError:
            self.model.model.to(self.device)

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

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        frame_height, frame_width = frame.shape[:2]

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            x1 = max(0, min(frame_width - 1, x1))
            y1 = max(0, min(frame_height - 1, y1))
            x2 = max(0, min(frame_width - 1, x2))
            y2 = max(0, min(frame_height - 1, y2))

            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            cls_id = int(box.cls[0]) if box.cls is not None else -1

            if isinstance(self.names, dict):
                raw_label = self.names.get(cls_id)
            else:
                raw_label = self.names[cls_id] if 0 <= cls_id < len(self.names) else None

            age_range, gender = self._parse_label(raw_label)
            age_estimate = self._estimate_age(age_range)

            detections.append(
                DetectionResult(
                    bbox=(x1, y1, w, h),
                    confidence=conf,
                    age_range=age_range,
                    age_estimate=age_estimate,
                    gender_label=gender.capitalize() if gender else None,
                    class_label=raw_label,
                    source="age_gender",
                )
            )

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
                "Download a YOLO model (e.g. yolov8n.pt) and provide its path."
            )

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)

        try:
            self.model.to(self.device)
        except AttributeError:
            self.model.model.to(self.device)

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

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        frame_height, frame_width = frame.shape[:2]

        for box in boxes:
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            if cls_id not in self.person_class_ids:
                continue

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            x1 = max(0, min(frame_width - 1, x1))
            y1 = max(0, min(frame_height - 1, y1))
            x2 = max(0, min(frame_width - 1, x2))
            y2 = max(0, min(frame_height - 1, y2))

            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            conf = float(box.conf[0]) if box.conf is not None else 0.0

            if isinstance(self.names, dict):
                raw_label = self.names.get(cls_id, "Person")
            else:
                raw_label = self.names[cls_id] if 0 <= cls_id < len(self.names) else "Person"

            label_text = raw_label or "Person"
            if isinstance(label_text, str) and label_text.islower():
                label_text = label_text.capitalize()

            detections.append(
                DetectionResult(
                    bbox=(x1, y1, w, h),
                    confidence=conf,
                    age_range=None,
                    age_estimate=None,
                    gender_label=None,
                    class_label=label_text,
                    source="person",
                )
            )

        return detections


class VisionPipeline:
    """Runs YOLO age/gender detection on incoming frames."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        person_model_path: str = PERSON_DETECTOR_MODEL_PATH,
    ):
        self.age_detector = YOLOAgeGenderDetector(model_path, device=device)
        self.person_detector = YOLOPersonDetector(person_model_path, device=device)
        self.frame_interval = 1  # process every frame by default

    def set_frame_interval(self, interval: int) -> None:
        self.frame_interval = max(1, interval)

    def process(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, List[DetectionResult]]:
        if frame_idx % self.frame_interval != 0:
            return frame, []

        age_detections = self.age_detector.predict(frame)
        person_detections = self.person_detector.predict(frame)
        detections = age_detections.copy()

        for person_det in person_detections:
            if any(self._iou(person_det.bbox, age_det.bbox) > 0.35 for age_det in age_detections):
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


class VideoWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(QtGui.QImage, list)
    status = QtCore.pyqtSignal(str)

    def __init__(self, pipeline: VisionPipeline, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.source: Union[int, str] = 0
        self._running = False
        self._frame_index = 0

    def set_source(self, source: Union[int, str]) -> None:
        self.source = source

    def stop(self) -> None:
        self._running = False
        self.wait(1_000)

    def run(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.status.emit("Unable to open video source.")
            return

        self._running = True
        self._frame_index = 0
        self.status.emit("Streaming...")
        last_emit_time = 0.0

        while self._running:
            ok, frame = cap.read()
            if not ok:
                self.status.emit("Stream ended or lost connection.")
                break

            annotated, detections = self.pipeline.process(frame, self._frame_index)
            self._frame_index += 1

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()

            now = time.time()
            if detections and (now - last_emit_time) > 0.75:
                self.status.emit(f"Detected {len(detections)} person(s).")
                last_emit_time = now

            self.frame_ready.emit(qt_image, detections)

        cap.release()
        self.status.emit("Idle.")


class OptiqWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        model_path: str,
        person_model_path: str,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Optiq Retail Analytics")
        self.resize(1280, 720)

        self.pipeline = VisionPipeline(model_path=model_path, person_model_path=person_model_path)
        self.worker = VideoWorker(self.pipeline)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.status.connect(self.update_status)

        self._build_ui()
        self._apply_dark_theme()

        self.statusBar().showMessage("Idle.")

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QtWidgets.QLabel("Video stream will appear here.")
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("color: #8f9ba8;")

        layout.addWidget(self._build_side_panel(), 0)
        layout.addWidget(self.video_label, 1)
        layout.addWidget(self._build_detection_panel(), 0)

        self.setCentralWidget(central)

    def _build_side_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setFixedWidth(320)
        panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        title = QtWidgets.QLabel("Optiq Retail Analytics")
        title.setStyleSheet("font-size: 18px; font-weight: 600; color: #e2e8f0;")
        layout.addWidget(title)

        layout.addWidget(self._build_source_group())
        layout.addWidget(self._build_settings_group())

        layout.addStretch(1)

        self.start_button = QtWidgets.QPushButton("Start Analysis")
        self.start_button.clicked.connect(self.start_stream)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_stream)
        self.stop_button.setEnabled(False)

        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        return panel

    def _build_source_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Capture Source")
        group_layout = QtWidgets.QVBoxLayout(group)

        self.source_selector = QtWidgets.QComboBox()
        self.source_selector.addItems(["Webcam", "Video File", "RTSP / CCTV"])
        self.source_selector.currentIndexChanged.connect(self._on_source_changed)

        self.webcam_index = QtWidgets.QSpinBox()
        self.webcam_index.setRange(0, 10)
        self.webcam_index.setValue(0)
        self.webcam_index.setPrefix("Camera #")

        self.file_controls = QtWidgets.QWidget()
        file_layout = QtWidgets.QHBoxLayout(self.file_controls)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(8)

        self.file_path_edit = QtWidgets.QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a video file...")
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.clicked.connect(self._browse_file)
        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(self.browse_button)

        self.rtsp_edit = QtWidgets.QLineEdit()
        self.rtsp_edit.setPlaceholderText("rtsp://user:pass@host:554/stream")

        group_layout.addWidget(self.source_selector)
        group_layout.addWidget(self.webcam_index)
        group_layout.addWidget(self.file_controls)
        group_layout.addWidget(self.rtsp_edit)

        self._on_source_changed(0)
        return group

    def _build_settings_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Processing Settings")
        layout = QtWidgets.QFormLayout(group)

        self.frame_skip_spin = QtWidgets.QSpinBox()
        self.frame_skip_spin.setRange(1, 10)
        self.frame_skip_spin.setValue(1)
        self.frame_skip_spin.valueChanged.connect(self._on_frame_skip_changed)

        layout.addRow("Frame skip", self.frame_skip_spin)
        return group

    def _build_detection_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QFrame()
        panel.setFixedWidth(260)
        panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(16, 24, 16, 24)
        layout.setSpacing(12)

        header = QtWidgets.QLabel("Detections")
        header.setStyleSheet("font-size: 16px; font-weight: 600; color: #e2e8f0;")
        layout.addWidget(header)

        self.detections_list = QtWidgets.QListWidget()
        self.detections_list.setStyleSheet("background-color: #1b2533; border: 1px solid #2b3645; color: #f1f5f9;")
        layout.addWidget(self.detections_list, 1)

        return panel

    def _apply_dark_theme(self) -> None:
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#101820"))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#e2e8f0"))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#17212b"))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#1b2533"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#1f2933"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#f1f5f9"))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#e2e8f0"))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#1b2533"))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#e2e8f0"))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#2563eb"))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#f8fafc"))
        self.setPalette(palette)

        self.setStyleSheet(
            """
            QWidget { background-color: #101820; color: #e2e8f0; }
            QGroupBox { border: 1px solid #27313f; border-radius: 6px; margin-top: 12px; padding: 12px; }
            QGroupBox:title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #93c5fd; }
            QPushButton { background-color: #2563eb; color: #f8fafc; padding: 8px 14px; border: none; border-radius: 6px; }
            QPushButton:disabled { background-color: #1f2937; color: #64748b; }
            QPushButton:hover { background-color: #1d4ed8; }
            QComboBox, QSpinBox, QLineEdit { background-color: #17212b; border: 1px solid #27313f; border-radius: 4px; padding: 6px; }
            QLabel { color: #e2e8f0; }
            QStatusBar { background-color: #0f172a; color: #cbd5f5; }
            """
        )

    def _on_source_changed(self, index: int) -> None:
        self.webcam_index.setVisible(index == 0)
        self.file_controls.setVisible(index == 1)
        self.rtsp_edit.setVisible(index == 2)

    def _on_frame_skip_changed(self, value: int) -> None:
        self.pipeline.set_frame_interval(value)

    def _browse_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select video file",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*.*)",
        )
        if path:
            self.file_path_edit.setText(path)

    def start_stream(self) -> None:
        source_mode = self.source_selector.currentIndex()
        source: Union[int, str]

        if source_mode == 0:
            source = int(self.webcam_index.value())
        elif source_mode == 1:
            path = self.file_path_edit.text().strip()
            if not path:
                QtWidgets.QMessageBox.warning(self, "Video file required", "Please select a video file to analyse.")
                return
            source = path
        else:
            url = self.rtsp_edit.text().strip()
            if not url:
                QtWidgets.QMessageBox.warning(self, "Stream URL required", "Please enter an RTSP / CCTV URL.")
                return
            source = url

        self.worker.set_source(source)
        self.worker.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_stream(self) -> None:
        if self.worker.isRunning():
            self.worker.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_frame(self, image: QtGui.QImage, detections: List[DetectionResult]) -> None:
        pixmap = QtGui.QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio))
        self._refresh_detection_list(detections)

    def update_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 2000)

    def _refresh_detection_list(self, detections: List[DetectionResult]) -> None:
        self.detections_list.clear()
        for det in detections:
            source_text = "Age/Gender" if det.source == "age_gender" else "Person"
            label_text = det.gender_label or det.class_label or "Person"
            if isinstance(label_text, str) and label_text.islower():
                label_text = label_text.capitalize()

            if det.source == "age_gender":
                age_text = det.age_range or "-"
                approx = f" (~{det.age_estimate:0.1f})" if det.age_estimate is not None else ""
            else:
                age_text = "-"
                approx = ""

            conf_text = f"{det.confidence * 100:0.0f}%"
            bbox_text = f"[x:{det.bbox[0]} y:{det.bbox[1]} w:{det.bbox[2]} h:{det.bbox[3]}]"
            item_text = f"{source_text}: {label_text} | Age {age_text}{approx} | Conf {conf_text} | {bbox_text}"
            self.detections_list.addItem(item_text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self.stop_stream()
        event.accept()


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)

    model_path = AGE_GENDER_MODEL_PATH
    person_model_path = PERSON_DETECTOR_MODEL_PATH
    try:
        window = OptiqWindow(model_path=model_path, person_model_path=person_model_path)
    except RuntimeError as exc:
        QtWidgets.QMessageBox.critical(None, "Optiq Retail Analytics", str(exc))
        sys.exit(1)

    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
