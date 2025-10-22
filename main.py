import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch


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
                "Download a YOLO model (for example yolov8n.pt) and provide its path."
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
    """Runs YOLO detections on incoming frames."""

    def __init__(
        self,
        model_path: str,
        person_model_path: str,
        device: Optional[str] = None,
        age_conf_threshold: float = 0.4,
        person_conf_threshold: float = 0.35,
        imgsz: int = 640,
    ):
        self.age_detector = YOLOAgeGenderDetector(
            model_path,
            device=device,
            conf_threshold=age_conf_threshold,
            imgsz=imgsz,
        )
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
    age_conf: float,
    person_conf: float,
    imgsz: int,
) -> VisionPipeline:
    device = _resolve_device(device_choice)
    return VisionPipeline(
        model_path=age_model_path,
        person_model_path=person_model_path,
        device=device,
        age_conf_threshold=age_conf,
        person_conf_threshold=person_conf,
        imgsz=imgsz,
    )


def _write_upload_to_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def prepare_video_source(
    source_type: str,
    camera_index: int,
    file_path: str,
    stream_url: str,
    uploaded_file,
) -> Tuple[Union[int, str], Optional[Path]]:
    if source_type == "Webcam":
        return camera_index, None

    if source_type == "Video file (path)":
        path = Path(file_path).expanduser()
        if not path.exists():
            raise RuntimeError(f"Video file not found: {path}")
        return str(path), None

    if source_type == "RTSP / CCTV":
        if not stream_url:
            raise RuntimeError("A valid RTSP / CCTV URL is required.")
        return stream_url, None

    if uploaded_file is None:
        raise RuntimeError("Upload a video file before running analysis.")
    temp_path = _write_upload_to_temp(uploaded_file)
    return str(temp_path), temp_path


def run_analysis(
    pipeline: VisionPipeline,
    source: Union[int, str],
    preview_stride: int,
    progress_bar,
    frame_placeholder,
    status_placeholder,
    stop_requested_fn: Callable[[], bool],
) -> Tuple[List[Dict[str, Union[str, float, int]]], Optional[np.ndarray], int, float, float]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Unable to open the selected video source.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames if total_frames > 0 else None

    detection_rows: List[Dict[str, Union[str, float, int]]] = []
    preview_image: Optional[np.ndarray] = None

    start_time = time.time()
    processed_frames = 0
    frame_idx = 0
    stopped_by_user = False

    try:
        while True:
            if stop_requested_fn():
                stopped_by_user = True
                break

            ok, frame = cap.read()
            if not ok:
                break

            annotated, detections = pipeline.process(frame, frame_idx)
            frame_idx += 1

            if frame_idx % pipeline.frame_interval != 0:
                continue

            processed_frames += 1

            if detections or preview_image is None:
                preview_image = annotated.copy()

            if processed_frames % max(1, preview_stride) == 0 or detections:
                rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    rgb_frame,
                    caption=f"Frame {frame_idx}",
                    use_container_width=True,
                    channels="RGB",
                )

            for det in detections:
                label_text = det.gender_label or det.class_label or "Person"
                if isinstance(label_text, str):
                    label_text = label_text.title()
                detection_rows.append(
                    {
                        "frame": frame_idx,
                        "source": "Age/Gender" if det.source == "age_gender" else "Person",
                        "label": label_text,
                        "age_range": det.age_range or "",
                        "age_estimate": det.age_estimate if det.age_estimate is not None else np.nan,
                        "confidence": det.confidence,
                        "bbox_x": det.bbox[0],
                        "bbox_y": det.bbox[1],
                        "bbox_w": det.bbox[2],
                        "bbox_h": det.bbox[3],
                    }
                )

            if total_frames:
                progress_ratio = min(1.0, frame_idx / total_frames)
                progress_value = int(progress_ratio * 100)
                progress_bar.progress(progress_value, text=f"Processed {processed_frames} frame(s)")
            else:
                progress_bar.progress(0, text=f"Processed {processed_frames} frame(s)")

            if processed_frames % max(1, preview_stride) == 0:
                status_placeholder.info(f"Frames processed: {processed_frames}")
    finally:
        cap.release()

    elapsed = time.time() - start_time
    if processed_frames == 0:
        if stopped_by_user:
            progress_bar.progress(0, text="Analysis stopped")
            status_placeholder.info("Analysis stopped before any frames were processed.")
            return detection_rows, preview_image, processed_frames, 0.0, elapsed
        raise RuntimeError("No frames were processed from the selected source.")

    fps = processed_frames / elapsed if elapsed > 0 else 0.0
    if stopped_by_user:
        progress_bar.progress(0, text="Analysis stopped")
        status_placeholder.warning(f"Analysis stopped by user after {processed_frames} frame(s).")
    else:
        progress_bar.progress(100, text="Analysis complete")
        status_placeholder.success(f"Completed in {elapsed:.1f}s ({fps:.2f} FPS)")
    return detection_rows, preview_image, processed_frames, fps, elapsed


def summarize_detections(rows: List[Dict[str, Union[str, float, int]]]) -> Dict[str, int]:
    age_gender = sum(1 for row in rows if row["source"] == "Age/Gender")
    generic = sum(1 for row in rows if row["source"] == "Person")
    return {"age_gender": age_gender, "person": generic, "total": len(rows)}


def render_results(
    detection_rows: List[Dict[str, Union[str, float, int]]],
    preview_image: Optional[np.ndarray],
    processed_frames: int,
    fps: float,
    elapsed: float,
) -> None:
    if preview_image is not None:
        st.subheader("Preview")
        st.image(
            cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB),
            caption="Most recent annotated frame",
            use_container_width=True,
        )

    summary = summarize_detections(detection_rows)
    st.subheader("Session summary")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Frames analysed", processed_frames)
    metric_cols[1].metric("Detections", summary["total"])
    metric_cols[2].metric("Age/Gender", summary["age_gender"])
    metric_cols[3].metric("Processing FPS", f"{fps:.2f}")
    st.caption(f"Elapsed time: {elapsed:.1f}s")

    st.subheader("Detections")
    if detection_rows:
        df = pd.DataFrame(detection_rows)
        styled = df.style.format(
            {
                "confidence": "{:.1%}",
                "age_estimate": "{:.1f}",
            },
            na_rep="–",
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download detections (CSV)",
            data=csv,
            file_name="optiq_detections.csv",
            mime="text/csv",
        )
    else:
        st.info("No detections were produced with the current configuration.")


def main() -> None:
    st.set_page_config(page_title="Optiq Retail Analytics", layout="wide")
    st.title("Optiq Retail Analytics")
    st.caption("Streamlit dashboard for age and gender analytics powered by YOLO.")

    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False

    with st.sidebar:
        st.header("Models")
        age_model_path = AGE_GENDER_MODEL_PATH
        person_model_path = PERSON_DETECTOR_MODEL_PATH
        st.caption("Using default model weight files bundled with the app.")
        device_choice = st.selectbox("Device", options=["auto", "cpu", "cuda"], index=0)
        age_conf = st.slider("Age/Gender confidence", 0.05, 0.95, 0.40, 0.05)
        person_conf = st.slider("Person confidence", 0.05, 0.95, 0.35, 0.05)
        imgsz = st.slider("Image size", 320, 960, 640, 32)

        st.header("Capture source")
        source_type = st.selectbox(
            "Source type",
            options=["Upload video", "Webcam", "Video file (path)", "RTSP / CCTV"],
            index=0,
        )
        uploaded_file = None
        file_path = ""
        stream_url = ""
        camera_index = 0

        if source_type == "Upload video":
            uploaded_file = st.file_uploader("Select a video file", type=["mp4", "mov", "mkv", "avi"])
        elif source_type == "Webcam":
            camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
        elif source_type == "Video file (path)":
            file_path = st.text_input("Absolute or relative file path")
        else:
            stream_url = st.text_input("RTSP / CCTV URL", value="rtsp://user:pass@host:554/stream")

        st.header("Processing")
        frame_skip = st.slider("Frame skip", 1, 10, 1)
        preview_stride = st.slider("Preview interval", 1, 30, 5)
        run_clicked = st.button("Run analysis", type="primary")
        stop_clicked = st.button("Stop analysis", type="secondary")
        if stop_clicked:
            st.session_state.stop_requested = True
        if run_clicked:
            st.session_state.stop_requested = False

    st.markdown(
        """
        Configure the capture source and press **Run analysis** to process a batch of frames.
        The dashboard reuses the original YOLO pipeline and displays aggregated detections,
        annotated previews, and downloadable results.
        """
    )

    if not run_clicked:
        st.info(
            "Adjust the settings in the sidebar and click **Run analysis** to begin. "
            "Use **Stop analysis** to halt processing at any time."
        )
        return

    temp_file: Optional[Path] = None
    try:
        with st.spinner("Loading detection models..."):
            pipeline = load_pipeline(
                age_model_path=age_model_path,
                person_model_path=person_model_path,
                device_choice=device_choice,
                age_conf=age_conf,
                person_conf=person_conf,
                imgsz=imgsz,
            )

        pipeline.set_frame_interval(frame_skip)

        source, temp_file = prepare_video_source(
            source_type=source_type,
            camera_index=int(camera_index),
            file_path=file_path,
            stream_url=stream_url,
            uploaded_file=uploaded_file,
        )

        progress_bar = st.progress(0, text="Initialising stream...")
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        results = run_analysis(
            pipeline=pipeline,
            source=source,
            preview_stride=preview_stride,
            progress_bar=progress_bar,
            frame_placeholder=frame_placeholder,
            status_placeholder=status_placeholder,
            stop_requested_fn=lambda: st.session_state.get("stop_requested", False),
        )
        detection_rows, preview_image, processed_frames, fps, elapsed = results

        progress_bar.empty()
        status_placeholder.empty()

        render_results(
            detection_rows=detection_rows,
            preview_image=preview_image,
            processed_frames=processed_frames,
            fps=fps,
            elapsed=elapsed,
        )
    except RuntimeError as exc:
        st.error(str(exc))
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)
        st.session_state.stop_requested = False


if __name__ == "__main__":
    main()
