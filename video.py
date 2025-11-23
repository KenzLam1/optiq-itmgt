import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python <3.9 fallback using backports.zoneinfo
    ZoneInfo = None  # type: ignore[assignment]

import cv2
import numpy as np

from data_store import append_detection_logs
from detections import DetectionLogEntry, DetectionSnapshot
from pipeline import VisionPipeline


@dataclass
class TrackState:
    center_x: float
    center_y: float
    logged_at: datetime


class DetectionDeduper:
    """Lightweight duplicate suppression using temporal/spatial proximity."""

    def __init__(self, ttl_seconds: float = 6.0, duplicate_distance: float = 0.06) -> None:
        self.ttl_seconds = ttl_seconds
        self.duplicate_distance = duplicate_distance
        self._tracks: List[TrackState] = []

    def should_log(self, timestamp: datetime, normalized_center: Tuple[float, float]) -> bool:
        self._prune(timestamp)
        for track in self._tracks:
            dx = normalized_center[0] - track.center_x
            dy = normalized_center[1] - track.center_y
            dist = float(np.hypot(dx, dy))
            if dist <= self.duplicate_distance:
                track.center_x = normalized_center[0]
                track.center_y = normalized_center[1]
                track.logged_at = timestamp
                return False
        self._tracks.append(
            TrackState(
                center_x=normalized_center[0],
                center_y=normalized_center[1],
                logged_at=timestamp,
            )
        )
        return True

    def _prune(self, timestamp: datetime) -> None:
        self._tracks = [
            track
            for track in self._tracks
            if (timestamp - track.logged_at).total_seconds() <= self.ttl_seconds
        ]


class DetectionLogBuffer:
    """Buffer detection logs before writing to SQLite to reduce I/O."""

    def __init__(self, flush_threshold: int = 50) -> None:
        self.flush_threshold = flush_threshold
        self._entries: List[DetectionLogEntry] = []

    def add(self, entry: DetectionLogEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) >= self.flush_threshold:
            self.flush()

    def flush(self) -> None:
        if not self._entries:
            return
        append_detection_logs(self._entries)
        self._entries.clear()

    def close(self) -> None:
        self.flush()


class AnalysisUI:
    """Encapsulate Streamlit preview/progress updates."""

    def __init__(self, progress_bar, frame_placeholder, status_placeholder, preview_stride: int) -> None:
        self.progress_bar = progress_bar
        self.frame_placeholder = frame_placeholder
        self.status_placeholder = status_placeholder
        self.preview_stride = max(1, preview_stride)

    def show_frame(
        self,
        annotated: np.ndarray,
        frame_idx: int,
        processed_frames: int,
        force: bool = False,
    ) -> None:
        if not force and processed_frames % self.preview_stride != 0:
            return
        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        self.frame_placeholder.image(
            rgb_frame,
            caption=f"Frame {frame_idx}",
            channels="RGB",
            width="stretch",
        )

    def update_progress(self, frame_idx: int, total_frames: Optional[int], processed_frames: int) -> None:
        if total_frames:
            progress_ratio = min(1.0, frame_idx / max(1, total_frames))
            progress_value = int(progress_ratio * 100)
            self.progress_bar.progress(progress_value, text=f"Processed {processed_frames} frame(s)")
        else:
            self.progress_bar.progress(0, text=f"Processed {processed_frames} frame(s)")

    def update_status(self, processed_frames: int) -> None:
        if processed_frames % self.preview_stride == 0:
            self.status_placeholder.info(f"Frames processed: {processed_frames}")

    def finish(self, *, processed_frames: int, fps: float, elapsed: float, stopped: bool) -> None:
        if stopped:
            self.progress_bar.progress(0, text="Analysis stopped")
            self.status_placeholder.warning(
                f"Analysis stopped by user after {processed_frames} frame(s)."
            )
        else:
            self.progress_bar.progress(100, text="Analysis complete")
            self.status_placeholder.success(f"Completed in {elapsed:.1f}s ({fps:.2f} FPS)")

    def stop_without_frames(self) -> None:
        self.progress_bar.progress(0, text="Analysis stopped")
        self.status_placeholder.info("Analysis stopped before any frames were processed.")


def _write_upload_to_temp(uploaded_file: Any) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def prepare_video_source(
    source_type: str,
    camera_index: int,
    uploaded_file: Optional[Any],
) -> Tuple[Union[int, str], Optional[Path]]:
    if source_type == "Webcam":
        return camera_index, None

    if uploaded_file is None:
        raise RuntimeError("Upload a video file before running analysis.")
    temp_path = _write_upload_to_temp(uploaded_file)
    return str(temp_path), temp_path


def _current_manila_time(now_utc: datetime) -> datetime:
    if ZoneInfo is not None:
        return now_utc.astimezone(ZoneInfo("Asia/Manila"))
    return now_utc.astimezone(timezone(timedelta(hours=8)))


def _normalized_center(bbox: Tuple[int, int, int, int], frame_width: int, frame_height: int) -> Tuple[float, float]:
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    center_x = (bbox_x + bbox_w / 2.0) / max(1, frame_width)
    center_y = (bbox_y + bbox_h / 2.0) / max(1, frame_height)
    return (
        float(np.clip(center_x, 0.0, 1.0)),
        float(np.clip(center_y, 0.0, 1.0)),
    )


def run_analysis(
    pipeline: VisionPipeline,
    source: Union[int, str],
    preview_stride: int,
    progress_bar,
    frame_placeholder,
    status_placeholder,
    stop_requested_fn: Callable[[], bool],
    run_id: str,
) -> Tuple[List[Dict[str, Union[str, float, int]]], Optional[np.ndarray], int, float, float]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Unable to open the selected video source.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames if total_frames > 0 else None

    detection_rows: List[Dict[str, Union[str, float, int]]] = []
    preview_image: Optional[np.ndarray] = None
    deduper = DetectionDeduper()
    log_buffer = DetectionLogBuffer()
    ui = AnalysisUI(progress_bar, frame_placeholder, status_placeholder, preview_stride)

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

            frame_height, frame_width = frame.shape[:2]
            annotated, detections = pipeline.process(frame, frame_idx)
            frame_idx += 1

            if frame_idx % pipeline.frame_interval != 0:
                continue

            processed_frames += 1
            frame_timestamp = datetime.now(timezone.utc)
            manila_ts = _current_manila_time(frame_timestamp)

            if detections or preview_image is None:
                preview_image = annotated.copy()

            ui.show_frame(annotated, frame_idx, processed_frames, force=bool(detections))

            for det in detections:
                normalized_center = _normalized_center(det.bbox, frame_width, frame_height)
                if not deduper.should_log(frame_timestamp, normalized_center):
                    continue

                snapshot = DetectionSnapshot.from_detection(frame_idx, det)
                detection_rows.append(snapshot.to_row())

                log_entry = DetectionLogEntry.from_detection(
                    run_id=run_id,
                    detection=det,
                    logged_at=manila_ts,
                    frame_idx=frame_idx,
                    processed_frame=processed_frames,
                    normalized_center=normalized_center,
                    frame_dimensions=(frame_width, frame_height),
                )
                log_buffer.add(log_entry)

            ui.update_progress(frame_idx, total_frames, processed_frames)
            ui.update_status(processed_frames)
    finally:
        log_buffer.close()
        cap.release()

    elapsed = time.time() - start_time
    if processed_frames == 0:
        if stopped_by_user:
            ui.stop_without_frames()
            return detection_rows, preview_image, processed_frames, 0.0, elapsed
        raise RuntimeError("No frames were processed from the selected source.")

    fps = processed_frames / elapsed if elapsed > 0 else 0.0
    ui.finish(processed_frames=processed_frames, fps=fps, elapsed=elapsed, stopped=stopped_by_user)
    return detection_rows, preview_image, processed_frames, fps, elapsed
