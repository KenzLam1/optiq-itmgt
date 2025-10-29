import tempfile
import time
from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python <3.9 fallback using backports.zoneinfo
    ZoneInfo = None  # type: ignore[assignment]
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from data_store import append_detection_logs
from pipeline import VisionPipeline


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
    file_path: str,
    stream_url: str,
    uploaded_file: Optional[Any],
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
    run_id: str,
) -> Tuple[List[Dict[str, Union[str, float, int]]], Optional[np.ndarray], int, float, float]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Unable to open the selected video source.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = total_frames if total_frames > 0 else None

    detection_rows: List[Dict[str, Union[str, float, int]]] = []
    preview_image: Optional[np.ndarray] = None
    log_entries: List[Dict[str, Any]] = []
    recent_tracks: List[Dict[str, Any]] = []

    start_time = time.time()
    processed_frames = 0
    frame_idx = 0
    stopped_by_user = False
    flush_threshold = 50

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
            if ZoneInfo is not None:
                manila_ts = frame_timestamp.astimezone(ZoneInfo("Asia/Manila"))
            else:
                phil_tz = timezone(timedelta(hours=8))
                manila_ts = frame_timestamp.astimezone(phil_tz)
            track_cutoff = frame_timestamp

            # Remove stale tracks
            track_ttl_seconds = 6.0
            recent_tracks = [
                track
                for track in recent_tracks
                if (track_cutoff - track["logged_at"]).total_seconds() <= track_ttl_seconds
            ]

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

                bbox_x, bbox_y, bbox_w, bbox_h = det.bbox
                center_x = (bbox_x + bbox_w / 2.0) / max(1, frame_width)
                center_y = (bbox_y + bbox_h / 2.0) / max(1, frame_height)
                normalized_center = (float(np.clip(center_x, 0.0, 1.0)), float(np.clip(center_y, 0.0, 1.0)))

                matched_track: Optional[Dict[str, Any]] = None
                duplicate_distance = 0.06
                for track in recent_tracks:
                    dx = normalized_center[0] - track["center_x"]
                    dy = normalized_center[1] - track["center_y"]
                    dist = np.hypot(dx, dy)
                    time_diff = (frame_timestamp - track["logged_at"]).total_seconds()
                    if dist <= duplicate_distance and time_diff <= track_ttl_seconds:
                        matched_track = track
                        break

                if matched_track:
                    matched_track["logged_at"] = frame_timestamp
                    matched_track["center_x"] = normalized_center[0]
                    matched_track["center_y"] = normalized_center[1]
                    continue

                recent_tracks.append(
                    {
                        "center_x": normalized_center[0],
                        "center_y": normalized_center[1],
                        "logged_at": frame_timestamp,
                    }
                )

                detection_rows.append(
                    {
                        "frame": frame_idx,
                        "source": "Age/Gender" if det.source == "age_gender" else "Person",
                        "label": label_text,
                        "age_range": det.age_range or "",
                        "age_estimate": det.age_estimate if det.age_estimate is not None else np.nan,
                        "confidence": det.confidence,
                        "bbox_x": bbox_x,
                        "bbox_y": bbox_y,
                        "bbox_w": bbox_w,
                        "bbox_h": bbox_h,
                    }
                )

                log_entries.append(
                    {
                        "run_id": run_id,
                        "logged_at": manila_ts,
                        "frame_idx": frame_idx,
                        "processed_frame": processed_frames,
                        "source": "Age/Gender" if det.source == "age_gender" else "Person",
                        "label": label_text,
                        "gender": det.gender_label.capitalize() if det.gender_label else "Unknown",
                        "age_range": det.age_range or "",
                        "age_estimate": det.age_estimate if det.age_estimate is not None else np.nan,
                        "confidence": float(det.confidence),
                        "bbox_x": bbox_x,
                        "bbox_y": bbox_y,
                        "bbox_w": bbox_w,
                        "bbox_h": bbox_h,
                        "center_x": normalized_center[0],
                        "center_y": normalized_center[1],
                        "frame_width": frame_width,
                        "frame_height": frame_height,
                    }
                )
                if len(log_entries) >= flush_threshold:
                    append_detection_logs(log_entries)
                    log_entries.clear()

            if total_frames:
                progress_ratio = min(1.0, frame_idx / total_frames)
                progress_value = int(progress_ratio * 100)
                progress_bar.progress(progress_value, text=f"Processed {processed_frames} frame(s)")
            else:
                progress_bar.progress(0, text=f"Processed {processed_frames} frame(s)")

            if processed_frames % max(1, preview_stride) == 0:
                status_placeholder.info(f"Frames processed: {processed_frames}")
    finally:
        if log_entries:
            append_detection_logs(log_entries)
            log_entries.clear()
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
