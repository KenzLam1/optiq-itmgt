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
import supervision as sv

from data_store import append_detection_logs
from pipeline import VisionPipeline


def _call_with_width(fn: Callable[..., Any], *args: Any, stretch: bool = True, **kwargs: Any):
    width_value = "stretch" if stretch else "content"
    try:
        return fn(*args, width=width_value, **kwargs)
    except TypeError as exc:
        if "width" not in str(exc):
            raise
        return fn(*args, use_container_width=stretch, **kwargs)


ZONE_COLORS = [
    "#ff6b6b",
    "#f4a261",
    "#ffd166",
    "#06d6a0",
    "#118ab2",
    "#8338ec",
]


class ZoneManager:
    def __init__(self, definitions: Optional[List[Dict[str, Any]]] = None) -> None:
        self.definitions = definitions or []
        self.instances: List[Dict[str, Any]] = []
        self.frame_key: Optional[Tuple[int, int]] = None
        self.frame_dims: Optional[Tuple[int, int]] = None

    def has_zones(self) -> bool:
        return bool(self.definitions)

    def prepare(self, frame_width: int, frame_height: int) -> None:
        if not self.has_zones():
            return
        key = (frame_width, frame_height)
        if self.frame_key == key:
            return
        self.frame_key = key
        self.frame_dims = (frame_width, frame_height)
        self.instances = []
        for idx, definition in enumerate(self.definitions):
            raw_points = definition.get("points") or []
            polygon = np.array(
                [
                    [float(pt["x"]) * frame_width, float(pt["y"]) * frame_height]
                    for pt in raw_points
                    if "x" in pt and "y" in pt
                ],
                dtype=np.float32,
            )
            if polygon.size == 0:
                continue
            epsilon = 1e-3
            polygon[:, 0] = np.clip(polygon[:, 0], 0.0, max(frame_width - 1 - epsilon, 0.0))
            polygon[:, 1] = np.clip(polygon[:, 1], 0.0, max(frame_height - 1 - epsilon, 0.0))
            if polygon.shape[0] < 3:
                continue
            try:
                zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=(frame_width, frame_height))
            except TypeError as exc:
                if "frame_resolution_wh" in str(exc):
                    zone = sv.PolygonZone(polygon=polygon)
                else:
                    raise
            color_hex = ZONE_COLORS[idx % len(ZONE_COLORS)]
            annotator = sv.PolygonZoneAnnotator(
                zone=zone,
                color=sv.Color.from_hex(color_hex),
                thickness=2,
                text_thickness=1,
                text_scale=0.5,
            )
            self.instances.append(
                {
                    "name": definition.get("name", f"Zone {idx + 1}"),
                    "zone": zone,
                    "annotator": annotator,
                }
            )

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        if not self.instances:
            return frame
        annotated = frame
        for instance in self.instances:
            annotated = instance["annotator"].annotate(scene=annotated)
        return annotated

    def assign(self, detections: List[Any]) -> List[Optional[str]]:
        if not self.instances or not detections:
            return [None] * len(detections)
        xyxy = np.array(
            [
                [
                    det.bbox[0],
                    det.bbox[1],
                    det.bbox[0] + det.bbox[2],
                    det.bbox[1] + det.bbox[3],
                ]
                for det in detections
            ],
            dtype=np.float32,
        )
        if xyxy.size == 0:
            return [None] * len(detections)
        if self.frame_dims is not None:
            frame_width, frame_height = self.frame_dims
            if frame_width > 0 and frame_height > 0:
                max_x = max(frame_width - 1.001, 0.0)
                max_y = max(frame_height - 1.001, 0.0)
                xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0.0, max_x)
                xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0.0, max_y)
        assignments: List[Optional[str]] = [None] * len(detections)
        for instance in self.instances:
            zone_xyxy = xyxy.copy()
            zone_resolution = getattr(instance["zone"], "frame_resolution_wh", None)
            if zone_resolution:
                zone_width, zone_height = zone_resolution
                if zone_width > 0 and zone_height > 0:
                    max_x = max(zone_width - 1.001, 0.0)
                    max_y = max(zone_height - 1.001, 0.0)
                    zone_xyxy[:, [0, 2]] = np.clip(zone_xyxy[:, [0, 2]], 0.0, max_x)
                    zone_xyxy[:, [1, 3]] = np.clip(zone_xyxy[:, [1, 3]], 0.0, max_y)
            zone_detections = sv.Detections(xyxy=zone_xyxy)
            mask = instance["zone"].trigger(detections=zone_detections)
            for idx, inside in enumerate(mask):
                if inside and assignments[idx] is None:
                    assignments[idx] = instance["name"]
        return assignments


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
    zones: Optional[List[Dict[str, Any]]] = None,
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
    zone_manager = ZoneManager(zones)

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
            zone_manager.prepare(frame_width, frame_height)
            annotated, detections = pipeline.process(frame, frame_idx)
            annotated = zone_manager.annotate(annotated)
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
                _call_with_width(
                    frame_placeholder.image,
                    rgb_frame,
                    caption=f"Frame {frame_idx}",
                    channels="RGB",
                )

            zone_hits = zone_manager.assign(detections)

            for idx, det in enumerate(detections):
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
                        "zone": zone_hits[idx] if idx < len(zone_hits) else None,
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
                        "zone": zone_hits[idx] if idx < len(zone_hits) else None,
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
