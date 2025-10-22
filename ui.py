from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from config import AGE_GENDER_MODEL_PATH, PERSON_DETECTOR_MODEL_PATH


@dataclass
class SidebarConfig:
    device_choice: str
    age_conf: float
    person_conf: float
    imgsz: int
    source_type: str
    uploaded_file: Optional[Any]
    file_path: str
    stream_url: str
    camera_index: int
    frame_skip: int
    preview_stride: int
    run_clicked: bool
    stop_clicked: bool


def ensure_session_state() -> None:
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False


def render_sidebar() -> SidebarConfig:
    with st.sidebar:
        st.header("Models")
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

    return SidebarConfig(
        device_choice=device_choice,
        age_conf=age_conf,
        person_conf=person_conf,
        imgsz=imgsz,
        source_type=source_type,
        uploaded_file=uploaded_file,
        file_path=file_path,
        stream_url=stream_url,
        camera_index=camera_index,
        frame_skip=frame_skip,
        preview_stride=preview_stride,
        run_clicked=run_clicked,
        stop_clicked=stop_clicked,
    )


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
            na_rep="N/A",
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


def render_intro() -> None:
    st.title("Optiq Retail Analytics")
    st.caption("Streamlit dashboard for age and gender analytics powered by YOLO.")
    st.markdown(
        """
        Configure the capture source and press **Run analysis** to process a batch of frames.
        The dashboard reuses the original YOLO pipeline and displays aggregated detections,
        annotated previews, and downloadable results.
        """
    )


def get_model_paths() -> Dict[str, str]:
    return {
        "age_model_path": AGE_GENDER_MODEL_PATH,
        "person_model_path": PERSON_DETECTOR_MODEL_PATH,
    }
