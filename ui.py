from dataclasses import dataclass
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import altair as alt
import cv2
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from config import AGE_GENDER_MODEL_PATH, PERSON_DETECTOR_MODEL_PATH
from data_store import initialize_database, load_detection_logs


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
    if "last_run_summary" not in st.session_state:
        st.session_state.last_run_summary = None
    if "last_preview_image" not in st.session_state:
        st.session_state.last_preview_image = None
    if "current_run_id" not in st.session_state:
        st.session_state.current_run_id = None
    if not st.session_state.get("db_initialized"):
        initialize_database()
        st.session_state.db_initialized = True


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


def _normalize_date_input(selection: Union[dt.date, List[dt.date], tuple]) -> tuple[dt.date, dt.date]:
    if isinstance(selection, tuple):
        if len(selection) == 2:
            return selection[0], selection[1]
        if len(selection) == 1:
            return selection[0], selection[0]
    if isinstance(selection, list):
        if len(selection) >= 2:
            return selection[0], selection[1]
        if len(selection) == 1:
            return selection[0], selection[0]
    return selection, selection


def _compute_age_range(df: pd.DataFrame) -> tuple[int, int]:
    valid_ages = df["age_estimate"].dropna()
    if valid_ages.empty:
        return 0, 100
    min_age = int(np.floor(valid_ages.min()))
    max_age = int(np.ceil(valid_ages.max()))
    return max(0, min_age), max(10, max_age)


def _apply_filters(
    df: pd.DataFrame,
    date_range: tuple[dt.date, dt.date],
    time_range: tuple[dt.time, dt.time],
    genders: List[str],
    age_bounds: tuple[int, int],
    include_unknown_age: bool,
) -> pd.DataFrame:
    if df.empty:
        return df

    start_date, end_date = date_range
    start_minutes = time_range[0].hour * 60 + time_range[0].minute
    end_minutes = time_range[1].hour * 60 + time_range[1].minute
    min_age, max_age = age_bounds

    filtered = df[
        (df["date"] >= start_date)
        & (df["date"] <= end_date)
    ].copy()

    if filtered.empty:
        return filtered

    if start_minutes <= end_minutes:
        time_mask = (filtered["minutes"] >= start_minutes) & (filtered["minutes"] <= end_minutes)
    else:
        # Wrap-around (e.g. 20:00 to 02:00)
        time_mask = (filtered["minutes"] >= start_minutes) | (filtered["minutes"] <= end_minutes)
    filtered = filtered[time_mask]

    if not genders:
        return filtered.iloc[0:0]
    filtered = filtered[filtered["gender"].isin(genders)]

    if not include_unknown_age:
        filtered = filtered[filtered["age_estimate"].notna()]

    age_mask = filtered["age_estimate"].between(min_age, max_age) | filtered["age_estimate"].isna()
    if not include_unknown_age:
        age_mask = filtered["age_estimate"].between(min_age, max_age)
    filtered = filtered[age_mask]

    return filtered


def render_analytics_dashboard() -> None:
    df = load_detection_logs().copy()
    if df.empty:
        st.info("No detection logs available yet. Run an analysis to populate the dataset.")
        return

    df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce")
    df = df.dropna(subset=["logged_at"])
    if df.empty:
        st.info("Detection logs are present but timestamps could not be parsed.")
        return

    df["date"] = df["logged_at"].dt.date
    df["minutes"] = df["logged_at"].dt.hour * 60 + df["logged_at"].dt.minute
    df["gender"] = df["gender"].fillna("Unknown")
    df["age_estimate"] = pd.to_numeric(df["age_estimate"], errors="coerce")

    min_date = df["date"].min()
    max_date = df["date"].max()
    date_default = (min_date, max_date) if min_date != max_date else (min_date, max_date)

    gender_options = sorted(df["gender"].unique().tolist())
    min_age, max_age = _compute_age_range(df)
    overall_min_age = 0
    overall_max_age = max(100, max_age)

    with st.expander("Filters", expanded=True):
        date_selection = st.date_input(
            "Date range",
            value=date_default,
            min_value=min_date,
            max_value=max_date,
        )
        start_date, end_date = _normalize_date_input(date_selection)

        time_selection = st.slider(
            "Time range",
            value=(dt.time(0, 0), dt.time(23, 59)),
            format="HH:mm",
        )

        selected_genders = st.multiselect(
            "Gender",
            options=gender_options,
            default=gender_options,
        )

        age_selection = st.slider(
            "Age range",
            min_value=overall_min_age,
            max_value=overall_max_age,
            value=(min_age, min(max_age, overall_max_age)),
        )

        include_unknown_age = st.checkbox("Include entries with unknown age", value=True)

    filtered = _apply_filters(
        df,
        date_range=(start_date, end_date),
        time_range=time_selection,
        genders=selected_genders,
        age_bounds=age_selection,
        include_unknown_age=include_unknown_age,
    )

    st.caption(f"Showing {len(filtered)} detection(s) after filtering.")

    st.subheader("Foot Traffic Overview")
    interval_labels = {
        "5T": "5 minutes",
        "15T": "15 minutes",
        "1H": "Hourly",
        "1D": "Daily",
    }
    interval_keys = list(interval_labels.keys())
    interval_choice = st.selectbox(
        "Aggregation interval",
        options=interval_keys,
        format_func=lambda key: interval_labels[key],
        index=1,
    )

    if filtered.empty:
        st.info("No detections match the selected filters.")
    else:
        traffic = (
            filtered.set_index("logged_at")
            .resample(interval_choice)
            .size()
            .reset_index(name="detections")
        )
        if traffic["detections"].sum() == 0:
            st.info("No detections found for the selected interval.")
        else:
            traffic_chart = (
                alt.Chart(traffic)
                .mark_line(point=True)
                .encode(
                    x=alt.X("logged_at:T", title="Timestamp"),
                    y=alt.Y("detections:Q", title="Detections"),
                    tooltip=["logged_at:T", "detections:Q"],
                )
            )
            st.altair_chart(traffic_chart, use_container_width=True)

    st.subheader("Detection Heatmap")
    heatmap_bins = st.slider("Heatmap granularity", min_value=5, max_value=30, value=10)
    heat_df = filtered.dropna(subset=["center_x", "center_y"])
    if heat_df.empty:
        st.info("No spatial data available for the selected filters.")
    else:
        heat_df = heat_df.copy()
        heat_df["x_bin"] = np.clip((heat_df["center_x"] * heatmap_bins).astype(int), 0, heatmap_bins - 1)
        heat_df["y_bin"] = np.clip((heat_df["center_y"] * heatmap_bins).astype(int), 0, heatmap_bins - 1)
        heat_counts = (
            heat_df.groupby(["x_bin", "y_bin"])
            .size()
            .reset_index(name="count")
        )
        if heat_counts.empty:
            st.info("No detections to plot in the heatmap.")
        else:
            heat_chart = (
                alt.Chart(heat_counts)
                .mark_rect()
                .encode(
                    x=alt.X("x_bin:O", title="Horizontal zone"),
                    y=alt.Y("y_bin:O", title="Vertical zone", sort="descending"),
                    color=alt.Color("count:Q", title="Detections"),
                    tooltip=["x_bin:O", "y_bin:O", "count:Q"],
                )
            )
            st.altair_chart(heat_chart, use_container_width=True)

    st.subheader("Age Distribution")
    age_df = filtered.dropna(subset=["age_estimate"])
    if age_df.empty:
        st.info("No age data available after applying the filters.")
    else:
        bins = [0, 13, 18, 25, 35, 45, 55, 65, 200]
        labels = ["0-12", "13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        age_df = age_df.copy()
        age_df["age_bucket"] = pd.cut(age_df["age_estimate"], bins=bins, labels=labels, right=False)
        age_counts = (
            age_df.groupby("age_bucket")
            .size()
            .reset_index(name="count")
        )
        age_counts = age_counts[age_counts["count"] > 0]
        if age_counts.empty:
            st.info("No age buckets to display.")
        else:
            age_chart = (
                alt.Chart(age_counts)
                .mark_arc()
                .encode(
                    theta=alt.Theta("count:Q", title="Detections"),
                    color=alt.Color("age_bucket:N", title="Age bucket"),
                    tooltip=["age_bucket:N", "count:Q"],
                )
            )
            st.altair_chart(age_chart, use_container_width=True)

    st.subheader("Gender Distribution")
    gender_counts = (
        filtered.assign(gender=filtered["gender"].fillna("Unknown"))
        .groupby("gender")
        .size()
        .reset_index(name="count")
    )
    gender_counts = gender_counts[gender_counts["count"] > 0]
    if gender_counts.empty:
        st.info("No gender data available after applying the filters.")
    else:
        gender_chart = (
            alt.Chart(gender_counts)
            .mark_arc()
            .encode(
                theta=alt.Theta("count:Q", title="Detections"),
                color=alt.Color("gender:N", title="Gender"),
                tooltip=["gender:N", "count:Q"],
            )
        )
        st.altair_chart(gender_chart, use_container_width=True)


def render_detection_log_preview(
    run_id: Optional[str],
    preview_image: Optional[np.ndarray],
    processed_frames: Optional[int],
    fps: Optional[float],
    elapsed: Optional[float],
) -> None:
    st.subheader("Latest Detections Preview")

    session_summary = st.session_state.get("last_run_summary")
    if run_id is None and session_summary:
        run_id = session_summary.get("run_id")
    if preview_image is None:
        preview_image = st.session_state.get("last_preview_image")
    if processed_frames is None and session_summary:
        processed_frames = session_summary.get("processed_frames")
    if fps is None and session_summary:
        fps = session_summary.get("fps")
    if elapsed is None and session_summary:
        elapsed = session_summary.get("elapsed")

    logs = load_detection_logs().copy()
    if logs.empty:
        if preview_image is not None:
            st.image(
                cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB),
                caption="Most recent annotated frame",
                use_container_width=True,
            )
        st.info("No detection logs recorded yet. Run an analysis to populate the dataset.")
        return

    logs["logged_at"] = pd.to_datetime(logs["logged_at"], errors="coerce")
    logs = logs.dropna(subset=["logged_at"])
    if logs.empty:
        st.info("Detection logs exist but timestamps could not be parsed.")
        return

    logs = logs.sort_values("logged_at")
    selected_run_id = run_id if run_id in logs["run_id"].values else None
    if selected_run_id is None:
        selected_run_id = logs.iloc[-1]["run_id"]
    if session_summary and session_summary.get("run_id") == selected_run_id:
        processed_frames = processed_frames or session_summary.get("processed_frames")
        fps = fps or session_summary.get("fps")
        elapsed = elapsed or session_summary.get("elapsed")
    run_logs = logs[logs["run_id"] == selected_run_id].copy()

    total_runs = logs["run_id"].nunique()
    st.caption(
        f"Displaying {len(run_logs)} detection(s) for run `{selected_run_id}` "
        f"(total logged runs: {total_runs})."
    )

    if preview_image is not None:
        st.image(
            cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB),
            caption="Most recent annotated frame",
            use_container_width=True,
        )

    metrics = st.columns(4)
    metrics[0].metric("Detections (run)", len(run_logs))
    metrics[1].metric("Unique labels", run_logs["label"].nunique())
    metrics[2].metric(
        "Avg confidence",
        f"{run_logs['confidence'].mean():.1%}" if not run_logs.empty else "—",
    )
    metrics[3].metric(
        "Sessions logged",
        total_runs,
    )
    if processed_frames is not None and fps is not None and elapsed is not None:
        st.caption(f"Run stats: {processed_frames} frames processed in {elapsed:.1f}s ({fps:.2f} FPS).")

    st.subheader("Run Log")
    display_cols = [
        "logged_at",
        "label",
        "gender",
        "age_range",
        "age_estimate",
        "confidence",
        "frame_idx",
        "processed_frame",
    ]
    existing_cols = [col for col in display_cols if col in run_logs.columns]
    preview_df = run_logs[existing_cols].sort_values("logged_at", ascending=False)
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download full detection log",
        data=logs.to_csv(index=False).encode("utf-8"),
        file_name="detections_log.csv",
        mime="text/csv",
    )


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
