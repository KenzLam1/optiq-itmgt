import uuid
from pathlib import Path
from typing import Optional

import streamlit as st

from data_store import clear_detection_logs, load_detection_logs
from pipeline import load_pipeline
from ui import (
    ensure_session_state,
    get_model_paths,
    render_intro,
    render_detection_log_preview,
    render_analytics_dashboard,
    render_sidebar,
)
from video import prepare_video_source, run_analysis


def main() -> None:
    st.set_page_config(page_title="Optiq Retail Analytics", layout="wide")

    ensure_session_state()
    render_intro()
    current_logs = load_detection_logs()
    st.session_state.latest_detection_logs = current_logs
    sidebar = render_sidebar()
    model_paths = get_model_paths()

    just_cleared = False
    if st.session_state.get("clear_confirmed"):
        clear_detection_logs()
        st.session_state.clear_confirmed = False
        st.session_state.last_run_summary = None
        st.session_state.last_preview_image = None
        current_logs = load_detection_logs()
        st.session_state.latest_detection_logs = current_logs
        just_cleared = True
        sidebar.run_clicked = False
        st.success("Detection logs have been cleared.")

    if sidebar.stop_clicked:
        st.session_state.stop_requested = True
    if sidebar.run_clicked:
        st.session_state.stop_requested = False

    if just_cleared or not sidebar.run_clicked:
        last_summary = st.session_state.get("last_run_summary")
        last_run_id = (last_summary or {}).get("run_id")
        last_preview_image = st.session_state.get("last_preview_image")
        current_logs = st.session_state.get("latest_detection_logs", current_logs)

        detections_tab, analytics_tab = st.tabs(["Detections", "Analytics"])
        with detections_tab:
            render_detection_log_preview(
                run_id=last_run_id,
                preview_image=last_preview_image,
                processed_frames=(last_summary or {}).get("processed_frames"),
                fps=(last_summary or {}).get("fps"),
                elapsed=(last_summary or {}).get("elapsed"),
                logs_df=current_logs,
            )
        with analytics_tab:
            render_analytics_dashboard(current_logs)
        return

    temp_file: Optional[Path] = None
    latest_summary = None
    latest_preview = None
    try:
        with st.spinner("Loading detection models..."):
            pipeline = load_pipeline(
                age_model_path=model_paths["age_model_path"],
                person_model_path=model_paths["person_model_path"],
                device_choice=sidebar.device_choice,
                age_conf=sidebar.age_conf,
                person_conf=sidebar.person_conf,
                imgsz=sidebar.imgsz,
            )

        pipeline.set_frame_interval(sidebar.frame_skip)

        source, temp_file = prepare_video_source(
            source_type=sidebar.source_type,
            camera_index=int(sidebar.camera_index),
            file_path=sidebar.file_path,
            stream_url=sidebar.stream_url,
            uploaded_file=sidebar.uploaded_file,
        )

        progress_bar = st.progress(0, text="Initialising stream...")
        frame_placeholder = st.empty()
        status_placeholder = st.empty()

        run_id = uuid.uuid4().hex
        st.session_state.current_run_id = run_id

        results = run_analysis(
            pipeline=pipeline,
            source=source,
            preview_stride=sidebar.preview_stride,
            progress_bar=progress_bar,
            frame_placeholder=frame_placeholder,
            status_placeholder=status_placeholder,
            stop_requested_fn=lambda: st.session_state.get("stop_requested", False),
            run_id=run_id,
        )
        detection_rows, preview_image, processed_frames, fps, elapsed = results

        progress_bar.empty()
        status_placeholder.empty()
        frame_placeholder.empty()

        st.session_state.last_run_summary = {
            "run_id": run_id,
            "processed_frames": processed_frames,
            "fps": fps,
            "elapsed": elapsed,
        }
        st.session_state.last_preview_image = preview_image
        st.session_state.current_run_id = None
        latest_summary = st.session_state.last_run_summary
        latest_preview = preview_image
        st.session_state.latest_detection_logs = load_detection_logs()
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()
        return
    except RuntimeError as exc:
        st.error(str(exc))
    finally:
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)
        st.session_state.stop_requested = False
        if st.session_state.get("current_run_id"):
            st.session_state.current_run_id = None

    current_logs = load_detection_logs()
    st.session_state.latest_detection_logs = current_logs

    summary_for_display = latest_summary or st.session_state.get("last_run_summary")
    preview_for_display = latest_preview or st.session_state.get("last_preview_image")
    detections_tab, analytics_tab = st.tabs(["Detections", "Analytics"])
    with detections_tab:
        render_detection_log_preview(
            run_id=(summary_for_display or {}).get("run_id"),
            preview_image=preview_for_display,
            processed_frames=(summary_for_display or {}).get("processed_frames"),
            fps=(summary_for_display or {}).get("fps"),
            elapsed=(summary_for_display or {}).get("elapsed"),
            logs_df=current_logs,
        )
    with analytics_tab:
        render_analytics_dashboard(current_logs)


if __name__ == "__main__":
    main()
