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

# Hardcoded values
DEFAULT_FRAME_SKIP = 1  # Process every frame 
DEFAULT_PREVIEW_STRIDE = 1  # Show every processed frame in the preview video
DEFAULT_IMGSZ = 640  # Inference image size 

# Pull latest detection logs from the database and store in session state
def _refresh_session_logs() -> None:
    logs = load_detection_logs()
    st.session_state.latest_detection_logs = logs # Update latest detection logs attribute in session state

# Handle any pending requests to clear detection logs
def _handle_pending_clear_request() -> bool:
    if not st.session_state.get("clear_confirmed"): # if no clear request pending, do nothing
        return False
    clear_detection_logs()  
    st.session_state.clear_confirmed = False    
    st.session_state.last_run_summary = None    
    st.session_state.last_preview_image = None  
    _refresh_session_logs()  # refresh the latest detection logs in session state
    st.success("Detection logs have been cleared.") 
    return True 


def _update_stop_flags(run_clicked: bool, stop_clicked: bool) -> None:
    if stop_clicked:
        st.session_state.stop_requested = True
    if run_clicked:
        st.session_state.stop_requested = False


def _execute_run(sidebar, model_paths) -> None:
    temp_file: Optional[Path] = None
    progress_bar = st.progress(0, text="Initialising stream...")
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    try:
        with st.spinner("Loading detection models..."):
            pipeline = load_pipeline(
                age_model_path=model_paths["age_model_path"],
                person_model_path=model_paths["person_model_path"],
                device_choice=sidebar.device_choice,
                enable_age_detector=sidebar.enable_age_detector,
                enable_person_detector=sidebar.enable_person_detector,
                age_conf=sidebar.age_conf,
                person_conf=sidebar.person_conf,
                imgsz=DEFAULT_IMGSZ,
            )
        device_summary = getattr(getattr(pipeline, "age_detector", None), "device", "unknown")
        st.caption(f"Models are running on: `{device_summary}`")

        pipeline.set_frame_interval(DEFAULT_FRAME_SKIP)

        source, temp_file = prepare_video_source(
            source_type=sidebar.source_type,
            camera_index=int(sidebar.camera_index),
            uploaded_file=sidebar.uploaded_file,
        )

        run_id = uuid.uuid4().hex
        st.session_state.current_run_id = run_id

        detection_rows, preview_image, processed_frames, fps, elapsed = run_analysis(
            pipeline=pipeline,
            source=source,
            preview_stride=DEFAULT_PREVIEW_STRIDE,
            progress_bar=progress_bar,
            frame_placeholder=frame_placeholder,
            status_placeholder=status_placeholder,
            stop_requested_fn=lambda: st.session_state.get("stop_requested", False),
            run_id=run_id,
        )

        st.session_state.last_run_summary = {
            "run_id": run_id,
            "processed_frames": processed_frames,
            "fps": fps,
            "elapsed": elapsed,
            "detections": len(detection_rows),
        }
        st.session_state.last_preview_image = preview_image
        st.session_state.current_run_id = None
        _refresh_session_logs()
        st.success(
            f"Run complete: processed {processed_frames} frame(s) in {elapsed:.1f}s "
            f"({fps:.2f} FPS)."
        )
    except RuntimeError as exc:
        st.error(str(exc))
    finally:
        progress_bar.empty()
        status_placeholder.empty()
        frame_placeholder.empty()
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)
        st.session_state.stop_requested = False
        if st.session_state.get("current_run_id"):
            st.session_state.current_run_id = None


def _acknowledge_stop() -> None:
    st.session_state.stop_requested = False
    _refresh_session_logs()
    st.info("Stop acknowledged. Latest detections refreshed.")


def _render_idle_state() -> None:
    summary = st.session_state.get("last_run_summary")
    if summary:
        st.success(
            f"Last run processed {summary['processed_frames']} frame(s) "
            f"at {summary['fps']:.2f} FPS."
        )
    else:
        st.info("Adjust the settings in the sidebar and click **Run analysis** to begin.")


def _render_run_tab(sidebar, model_paths, just_cleared: bool) -> None:
    st.subheader("Run Analysis")
    if just_cleared:
        st.info("Detection logs cleared. Ready for a fresh analysis run.")
    if st.session_state.get("stop_requested"):
        st.warning("Stop requested. Any active run will halt as soon as possible.")

    run_clicked = sidebar.run_clicked and not just_cleared

    if run_clicked:
        _execute_run(sidebar, model_paths)
    elif sidebar.stop_clicked:
        _acknowledge_stop()
    else:
        _render_idle_state()


def main() -> None:
    st.set_page_config(page_title="Optiq Retail Analytics", layout="wide")  # set page metadata

    ensure_session_state()  # initialize session state variables
    render_intro()  # display title and caption
    _refresh_session_logs()     # load latest detection logs into session state

    sidebar = render_sidebar()
    model_paths = get_model_paths()

    just_cleared = _handle_pending_clear_request()
    _update_stop_flags(run_clicked=sidebar.run_clicked, stop_clicked=sidebar.stop_clicked)

    run_tab, preview_tab, analytics_tab = st.tabs(
        ["Run Analysis", "Latest Detections", "Analytics"]
    )

    with run_tab:
        _render_run_tab(sidebar, model_paths, just_cleared)

    current_logs = st.session_state.get("latest_detection_logs", load_detection_logs())
    summary_for_display = st.session_state.get("last_run_summary")
    preview_for_display = st.session_state.get("last_preview_image")

    with preview_tab:
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
