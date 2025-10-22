from pathlib import Path
from typing import Optional

import streamlit as st

from pipeline import load_pipeline
from ui import (
    ensure_session_state,
    get_model_paths,
    render_intro,
    render_results,
    render_sidebar,
)
from video import prepare_video_source, run_analysis


def main() -> None:
    st.set_page_config(page_title="Optiq Retail Analytics", layout="wide")

    ensure_session_state()
    render_intro()
    sidebar = render_sidebar()
    model_paths = get_model_paths()

    if sidebar.stop_clicked:
        st.session_state.stop_requested = True
    if sidebar.run_clicked:
        st.session_state.stop_requested = False

    if not sidebar.run_clicked:
        st.info(
            "Adjust the settings in the sidebar and click **Run analysis** to begin. "
            "Use **Stop analysis** to halt processing at any time."
        )
        return

    temp_file: Optional[Path] = None
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

        results = run_analysis(
            pipeline=pipeline,
            source=source,
            preview_stride=sidebar.preview_stride,
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

