from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import altair as alt
import cv2
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from data_store import initialize_database, load_detection_logs
from hardware import available_device_choices

AGE_GENDER_MODEL_PATH = "age-gender_detector.pt"
PERSON_DETECTOR_MODEL_PATH = "person_detector.pt"

DEVICE_LABELS = {
    "auto": "Auto (best available)",
    "cpu": "CPU only",
    "cuda": "NVIDIA CUDA",
    "mps": "Apple Silicon (MPS)",
}


@dataclass
class SidebarConfig:
    """Container for all sidebar configuration options selected by the user."""

    device_choice: str
    enable_age_detector: bool
    enable_person_detector: bool
    age_conf: float
    person_conf: float
    source_type: str
    uploaded_file: Optional[Any]
    camera_index: int
    run_clicked: bool
    stop_clicked: bool
    clear_db_requested: bool

"""helper to ensure session state variables are initialized."""
def ensure_session_state() -> None:
    if "stop_requested" not in st.session_state: #If stop requested key doesnt exist, create it and set to false (no request).
        st.session_state.stop_requested = False
    if "last_run_summary" not in st.session_state: #Stores info about the last run. Starts as None (no summary yet).
        st.session_state.last_run_summary = None
    if "last_preview_image" not in st.session_state: #Stores the last preview image shown in the UI. Starts as None (no image yet).
        st.session_state.last_preview_image = None
    if "current_run_id" not in st.session_state: #Stores the current run ID. Starts as None (no current run yet).
        st.session_state.current_run_id = None
    if "refresh_requested" not in st.session_state: #Tracks if a refresh is requested. Starts as False (no request).
        st.session_state.refresh_requested = False
    if "show_clear_prompt" not in st.session_state: #Tracks if the clear logs prompt should be shown. Starts as False (no prompt).
        st.session_state.show_clear_prompt = False
    if "clear_confirmed" not in st.session_state: #Tracks if the user confirmed clearing logs. Starts as False (not confirmed).
        st.session_state.clear_confirmed = False
    if not st.session_state.get("db_initialized"): #Checks if the database has been initialized. If not, it initializes it and sets the flag to True to prevent reinitializing.
        initialize_database()
        st.session_state.db_initialized = True

"""Render the sidebar UI and return the selected configuration as a SidebarConfig object."""
def render_sidebar() -> SidebarConfig:
    with st.sidebar:
        st.header("Models")
        device_options, mps_available = available_device_choices()  #Get available device options
        device_choice = st.selectbox(
            "Device",
            options=device_options,
            index=0,
            format_func=lambda opt: DEVICE_LABELS.get(opt, opt.upper()), #Display user-friendly labels, otherwise uppercase key
        )
        if mps_available:
            st.caption("Apple Silicon detected — pick 'Apple Silicon (MPS)' for higher FPS on Mac.")
        enable_age_detector = st.toggle(
            "Run age/gender detector",
            value=True,
            help="Disable to skip age/gender predictions for higher FPS on slower machines.",
        )
        enable_person_detector = st.toggle(
            "Run person detector (second model)",
            value=True,
            help="Disable to run only the age/gender model for higher FPS on slower machines.",
        )
        if not enable_age_detector and not enable_person_detector:
            st.error("Enable at least one detector to run analysis.")
        age_conf = st.slider("Age/Gender confidence", 0.05, 0.95, 0.40, 0.05)
        person_conf = st.slider("Person confidence", 0.05, 0.95, 0.35, 0.05)

        st.header("Capture source")
        source_type = st.selectbox(
            "Source type",
            options=["Upload video", "Webcam"],
            index=0,
        )
        # Initialize variables with default values. Holds source-specific inputs
        uploaded_file = None    
        camera_index = 0

        if source_type == "Upload video":
            uploaded_file = st.file_uploader("Select a video file", type=["mp4", "mov", "mkv", "avi"])
        elif source_type == "Webcam":
            camera_index = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)

        run_clicked = st.button("Run analysis", type="primary")
        stop_clicked = st.button("Stop analysis", type="secondary")
        clear_db_requested = st.button("⚠️ Clear detection logs", type="secondary", key="clear_logs_button")
        if clear_db_requested:
            st.session_state.show_clear_prompt = True   # Show confirmation prompt

        if st.session_state.get("show_clear_prompt"):
            st.warning("Clearing logs will delete all stored detections and analytics history.")
            confirm = st.button("Confirm clear", type="primary", key="confirm_clear_logs")
            cancel = st.button("Cancel", type="secondary", key="cancel_clear_logs")
            if confirm:
                st.session_state.clear_confirmed = True     # Set flag to confirm clearing logs
                st.session_state.show_clear_prompt = False  # Hide the prompt after confirmation
            elif cancel:
                st.session_state.show_clear_prompt = False  # Hide the prompt if cancelled

    return SidebarConfig(
        device_choice=device_choice,
        enable_age_detector=enable_age_detector,
        enable_person_detector=enable_person_detector,
        age_conf=age_conf,
        person_conf=person_conf,
        source_type=source_type,
        uploaded_file=uploaded_file,
        camera_index=camera_index,
        run_clicked=run_clicked,
        stop_clicked=stop_clicked,
        clear_db_requested=clear_db_requested,
    )

"""Returns a consistent tuple of start and end date range from various input formats."""
def _normalize_date_input(selection: dt.date | tuple[dt.date, dt.date]) -> tuple[dt.date, dt.date]:
    if isinstance(selection, tuple):
        if len(selection) == 2:
            return selection[0], selection[1]
        if len(selection) == 1:
            return selection[0], selection[0]
    return selection, selection

"""Computes age range for default behavior of age range filter"""
def _compute_age_range(df: pd.DataFrame) -> tuple[int, int]:
    valid_ages = df["age_estimate"].dropna()
    if valid_ages.empty:    # no age data 
        return 0, 100
    min_age = int(np.floor(valid_ages.min()))
    max_age = int(np.ceil(valid_ages.max()))
    return max(0, min_age), max(10, max_age) # Returns min and max age. Min age is at least 0, max age is at least 10

"""Takes in the full detection logs DataFrame and applies the user specified filters, returning the filtered DataFrame."""
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
    start_minutes = time_range[0].hour * 60 + time_range[0].minute  #Convert start time to total minutes
    end_minutes = time_range[1].hour * 60 + time_range[1].minute    #Convert end time to total minutes
    min_age, max_age = age_bounds

    filtered = df[      # Filter by date range
        (df["date"] >= start_date)
        & (df["date"] <= end_date)
    ].copy()

    if filtered.empty:
        return filtered # Return empty DataFrame if no data after date filtering
    
        # Normal case
    if start_minutes <= end_minutes:
        time_mask = (filtered["minutes"] >= start_minutes) & (filtered["minutes"] <= end_minutes)
    else:
        # Wrap-around case (e.g. 20:00 to 02:00)
        time_mask = (filtered["minutes"] >= start_minutes) | (filtered["minutes"] <= end_minutes)
    filtered = filtered[time_mask]

    if not genders:
        return filtered.iloc[0:0]   # Return empty dataframe if no genders selected
    filtered = filtered[filtered["gender"].isin(genders)]

    if include_unknown_age:
        age_mask = filtered["age_estimate"].between(min_age, max_age) | filtered["age_estimate"].isna()
    else:
        age_mask = filtered["age_estimate"].between(min_age, max_age)
    filtered = filtered[age_mask]

    return filtered


def render_analytics_dashboard(logs_df: Optional[pd.DataFrame] = None) -> None:
    df = logs_df.copy() if logs_df is not None else load_detection_logs().copy()
    if df.empty:
        st.info("No detection logs available yet. Run an analysis to populate the dataset.")
        return

    df["logged_at_local"] = df["logged_at"].dt.tz_localize(None)    # Remove timezone info for local time operations
    df["date"] = df["logged_at_local"].dt.date  # Extract date part for date filtering
    df["minutes"] = df["logged_at_local"].dt.hour * 60 + df["logged_at_local"].dt.minute    # Extract total minutes for time filtering
    df["gender"] = df["gender"].fillna("Unknown")   # Repalce missing genders with "Unknown"
    df["age_estimate"] = pd.to_numeric(df["age_estimate"], errors="coerce") #Ensure age_estimate is numeric, errors become NaN

    # Prepare filter values
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_default = (min_date, max_date) 
    gender_options = sorted(df["gender"].unique().tolist()) # Get unique genders from the data and sort them alphabetically
    min_age, max_age = _compute_age_range(df)
    overall_min_age = 0
    overall_max_age = max(100, max_age)     # at least 100 

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
            default=gender_options,     # Marks all gender options as selected by default
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
        "5min": "5 minutes",
        "15min": "15 minutes",
        "1H": "Hourly",
        "1D": "Daily",
    }
    interval_keys = list(interval_labels.keys())    
    interval_choice = st.selectbox(
        "Aggregation interval",
        options=interval_keys,
        format_func=lambda key: interval_labels[key],   # Use user-friendly labels
        index=1,
    )

    if filtered.empty:
        st.info("No detections match the selected filters.")
    else:
        traffic = (     # Creates a 2 column DataFrame with counts of detections per selected interval
            filtered.set_index("logged_at_local")   # Set timestamp as index for resampling. Timestamp is no longer a column.
            .resample(interval_choice)  # Resample by the selected interval (e.g. 5 minute buckets if "5min" selected)
            .size() # Count number of detections in each interval. Returns a series with timestamp index and counts as values.
            .reset_index(name="detections") # Turn the series back into a normal df. Counts are named "detections".
        )
        if traffic["detections"].sum() == 0:
            st.info("No detections found for the selected interval.")
        else:
            traffic_chart = (
                alt.Chart(traffic)  # Create Altair chart from the traffic DataFrame
                .mark_line(point=True) # Draw a line with points at each data point
                .encode(
                    x=alt.X("logged_at_local:T", title="Timestamp"),    # X axis is the timestamp. This is a time value.
                    y=alt.Y("detections:Q", title="Detections"), # Y axis is the count of detections. This is a quantitative value.
                    tooltip=["logged_at_local:T", "detections:Q"],      # Tooltip shows timestamp and count when hovering
                )
            )
            st.altair_chart(traffic_chart, width="stretch") 

    st.subheader("Age Distribution")
    age_df = filtered.dropna(subset=["age_estimate"])   
    if age_df.empty:
        st.info("No age data available after applying the filters.")
    else:
        bins = [0, 13, 18, 25, 35, 45, 55, 65, 200] # Define age buckets
        labels = ["0-12", "13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]  # how each age buckets are displayed
        age_df = age_df.copy()
        age_df["age_bucket"] = pd.cut(age_df["age_estimate"], bins=bins, labels=labels, right=False)    # Take age_estimate and assign to age_bucket based on defined bins. Left side is inclusive and right side is exclusive.
        age_counts = (
            age_df.groupby("age_bucket", observed=True)   # Group by age bucket and only include buckets that count > 0
            .size() # Count number of detections in each age bucket
            .reset_index(name="count") # Turn the series back into a normal df. Counts are named "count".
        )
        if age_counts.empty:
            st.info("No age buckets to display.")
        else:
            age_chart = (
                alt.Chart(age_counts) # Create Altair chart from the age_counts DataFrame
                .mark_arc() # Draw a pie chart
                .encode( 
                    theta=alt.Theta("count:Q", title="Detections"), # Angle (size) of each slice is based on count of detections. Quantitative data.
                    color=alt.Color("age_bucket:N", title="Age bucket"), # Color of each slice is based on age bucket. Nominal data.
                    tooltip=["age_bucket:N", "count:Q"],   # Tooltip shows age bucket and count when hovering.
                )
            )
            st.altair_chart(age_chart, width="stretch")

    st.subheader("Gender Distribution")
    gender_counts = (
        filtered[filtered["gender"] != "Unknown"] # Exclude "Unknown" genders   
        .groupby("gender")  # Group by gender 
        .size() # Count number of detections per gender
        .reset_index(name="count") # Turn the series back into a normal df. Counts are named "count".
    )
    if gender_counts.empty: 
        st.info("No gender data available after applying the filters.")
    else:
        gender_chart = (
            alt.Chart(gender_counts) # Create Altair chart from the gender_counts DataFrame
            .mark_arc() # Draw a pie chart
            .encode(
                theta=alt.Theta("count:Q", title="Detections"), # Angle (size) of each slice is based on count of detections. Quantitative data.
                color=alt.Color("gender:N", title="Gender"), # Color of each slice is based on gender. Nominal data.
                tooltip=["gender:N", "count:Q"],  # Tooltip shows gender and count when hovering.
            )
        )
        st.altair_chart(gender_chart, width="stretch")


def render_detection_log_preview(
    run_id: Optional[str],
    preview_image: Optional[np.ndarray],
    processed_frames: Optional[int],
    fps: Optional[float],
    elapsed: Optional[float],
    logs_df: Optional[pd.DataFrame] = None,
) -> None:
    st.subheader("Latest Detections Preview")
    # Load session state values for later use
    session_summary = st.session_state.get("last_run_summary") or {}    #uses empty dict if no summary yet
    last_preview_image = st.session_state.get("last_preview_image")
    
    # Use provided logs_df or load from data store if not provided. Makes a copy to avoid modifications.
    logs = logs_df.copy() if logs_df is not None else load_detection_logs().copy()
    if logs.empty:  
        st.info("No detection logs recorded yet. Run an analysis to populate the dataset.")
        return  #Early exit if no logs available

    logs = logs.sort_values("logged_at")    #Sort logs by timestamp

    # Decide run ID
    selected_run_id = run_id or session_summary.get("run_id") or logs.iloc[-1]["run_id"]
  
   #If session summary exists and run ID matches the session summary, fill in values if missing. 
    if session_summary.get("run_id") == selected_run_id:
        if preview_image is None:
            preview_image = last_preview_image
        if processed_frames is None:
            processed_frames = session_summary.get("processed_frames")
        if fps is None:
            fps = session_summary.get("fps")
        if elapsed is None:
            elapsed = session_summary.get("elapsed")

    run_logs = logs[logs["run_id"] == selected_run_id].copy()   # subset of the full logs for the selected run ID 

    total_runs = logs["run_id"].nunique() # total number of unique run IDs in the logs
    st.caption(
        f"Displaying {len(run_logs)} detection(s) for run `{selected_run_id}` "
        f"(total logged runs: {total_runs})."
    )

    if preview_image is not None: 
        st.image(
            cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB),
            caption="Most recent annotated frame",
            width="stretch",
        )

    metrics = st.columns(4)
    metrics[0].metric("Detections (run)", len(run_logs))
    metrics[1].metric("Unique labels", run_logs["label"].nunique())
    metrics[2].metric(
        "Avg confidence",
        f"{run_logs['confidence'].mean():.1%}" if not run_logs.empty else "N/A",
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
    ]
    existing_cols = [col for col in display_cols if col in run_logs.columns]    #Only include columns that exist in the DataFrame
    preview_df = run_logs[existing_cols].sort_values("logged_at", ascending=False).copy()   #Sort by timestamp descending
    if "logged_at" in preview_df.columns:   
        preview_df["logged_at"] = preview_df["logged_at"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S") #Format timestamp to plain string for better readability
        st.dataframe(preview_df, hide_index=True, width="stretch")

    download_df = logs.copy()
    download_df["logged_at"] = download_df["logged_at"].dt.tz_localize(None)
    st.download_button(
        "Download full detection log",
        data=download_df.to_csv(index=False).encode("utf-8"),
        file_name="detections_log.csv",
        mime="text/csv",
    )


def render_intro() -> None:
    st.title("Optiq Retail Analytics")
    st.caption("Analytics dashboard for Foot Traffic and Demographic analytics powered by AI.")

"""Returns a dict mapping the two models to their respective file paths."""
def get_model_paths() -> Dict[str, str]:
    return {
        "age_model_path": AGE_GENDER_MODEL_PATH,
        "person_model_path": PERSON_DETECTOR_MODEL_PATH,
    }
