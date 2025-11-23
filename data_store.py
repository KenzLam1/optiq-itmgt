from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Union

import pandas as pd

from detections import DetectionLogEntry

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "detections.db"

TABLE_NAME = "detections"
"""Tuple of all the columns names in the detection logs table"""
TABLE_COLUMNS: Sequence[str] = ( 
    "run_id",               #ID for a specific run/session of your video analysis.
    "logged_at",            #Timestamp indicating when the detection was logged.
    "frame_idx",            #Index of the frame in the video where the detection occurred.
    "processed_frame",      #Indicates whether the frame was processed (1) or not (0).                     
    "source",               #Source of the video (e.g., file or camera ID).
    "label",                #Label assigned to the detected object (e.g., "person").
    "gender",               #Estimated gender of the detected person.
    "age_range",            #Estimated age range of the detected person (e.g., "20-30").
    "age_estimate",         #Estimated age of the detected person as a numerical value.
    "confidence",           #Confidence score of the detection (e.g., 0.85 for 85% confidence).
    "bbox_x",               #X-coordinate of the bounding box for the detected object.
    "bbox_y",               #Y-coordinate of the bounding box for the detected object.
    "bbox_w",               #Width of the bounding box for the detected object.
    "bbox_h",               #Height of the bounding box for the detected object.
    "center_x",             #X-coordinate of the center point of the detected object.
    "center_y",             #Y-coordinate of the center point of the detected object.
    "frame_width",          #Width of the video frame.
    "frame_height",         #Height of the video frame.
)

"""creates the data directory if it does not exist"""
def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

"""connects to the sqlite database, creating it if it does not exist"""
def _connect() -> sqlite3.Connection:
    _ensure_data_dir()
    conn = sqlite3.connect(DB_PATH) # Opens (or creates, if missing) the database file
    conn.execute("PRAGMA journal_mode=WAL;") # configure sqlite with WAL mode (suposedly faster)
    conn.execute("PRAGMA synchronous=NORMAL;") # configure sqlite with normal sync
    return conn

"""Makes sure the right DB structure (schema) exists""" 
def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(   
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            run_id TEXT NOT NULL,
            logged_at TEXT NOT NULL,
            frame_idx INTEGER,
            processed_frame INTEGER,
            source TEXT,
            label TEXT,
            gender TEXT,
            age_range TEXT,
            age_estimate REAL,
            confidence REAL,
            bbox_x REAL,
            bbox_y REAL,
            bbox_w REAL,
            bbox_h REAL,
            center_x REAL,
            center_y REAL,
            frame_width INTEGER,
            frame_height INTEGER
        );
        """
    )
    # Create indexes for faster querying for timestamp and run_id
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_logged_at ON {TABLE_NAME} (logged_at);")
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_run_id ON {TABLE_NAME} (run_id);")

"""Initializes the database by ensuring the data directory and schema exist"""
def initialize_database() -> None:
    with _connect() as conn:
        _ensure_schema(conn)

"""Turn any row into a dictionary with the right columns and formats so its ready to go into the database"""
def _normalize_row(row: Union[DetectionLogEntry, Dict[str, Any]]) -> Dict[str, Any]: 
    payload = row.to_row() if isinstance(row, DetectionLogEntry) else row   #If row is a DetectionLogEntry, convert it to a dictionary; otherwise, assume it's already a dictionary.
    normalized: Dict[str, Any] = {}     # Prepare an empty dictionary to hold values.  
    for column in TABLE_COLUMNS:
        value = payload.get(column) # Get the value for each expected column
        if column == "logged_at" and value is not None: # Convert datetime to ISO format string for storage 
            if hasattr(value, "isoformat"):
                value = value.isoformat()
        normalized[column] = value  # Assign the (possibly converted) value to the normalized dictionary
    return normalized

"""Append multiple detection log entries to the SQLite database"""
def append_detection_logs(entries: Iterable[Union[DetectionLogEntry, Dict[str, Any]]]) -> None:
    rows = [_normalize_row(entry) for entry in entries] # Normalize each entry to ensure it matches the database schema
    if not rows:
        return  # if there are no rows to insert, exit early

    with _connect() as conn:    # Connect to the database
        _ensure_schema(conn)    # Ensure the database schema is set up before inserting data
        placeholders = ",".join("?" for _ in TABLE_COLUMNS) # Build the correct number of ? placeholders for the SQL insert
        # sample: INSERT INTO detections (run_id, logged_at, frame_idx, ...) VALUES (?, ?, ?, ...)
        insert_sql = f"""       
            INSERT INTO {TABLE_NAME} ({",".join(TABLE_COLUMNS)})    
            VALUES ({placeholders});    
        """
        values = [tuple(row.get(col) for col in TABLE_COLUMNS) for row in rows]  # Prepare the values for insertion. For each row dict, create a tuple of values in the correct order.
        conn.executemany(insert_sql, values)    # Execute the batch insert. Runs insert statement for every tuple in values.
        conn.commit()   

"""Clear all detection logs from the database"""
def clear_detection_logs() -> None:
    if not DB_PATH.exists(): 
        return                  
    with _connect() as conn:    
        _ensure_schema(conn)    
        conn.execute(f"DELETE FROM {TABLE_NAME};")  
        conn.commit()   

"""Pull detection logs from the sql database and return as a pandas DataFrame. Cleans and adjusts timezone as needed."""
def load_detection_logs() -> pd.DataFrame:
    if not DB_PATH.exists(): # if database does not exist, return empty DataFrame with expected table columns
        return pd.DataFrame(columns=TABLE_COLUMNS)

    with _connect() as conn:
        _ensure_schema(conn)
        query = f"SELECT * FROM {TABLE_NAME} ORDER BY logged_at;"   # Query to select all detection logs ordered by timestamp
        df = pd.read_sql_query(query, conn) # Load the query results into a pandas DataFrame

    if df.empty:
        return df   # Early return if DataFrame exists but has no data  
    
    df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True)  # Convert logged_at column to pandas datetime. Invalid timestamps become NaT.
    df = df.dropna(subset=["logged_at"])    # Drop rows where logged_at is NaT (invalid timestamps)

    if df.empty:
        return df   # Early return if DataFrame is empty after dropping invalid timestamps
    
    df["logged_at"] = df["logged_at"].dt.tz_convert("Asia/Manila")  # Convert timestamps to Manila timezone
    return df   # Return the cleaned and timezone-adjusted DataFrame
