from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Union

import pandas as pd

from detections import DetectionLogEntry

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "detections.db"

TABLE_NAME = "detections"
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

#creates the data directory if it does not exist
def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

#connects to the sqlite database, creating it if it does not exist
def _connect() -> sqlite3.Connection:
    _ensure_data_dir()
    conn = sqlite3.connect(DB_PATH) # Opens (or creates, if missing) the database file
    conn.execute("PRAGMA journal_mode=WAL;") # configure sqlite with WAL mode (suposedly faster)
    conn.execute("PRAGMA synchronous=NORMAL;") # configure sqlite with normal sync
    return conn

#Makes sure the right DB structure (schema) exists
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


def initialize_database() -> None:
    with _connect() as conn:
        _ensure_schema(conn)


def _normalize_row(row: Union[DetectionLogEntry, Dict[str, Any]]) -> Dict[str, Any]:
    payload = row.to_row() if isinstance(row, DetectionLogEntry) else row
    normalized: Dict[str, Any] = {}
    for column in TABLE_COLUMNS:
        value = payload.get(column)
        if column == "logged_at" and value is not None:
            if hasattr(value, "isoformat"):
                value = value.isoformat()
        normalized[column] = value
    return normalized


def append_detection_logs(entries: Iterable[Union[DetectionLogEntry, Dict[str, Any]]]) -> None:
    rows = [_normalize_row(entry) for entry in entries]
    if not rows:
        return

    with _connect() as conn:
        _ensure_schema(conn)
        placeholders = ",".join("?" for _ in TABLE_COLUMNS)
        insert_sql = f"""
            INSERT INTO {TABLE_NAME} ({",".join(TABLE_COLUMNS)})
            VALUES ({placeholders});
        """
        values = [tuple(row.get(col) for col in TABLE_COLUMNS) for row in rows]
        conn.executemany(insert_sql, values)
        conn.commit()


def clear_detection_logs() -> None:
    if not DB_PATH.exists():
        return
    with _connect() as conn:
        _ensure_schema(conn)
        conn.execute(f"DELETE FROM {TABLE_NAME};")
        conn.commit()

#Pull detection logs from the sql database and return as a pandas DataFrame.
def load_detection_logs() -> pd.DataFrame:
    if not DB_PATH.exists(): # if database does not exist, return empty DataFrame with expected table columns
        return pd.DataFrame(columns=TABLE_COLUMNS)

    with _connect() as conn:
        _ensure_schema(conn)
        query = f"SELECT * FROM {TABLE_NAME} ORDER BY logged_at;"
        df = pd.read_sql_query(query, conn)
    if df.empty:
        return df
    df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["logged_at"])
    if df.empty:
        return df
    df["logged_at"] = df["logged_at"].dt.tz_convert("Asia/Manila")
    return df
