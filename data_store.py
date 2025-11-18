from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Union

import pandas as pd

from detections import DetectionLogEntry

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "detections.db"
LEGACY_CSV_PATH = DATA_DIR / "detections_log.csv"

TABLE_NAME = "detections"
TABLE_COLUMNS: Sequence[str] = (
    "run_id",
    "logged_at",
    "frame_idx",
    "processed_frame",
    "source",
    "label",
    "gender",
    "age_range",
    "age_estimate",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "center_x",
    "center_y",
    "frame_width",
    "frame_height",
)


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    _ensure_data_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


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
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_logged_at ON {TABLE_NAME} (logged_at);"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_run_id ON {TABLE_NAME} (run_id);"
    )


def initialize_database() -> None:
    with _connect() as conn:
        _ensure_schema(conn)
    _bootstrap_from_legacy_csv_if_needed()


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


def load_detection_logs() -> pd.DataFrame:
    _bootstrap_from_legacy_csv_if_needed()
    if not DB_PATH.exists():
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


def _bootstrap_from_legacy_csv_if_needed() -> None:
    if not LEGACY_CSV_PATH.exists():
        return

    db_exists = DB_PATH.exists()
    with _connect() as conn:
        _ensure_schema(conn)
        if db_exists:
            count = conn.execute(f"SELECT COUNT(1) FROM {TABLE_NAME};").fetchone()[0]
            if count:
                return

        legacy_df = pd.read_csv(LEGACY_CSV_PATH)
        if legacy_df.empty:
            return

        if "logged_at" in legacy_df.columns:
            legacy_df["logged_at"] = pd.to_datetime(legacy_df["logged_at"], errors="coerce", utc=True)
            legacy_df = legacy_df.dropna(subset=["logged_at"])
            legacy_df["logged_at"] = legacy_df["logged_at"].dt.tz_convert("Asia/Manila")

        missing_cols = [col for col in TABLE_COLUMNS if col not in legacy_df.columns]
        for col in missing_cols:
            legacy_df[col] = None
        legacy_df = legacy_df[list(TABLE_COLUMNS)]

        placeholders = ",".join("?" for _ in TABLE_COLUMNS)
        insert_sql = f"""
            INSERT INTO {TABLE_NAME} ({",".join(TABLE_COLUMNS)})
            VALUES ({placeholders});
        """
        values = [
            tuple(
                row[col].isoformat() if col == "logged_at" and hasattr(row[col], "isoformat") else row[col]
                for col in TABLE_COLUMNS
            )
            for row in legacy_df.to_dict(orient="records")
        ]
        conn.executemany(insert_sql, values)
        conn.commit()
