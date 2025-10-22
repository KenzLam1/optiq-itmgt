from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Sequence

import pandas as pd

DATA_DIR = Path("data")
LOG_PATH = DATA_DIR / "detections_log.csv"

LOG_COLUMNS: Sequence[str] = (
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


def append_detection_logs(entries: Iterable[dict[str, Any]]) -> None:
    rows = list(entries)
    if not rows:
        return

    _ensure_data_dir()

    df = pd.DataFrame(rows, columns=LOG_COLUMNS)
    if "logged_at" in df.columns:
        df["logged_at"] = pd.to_datetime(df["logged_at"])

    if LOG_PATH.exists():
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, mode="w", header=True, index=False)


def load_detection_logs() -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)

    df = pd.read_csv(LOG_PATH, parse_dates=["logged_at"])
    return df

