from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DetectionResult:
    """Container for a single detection and its age/gender estimate."""

    bbox: Tuple[int, int, int, int]
    confidence: float
    age_range: Optional[str]
    age_estimate: Optional[float]
    gender_label: Optional[str]
    class_label: Optional[str]
    source: str = "age_gender"

