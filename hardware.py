from __future__ import annotations

from typing import Optional

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]


def select_runtime_device(explicit_device: Optional[str] = None) -> str:
    """Picks the best device for Ultralytics models."""
    if explicit_device:
        return explicit_device

    if torch is None:
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "cuda:0"     # means "first GPU" in pytorch
    except Exception:  # noqa: BLE001
        pass

    try:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass

    return "cpu"


def resolve_user_device_choice(choice: str) -> Optional[str]:
    """Maps sidebar device choices to device identifiers that the code understands."""
    normalized = choice.lower()    # makes the function case-insensitive
    if normalized == "auto":
        return None         # lets select_runtime_device pick the best device
    if normalized == "cuda":
        return "cuda:0"   # means "first GPU" in pytorch
    return normalized


def available_device_choices() -> tuple[list[str], bool]:
    """Return supported sidebar device options."""
    options: list[str] = ["auto", "cpu"]
    mps_available = False

    if torch is None:
        return options, mps_available

    try:
        if torch.cuda.is_available():
            options.append("cuda")
    except Exception:  # noqa: BLE001
        pass

    try:
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            options.append("mps")
            mps_available = True
    except Exception:  # noqa: BLE001
        pass

    return options, mps_available
