from typing import Any, Callable


def call_with_width(
    fn: Callable[..., Any],
    *args: Any,
    stretch: bool = True,
    **kwargs: Any,
) -> Any:
    """Streamlit width helper compatible with multiple releases."""
    width_value = "stretch" if stretch else "content"
    try:
        return fn(*args, width=width_value, **kwargs)
    except TypeError as exc:
        if "width" not in str(exc):
            raise
        return fn(*args, use_container_width=stretch, **kwargs)

