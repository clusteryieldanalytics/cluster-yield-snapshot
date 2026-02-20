"""Shared utility functions."""

from __future__ import annotations

from typing import Optional


def fmt_bytes(n: int | float | None) -> str:
    """Format byte count to human-readable string."""
    if n is None:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} PB"


def parse_int(val: str | int | None) -> Optional[int]:
    """Parse a config value to int, returning None on failure."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def esc_html(s: str) -> str:
    """Escape HTML entities."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
