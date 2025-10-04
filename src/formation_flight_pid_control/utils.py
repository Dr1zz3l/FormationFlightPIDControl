from __future__ import annotations


def clamp(value: float, lower: float, upper: float) -> float:
    """Return *value* limited to the inclusive range [lower, upper]."""
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value
