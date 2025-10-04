from __future__ import annotations

from typing import Final, Sequence

import numpy as np
import quaternion

AIRCRAFT_GEOMETRY_BODY: Final = {
    "nose": np.array([40.0, 0.0, 0.0]),
    "tail": np.array([-40.0, 0.0, 0.0]),
    "wing_r": np.array([0.0, 44.0, 0.0]),
    "wing_l": np.array([0.0, -44.0, 0.0]),
    "tail_r": np.array([-40.0, 16.0, 0.0]),
    "tail_l": np.array([-40.0, -16.0, 0.0]),
    "tail_v": np.array([-40.0, 0.0, -16.0]),
}

NED_TO_ENU: Final = np.array([
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, -1],
])


def _normalized_quaternion(quat_like: Sequence[float]) -> quaternion.quaternion:
    quat_array = np.asarray(quat_like, dtype=float)
    if quat_array.shape != (4,):
        raise ValueError("Quaternion must have 4 components (w, x, y, z)")
    quat = quaternion.quaternion(*quat_array)
    return quat / np.abs(quat)


def rotate_body_to_earth(quat_like: Sequence[float], vector_body: Sequence[float]) -> np.ndarray:
    """Rotate a body-frame vector into the earth frame using ``quat_like``."""
    quat = _normalized_quaternion(quat_like)
    vec = np.asarray(vector_body, dtype=float)
    rotated = quaternion.rotate_vectors(quat, vec)
    return np.asarray(rotated, dtype=float)


def rotate_earth_to_body(quat_like: Sequence[float], vector_earth: Sequence[float]) -> np.ndarray:
    """Rotate an earth-frame vector into the body frame using ``quat_like``."""
    quat = _normalized_quaternion(quat_like).conjugate()
    vec = np.asarray(vector_earth, dtype=float)
    rotated = quaternion.rotate_vectors(quat, vec)
    return np.asarray(rotated, dtype=float)
