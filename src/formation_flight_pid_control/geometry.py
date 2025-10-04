from __future__ import annotations

from typing import Final

import numpy as np

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


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Return a unit-length copy of ``quaternion``."""

    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("Quaternion norm is zero")
    return q / norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (w, x, y, z)."""

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_to_rotmat(quaternion: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping body-frame vectors to world frame."""

    qw, qx, qy, qz = normalize_quaternion(quaternion)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )


def world2body(quaternion: np.ndarray) -> np.ndarray:
    """Rotation matrix from world (NED) frame to aircraft body frame."""

    return quat_to_rotmat(quaternion).T
