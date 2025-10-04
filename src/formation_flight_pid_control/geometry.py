from __future__ import annotations

import math
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


def earth2body(phi: float, theta: float, psi: float) -> np.ndarray:
    """Rotation matrix from earth (NED) frame to aircraft body frame."""
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)

    r_roll = np.array(
        [
            [1, 0, 0],
            [0, c_phi, s_phi],
            [0, -s_phi, c_phi],
        ]
    )

    r_pitch = np.array(
        [
            [c_theta, 0, -s_theta],
            [0, 1, 0],
            [s_theta, 0, c_theta],
        ]
    )

    r_yaw = np.array(
        [
            [c_psi, s_psi, 0],
            [-s_psi, c_psi, 0],
            [0, 0, 1],
        ]
    )

    return r_roll @ r_pitch @ r_yaw


def euler_rates_matrix(phi: float, theta: float) -> np.ndarray:
    """Map body angular rates to Euler angle rates for ZYX rotations."""
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    sin_theta, cos_theta = math.sin(theta), math.cos(theta)
    tan_theta = math.tan(theta)

    return np.array(
        [
            [1.0, sin_phi * tan_theta, cos_phi * tan_theta],
            [0.0, cos_phi, -sin_phi],
            [0.0, sin_phi / cos_theta, cos_phi / cos_theta],
        ]
    )
