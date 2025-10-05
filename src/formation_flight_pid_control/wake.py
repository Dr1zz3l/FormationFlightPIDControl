"""Wake vortex data structures and helper functions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VortexBead:
    pos_w: np.ndarray
    gamma: float
    age: float


@dataclass
class TipWake:
    beads: list[VortexBead] = field(default_factory=list)


@dataclass
class AircraftWake:
    left: TipWake = field(default_factory=TipWake)
    right: TipWake = field(default_factory=TipWake)


def unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return np.zeros_like(v)
    return v / norm


def lamb_oseen_induced_vel_at_point(
    point_w: np.ndarray,
    line_point_w: np.ndarray,
    line_dir_w: np.ndarray,
    gamma: float,
    core_radius: float,
) -> np.ndarray:
    """Return the induced velocity using a Lamb–Oseen-smoothed vortex line."""

    r_vec = point_w - line_point_w
    t_hat = unit(line_dir_w)
    r_ax = np.dot(r_vec, t_hat) * t_hat
    r_perp = r_vec - r_ax
    r_mag = float(np.linalg.norm(r_perp))
    if r_mag < 1e-6:
        return np.zeros(3)

    r_hat = r_perp / r_mag
    theta_hat = np.cross(t_hat, r_hat)

    v_theta = (gamma / (2.0 * np.pi * r_mag)) * (
        1.0 - np.exp(-(r_mag * r_mag) / (core_radius * core_radius))
    )
    return v_theta * theta_hat


def axial_decay_factor(axial_dist: float, Ld: float) -> float:
    if Ld <= 0.0:
        return 1.0
    return float(np.exp(-max(0.0, axial_dist) / Ld))

