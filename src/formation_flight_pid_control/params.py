from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Params:
    mtom: float = 16.0
    gravity: float = 9.81
    S_wing_ref_area: float = 0.6
    c_bar: float = 0.5
    x_ac: float = 0.2
    x_cg: float = 0.15
    b_span: float = 2.4
    AR: float = b_span / c_bar
    rho: float = 1.225
    e_const: float = 0.6
    alpha_stall: float = 0.25

    thrust_max: float = 35.0

    CL0: float = 0.0
    CL_alpha: float = 5.0
    CY_beta: float = -0.9
    CD0: float = 0.01
    CMAC: float = 0.05
    CL_max: float = CL0 + CL_alpha * alpha_stall
    Cl_beta: float = -0.05

    K_roll: float = 6.0 / 1000
    K_pitch: float = 12.0 / 1000
    K_yaw: float = 0.8 / 1000

    Bp: float = 2.0
    Bq: float = 2.8
    Br: float = 2.0

    J: np.ndarray = field(default_factory=lambda: np.diag([0.35, 0.45, 0.60]))

    v_eps: float = 1e-3
    cth_eps: float = 1e-3
