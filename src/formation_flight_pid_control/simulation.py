from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .geometry import earth2body, normalize_quaternion, quat_multiply
from .params import Params
from .utils import clamp


class Airplane6DoFLite:
    def __init__(self, aircraft: Params):
        self.aircraft = aircraft
        self.J = np.diag([aircraft.Jx, aircraft.Jy, aircraft.Jz])
        self.Jinv = np.diag([1.0 / aircraft.Jx, 1.0 / aircraft.Jy, 1.0 / aircraft.Jz])
        self.state = np.zeros(13)
        self.state[2] = -700.0
        self.state[0] = 200.0
        self.state[1] = 200.0
        self.state[3] = 26.0
        self.state[6] = 1.0

    def forces_and_moments(
        self,
        state: np.ndarray,
        u: np.ndarray,
        ext_F_body: Optional[np.ndarray] = None,
        ext_M_body: Optional[np.ndarray] = None,
    ):
        aircraft = self.aircraft
        x, y, z, vx, vy, vz, qw, qx, qy, qz, p_rate, q_rate, r_rate = state
        throttle, roll_cmd, pitch_cmd, yaw_cmd = u

        quat = normalize_quaternion(np.array([qw, qx, qy, qz]))
        R_be = earth2body(quat)
        R_eb = R_be.T

        vel_body = R_be @ np.array([vx, vy, vz])
        u_body, v_body, w_body = vel_body
        vel_norm = max(aircraft.v_eps, float(np.linalg.norm(vel_body)))

        alpha = math.atan2(w_body, u_body)
        beta = math.asin(clamp(v_body / vel_norm, -1.0, 1.0))

        qbar = 0.5 * aircraft.rho * vel_norm**2

        T_force = aircraft.thrust_max * clamp(throttle, 0.0, 1.0)
        F_thrust_body = np.array([T_force, 0.0, 0.0])

        V_hat = vel_body / vel_norm

        if alpha <= aircraft.alpha_stall:
            CL_tot = aircraft.CL0 + aircraft.CL_alpha * alpha
        else:
            CL_tot = aircraft.CL_max * (
                1 - (abs(alpha) - aircraft.alpha_stall) / (np.pi / 2 - aircraft.alpha_stall)
            )
            CL_tot *= math.copysign(1.0, alpha)
            print("STALL")

        L_force = CL_tot * qbar * aircraft.S_wing_ref_area
        lift_dir = np.array([V_hat[2], 0, -V_hat[0]])
        lift_dir /= np.linalg.norm(lift_dir)
        F_lift_body = L_force * lift_dir

        CY_tot = aircraft.CY_beta * beta
        Y_force = CY_tot * qbar * aircraft.S_wing_ref_area
        side_dir = np.cross(V_hat, lift_dir)
        side_dir /= np.linalg.norm(side_dir)
        F_side_body = Y_force * side_dir

        CD_induced = CL_tot**2 / (np.pi * aircraft.AR * aircraft.e_const)
        CD_tot = aircraft.CD0 + CD_induced
        D_force = CD_tot * qbar * aircraft.S_wing_ref_area
        drag_dir = -V_hat
        drag_dir /= np.linalg.norm(drag_dir)
        F_drag_body = D_force * drag_dir

        F_body = F_side_body + F_thrust_body + F_lift_body + F_drag_body

        if ext_F_body is not None:
            F_body = F_body + ext_F_body

        F_earth = R_eb @ F_body + np.array([0.0, 0.0, aircraft.mtom * aircraft.gravity])

        M_cmd = np.array(
            [
                aircraft.K_roll * clamp(roll_cmd, -1.0, 1.0),
                aircraft.K_pitch * clamp(pitch_cmd, -1.0, 1.0),
                aircraft.K_yaw * clamp(yaw_cmd, -1.0, 1.0),
            ]
        ) * qbar
        M_damp = -np.array([aircraft.Bp * p_rate, aircraft.Bq * q_rate, aircraft.Br * r_rate])

        C_M = aircraft.CMAC + (aircraft.x_cg - aircraft.x_ac) / aircraft.c_bar * CL_tot
        M_pitch_CG_AC = qbar * aircraft.S_wing_ref_area * aircraft.c_bar * C_M
        M_CG_AC = np.array([0.0, M_pitch_CG_AC, 0.0])

        Cl_beta = aircraft.Cl_beta
        L_roll = Cl_beta * beta * qbar * aircraft.S_wing_ref_area * aircraft.b_span
        M_lateral_body = np.array([L_roll, 0.0, 0.0])

        M_body = M_damp + M_cmd + M_CG_AC + M_lateral_body

        if ext_M_body is not None:
            M_body = M_body + ext_M_body

        return F_earth, M_body

    def f(
        self,
        state: np.ndarray,
        u: np.ndarray,
        ext_F_body: Optional[np.ndarray] = None,
        ext_M_body: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        aircraft = self.aircraft
        x, y, z, vx, vy, vz, qw, qx, qy, qz, p_rate, q_rate, r_rate = state

        F_earth, M_body = self.forces_and_moments(state, u, ext_F_body, ext_M_body)

        accel_world = F_earth / aircraft.mtom

        omega = np.array([p_rate, q_rate, r_rate])
        omega_dot = self.Jinv @ (M_body - np.cross(omega, self.J @ omega))

        quat = normalize_quaternion(np.array([qw, qx, qy, qz]))
        omega_quat = np.array([0.0, *omega])
        quat_dot = 0.5 * quat_multiply(quat, omega_quat)

        xdot = np.zeros_like(state)
        xdot[0:3] = np.array([vx, vy, vz])
        xdot[3:6] = accel_world
        xdot[6:10] = quat_dot
        xdot[10:13] = omega_dot
        return xdot

    def step(
        self,
        u: np.ndarray,
        dt: float,
        ext_F_body: Optional[np.ndarray] = None,
        ext_M_body: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        state = self.state
        k1 = self.f(state, u, ext_F_body, ext_M_body)
        k2 = self.f(state + 0.5 * dt * k1, u, ext_F_body, ext_M_body)
        k3 = self.f(state + 0.5 * dt * k2, u, ext_F_body, ext_M_body)
        k4 = self.f(state + dt * k3, u, ext_F_body, ext_M_body)
        self.state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self.state[6:10] = normalize_quaternion(self.state[6:10])
        return self.state.copy()
