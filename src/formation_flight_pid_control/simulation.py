from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .geometry import normalize_quaternion, quat_multiply, world2body
from .params import Params
from .utils import clamp


class Airplane6DoFLite:
    """Lightweight 6-DoF rigid-body airplane model.

    The simulator follows the standard equations of motion for a small aircraft
    with the state vector ``[x, y, z, u, v, w, q_w, q_x, q_y, q_z, p, q, r]``.
    Position ``(x, y, z)`` is expressed in the world/inertial frame using the
    aerospace convention of +Z downward. The velocity components ``(u, v, w)``
    and angular rates ``(p, q, r)`` live in the body frame. Attitude is stored
    as a quaternion that rotates vectors from the world frame into the body
    frame. Each step performs the following operations:

    * Convert the body-frame velocity and attitude quaternion into world-frame
      orientation/velocity so that forces can be resolved consistently.
    * Evaluate aerodynamic force and moment coefficients based on angle of
      attack ``alpha`` and sideslip ``beta`` and apply simple stall and damping
      models.
    * Sum thrust, aerodynamic, and external forces/moments; then transform them
      to the world frame to obtain linear acceleration while adding gravity.
    * Integrate position, velocity, attitude, and angular rate using a fixed
      time-step 4th-order Runge–Kutta scheme.

    The model is intentionally simple (no landing gear, symmetric inertia, and
    small-angle control derivatives) yet sufficient for control testing.
    """
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
        R_bw = world2body(quat).T  # rotation that maps body-frame vectors into world frame

        vel_body = np.array([vx, vy, vz])
        u_body, v_body, w_body = vel_body
        vel_norm = max(aircraft.v_eps, float(np.linalg.norm(vel_body)))

        alpha = math.atan2(w_body, u_body)
        beta = math.asin(clamp(v_body / vel_norm, -1.0, 1.0))

        qbar = 0.5 * aircraft.rho * vel_norm**2

        T_force = aircraft.thrust_max * clamp(throttle, 0.0, 1.0)
        F_thrust_body = np.array([T_force, 0.0, 0.0])  # thrust along body x-axis

        V_hat = vel_body / vel_norm

        if alpha <= aircraft.alpha_stall:
            CL_tot = aircraft.CL0 + aircraft.CL_alpha * alpha
        else:
            CL_tot = aircraft.CL_max * (
                1 - (abs(alpha) - aircraft.alpha_stall) / (np.pi / 2 - aircraft.alpha_stall)
            )
            CL_tot *= math.copysign(1.0, alpha)

        L_force = CL_tot * qbar * aircraft.S_wing_ref_area
        lift_dir = np.cross(np.cross(V_hat, np.array([0.0, 1.0, 0.0])), V_hat)
        lift_norm = np.linalg.norm(lift_dir)
        if lift_norm > aircraft.v_eps:
            lift_dir /= lift_norm
        else:
            lift_dir = np.zeros(3)
        F_lift_body = L_force * lift_dir  # lift always perpendicular to velocity

        CY_tot = aircraft.CY_beta * beta
        Y_force = CY_tot * qbar * aircraft.S_wing_ref_area
        side_dir = np.cross(V_hat, lift_dir)
        side_norm = np.linalg.norm(side_dir)
        if side_norm > aircraft.v_eps:
            side_dir /= side_norm
        else:
            side_dir = np.zeros(3)
        F_side_body = Y_force * side_dir

        CD_induced = CL_tot**2 / (np.pi * aircraft.AR * aircraft.e_const)
        CD_tot = aircraft.CD0 + CD_induced
        D_force = CD_tot * qbar * aircraft.S_wing_ref_area
        drag_dir = -V_hat
        F_drag_body = D_force * drag_dir  # drag opposes relative airflow

        F_body = F_side_body + F_thrust_body + F_lift_body + F_drag_body

        if ext_F_body is not None:
            F_body = F_body + ext_F_body

        # Resolve forces in world frame and add gravity acting in +Z direction
        F_world = R_bw @ F_body + np.array([0.0, 0.0, aircraft.mtom * aircraft.gravity])

        M_cmd = np.array(
            [
                aircraft.K_roll * clamp(roll_cmd, -1.0, 1.0),
                aircraft.K_pitch * clamp(pitch_cmd, -1.0, 1.0),
                aircraft.K_yaw * clamp(yaw_cmd, -1.0, 1.0),
            ]
        ) * qbar
        # Linear damping models rotational friction around each axis
        M_damp = -np.array([aircraft.Bp * p_rate, aircraft.Bq * q_rate, aircraft.Br * r_rate])

        C_M = aircraft.CMAC + (aircraft.x_cg - aircraft.x_ac) / aircraft.c_bar * CL_tot
        M_pitch_CG_AC = qbar * aircraft.S_wing_ref_area * aircraft.c_bar * C_M
        M_CG_AC = np.array([0.0, M_pitch_CG_AC, 0.0])  # pitching moment from lift offset

        Cl_beta = aircraft.Cl_beta
        L_roll = Cl_beta * beta * qbar * aircraft.S_wing_ref_area * aircraft.b_span
        M_lateral_body = np.array([L_roll, 0.0, 0.0])  # lateral force induces rolling moment

        M_body = M_damp + M_cmd + M_CG_AC + M_lateral_body

        if ext_M_body is not None:
            M_body = M_body + ext_M_body

        return F_world, M_body

    def f(
        self,
        state: np.ndarray,
        u: np.ndarray,
        ext_F_body: Optional[np.ndarray] = None,
        ext_M_body: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        aircraft = self.aircraft
        x, y, z, vx, vy, vz, qw, qx, qy, qz, p_rate, q_rate, r_rate = state

        quat = normalize_quaternion(np.array([qw, qx, qy, qz]))
        R_bw = world2body(quat).T

        F_world, M_body = self.forces_and_moments(state, u, ext_F_body, ext_M_body)

        accel_world = F_world / aircraft.mtom  # Newton's 2nd law in world frame

        omega = np.array([p_rate, q_rate, r_rate])
        # Euler rotational dynamics with diagonal inertia tensor
        omega_dot = self.Jinv @ (M_body - np.cross(omega, self.J @ omega))

        omega_quat = np.array([0.0, *omega])
        quat_dot = 0.5 * quat_multiply(quat, omega_quat)

        xdot = np.zeros_like(state)
        vel_body = np.array([vx, vy, vz])
        # Integrate position in world frame using transformed body velocity
        xdot[0:3] = R_bw @ vel_body
        # Translational acceleration, quaternion rate, and angular acceleration
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
