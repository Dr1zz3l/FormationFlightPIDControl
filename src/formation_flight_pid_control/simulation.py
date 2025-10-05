from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import quaternion

from .geometry import rotate_body_to_earth, rotate_earth_to_body, _normalized_quaternion
from .params import Params
from .utils import clamp


@dataclass
class AirplaneState:
    position: np.ndarray
    pose: np.quaternion
    velocity: np.ndarray
    angular_rates: np.ndarray


class Airplane6DoFLite:
    """Lightweight 6-DoF rigid-body aircraft model.

    The model integrates translational and rotational dynamics driven by
    aerodynamic lift, drag, side-force, thrust, and simple control-derived
    moments. Aerodynamic coefficients follow a quasi-steady, low-angle
    approximation with a linear lift curve up to a prescribed stall angle and
    induced drag based on span efficiency. Attitude is represented with
    quaternions and integrated alongside angular rates to avoid gimbal lock.
    """
    def __init__(self, params: Params, state: AirplaneState = None):
        self.params = params
        self.J = np.diag([params.Jx, params.Jy, params.Jz])
        self.Jinv = np.diag([1.0 / params.Jx, 1.0 / params.Jy, 1.0 / params.Jz])

        self.state = state if state is not None else AirplaneState(
            position=np.array([200.0, 200.0, -700.0]),
            pose=np.quaternion(1.0, 0.0, 0.0, 0.0),
            velocity=np.array([26.0, 0.0, 0.0]),
            angular_rates=np.array([0.0, 0.0, 0.0]),
        )

    def forces_and_moments(
        self,
        state: AirplaneState,
        u: np.ndarray,
        ext_F_body: Optional[np.ndarray] = None,
        ext_M_body: Optional[np.ndarray] = None,
    ):
        aircraft = self.params
        throttle, roll_cmd, pitch_cmd, yaw_cmd = u

        # Extract state components in a clean way
        position = state.position
        velocity = state.velocity
        pose = state.pose
        angular_rates = state.angular_rates
        
        quat = _normalized_quaternion([pose.w, pose.x, pose.y, pose.z])
        quat_array = np.array([quat.w, quat.x, quat.y, quat.z])
        vel_body = rotate_earth_to_body(quat_array, velocity)
        u_body, v_body, w_body = vel_body
        vel_norm = max(aircraft.v_eps, float(np.linalg.norm(vel_body)))
        p_rate, q_rate, r_rate = angular_rates

        alpha = math.atan2(w_body, u_body)
        beta = math.asin(clamp(v_body / vel_norm, -1.0, 1.0))

        # Dynamic pressure for wing forces that scale with velocity squared.
        qbar = 0.5 * aircraft.rho * vel_norm**2

        T_force = aircraft.thrust_max * clamp(throttle, 0.0, 1.0)
        F_thrust_body = np.array([T_force, 0.0, 0.0])

        V_hat = vel_body / vel_norm

        # Lift follows a linear curve up to the stall angle, then tapers.
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

        # Side-force from sideslip is aligned with the lateral unit vector.
        CY_tot = aircraft.CY_beta * beta
        Y_force = CY_tot * qbar * aircraft.S_wing_ref_area
        side_dir = np.cross(V_hat, lift_dir)
        side_dir /= np.linalg.norm(side_dir)
        F_side_body = Y_force * side_dir

        # Total drag is zero-lift plus induced drag from lift production.
        CD_induced = CL_tot**2 / (np.pi * aircraft.AR * aircraft.e_const)
        CD_tot = aircraft.CD0 + CD_induced
        D_force = CD_tot * qbar * aircraft.S_wing_ref_area
        drag_dir = -V_hat
        drag_dir /= np.linalg.norm(drag_dir)
        F_drag_body = D_force * drag_dir

        # Sum of aerodynamic and thrust forces expressed in the body frame.
        F_body = F_side_body + F_thrust_body + F_lift_body + F_drag_body

        if ext_F_body is not None:
            F_body = F_body + ext_F_body

        F_earth = rotate_body_to_earth(quat_array, F_body) + np.array(
            [0.0, 0.0, aircraft.mtom * aircraft.gravity]
        )

        M_cmd = np.array(
            [
                aircraft.K_roll * clamp(roll_cmd, -1.0, 1.0),
                aircraft.K_pitch * clamp(pitch_cmd, -1.0, 1.0),
                aircraft.K_yaw * clamp(yaw_cmd, -1.0, 1.0),
            ]
        ) * qbar
        # Linear damping moments oppose rotation about each axis.
        M_damp = -np.array([aircraft.Bp * p_rate, aircraft.Bq * q_rate, aircraft.Br * r_rate])

        C_M = aircraft.CMAC + (aircraft.x_cg - aircraft.x_ac) / aircraft.c_bar * CL_tot
        M_pitch_CG_AC = qbar * aircraft.S_wing_ref_area * aircraft.c_bar * C_M
        M_CG_AC = np.array([0.0, M_pitch_CG_AC, 0.0])

        Cl_beta = aircraft.Cl_beta
        L_roll = Cl_beta * beta * qbar * aircraft.S_wing_ref_area * aircraft.b_span
        M_lateral_body = np.array([L_roll, 0.0, 0.0])

        # Net moments include control commands, damping, and aerodynamic coupling.
        M_body = M_damp + M_cmd + M_CG_AC + M_lateral_body

        if ext_M_body is not None:
            M_body = M_body + ext_M_body

        return F_earth, M_body

    def f(
        self,
        state: AirplaneState,
        u: np.ndarray,
        ext_F_body: Optional[np.ndarray] = None,
        ext_M_body: Optional[np.ndarray] = None,
    ) -> AirplaneState:
        aircraft = self.params

        F_earth, M_body = self.forces_and_moments(state, u, ext_F_body, ext_M_body)

        # Newton's second law for translational acceleration in earth axes.
        accel_world = F_earth / aircraft.mtom

        omega = state.angular_rates
        # Euler's equation for rotational acceleration with gyroscopic coupling.
        omega_dot = self.Jinv @ (M_body - np.cross(omega, self.J @ omega))

        # For quaternion derivatives, use the standard quaternion kinematics
        quat = _normalized_quaternion([state.pose.w, state.pose.x, state.pose.y, state.pose.z])
        omega_quat = quaternion.quaternion(0.0, *omega)
        quat_dot = 0.5 * quat * omega_quat

        # Return the derivatives as an AirplaneState
        return AirplaneState(
            position=state.velocity,  # dx/dt = velocity
            velocity=accel_world,     # dv/dt = acceleration
            pose=quat_dot,           # dq/dt = quaternion derivative
            angular_rates=omega_dot   # domega/dt = angular acceleration
        )

    def step(
        self,
        u: np.ndarray,
        dt: float,
        ext_F_body: Optional[np.ndarray] = None,
        ext_M_body: Optional[np.ndarray] = None,
    ) -> AirplaneState:
        state = self.state
        k1 = self.f(state, u, ext_F_body, ext_M_body)
        k2 = self.f(self._add_state_derivative(state, k1, 0.5 * dt), u, ext_F_body, ext_M_body)
        k3 = self.f(self._add_state_derivative(state, k2, 0.5 * dt), u, ext_F_body, ext_M_body)
        k4 = self.f(self._add_state_derivative(state, k3, dt), u, ext_F_body, ext_M_body)
        
        self.state = self._rk4_combine(state, k1, k2, k3, k4, dt)
        
        # Ensure quaternion stays normalized
        quat = _normalized_quaternion([self.state.pose.w, self.state.pose.x, self.state.pose.y, self.state.pose.z])
        self.state.pose = quat
        
        return self.state
    
    def _add_state_derivative(self, state: AirplaneState, derivative: AirplaneState, dt: float) -> AirplaneState:
        """Add a scaled derivative to a state for RK4 intermediate steps."""
        # For quaternions, add the quaternion derivative and then normalize
        new_pose = state.pose + dt * derivative.pose
        new_pose = new_pose.normalized()  # Keep quaternion normalized
        
        return AirplaneState(
            position=state.position + dt * derivative.position,
            velocity=state.velocity + dt * derivative.velocity,
            pose=new_pose,
            angular_rates=state.angular_rates + dt * derivative.angular_rates
        )
    
    def _rk4_combine(self, state: AirplaneState, k1: AirplaneState, k2: AirplaneState, k3: AirplaneState, k4: AirplaneState, dt: float) -> AirplaneState:
        """Combine RK4 derivatives with proper quaternion integration."""
        # Standard RK4 combination for quaternions (treating them as 4D vectors temporarily)
        new_pose = state.pose + (dt/6) * (k1.pose + 2*k2.pose + 2*k3.pose + k4.pose)
        new_pose = new_pose.normalized()  # Renormalize after integration
        
        return AirplaneState(
            position=state.position + (dt/6) * (k1.position + 2*k2.position + 2*k3.position + k4.position),
            velocity=state.velocity + (dt/6) * (k1.velocity + 2*k2.velocity + 2*k3.velocity + k4.velocity),
            pose=new_pose,
            angular_rates=state.angular_rates + (dt/6) * (k1.angular_rates + 2*k2.angular_rates + 2*k3.angular_rates + k4.angular_rates)
        )
