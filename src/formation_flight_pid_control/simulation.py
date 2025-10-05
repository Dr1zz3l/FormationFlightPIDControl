from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, NamedTuple

import numpy as np
import quaternion

from .geometry import rotate_vector_by_quaternion
from .params import Params
from .utils import clamp


class ControlInputs(NamedTuple):
    """Aircraft control inputs from pilot or autopilot."""
    throttle: float    # Engine throttle [0, 1]
    roll_cmd: float    # Roll command [-1, 1] 
    pitch_cmd: float   # Pitch command [-1, 1]
    yaw_cmd: float     # Yaw command [-1, 1]
    
    @classmethod
    def from_array(cls, u_array: np.ndarray) -> 'ControlInputs':
        """Create ControlInputs from 4-element numpy array."""
        if len(u_array) != 4:
            raise ValueError(f"Control array must have 4 elements, got {len(u_array)}")
        return cls(*u_array)


@dataclass(frozen=True)
class AirplaneState:
    position: np.ndarray
    pose: np.quaternion
    velocity: np.ndarray
    angular_rates: np.ndarray

class Airplane():
    def __init__(self, params: Params, state: AirplaneState = None) -> None:
        
        self.params = params
        
        # Inertia matrix and its inverse for rotational dynamics
        # Use the J matrix from params, or create individual components
        if hasattr(params, 'J') and isinstance(params.J, np.ndarray):
            self.J = params.J
            self.J_inv = np.linalg.inv(self.J)
        else:
            # Fallback: use individual moments of inertia
            Jx, Jy, Jz = 0.35, 0.45, 0.60  # Default values
            self.J = np.diag([Jx, Jy, Jz])
            self.J_inv = np.diag([1.0/Jx, 1.0/Jy, 1.0/Jz])
        
        self.state = state if state is not None else AirplaneState(
            position=np.array([200.0, 200.0, -700.0]),
            pose=np.quaternion(1.0, 0.0, 0.0, 0.0),
            velocity=np.array([26.0, 0.0, 0.0]),
            angular_rates=np.array([0.0, 0.0, 0.0]),
        )

    def step(
            self, 
            dt: float, 
            control_inputs: ControlInputs, 
            ext_F_body: np.ndarray = np.zeros(3), 
            ext_M_body: np.ndarray = np.zeros(3),
            ) -> None:
        
        """Advance the simulation by one time step using RK4 integration."""

        current_state = self.state
        
        # Calculate the 4 RK4 slopes
        k1 = self._dynamics(current_state, control_inputs, ext_F_body, ext_M_body)
        k2 = self._dynamics(self._add_states(current_state, k1, dt/2), control_inputs, ext_F_body, ext_M_body)
        k3 = self._dynamics(self._add_states(current_state, k2, dt/2), control_inputs, ext_F_body, ext_M_body)
        k4 = self._dynamics(self._add_states(current_state, k3, dt), control_inputs, ext_F_body, ext_M_body)
        
        # Combine using RK4 formula
        new_state = self._rk4_combine(current_state, k1, k2, k3, k4, dt)
        
        # Normalize quaternion to prevent drift
        pose_norm = np.abs(new_state.pose)
        if pose_norm < 1e-12:  # Quaternion became degenerate
            # Reset to identity quaternion if corrupted
            normalized_pose = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
        else:
            normalized_pose = new_state.pose / pose_norm
        new_state = replace(new_state, pose=normalized_pose)

        self.state = new_state

    def _calculate_forces_and_moments(self, state: AirplaneState, controls: ControlInputs, ext_F_body: np.ndarray, ext_M_body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate aerodynamic and thrust forces and moments."""
        
        # Unpack state and control inputs
        # Transform world velocity to body frame (world to body rotation)
        V_body = rotate_vector_by_quaternion(state.pose.conjugate(), state.velocity)
        
        # Calculate airspeed with safety check
        airspeed = np.linalg.norm(V_body)
        if airspeed < 1e-6:  # Avoid division by zero
            alpha = 0.0
            beta = 0.0
        else:
            alpha = math.atan2(-V_body[2], V_body[0])  # Angle of attack
            beta = math.asin(clamp(V_body[1] / airspeed, -1.0, 1.0))  # Sideslip angle

        # Simplified aerodynamic coefficients using available parameters
        C_L = self.params.CL0 + self.params.CL_alpha * alpha
        C_D = self.params.CD0 + 0.1 * alpha**2  # Simple induced drag
        C_Y = self.params.CY_beta * beta
        
        # Moment coefficients
        C_l = self.params.Cl_beta * beta  # Roll moment from sideslip
        C_m = self.params.CMAC  # Pitching moment (simplified)
        C_n = -0.1 * beta  # Yaw stability (negative for weathercock stability)

        # Dynamic pressure
        q_dyn = 0.5 * self.params.rho * np.linalg.norm(V_body)**2
        S = self.params.S_wing_ref_area
        b = self.params.b_span

        # Aerodynamic forces in body frame (X=forward, Y=right, Z=down)
        # Standard aircraft body axes: drag opposes velocity, lift is perpendicular up
        F_aero_body = np.array([
            -q_dyn * S * C_D,     # Drag opposes forward motion
            q_dyn * S * C_Y,      # Side force (positive = right)
            -q_dyn * S * C_L      # Lift (negative Z = up in NED convention)
        ])

        # Aerodynamic moments in body frame (L=roll, M=pitch, N=yaw)
        M_aero_body = np.array([
            q_dyn * S * b * C_l,                    # Roll moment (use wingspan b)
            q_dyn * S * self.params.c_bar * C_m,    # Pitch moment (use chord c_bar)
            q_dyn * S * b * C_n                     # Yaw moment (use wingspan b)
        ])  

        # Thrust force in body frame
        thrust = clamp(controls.throttle * self.params.thrust_max, 0.0, self.params.thrust_max)
        F_thrust_body = np.array([thrust, 0.0, 0.0])
        F_total_body = F_aero_body + F_thrust_body + ext_F_body

        # Convert total force to world frame (body to world rotation)
        F_total_world = rotate_vector_by_quaternion(state.pose, F_total_body)
        # Gravity points down (assuming ENU convention: +Z is up, so gravity is +Z)
        F_gravity_world = np.array([0.0, 0.0, self.params.mtom * self.params.gravity])
        F_world = F_total_world + F_gravity_world
        M_body = M_aero_body + ext_M_body

        return F_world, M_body

    def _dynamics(self, state: AirplaneState, controls: ControlInputs, ext_F_body: np.ndarray, ext_M_body: np.ndarray) -> AirplaneState:
        """Calculate state derivatives for integration."""
        
        # 1. Calculate forces and moments (aerodynamics + thrust + gravity)
        F_world, M_body = self._calculate_forces_and_moments(state, controls, ext_F_body, ext_M_body)
        
        # 2. Translational dynamics: F = ma -> a = F/m
        acceleration_world = F_world / self.params.mtom
        
        # 3. Rotational dynamics: M = J*omega_dot + omega x (J*omega)
        omega = state.angular_rates
        angular_acceleration = self.J_inv @ (M_body - np.cross(omega, self.J @ omega))
        
        # 4. Quaternion kinematics: dq/dt = 0.5 * q * omega_quat
        omega_quat = np.quaternion(0, *omega)
        pose_derivative = 0.5 * state.pose * omega_quat
        
        return AirplaneState(
            position=state.velocity,        # dx/dt = velocity
            velocity=acceleration_world,    # dv/dt = acceleration  
            pose=pose_derivative,          # dq/dt = quaternion derivative
            angular_rates=angular_acceleration  # domega/dt = angular acceleration
        )


    def _add_states(self, state: AirplaneState, derivative: AirplaneState, scale: float) -> AirplaneState:
        """Add scaled derivative to state for RK4 intermediate steps."""
        return AirplaneState(
            position=state.position + scale * derivative.position,
            velocity=state.velocity + scale * derivative.velocity,
            pose=state.pose + scale * derivative.pose,
            angular_rates=state.angular_rates + scale * derivative.angular_rates
        )
    
    def _rk4_combine(self, state: AirplaneState, k1: AirplaneState, k2: AirplaneState, 
                k3: AirplaneState, k4: AirplaneState, dt: float) -> AirplaneState:
        """Apply RK4 weighted combination formula."""
        return AirplaneState(
            position=state.position + (dt/6) * (k1.position + 2*k2.position + 2*k3.position + k4.position),
            velocity=state.velocity + (dt/6) * (k1.velocity + 2*k2.velocity + 2*k3.velocity + k4.velocity),
            pose=state.pose + (dt/6) * (k1.pose + 2*k2.pose + 2*k3.pose + k4.pose),
            angular_rates=state.angular_rates + (dt/6) * (k1.angular_rates + 2*k2.angular_rates + 2*k3.angular_rates + k4.angular_rates)
        )