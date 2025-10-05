from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .geometry import rotate_vector_by_quaternion
from .simulation import AirplaneState, ControlInputs


class PIDFollower:
    def __init__(self) -> None:
        self.e_int_body = np.zeros(3)

    @staticmethod
    def pilot_leader(t: float, state: AirplaneState) -> ControlInputs:
        """Generate control inputs for leader aircraft following a circular pattern."""
        throttle = 0.5
        base_roll = 0.15   # constant turn component
        base_yaw = 0.15
        wiggle = 0.1 * np.sin(0.1 * t)  # oscillation for more interesting pattern
        
        return ControlInputs(
            throttle=throttle,
            roll_cmd=base_roll + wiggle,
            pitch_cmd=0.3,
            yaw_cmd=base_yaw + wiggle
        )

    def pilot_follower(
        self,
        t: float,
        follower_state: AirplaneState,
        leader_state: AirplaneState,
        dt: float,
    ) -> Tuple[ControlInputs, Optional[np.ndarray], Optional[np.ndarray]]:
        """Generate control inputs for follower aircraft to maintain formation."""
        # Desired relative offset in leader body frame
        offset_body = np.array([-60.0, -80.0, 0.0])

        # Calculate desired position in world frame
        # Rotate offset from leader body frame to world frame, then add to leader position
        offset_world = rotate_vector_by_quaternion(leader_state.pose, offset_body)
        pos_desired = leader_state.position + offset_world

        # Position and velocity errors in world frame
        e_position_world = pos_desired - follower_state.position
        e_velocity_world = leader_state.velocity - follower_state.velocity

        # Transform errors to follower body frame for control
        # Use conjugate (inverse) quaternion to rotate from world to body frame
        e_position_body = rotate_vector_by_quaternion(follower_state.pose.conjugate(), e_position_world)
        e_velocity_body = rotate_vector_by_quaternion(follower_state.pose.conjugate(), e_velocity_world)

        # Update integral error
        self.e_int_body += e_position_body * dt

        # PID gains for each body axis
        Kp_body = np.array([0.3, 0.05, 0.2])    # Position gains
        Kv_body = np.array([3.0, 0.5, 0.5])     # Velocity gains  
        Ki_body = np.array([0.0005, 0.0005, 0.00025])  # Integral gains

        # Calculate desired acceleration using PID controller
        a_desired_body = (Kp_body * e_position_body + 
                         Kv_body * e_velocity_body + 
                         Ki_body * self.e_int_body)

        # Map desired acceleration to control inputs
        throttle_gain = 0.2
        roll_gain = 0.01
        pitch_gain = 0.1
        yaw_gain = 0.1

        throttle = throttle_gain * (a_desired_body[0] - a_desired_body[2])
        roll_cmd = roll_gain * a_desired_body[1]
        pitch_cmd = -pitch_gain * a_desired_body[2]
        yaw_cmd = yaw_gain * a_desired_body[1]

        # Calculate formation interaction effects
        separation_vector = pos_desired - follower_state.position
        distance = np.linalg.norm(separation_vector)
        separation_body = rotate_vector_by_quaternion(follower_state.pose.conjugate(), separation_vector)
        lateral_separation = separation_body[1]

        # Formation interaction parameters
        interaction_range = 120.0
        max_roll_moment = 1.0
        max_extra_lift = 100.0

        # Apply wingtip vortex effects when in close formation
        if 1e-6 < distance < interaction_range:
            strength = (1.0 - (distance / interaction_range)) ** 2
            roll_sign = -1.0 if abs(lateral_separation) > 1e-6 else 0.0
            
            # External forces and moments from formation effects
            F_external = np.array([0.0, 0.0, -max_extra_lift * strength])
            M_external = np.array([roll_sign * max_roll_moment * strength, 0.0, 0.0])
        else:
            F_external = None
            M_external = None

        # Clamp control inputs to realistic limits
        controls = ControlInputs(
            throttle=np.clip(throttle, 0.0, 1.0),
            roll_cmd=np.clip(roll_cmd, -0.5, 0.5),
            pitch_cmd=np.clip(pitch_cmd, -0.5, 0.5),
            yaw_cmd=np.clip(yaw_cmd, -0.3, 0.3)
        )

        return controls, F_external, M_external

    def pilot_follower2(
        self,
        t: float,
        follower_state: AirplaneState,
        leader_state: AirplaneState,
        dt: float,
    ) -> Tuple[ControlInputs, Optional[np.ndarray], Optional[np.ndarray]]:
        """Alias for pilot_follower - same formation logic."""
        return self.pilot_follower(t, follower_state, leader_state, dt)
