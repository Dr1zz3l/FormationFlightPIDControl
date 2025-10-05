from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .airplane import AirplaneState
from .geometry import rotate_body_to_earth, rotate_earth_to_body


class PIDFollower:
    def __init__(self) -> None:
        self.e_int_body = np.zeros(3)

    @staticmethod
    def pilot_leader(t: float, state: AirplaneState) -> np.ndarray:
        throttle = 0.5
        base_roll = 0.15
        base_yaw = 0.15
        wiggle = 0.1 * np.sin(0.1 * t)
        roll_cmd = base_roll + wiggle
        pitch_cmd = 0.3
        yaw_cmd = base_yaw + wiggle
        return np.array([throttle, roll_cmd, pitch_cmd, yaw_cmd])

    def pilot_follower(
        self,
        t: float,
        state_f: AirplaneState,
        state_L: AirplaneState,
        dt: float,
    ) -> Tuple[np.ndarray]:
        offset_body = np.array([-60.0, -80.0, 0.0])

        pos_F = state_f.position
        vel_F = state_f.velocity
        quat_f = np.array([state_f.pose.w, state_f.pose.x, state_f.pose.y, state_f.pose.z])

        pos_L = state_L.position
        vel_L = state_L.velocity
        quat_L = np.array([state_L.pose.w, state_L.pose.x, state_L.pose.y, state_L.pose.z])

        pos_des = pos_L + rotate_body_to_earth(quat_L, offset_body)

        e_p_world = pos_des - pos_F
        e_v_world = vel_L - vel_F

        e_p_body = rotate_earth_to_body(quat_f, e_p_world)
        e_v_body = rotate_earth_to_body(quat_f, e_v_world)

        self.e_int_body += e_p_body * dt

        Kp_body = np.array([0.3, 0.05, 0.2])
        Kv_body = np.array([3.0, 0.5, 0.5])
        Ki_body = np.array([0.0005, 0.0005, 0.00025])

        a_des_body = Kp_body * e_p_body + Kv_body * e_v_body + Ki_body * self.e_int_body

        throttle_gain = 0.2
        roll_gain = 0.01
        pitch_gain = 0.1
        yaw_gain = 0.1

        throttle = throttle_gain * (a_des_body[0] - a_des_body[2])
        roll_cmd = roll_gain * a_des_body[1]
        pitch_cmd = -pitch_gain * a_des_body[2]
        yaw_cmd = yaw_gain * a_des_body[1]

        throttle = np.clip(throttle, 0.0, 1.0)
        roll_cmd = np.clip(roll_cmd, -0.5, 0.5)
        pitch_cmd = np.clip(pitch_cmd, -0.5, 0.5)
        yaw_cmd = np.clip(yaw_cmd, -0.3, 0.3)

        return np.array([throttle, roll_cmd, pitch_cmd, yaw_cmd])