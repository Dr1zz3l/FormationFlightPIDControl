from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .geometry import rotate_body_to_earth, rotate_earth_to_body


class PIDFollower:
    def __init__(self) -> None:
        self.e_int_body = np.zeros(3)

    @staticmethod
    def pilot_leader(t: float, state: np.ndarray) -> np.ndarray:
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
        state_f: np.ndarray,
        state_L: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        offset_body = np.array([-60.0, -80.0, 0.0])

        pos_F = state_f[0:3]
        vel_F = state_f[3:6]
        quat_f = state_f[6:10]

        pos_L = state_L[0:3]
        vel_L = state_L[3:6]
        quat_L = state_L[6:10]

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

        r_world = pos_des - pos_F
        dist = np.linalg.norm(r_world)
        r_body = rotate_earth_to_body(quat_f, r_world)
        lateral = r_body[1]

        r_max = 120.0
        max_roll_moment = 1.0
        max_extra_lift = 100.0

        if 1e-6 < dist < r_max:
            strength = (1.0 - (dist / r_max)) ** 2
            roll_sign = -1.0 if abs(lateral) > 1e-6 else 0.0
            M_roll = roll_sign * max_roll_moment * strength
            F_lift = max_extra_lift * strength
            F_ext_body = np.array([0.0, 0.0, -F_lift])
            M_ext_body = np.array([M_roll, 0.0, 0.0])
        else:
            F_ext_body = None
            M_ext_body = None

        throttle = np.clip(throttle, 0.0, 1.0)
        roll_cmd = np.clip(roll_cmd, -0.5, 0.5)
        pitch_cmd = np.clip(pitch_cmd, -0.5, 0.5)
        yaw_cmd = np.clip(yaw_cmd, -0.3, 0.3)

        return np.array([throttle, roll_cmd, pitch_cmd, yaw_cmd]), F_ext_body, M_ext_body

    def pilot_follower2(
        self,
        t: float,
        state_f: np.ndarray,
        state_L: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        return self.pilot_follower(t, state_f, state_L, dt)
