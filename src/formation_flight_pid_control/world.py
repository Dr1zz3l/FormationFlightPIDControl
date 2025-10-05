

from __future__ import annotations

from typing import List

import numpy as np

from .airplane import Airplane6DoFLite
from .controllers import PIDFollower
from .formation import build_formation
from .geometry import rotate_body_to_earth, rotate_earth_to_body
from .params import Params
from .visualization import AircraftVisual
from .wake import (
    AircraftWake,
    VortexBead,
    axial_decay_factor,
    lamb_oseen_induced_vel_at_point,
    unit,
)


class World:
    """World simulation containing aircraft formation and physics step logic."""
    
    def __init__(
        self,
        params: Params,
        leader_config: tuple[str, str] = ("Leader", "k"),
        follower_specs: List[tuple[str, str, np.ndarray]] = None
    ) -> None:
        """Initialize world with aircraft formation.
        
        Args:
            params: Aircraft physical parameters
            leader_config: (label, color) for leader aircraft
            follower_specs: List of (label, color, offset_position) for followers
        """
        self.params = params
        
        # Build the formation with aircraft and controllers
        self.formation = build_formation(params, leader_config, follower_specs)
        
        # Extract airplanes for easier access
        self.airplanes = [aircraft.sim for aircraft in self.formation]

        # Time tracking
        self.t = 0.0

        self._init_wake_state()

    def _init_wake_state(self) -> None:
        self._aircraft_wakes = {ac.sim: AircraftWake() for ac in self.formation}
        self._wake_accum_dist = {ac.sim: 0.0 for ac in self.formation}

    def _estimate_circulation(self, sim: Airplane6DoFLite) -> float:
        quat = np.array([sim.state.pose.w, sim.state.pose.x, sim.state.pose.y, sim.state.pose.z])
        vel_body = rotate_earth_to_body(quat, sim.state.velocity)
        V = max(1e-3, float(np.linalg.norm(vel_body)))
        u_body, v_body, w_body = vel_body
        alpha = np.arctan2(w_body, u_body)
        CL = self.params.CL0 + self.params.CL_alpha * alpha
        CL = float(np.clip(CL, -self.params.CL_max, self.params.CL_max))
        qbar = 0.5 * self.params.rho * V * V
        L = CL * qbar * self.params.S_wing_ref_area
        Gamma_tot = (L / (self.params.rho * V * self.params.b_span)) * self.params.wake_gamma_scale
        return Gamma_tot

    def _wingtip_world_positions(
        self, sim: Airplane6DoFLite
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        quat = np.array([sim.state.pose.w, sim.state.pose.x, sim.state.pose.y, sim.state.pose.z])
        ex_b = rotate_body_to_earth(quat, np.array([1.0, 0.0, 0.0]))
        ey_b = rotate_body_to_earth(quat, np.array([0.0, 1.0, 0.0]))
        cg_w = sim.state.position
        half_span = 0.5 * self.params.b_span
        left_tip_w = cg_w - ey_b * half_span
        right_tip_w = cg_w + ey_b * half_span
        return cg_w, left_tip_w, right_tip_w, ex_b

    def _update_emit_wake(self, dt: float) -> None:
        if not self.params.wake_enable:
            return

        for ac in self.formation:
            sim = ac.sim
            wake = self._aircraft_wakes[sim]
            cg_w, left_tip_w, right_tip_w, xhat_w = self._wingtip_world_positions(sim)
            Vmag = float(np.linalg.norm(sim.state.velocity))
            if Vmag < 1e-2:
                continue

            Gamma_tot = self._estimate_circulation(sim)
            Gamma_tip = 0.5 * Gamma_tot
            gamma_left = +Gamma_tip
            gamma_right = -Gamma_tip

            self._wake_accum_dist[sim] += Vmag * dt
            if self._wake_accum_dist[sim] >= self.params.wake_segment_length or not wake.left.beads:
                self._wake_accum_dist[sim] = 0.0
                wake.left.beads.insert(
                    0,
                    VortexBead(pos_w=left_tip_w.copy(), gamma=gamma_left, age=0.0),
                )
                wake.right.beads.insert(
                    0,
                    VortexBead(pos_w=right_tip_w.copy(), gamma=gamma_right, age=0.0),
                )
                if len(wake.left.beads) > self.params.wake_history_len:
                    wake.left.beads.pop()
                if len(wake.right.beads) > self.params.wake_history_len:
                    wake.right.beads.pop()

            for tip in (wake.left, wake.right):
                for bead in tip.beads:
                    bead.pos_w = bead.pos_w + sim.state.velocity * dt
                    bead.age += dt
                    if self.params.wake_decay_time > 0.0:
                        bead.gamma *= np.exp(-dt / self.params.wake_decay_time)
                    ax = np.dot((bead.pos_w - cg_w), xhat_w)
                    bead.gamma *= axial_decay_factor(ax, self.params.wake_axial_decay_len)

    def _halfspan_sample_points_world(
        self, sim: Airplane6DoFLite
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        quat = np.array([sim.state.pose.w, sim.state.pose.x, sim.state.pose.y, sim.state.pose.z])
        xhat_w = rotate_body_to_earth(quat, np.array([1.0, 0.0, 0.0]))
        yhat_w = rotate_body_to_earth(quat, np.array([0.0, 1.0, 0.0]))
        cg_w = sim.state.position
        half_span = 0.5 * self.params.b_span * 0.9
        left_w = cg_w - yhat_w * half_span
        right_w = cg_w + yhat_w * half_span
        return cg_w, xhat_w, yhat_w, left_w, right_w

    def _induced_velocity_from_all_wakes(
        self, receiver_sim: Airplane6DoFLite, point_w: np.ndarray
    ) -> np.ndarray:
        if not self.params.wake_enable:
            return np.zeros(3)

        v_ind = np.zeros(3)
        rc = self.params.wake_core_radius
        for ac in self.formation:
            src_sim = ac.sim
            if src_sim is receiver_sim:
                continue
            wake = self._aircraft_wakes[src_sim]
            for tip in (wake.left, wake.right):
                beads = tip.beads
                if not beads:
                    continue
                for i, bead in enumerate(beads):
                    if i + 1 < len(beads):
                        next_bead = beads[i + 1]
                        line_dir = unit(next_bead.pos_w - bead.pos_w)
                        if np.linalg.norm(line_dir) < 1e-6:
                            _, _, _, xhat_w = self._wingtip_world_positions(src_sim)
                            line_dir = xhat_w
                    else:
                        _, _, _, xhat_w = self._wingtip_world_positions(src_sim)
                        line_dir = xhat_w
                    v_ind += lamb_oseen_induced_vel_at_point(
                        point_w, bead.pos_w, line_dir, bead.gamma, rc
                    )
        return v_ind

    def _wake_forces_moments_on(
        self, receiver_sim: Airplane6DoFLite
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.params.wake_enable:
            return np.zeros(3), np.zeros(3)

        cg_w, _, _, left_w, right_w = self._halfspan_sample_points_world(receiver_sim)

        v_left_w = self._induced_velocity_from_all_wakes(receiver_sim, left_w)
        v_right_w = self._induced_velocity_from_all_wakes(receiver_sim, right_w)

        quat = np.array([receiver_sim.state.pose.w, receiver_sim.state.pose.x, receiver_sim.state.pose.y, receiver_sim.state.pose.z])
        v_body = rotate_earth_to_body(quat, receiver_sim.state.velocity)
        Vmag = max(1e-3, float(np.linalg.norm(v_body)))

        v_left_b = rotate_earth_to_body(quat, v_left_w)
        v_right_b = rotate_earth_to_body(quat, v_right_w)

        w_left = v_left_b[2]
        w_right = v_right_b[2]
        d_alpha_left = -w_left / Vmag
        d_alpha_right = -w_right / Vmag

        qbar = 0.5 * self.params.rho * Vmag * Vmag
        S_half = 0.5 * self.params.S_wing_ref_area

        dL_left = qbar * S_half * self.params.CL_alpha * d_alpha_left
        dL_right = qbar * S_half * self.params.CL_alpha * d_alpha_right

        V_hat_body = v_body / Vmag
        lift_dir_b = np.array([V_hat_body[2], 0.0, -V_hat_body[0]])
        norm_lift = float(np.linalg.norm(lift_dir_b))
        if norm_lift < 1e-6:
            lift_dir_b = np.array([0.0, 0.0, -1.0])
        else:
            lift_dir_b = lift_dir_b / norm_lift

        left_arm_b = rotate_earth_to_body(quat, left_w - cg_w)
        right_arm_b = rotate_earth_to_body(quat, right_w - cg_w)

        F_left_b = dL_left * lift_dir_b
        F_right_b = dL_right * lift_dir_b

        M_left_b = np.cross(left_arm_b, F_left_b)
        M_right_b = np.cross(right_arm_b, F_right_b)

        F_extra = F_left_b + F_right_b
        M_extra = M_left_b + M_right_b

        return F_extra, M_extra

    def simulation_step(self, dt: float) -> None:
        """Execute a single simulation step for all aircraft.
        
        Args:
            dt: Time step in seconds
        """
        # Leader control
        leader = self.formation[0]
        u_leader = PIDFollower.pilot_leader(self.t, leader.sim.state)
        leader.sim.step(u_leader, dt)
        leader.throttle_history.append(u_leader[0])

        self._update_emit_wake(dt)

        for follower, target in zip(self.formation[1:], self.formation[:-1]):
            if follower.pid is None:
                raise ValueError("Follower missing PID controller")
            u_cmd, force_pid, moment_pid = follower.pid.pilot_follower(
                self.t, follower.sim.state, target.sim.state, dt
            )
            F_wake, M_wake = self._wake_forces_moments_on(follower.sim)

            F_total = np.zeros(3)
            M_total = np.zeros(3)
            if force_pid is not None:
                F_total += force_pid
            if moment_pid is not None:
                M_total += moment_pid

            F_total += F_wake
            M_total += M_wake

            follower.sim.step(u_cmd, dt, ext_F_body=F_total, ext_M_body=M_total)
            follower.throttle_history.append(u_cmd[0])

        self.t += dt
    
    def get_formation(self) -> List[AircraftVisual]:
        """Get the formation for visualization."""
        return self.formation
    
    def get_time(self) -> float:
        """Get current simulation time."""
        return self.t



    