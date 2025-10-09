

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .airplane import Airplane6DoFLite, AirplaneState
from .controllers import PIDFollower
from .formation import build_formation
from .geometry import AIRCRAFT_GEOMETRY_BODY, rotate_body_to_earth, rotate_earth_to_body, _normalized_quaternion
from .params import Params
from .visualization import AircraftVisual


def vortex_induced_velocity_at_point(r_eval_e: np.ndarray,
                                     r0_e: np.ndarray,
                                     d_hat_e: np.ndarray,
                                     Gamma_tip: float,
                                     r_core: float,
                                     lam_decay: float) -> np.ndarray:
    """Regularized semi-infinite straight vortex starting at r0_e, along +d_hat_e."""
    Delta = r_eval_e - r0_e
    s = max(0.0, float(np.dot(Delta, d_hat_e)))           # distance along filament
    R_vec = Delta - s * d_hat_e                           # perpendicular from filament
    R2 = np.dot(R_vec, R_vec) + r_core**2
    # Infinite-line induced velocity with axial decay:
    v_ind = (Gamma_tip / (2.0 * np.pi)) * np.cross(d_hat_e, R_vec) / R2
    if lam_decay > 0.0:
        v_ind *= np.exp(-s / lam_decay)
    return v_ind

def compute_tip_circulation(params, qbar: float, CL: float, V: float) -> float:
    """Γ_tip ≈ L/(ρ V b) with L = q S CL."""
    L = qbar * params.S_wing_ref_area * CL
    return L / (params.rho * max(V, params.v_eps) * params.b_span)

def wake_forces_moments_on_follower(follower: Airplane6DoFLite,
                                    others: list[Airplane6DoFLite],
                                    r_core=2.0, lam_decay=200.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute external F_body, M_body on 'follower' from all other aircraft wakes.
    r_core ~ vortex core radius [m], lam_decay ~ axial decay length [m].
    """

    params = follower.params

    F_body = np.zeros(3)
    M_body = np.zeros(3)

    # Half-wing sample points in EARTH frame for follower:
    quat_f = np.array([follower.state.pose.w, follower.state.pose.x, follower.state.pose.y, follower.state.pose.z])
    cg_e   = follower.state.position
    L_mid_b = np.array([0.0, -params.b_span/4.0, 0.0])
    R_mid_b = np.array([0.0, +params.b_span/4.0, 0.0])
    L_mid_e = cg_e + rotate_body_to_earth(quat_f, L_mid_b)
    R_mid_e = cg_e + rotate_body_to_earth(quat_f, R_mid_b)

    # Follower freestream (use air-relative if you have wind):
    V_e = follower.state.velocity
    V_mag = max(params.v_eps, float(np.linalg.norm(V_e)))
    qbar = 0.5 * params.rho * V_mag**2

    # Unit velocity directions:
    V_hat_e = V_e / V_mag
    # Lift direction in BODY frame (same as your main model):
    V_hat_b = rotate_earth_to_body(quat_f, V_e) / V_mag
    lift_dir_b = np.array([V_hat_b[2], 0.0, -V_hat_b[0]])
    lift_dir_b /= np.linalg.norm(lift_dir_b)

    # Sum induced velocities at the half-span points from all other aircraft
    vL_e = np.zeros(3); vR_e = np.zeros(3)
    for other in others:
        if other is follower:
            continue
        # Other’s wingtip locations and vortex direction (opposite its airspeed):
        tip_L_e, tip_R_e = other.wingtips_world()
        V_other_e = other.state.velocity
        V_other_mag = max(params.v_eps, float(np.linalg.norm(V_other_e)))
        d_hat_e = -V_other_e / V_other_mag

        # Build Γ from other's current CL (recompute like in your forces)
        # Minimal: reuse follower's CL_alpha approx for an estimate at other's alpha
        vel_b = rotate_earth_to_body(np.array([other.state.pose.w, other.state.pose.x,
                                               other.state.pose.y, other.state.pose.z]),
                                     other.state.velocity)
        alpha_other = math.atan2(vel_b[2], max(params.v_eps, vel_b[0]))
        if alpha_other <= params.alpha_stall:
            CL_other = params.CL0 + params.CL_alpha * alpha_other
        else:
            CL_other = params.CL_max * (1 - (abs(alpha_other)-params.alpha_stall)/(np.pi/2-params.alpha_stall))
            CL_other *= math.copysign(1.0, alpha_other)

        qbar_other = 0.5 * params.rho * V_other_mag**2
        Gamma_tip = compute_tip_circulation(params, qbar_other, CL_other, V_other_mag)

        # Induced velocity contributions at L/R midpoints:
        vL_e += vortex_induced_velocity_at_point(L_mid_e, tip_L_e, d_hat_e, +Gamma_tip, r_core, lam_decay)
        vL_e += vortex_induced_velocity_at_point(L_mid_e, tip_R_e, d_hat_e, +Gamma_tip, r_core, lam_decay)
        vR_e += vortex_induced_velocity_at_point(R_mid_e, tip_L_e, d_hat_e, +Gamma_tip, r_core, lam_decay)
        vR_e += vortex_induced_velocity_at_point(R_mid_e, tip_R_e, d_hat_e, +Gamma_tip, r_core, lam_decay)

    # Convert to BODY frame of follower
    vL_b = rotate_earth_to_body(quat_f, vL_e)
    vR_b = rotate_earth_to_body(quat_f, vR_e)

    # Small-angle Δalpha on each half (use body z-component along AoA plane)
    dAlpha_L = vL_b[2] / V_mag
    dAlpha_R = vR_b[2] / V_mag

    # Incremental lifts on halves
    S_half = 0.5 * params.S_wing_ref_area
    dL_L = qbar * S_half * params.CL_alpha * dAlpha_L
    dL_R = qbar * S_half * params.CL_alpha * dAlpha_R

    # Forces in BODY (apply along lift_dir)
    F_body += dL_L * lift_dir_b + dL_R * lift_dir_b

    # Rolling moment from spanwise force difference
    M_body[0] += (params.b_span/2.0) * (dL_R - dL_L)

    return F_body, M_body




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
    
    def simulation_step(self, dt: float) -> None:
        """Execute a single simulation step for all aircraft.
        
        Args:
            dt: Time step in seconds
        """
        # Compute wake forces for all aircraft
        for i, aircraft in enumerate(self.formation):
            others = [self.formation[j].sim for j, o in enumerate(self.formation) if j != i]
            dF_body, dM_body = wake_forces_moments_on_follower(aircraft.sim, others, 
                                                               r_core=2.0, lam_decay=200.0)
            
            if i == 2:  
                print("wake forces:", dF_body, dM_body, "                    ", end="\r", flush=True)
            
            #!!! for testing purposes only
            # dF_body, dM_body = np.zeros(3), np.zeros(3)


            # Apply control based on aircraft role
            if i == 0:  # Leader
                u_cmd = PIDFollower.pilot_leader(self.t, aircraft.sim.state)
            else:  # Follower
                if aircraft.pid is None:
                    raise ValueError("Follower missing PID controller")
                target = self.formation[i-1]  # Follow previous aircraft in chain
                u_cmd = aircraft.pid.pilot_follower(
                    self.t, aircraft.sim.state, target.sim.state, dt
                )
            
            # Step with wake forces as external forces
            aircraft.sim.step(u_cmd, dt, ext_F_body=dF_body, ext_M_body=dM_body)

            #!!! for testing purposes only
            #subract position of leader from plane so they stay in view
            #aircraft.sim.state.position -= self.formation[0].sim.state.position

            aircraft.throttle_history.append(u_cmd[0])

        self.t += dt
    
    def get_formation(self) -> List[AircraftVisual]:
        """Get the formation for visualization."""
        return self.formation
    
    def get_time(self) -> float:
        """Get current simulation time."""
        return self.t