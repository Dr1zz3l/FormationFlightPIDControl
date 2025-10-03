#!/usr/bin/env python3
"""
Simple but thorough 6DOF airplane model in discrete time with live visualization.
Author: Peter Ryseck
Date: 09/27/2025

States (all in East-North-Up/world unless noted):
- position (x, y, z) [m]
- velocity (vx, vy, vz) [m/s]
- Euler angles (phi, theta, psi) = roll, pitch, yaw [rad]
- body rates (p, q, r) [rad/s]

Inputs (pilot commands):
- throttle in [0, 1]
- roll_cmd in [-1, 1]
- pitch_cmd in [-1, 1]
- yaw_cmd in [-1, 1]

Physics:
- Gravity in world frame
- Thrust along +x body axis, scaled by throttle
- Simple aero forces: lift ~ alpha, sideforce ~ beta, drag ~ induced lift and profile drag
- Simple control torques: proportional to stick commands + rate damping

Integration: RK4 fixed step
Visualization: matplotlib 3D live animation of body axes and trail

Dependencies: numpy, matplotlib

Tip: Adjust Params at the bottom for different handling.
"""
from dataclasses import dataclass
import numpy as np
import math
import time

# ----------------------------- Helpers -----------------------------

def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

import numpy as np

draw_every = 5
MAX_TRAIL_LENGTH = 500
e_int_body = np.zeros(3)  # start at zero

def earth2body(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Compute the rotation matrix to transform coordinates from the Earth frame (NED)
    to the body frame of an aircraft.

    Coordinate Frames
    -----------------
    - Earth frame (NED): X points North, Y points East, Z points Down.
    - Body frame: X points forward (out the nose), Y points right (out the right wing), Z points down.

    Parameters
    ----------
    phi : float
        Roll angle in radians.
    theta : float
        Pitch angle in radians.
    psi : float
        Yaw angle in radians.

    Returns
    -------
    np.ndarray
        A 3x3 rotation matrix for converting vectors from Earth frame to body frame.
    """
    # Precompute sine and cosine of the angles
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)

    # Rotation matrix for roll (X-axis)
    R_roll = np.array([
        [1, 0,     0],
        [0, c_phi, s_phi],
        [0, -s_phi, c_phi]
    ])

    # Rotation matrix for pitch (Y-axis)
    R_pitch = np.array([
        [c_theta, 0, -s_theta],
        [0,       1, 0],
        [s_theta, 0, c_theta]
    ])

    # Rotation matrix for yaw (Z-axis)
    R_yaw = np.array([
        [c_psi,  s_psi, 0],
        [-s_psi, c_psi, 0],
        [0,      0,     1]
    ])

    # Combined rotation matrix: body = R_roll * R_pitch * R_yaw * earth
    return R_roll @ R_pitch @ R_yaw


def euler_rates_matrix(phi: float, theta: float) -> np.ndarray:
    """
    Converts body angular rates [p, q, r] to Euler angle rates [φ̇, θ̇, ψ̇]
    for ZYX rotation order (yaw → pitch → roll).

    Assumes small pitch angles (|θ| < 89°) to avoid gimbal lock.

    Parameters
    ----------
    phi : float
        Roll angle (φ) in radians.
    theta : float
        Pitch angle (θ) in radians.

    Returns
    -------
    np.ndarray
        A 3x3 transformation matrix that maps body rates [p, q, r] to 
        Euler angle rates [φ̇, θ̇, ψ̇].
    """
    sin_phi,  cos_phi  = np.sin(phi), np.cos(phi)
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    tan_theta = np.tan(theta)

    return np.array([
        [1.0, sin_phi * tan_theta,  cos_phi * tan_theta],
        [0.0, cos_phi,             -sin_phi],
        [0.0, sin_phi / cos_theta,  cos_phi / cos_theta],
    ])


# ----------------------------- Parameters -----------------------------

@dataclass
class Params:
    mtom: float = 16.0            # mass [kg]
    gravity: float = 9.81         # gravity [m/s^2]
    S_wing_ref_area: float = 0.6  # wing reference area [m^2]
    c_bar: float = 0.5            # MAC [m]
    x_ac: float = 0.2             # Vehicle aerodynamic center aka neutral point [m] (Frame is x pointing back in this case, origin at leading edge of wing)
    x_cg: float = 0.15             # CG ahead of the AC, positive static margin [m] (Frame is x pointing back in this case, origin at leading edge of wing)
    b_span: float = 2.4           # span [m]
    AR: float = b_span / c_bar    # aspect ratio
    rho: float = 1.225            # air density [kg/m^3]
    e_const: float = 0.6          # Oswald efficiency factor (typically [0.7, 0.95])
    alpha_stall = 0.25            # typical stall angle 12°–18° → about 0.21–0.31 radians

    thrust_max: float = 35.0     # max thrust [N]

    # Aero coefficients (very simplified)
    CL0: float = 0.0
    CL_alpha: float = 5.0         # per rad
    CY_beta: float = -0.9         # per rad
    CD0: float = 0.01
    CMAC = 0.05                   # approximation for aerodynamic center moment coefficient, should be positive for stability (includes the tail contribution). AKA CM0. Should be equal to ((aircraft.x_cg - aircraft.x_ac) / aircraft.c_bar * CL_tot) in steady level flight
    CL_max = CL0 + CL_alpha * alpha_stall
    Cl_beta = -0.05               # per rad - lateral stability

    # Control torque gains (maps stick to Nm, these are similar to pseudo aileron, elevator, rudder)
    K_roll: float = 6.0/1000
    K_pitch: float = 12.0/1000
    K_yaw: float = 0.8/1000

    # Rate damping (Nm per rad/s)
    Bp: float = 2.0
    Bq: float = 2.8
    Br: float = 2.0

    # Inertia (diagonal for simplicity)
    Jx: float = 0.35
    Jy: float = 0.45
    Jz: float = 0.60

    # Softeners
    v_eps: float = 1e-3
    cth_eps: float = 1e-3


# ----------------------------- Dynamics -----------------------------

class Airplane6DoFLite:
    def __init__(self, aircraft: Params):
        self.aircraft = aircraft
        self.J = np.diag([aircraft.Jx, aircraft.Jy, aircraft.Jz])
        self.Jinv = np.diag([1.0/aircraft.Jx, 1.0/aircraft.Jy, 1.0/aircraft.Jz])
        # State vector: [x y z vx vy vz phi theta psi p q r]
        self.state = np.zeros(12)
        # Start a bit above ground pointing forward
        self.state[2] = -700.0  # z down
        self.state[0] = 200.0  # x forward
        self.state[1] = 200.0  # y right
        self.state[3] = 26.0 # forward speed [m/s] (for 12 kg, use 26m/s)
        self.state[7] = 5 * np.pi/180

    # --------------- Aerodynamics & forces ---------------
    def forces_and_moments(self, state, u, ext_F_body=None, ext_M_body=None):
        aircraft = self.aircraft
        x, y, z, vx, vy, vz, phi, theta, psi, p_rate, q_rate, r_rate = state
        throttle, roll_cmd, pitch_cmd, yaw_cmd = u

        R_be = earth2body(phi, theta, psi)
        R_eb = R_be.T

        # velocity in body frame
        vel_body = R_be @ np.array([vx, vy, vz])
        u_body, v_body, w_body = vel_body
        vel_norm = max(aircraft.v_eps, float(np.linalg.norm(vel_body)))

        # angles of attack and sideslip
        alpha = math.atan2(w_body, u_body)
        beta  = math.asin(clamp(v_body / vel_norm, -1.0, 1.0))

        qbar = 0.5 * aircraft.rho * vel_norm ** 2

        # Thrust (body +x)
        T_force = aircraft.thrust_max * clamp(throttle, 0.0, 1.0)
        F_thrust_body = np.array([T_force, 0.0, 0.0])

        V_hat = vel_body / vel_norm

        # Lift along -z body
        # Modeling stall after alpha_stall
        if alpha <= aircraft.alpha_stall:
            CL_tot = aircraft.CL0 + aircraft.CL_alpha * alpha
        else:
            # linear falloff to zero by 90 deg
            CL_tot = aircraft.CL_max * (1 - (abs(alpha) - aircraft.alpha_stall) / (np.pi/2 - aircraft.alpha_stall))
            CL_tot *= np.sign(alpha)
            print("STALL")

        L_force = CL_tot * qbar * aircraft.S_wing_ref_area
        # lift vector perpendicular to V in x-z plane (assuming no sideslip)
        lift_dir = np.array([V_hat[2], 0, -V_hat[0]])
        lift_dir /= np.linalg.norm(lift_dir)
        F_lift_body = L_force * lift_dir

        # Sideforce from beta along +y body
        CY_tot = aircraft.CY_beta * beta
        Y_force = CY_tot * qbar * aircraft.S_wing_ref_area
        # F_side_wind = np.array([0.0, Y_force, 0.0])
        side_dir = np.cross(V_hat, lift_dir)  # right-handed: positive sideforce direction
        side_dir /= np.linalg.norm(side_dir)
        F_side_body = Y_force * side_dir

        # Drag along the -x body
        CD_induced = CL_tot ** 2 / (np.pi * aircraft.AR * aircraft.e_const)
        CD_tot = aircraft.CD0 + CD_induced  # profile plus induced drag
        D_force = CD_tot * qbar * aircraft.S_wing_ref_area
        drag_dir = -V_hat  # drag direction is opposite the velocity unit vector
        drag_dir /= np.linalg.norm(drag_dir)  # normalized just for symmetry with lift
        F_drag_body = D_force * drag_dir

        # Sum forces in body then to earth
        F_body = F_side_body + F_thrust_body + F_lift_body + F_drag_body

        # Add external interaction force if provided (assumed in body frame)
        if ext_F_body is not None:
            F_body = F_body + ext_F_body

        F_earth = R_eb @ F_body + np.array([0.0, 0.0, aircraft.mtom * aircraft.gravity])  # gravity is -z in ENU

        # Control moments (proportional to stick + rate damping)
        M_cmd = np.array([
            aircraft.K_roll  * clamp(roll_cmd,  -1.0, 1.0),
            aircraft.K_pitch * clamp(pitch_cmd, -1.0, 1.0),
            aircraft.K_yaw   * clamp(yaw_cmd,   -1.0, 1.0),
        ]) * qbar  # dependent on airspeed
        M_damp = -np.array([aircraft.Bp * p_rate, aircraft.Bq * q_rate, aircraft.Br * r_rate])
        
        # Aerodynamic moment and CG balance
        C_M = aircraft.CMAC + (aircraft.x_cg - aircraft.x_ac) / aircraft.c_bar * CL_tot
        M_pitch_CG_AC = qbar * aircraft.S_wing_ref_area * aircraft.c_bar * C_M
        M_CG_AC = np.array([
            0,
            M_pitch_CG_AC,
            0,
        ])

        # Dihedral effect: rolling moment from sideslip
        # Lateral stability derivative (roll per rad of sideslip)
        Cl_beta = aircraft.Cl_beta  # typically negative for stable dihedral
        L_roll = Cl_beta * beta * qbar * aircraft.S_wing_ref_area * aircraft.b_span
        M_lateral_body = np.array([L_roll, 0.0, 0.0])

        # Sum all moments
        M_body = M_damp + M_cmd + M_CG_AC + M_lateral_body

        # Add external interaction moment if provided (assumed in body frame)
        if ext_M_body is not None:
            M_body = M_body + ext_M_body

        return F_earth, M_body

    # --------------- Continuous dynamics f(x,u) ---------------
    def f(self, state, u, ext_F_body=None, ext_M_body=None):
        aircraft = self.aircraft
        x, y, z, vx, vy, vz, phi, theta, psi, p_rate, q_rate, r_rate = state

        F_earth, M_body = self.forces_and_moments(state, u, ext_F_body, ext_M_body)

        # Translational dynamics in world
        accel_world = F_earth / aircraft.mtom

        # Rotational dynamics in body
        omega = np.array([p_rate, q_rate, r_rate])
        omega_dot = self.Jinv @ (M_body - np.cross(omega, self.J @ omega))

        # Euler angle kinematics
        T = euler_rates_matrix(phi, theta)
        eulerdot = T @ omega

        # Pack derivatives
        xdot = np.zeros_like(state)
        xdot[0:3] = np.array([vx, vy, vz])
        xdot[3:6] = accel_world
        xdot[6:9] = eulerdot
        xdot[9:12] = omega_dot
        return xdot

    # --------------- Discrete step (RK4) ---------------
    # Runge–Kutta 4th order numerical integration
    def step(self, u, dt, ext_F_body=None, ext_M_body=None):
        state = self.state
        k1 = self.f(state, u, ext_F_body, ext_M_body)
        k2 = self.f(state + 0.5*dt*k1, u, ext_F_body, ext_M_body)
        k3 = self.f(state + 0.5*dt*k2, u, ext_F_body, ext_M_body)
        k4 = self.f(state + dt*k3, u, ext_F_body, ext_M_body)
        self.state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        # Keep theta away from singularities for stability
        self.state[6] = ((self.state[6] + math.pi) % (2*math.pi)) - math.pi   # phi wrap
        self.state[7] = clamp(self.state[7], -math.radians(89.0), math.radians(89.0))
        self.state[8] = ((self.state[8] + math.pi) % (2*math.pi)) - math.pi   # psi wrap
        return self.state.copy()


# ----------------------------- Live Viz -----------------------------

class PIDFollower:
    def __init__(self):
        self.e_int_body = np.zeros(3)
        self.e_prev_body = np.zeros(3)

    @staticmethod
    def pilot_leader(t, state):
        throttle = 0.5

        # --- Base circle bias ---
        base_roll = 0.15   # constant turn component
        base_yaw  = 0.15

        # --- Oscillation on top of circle ---
        wiggle = 0.1 * np.sin(0.1 * t)  # oscillates left/right

        roll_cmd = base_roll + wiggle
        pitch_cmd = 0.3
        yaw_cmd = base_yaw + wiggle

        return np.array([throttle, roll_cmd, pitch_cmd, yaw_cmd])

    def pilot_follower(self, t, state_f, state_L, dt):
        # Desired relative offset in LEADER BODY frame
        offset_body = np.array([-60.0, -80.0, 0.0])

        # --- Extract follower and leader states ---
        pos_F = state_f[0:3]
        vel_F = state_f[3:6]
        phi_f, theta_f, psi_f = state_f[6:9]

        pos_L = state_L[0:3]
        vel_L = state_L[3:6]
        phi_L, theta_L, psi_L = state_L[6:9]

        # --- Transform offset from leader body frame → world frame ---
        R_leader = earth2body(phi_L, theta_L, psi_L)
        pos_des = pos_L + R_leader.T @ offset_body

        # --- Position & velocity errors ---
        e_p_world = pos_des - pos_F
        e_v_world = vel_L - vel_F

        R_f = earth2body(phi_f, theta_f, psi_f)
        e_p_body = R_f @ e_p_world
        e_v_body = R_f @ e_v_world

        # --- Update integral error (instance-specific, no global) ---
        self.e_int_body += e_p_body * dt

        # --- Gains ---
        Kp_body = np.array([0.3, 0.05, 0.2])
        Kv_body = np.array([3.0, 0.5, 0.5])
        Ki_body = np.array([0.0005, 0.0005, 0.00025])  # tune these

        # --- Desired acceleration with PID ---
        a_des_body = Kp_body * e_p_body + Kv_body * e_v_body + Ki_body * self.e_int_body

        # --- Map to controls ---
        throttle_gain = 0.2
        roll_gain     = 0.01
        pitch_gain    = 0.1
        yaw_gain      = 0.1

        throttle = throttle_gain * (a_des_body[0] - a_des_body[2])
        roll_cmd = roll_gain * a_des_body[1]
        pitch_cmd = -pitch_gain * a_des_body[2]
        yaw_cmd = yaw_gain * a_des_body[1]

        # Interactions
        r_world = pos_des - pos_F
        dist = np.linalg.norm(r_world)
        r_body = R_f @ r_world
        lateral = r_body[1]

        r_max = 120.0
        max_roll_moment = 1.0
        max_extra_lift = 100.0

        if dist < r_max and dist > 1e-6:
            strength = (1.0 - (dist / r_max))**2
            roll_sign = -1.0 if abs(lateral) > 1e-6 else 0.0
            M_roll = roll_sign * max_roll_moment * strength
            F_lift = max_extra_lift * strength
            F_ext_body = np.array([0.0, 0.0, -F_lift])
            M_ext_body = np.array([M_roll, 0.0, 0.0])
        else:
            F_ext_body = None
            M_ext_body = None

        # Clamp
        throttle = np.clip(throttle, 0.0, 1.0)
        roll_cmd  = np.clip(roll_cmd, -0.5, 0.5)
        pitch_cmd = np.clip(pitch_cmd, -0.5, 0.5)
        yaw_cmd   = np.clip(yaw_cmd, -0.3, 0.3)

        return np.array([throttle, roll_cmd, pitch_cmd, yaw_cmd]), F_ext_body, M_ext_body

    def pilot_follower2(self, t, state_f, state_L, dt):
        # Just reuse the same follower logic
        return self.pilot_follower(t, state_f, state_L, dt)
    

def demo_sim():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from matplotlib.animation import FuncAnimation

    p = Params()
    sim1 = Airplane6DoFLite(p)        # Leader
    sim2 = Airplane6DoFLite(p)        # Follower
    sim3 = Airplane6DoFLite(p)        # Follower
    sim4 = Airplane6DoFLite(p)        # Follower
    sim5 = Airplane6DoFLite(p)        # Follower

    pid2 = PIDFollower()
    pid3 = PIDFollower()
    pid4 = PIDFollower()
    pid5 = PIDFollower()

    # Offset follower to start a bit behind leader
    sim2.state[0] -= 120.0
    sim2.state[1] -= 200.0
    sim2.state[2] += 100.0

    sim3.state[0] -= 400.0
    sim3.state[2] -= 150.0
    sim3.state[1] -= 150.0

    sim4.state[0] -= 600.0
    sim4.state[1] -= 500.0

    sim5.state[0] -= 600.0
    sim5.state[1] -= 500.0
    
    dt = 0.05
    Tfinal = 10000.0
    steps = int(Tfinal / dt)
    t = 0.0

    # For plotting
    # Trails
    trail1_x, trail1_y, trail1_z = [], [], []
    trail2_x, trail2_y, trail2_z = [], [], []
    trail3_x, trail3_y, trail3_z = [], [], []
    trail4_x, trail4_y, trail4_z = [], [], []
    trail5_x, trail5_y, trail5_z = [], [], []

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, 2000)  # these should be scaled to the same length to make sure things aren't warped
    ax.set_ylim(-1000, 1000)
    ax.set_zlim(0, 2000)
    ax.view_init(elev=18, azim=-160)

    # White background for figure and axes
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Make 3D panes white instead of gray
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')

    # Hide tick values
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.tight_layout()

    # Geometry scale
    scale = 40.0

    # Define airplane geometry in body frame
    nose_b   = np.array([ scale,   0.0,   0.0])
    tail_b   = np.array([-scale,   0.0,   0.0])
    wing_r_b = np.array([ 0.0,   scale*1.1, 0.0])
    wing_l_b = np.array([ 0.0,  -scale*1.1, 0.0])
    tail_r_b  = tail_b + np.array([0.0,  scale*0.4, 0.0])
    tail_l_b  = tail_b + np.array([0.0, -scale*0.4, 0.0])
    tail_v_b  = np.array([-scale, 0.0,  -scale*0.4])

    # NED→ENU transform for plotting
    C_ned_to_enu = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0,-1],
    ])

    # Make a set of line handles for each aircraft
    def make_aircraft_handles(color_body, color_trail):
        body_line, = ax.plot([], [], [], linewidth=1.5, color=color_body)
        wing_line, = ax.plot([], [], [], linewidth=1.5, color=color_body)
        htail_line, = ax.plot([], [], [], linewidth=1.5, color=color_body)
        vtail_line, = ax.plot([], [], [], linewidth=1.5, color=color_body)
        trail_line, = ax.plot([], [], [], linewidth=1, color=color_trail)
        return dict(
            body=body_line, wing=wing_line,
            htail=htail_line, vtail=vtail_line,
            trail=trail_line
        )
    
    ac1_lines = make_aircraft_handles("k", "k")
    ac2_lines = make_aircraft_handles("r", "r")
    ac3_lines = make_aircraft_handles("g", "g")
    ac4_lines = make_aircraft_handles("b", "b")
    ac5_lines = make_aircraft_handles("orange", "orange")

    def init():
        for hdict in (ac1_lines, ac2_lines, ac3_lines, ac4_lines):
            for ln in hdict.values():
                ln.set_data([], [])
                ln.set_3d_properties([])
        return list(ac1_lines.values()) + list(ac2_lines.values()) + list(ac3_lines.values()) + list(ac4_lines.values())


    def draw_aircraft(sim, trails, lines):
        x, y, z = sim.state[0:3]
        phi, theta, psi = sim.state[6:9]
        pos_ned = np.array([x, y, z])

        # Rotation
        R_be = earth2body(phi, theta, psi)
        R_eb = R_be.T

        def to_plot(p_b):
            p_ned = (R_eb @ p_b) + pos_ned
            return C_ned_to_enu @ p_ned

        nose   = to_plot(nose_b)
        tail   = to_plot(tail_b)
        wing_r = to_plot(wing_r_b)
        wing_l = to_plot(wing_l_b)
        tail_r = to_plot(tail_r_b)
        tail_l = to_plot(tail_l_b)
        tail_v = to_plot(tail_v_b)

        # Draw geometry
        lines["body"].set_data([tail[0], nose[0]], [tail[1], nose[1]])
        lines["body"].set_3d_properties([tail[2], nose[2]])

        lines["wing"].set_data([wing_l[0], wing_r[0]], [wing_l[1], wing_r[1]])
        lines["wing"].set_3d_properties([wing_l[2], wing_r[2]])

        lines["htail"].set_data([tail_l[0], tail_r[0]], [tail_l[1], tail_r[1]])
        lines["htail"].set_3d_properties([tail_l[2], tail_r[2]])

        lines["vtail"].set_data([tail[0], tail_v[0]], [tail[1], tail_v[1]])
        lines["vtail"].set_3d_properties([tail[2], tail_v[2]])

        # Trail
        p_plot = C_ned_to_enu @ pos_ned
        trails[0].append(p_plot[0])
        trails[1].append(p_plot[1])
        trails[2].append(p_plot[2])
        if len(trails[0]) > MAX_TRAIL_LENGTH:
            trails[0].pop(0); trails[1].pop(0); trails[2].pop(0)

        lines["trail"].set_data(trails[0], trails[1])
        lines["trail"].set_3d_properties(trails[2])


    def update(frame):
        nonlocal t

        if t >= Tfinal:
            plt.close(fig)   # stop animation when done
            return []
    
        for _ in range(draw_every):
            # Leader follows its own pilot
            u1 = PIDFollower.pilot_leader(t, sim1.state)
            sim1.step(u1, dt)

            # Follower 1 tracks leader
            u2, F_ext2, M_ext2 = pid2.pilot_follower(t, sim2.state, sim1.state, dt)
            sim2.step(u2, dt, ext_F_body=F_ext2, ext_M_body=M_ext2)

            # Follower 2 tracks follower 1
            u3, F_ext3, M_ext3 = pid3.pilot_follower(t, sim3.state, sim2.state, dt)
            sim3.step(u3, dt, ext_F_body=F_ext3, ext_M_body=M_ext3)

            # Follower 2 tracks follower 1
            u4, F_ext4, M_ext4 = pid4.pilot_follower(t, sim4.state, sim3.state, dt)
            sim4.step(u4, dt, ext_F_body=F_ext4, ext_M_body=M_ext4)

            # Follower 2 tracks follower 1
            u5, F_ext5, M_ext5 = pid5.pilot_follower(t, sim5.state, sim4.state, dt)
            sim5.step(u5, dt, ext_F_body=F_ext5, ext_M_body=M_ext5)

            t += dt

        draw_aircraft(sim1, (trail1_x, trail1_y, trail1_z), ac1_lines)
        draw_aircraft(sim2, (trail2_x, trail2_y, trail2_z), ac2_lines)
        draw_aircraft(sim3, (trail3_x, trail3_y, trail3_z), ac3_lines)
        draw_aircraft(sim4, (trail4_x, trail4_y, trail4_z), ac4_lines)
        draw_aircraft(sim5, (trail5_x, trail5_y, trail5_z), ac5_lines)

        ax.set_box_aspect([1, 1, 1])
        return list(ac1_lines.values()) + list(ac2_lines.values()) + list(ac3_lines.values()) + list(ac4_lines.values()) + list(ac5_lines.values())
        

    ani = FuncAnimation(fig, update, init_func=init, frames=steps, interval=dt*1000, blit=False)
    plt.show()


def main():
    """Main entry point for the formation flight simulation."""
    demo_sim()


if __name__ == "__main__":
    main()
