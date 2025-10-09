from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from .geometry import (
    AIRCRAFT_GEOMETRY_BODY,
    NED_TO_ENU,
    rotate_body_to_earth,
)

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from .airplane import Airplane6DoFLite
    from .controllers import PIDFollower


MAX_TRAIL_LENGTH = 500
VORTEX_HISTORY_LENGTH = 32
VORTEX_CIRCLE_POINTS = 24
VORTEX_RING_SPACING = 25.0
VORTEX_PHASE_STEP = 0.35


@dataclass
class VortexRing:
    center: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray
    radius: float
    strength: float
    phase: float


@dataclass
class Trail:
    x: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_TRAIL_LENGTH))
    y: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_TRAIL_LENGTH))
    z: Deque[float] = field(default_factory=lambda: deque(maxlen=MAX_TRAIL_LENGTH))

    def append_point(self, point_enu: np.ndarray) -> None:
        self.x.append(point_enu[0])
        self.y.append(point_enu[1])
        self.z.append(point_enu[2])

    def as_lists(self) -> tuple[list[float], list[float], list[float]]:
        return list(self.x), list(self.y), list(self.z)


@dataclass
class AircraftVisual:
    sim: "Airplane6DoFLite"
    label: str
    color: str
    pid: Optional["PIDFollower"] = None
    is_leader: bool = False
    trail: Trail = field(default_factory=Trail)
    throttle_history: List[float] = field(default_factory=list)
    line_handles: Dict[str, Any] = field(default_factory=dict)
    throttle_line: Optional[Any] = None
    average_throttle_text: Optional[Any] = None
    vortex_history_left: Deque[VortexRing] = field(
        default_factory=lambda: deque(maxlen=VORTEX_HISTORY_LENGTH)
    )
    vortex_history_right: Deque[VortexRing] = field(
        default_factory=lambda: deque(maxlen=VORTEX_HISTORY_LENGTH)
    )
    vortex_lines_left: List[Any] = field(default_factory=list)
    vortex_lines_right: List[Any] = field(default_factory=list)
    vortex_phase: float = 0.0


def configure_figure():
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax_3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_throttle = fig.add_subplot(gs[1, 0])

    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_xlim(0, 2000)
    ax_3d.set_ylim(-1000, 1000)
    ax_3d.set_zlim(0, 2000)
    ax_3d.view_init(elev=18, azim=-160)

    fig.patch.set_facecolor("white")
    ax_3d.set_facecolor("white")
    ax_throttle.set_facecolor("white")

    for axis in (ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis):
        axis.pane.set_facecolor("white")

    ax_3d.set_xticks([])
    ax_3d.set_yticks([])
    ax_3d.set_zticks([])

    ax_throttle.set_ylim(0.0, 1.0)
    ax_throttle.set_xlim(0.0, 60.0)
    ax_throttle.set_ylabel("Throttle")
    ax_throttle.set_xlabel("Time [s]")
    ax_throttle.grid(True, which="both", linestyle="--", linewidth=0.5)

    fig.tight_layout()
    return fig, ax_3d, ax_throttle


def create_aircraft_lines(ax_3d, color: str) -> Dict[str, Any]:
    body_line, = ax_3d.plot([], [], [], linewidth=1.5, color=color)
    wing_line, = ax_3d.plot([], [], [], linewidth=1.5, color=color)
    htail_line, = ax_3d.plot([], [], [], linewidth=1.5, color=color)
    vtail_line, = ax_3d.plot([], [], [], linewidth=1.5, color=color)
    trail_line, = ax_3d.plot([], [], [], linewidth=1.0, color=color)
    return {
        "body": body_line,
        "wing": wing_line,
        "htail": htail_line,
        "vtail": vtail_line,
        "trail": trail_line,
    }


def attach_aircraft_lines(ax_3d, ax_throttle, formation: Sequence[AircraftVisual]) -> None:
    for index, member in enumerate(formation):
        member.line_handles = create_aircraft_lines(ax_3d, member.color)
        throttle_line, = ax_throttle.plot([], [], color=member.color, linewidth=1.5, label=member.label)
        member.throttle_line = throttle_line
        text_y = 0.93 - index * 0.08
        member.average_throttle_text = ax_throttle.text(
            0.98,
            text_y,
            f"{member.label}: avg throttle --",
            transform=ax_throttle.transAxes,
            ha="right",
            va="top",
            color=member.color,
            fontsize=9,
        )
        if member.is_leader:
            member.vortex_lines_left = [
                ax_3d.plot([], [], [], color=member.color, linewidth=1.2, alpha=0.0)[0]
                for _ in range(VORTEX_HISTORY_LENGTH)
            ]
            member.vortex_lines_right = [
                ax_3d.plot([], [], [], color=member.color, linewidth=1.2, alpha=0.0)[0]
                for _ in range(VORTEX_HISTORY_LENGTH)
            ]

    ax_throttle.legend(loc="upper left")


def collect_line_artists(formation: Sequence[AircraftVisual]) -> list[Any]:
    artists: list[Any] = []
    for member in formation:
        artists.extend(member.line_handles.values())
        if member.throttle_line is not None:
            artists.append(member.throttle_line)
        if member.vortex_lines_left:
            artists.extend(member.vortex_lines_left)
        if member.vortex_lines_right:
            artists.extend(member.vortex_lines_right)
    return artists


def body_to_world_points(sim: Airplane6DoFLite) -> Dict[str, np.ndarray]:
    position = sim.state.position
    pose = sim.state.pose
    pos_ned = position
    quat_array = np.array([pose.w, pose.x, pose.y, pose.z])
    points = {"center": NED_TO_ENU @ pos_ned}
    for name, body_vec in AIRCRAFT_GEOMETRY_BODY.items():
        ned_point = rotate_body_to_earth(quat_array, body_vec) + pos_ned
        points[name] = NED_TO_ENU @ ned_point
    return points


def update_aircraft_visual(member: AircraftVisual) -> None:
    if not member.line_handles:
        return

    points = body_to_world_points(member.sim)

    body_line = member.line_handles["body"]
    body_line.set_data(
        [points["tail"][0], points["nose"][0]],
        [points["tail"][1], points["nose"][1]],
    )
    body_line.set_3d_properties([points["tail"][2], points["nose"][2]])

    wing_line = member.line_handles["wing"]
    wing_line.set_data(
        [points["wing_l"][0], points["wing_r"][0]],
        [points["wing_l"][1], points["wing_r"][1]],
    )
    wing_line.set_3d_properties([points["wing_l"][2], points["wing_r"][2]])

    htail_line = member.line_handles["htail"]
    htail_line.set_data(
        [points["tail_l"][0], points["tail_r"][0]],
        [points["tail_l"][1], points["tail_r"][1]],
    )
    htail_line.set_3d_properties([points["tail_l"][2], points["tail_r"][2]])

    vtail_line = member.line_handles["vtail"]
    vtail_line.set_data(
        [points["tail"][0], points["tail_v"][0]],
        [points["tail"][1], points["tail_v"][1]],
    )
    vtail_line.set_3d_properties([points["tail"][2], points["tail_v"][2]])

    member.trail.append_point(points["center"])
    trail_x, trail_y, trail_z = member.trail.as_lists()
    trail_line = member.line_handles["trail"]
    trail_line.set_data(trail_x, trail_y)
    trail_line.set_3d_properties(trail_z)

    if member.is_leader:
        update_leader_vortex(member, points)


def _ring_points(ring: VortexRing) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, VORTEX_CIRCLE_POINTS, endpoint=True) + ring.phase
    circle = ring.center + ring.radius * (
        np.outer(np.cos(angles), ring.basis_u) + np.outer(np.sin(angles), ring.basis_v)
    )
    return circle[:, 0], circle[:, 1], circle[:, 2]


def _compute_vortex_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dir_unit = direction / max(float(np.linalg.norm(direction)), 1e-6)
    up_candidate = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(dir_unit, up_candidate)) > 0.9:
        up_candidate = np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(dir_unit, up_candidate)
    basis_u_norm = np.linalg.norm(basis_u)
    if basis_u_norm < 1e-6:
        basis_u = np.array([0.0, 1.0, 0.0])
        basis_u_norm = 1.0
    basis_u = basis_u / basis_u_norm
    basis_v = np.cross(dir_unit, basis_u)
    basis_v_norm = np.linalg.norm(basis_v)
    if basis_v_norm < 1e-6:
        basis_v = np.array([0.0, 0.0, 1.0])
        basis_v_norm = 1.0
    basis_v = basis_v / basis_v_norm
    return dir_unit, basis_u, basis_v


def update_leader_vortex(member: AircraftVisual, points: Dict[str, np.ndarray]) -> None:
    if not member.vortex_lines_left or not member.vortex_lines_right:
        return

    velocity_enu = NED_TO_ENU @ member.sim.state.velocity
    direction = velocity_enu
    if np.linalg.norm(direction) < 1e-3:
        direction = points["nose"] - points["center"]
    dir_unit, basis_u, basis_v = _compute_vortex_basis(direction)

    throttle = member.throttle_history[-1] if member.throttle_history else 0.0
    throttle = float(np.clip(throttle, 0.0, 1.0))

    radius = 4.0 + 10.0 * throttle
    spacing = VORTEX_RING_SPACING * (0.8 + 0.4 * (1.0 - throttle))

    for ring in member.vortex_history_left:
        ring.phase += VORTEX_PHASE_STEP
        ring.radius *= 0.995
        ring.strength *= 0.99
    for ring in member.vortex_history_right:
        ring.phase -= VORTEX_PHASE_STEP
        ring.radius *= 0.995
        ring.strength *= 0.99

    center_left = (points["wing_l"] - dir_unit * spacing).copy()
    center_right = (points["wing_r"] - dir_unit * spacing).copy()

    member.vortex_history_left.append(
        VortexRing(
            center=center_left,
            basis_u=basis_u.copy(),
            basis_v=basis_v.copy(),
            radius=radius,
            strength=throttle,
            phase=0.0,
        )
    )

    member.vortex_history_right.append(
        VortexRing(
            center=center_right,
            basis_u=basis_u.copy(),
            basis_v=-basis_v.copy(),
            radius=radius,
            strength=throttle,
            phase=0.0,
        )
    )

    _update_vortex_lines(member.vortex_history_left, member.vortex_lines_left)
    _update_vortex_lines(member.vortex_history_right, member.vortex_lines_right)


def _update_vortex_lines(
    history: Deque[VortexRing],
    lines: Sequence[Any],
) -> None:
    rings = list(history)
    count = len(rings)
    for idx, line in enumerate(lines):
        if idx < count:
            ring = rings[count - 1 - idx]
            x, y, z = _ring_points(ring)
            line.set_data(x, y)
            line.set_3d_properties(z)
            fade = 1.0 - (idx / max(count - 1, 1)) * 0.8
            base_alpha = ring.strength
            alpha = float(np.clip(base_alpha * fade, 0.0, 1.0))
            line.set_alpha(alpha)
        else:
            line.set_data([], [])
            line.set_3d_properties([])
            line.set_alpha(0.0)


def update_throttle_plot(ax, time_history: Sequence[float], formation: Sequence[AircraftVisual]) -> None:
    if not time_history:
        return

    times = np.array(time_history)
    t_max = times[-1]
    if t_max <= 60.0:
        ax.set_xlim(0.0, max(60.0, t_max))
    else:
        ax.set_xlim(t_max - 60.0, t_max)

    for member in formation:
        if member.throttle_line is not None:
            member.throttle_line.set_data(times, member.throttle_history)
        if member.average_throttle_text is not None:
            if member.throttle_history:
                avg_throttle = float(np.mean(member.throttle_history))
                member.average_throttle_text.set_text(
                    f"{member.label}: avg throttle {avg_throttle:.2f}"
                )
            else:
                member.average_throttle_text.set_text(
                    f"{member.label}: avg throttle --"
                )
