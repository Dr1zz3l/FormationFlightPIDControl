from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from .geometry import AIRCRAFT_GEOMETRY_BODY, NED_TO_ENU, world2body

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from .controllers import PIDFollower
    from .simulation import Airplane6DoFLite


MAX_TRAIL_LENGTH = 500


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
    trail: Trail = field(default_factory=Trail)
    throttle_history: List[float] = field(default_factory=list)
    line_handles: Dict[str, Any] = field(default_factory=dict)
    throttle_line: Optional[Any] = None


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
    for member in formation:
        member.line_handles = create_aircraft_lines(ax_3d, member.color)
        throttle_line, = ax_throttle.plot([], [], color=member.color, linewidth=1.5, label=member.label)
        member.throttle_line = throttle_line

    ax_throttle.legend(loc="upper right")


def collect_line_artists(formation: Sequence[AircraftVisual]) -> list[Any]:
    artists: list[Any] = []
    for member in formation:
        artists.extend(member.line_handles.values())
        if member.throttle_line is not None:
            artists.append(member.throttle_line)
    return artists


def body_to_world_points(sim: Airplane6DoFLite) -> Dict[str, np.ndarray]:
    x, y, z = sim.state[0:3]
    quat = sim.state[6:10]
    pos_ned = np.array([x, y, z])
    R_bw = world2body(quat).T

    points = {"center": NED_TO_ENU @ pos_ned}
    for name, body_vec in AIRCRAFT_GEOMETRY_BODY.items():
        ned_point = (R_bw @ body_vec) + pos_ned
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
