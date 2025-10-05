#!/usr/bin/env python3
"""Entry point for the formation flight PID control demonstration."""
from __future__ import annotations

from typing import List

from .controllers import PIDFollower
from .formation import build_formation
from .params import Params
from .visualization import (
    attach_aircraft_lines,
    collect_line_artists,
    configure_figure,
    update_aircraft_visual,
    update_throttle_plot,
)

DRAW_EVERY = 5


def demo_sim() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - imported for side effects

    params = Params()
    formation = build_formation(params)

    dt = 0.05  # Restored original timestep - quaternion integration fixed
    Tfinal = 10000.0
    steps = int(Tfinal / dt)
    t = 0.0

    fig, ax_3d, ax_throttle = configure_figure()
    attach_aircraft_lines(ax_3d, ax_throttle, formation)

    time_history: List[float] = []

    def init():
        for member in formation:
            for line in member.line_handles.values():
                line.set_data([], [])
                line.set_3d_properties([])
            if member.throttle_line is not None:
                member.throttle_line.set_data([], [])
        return collect_line_artists(formation)

    def update(_frame_index: int):
        nonlocal t

        if t >= Tfinal:
            plt.close(fig)
            return []

        for _ in range(DRAW_EVERY):
            leader = formation[0]
            u_leader = PIDFollower.pilot_leader(t, leader.sim.state)
            leader.sim.step(u_leader, dt)
            leader.throttle_history.append(u_leader[0])

            for follower, target in zip(formation[1:], formation[:-1]):
                if follower.pid is None:
                    raise ValueError("Follower missing PID controller")
                u_cmd, force, moment = follower.pid.pilot_follower(
                    t, follower.sim.state, target.sim.state, dt
                )
                follower.sim.step(u_cmd, dt, ext_F_body=force, ext_M_body=moment)
                follower.throttle_history.append(u_cmd[0])

            t += dt
            time_history.append(t)

        for member in formation:
            update_aircraft_visual(member)

        update_throttle_plot(ax_throttle, time_history, formation)
        ax_3d.set_box_aspect([1, 1, 1])
        return collect_line_artists(formation)

    animation = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=steps,
        interval=dt * 1000,
        blit=False,
    )
    fig._formation_animation = animation  # type: ignore[attr-defined]
    plt.show()


def main() -> None:
    demo_sim()


if __name__ == "__main__":
    main()
