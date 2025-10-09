"""Formation flight simulation class."""
from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - imported for side effects

from .params import Params
from .visualization import (
    attach_aircraft_lines,
    collect_line_artists,
    configure_figure,
    update_aircraft_visual,
    update_throttle_plot,
)
from .world import World


class FormationFlightSimulation:
    """Formation flight simulation with configurable parameters."""
    
    def __init__(
        self,
        params: Params,
        leader_config: tuple[str, str] = ("Leader", "k"),
        follower_specs: List[tuple[str, str, np.ndarray]] = None,
        dt: float = 0.05,
        tfinal: float = 10000.0,
        draw_every: int = 5
    ):
        """Initialize formation flight simulation.
        
        Args:
            params: Aircraft physical parameters
            leader_config: (label, color) for leader aircraft
            follower_specs: List of (label, color, offset_position) for followers
            dt: Integration time step in seconds
            tfinal: Total simulation time in seconds
            draw_every: Number of integration steps between animation frames
        """
        self.params = params
        self.dt = dt
        self.tfinal = tfinal
        self.draw_every = draw_every
        
        # Create the world with formation
        self.world = World(params, leader_config, follower_specs)
        
        # Initialize simulation state
        self.steps = int(tfinal / dt)
        self.time_history: List[float] = []
        
        # Initialize visualization
        self.fig, self.ax_3d, self.ax_throttle = configure_figure()
        attach_aircraft_lines(self.ax_3d, self.ax_throttle, self.world.get_formation())
        
    def _init_animation(self):
        """Initialize animation frame."""
        formation = self.world.get_formation()
        for member in formation:
            for line in member.line_handles.values():
                line.set_data([], [])
                line.set_3d_properties([])
            if member.throttle_line is not None:
                member.throttle_line.set_data([], [])
            if member.vortex_lines_left or member.vortex_lines_right:
                for vortex_line in member.vortex_lines_left + member.vortex_lines_right:
                    vortex_line.set_data([], [])
                    vortex_line.set_3d_properties([])
                    vortex_line.set_alpha(0.0)
                member.vortex_history_left.clear()
                member.vortex_history_right.clear()
                member.vortex_phase = 0.0
        return collect_line_artists(formation)

    def _update_animation(self, _frame_index: int):
        """Update animation frame."""
        if self.world.get_time() >= self.tfinal:
            plt.close(self.fig)
            return []

        # Run multiple simulation steps per animation frame
        for _ in range(self.draw_every):
            self.world.simulation_step(self.dt)
            self.time_history.append(self.world.get_time())

        # Update visualization
        formation = self.world.get_formation()
        for member in formation:
            update_aircraft_visual(member)

        update_throttle_plot(self.ax_throttle, self.time_history, formation)
        self.ax_3d.set_box_aspect([1, 1, 1])
        return collect_line_artists(formation)
    


    def run(self):
        """Run the formation flight simulation with visualization."""
        animation = FuncAnimation(
            self.fig,
            self._update_animation,
            init_func=self._init_animation,
            frames=self.steps,
            interval=self.dt * 1000,
            blit=False,
        )
        
        # Keep a reference to prevent garbage collection
        self.fig._formation_animation = animation  # type: ignore[attr-defined]
        
        plt.show()


def demo_sim() -> None:
    """Run default formation flight demonstration."""
    params = Params()
    
    # Define formation configuration
    leader_config = ("Leader", "k")
    follower_specs = [
        ("Follower 1", "r", np.array([-120.0, -200.0, 100.0])),
        ("Follower 2", "g", np.array([-400.0, -150.0, -150.0])),
        # ("Follower 3", "b", np.array([-600.0, -500.0, 0.0])),
        # ("Follower 4", "orange", np.array([-600.0, -500.0, 0.0])),
    ]
    
    # Create and run simulation
    sim = FormationFlightSimulation(
        params=params,
        leader_config=leader_config,
        follower_specs=follower_specs,
        dt=0.05,  # Restored original timestep - quaternion integration fixed
        tfinal=10000.0,
        draw_every=5
    )
    
    sim.run()
