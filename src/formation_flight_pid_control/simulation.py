"""Formation flight simulation class."""
from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - imported for side effects

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
        
        # Build the formation
        self.formation = build_formation(params, leader_config, follower_specs)
        
        # Initialize simulation state
        self.t = 0.0
        self.steps = int(tfinal / dt)
        self.time_history: List[float] = []
        
        # Initialize visualization
        self.fig, self.ax_3d, self.ax_throttle = configure_figure()
        attach_aircraft_lines(self.ax_3d, self.ax_throttle, self.formation)
        
    def _init_animation(self):
        """Initialize animation frame."""
        for member in self.formation:
            for line in member.line_handles.values():
                line.set_data([], [])
                line.set_3d_properties([])
            if member.throttle_line is not None:
                member.throttle_line.set_data([], [])
        return collect_line_artists(self.formation)

    def _update_animation(self, _frame_index: int):
        """Update animation frame."""
        if self.t >= self.tfinal:
            plt.close(self.fig)
            return []

        # Run multiple simulation steps per animation frame
        for _ in range(self.draw_every):
            self._simulation_step()

        # Update visualization
        for member in self.formation:
            update_aircraft_visual(member)

        update_throttle_plot(self.ax_throttle, self.time_history, self.formation)
        self.ax_3d.set_box_aspect([1, 1, 1])
        return collect_line_artists(self.formation)
    
    def _simulation_step(self):
        """Execute a single simulation step for all aircraft."""
        # Leader control
        leader = self.formation[0]
        u_leader = PIDFollower.pilot_leader(self.t, leader.sim.state)
        leader.sim.step(u_leader, self.dt)
        leader.throttle_history.append(u_leader[0])

        # Follower control
        for follower, target in zip(self.formation[1:], self.formation[:-1]):
            if follower.pid is None:
                raise ValueError("Follower missing PID controller")
            u_cmd, force, moment = follower.pid.pilot_follower(
                self.t, follower.sim.state, target.sim.state, self.dt
            )
            follower.sim.step(u_cmd, self.dt, ext_F_body=force, ext_M_body=moment)
            follower.throttle_history.append(u_cmd[0])

        self.t += self.dt
        self.time_history.append(self.t)

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
        
    def run_headless(self, max_steps: int = None) -> dict:
        """Run simulation without visualization for testing/analysis.
        
        Args:
            max_steps: Maximum number of steps to run (defaults to all)
            
        Returns:
            Dictionary with simulation results
        """
        max_steps = max_steps or self.steps
        
        for step in range(min(max_steps, self.steps)):
            if self.t >= self.tfinal:
                break
            self._simulation_step()
        
        # Return final positions and states
        results = {
            'final_time': self.t,
            'steps_completed': step + 1,
            'aircraft_states': []
        }
        
        for aircraft in self.formation:
            results['aircraft_states'].append({
                'label': aircraft.label,
                'position': aircraft.sim.state.position.copy(),
                'velocity': aircraft.sim.state.velocity.copy(),
                'pose': aircraft.sim.state.pose,
                'angular_rates': aircraft.sim.state.angular_rates.copy()
            })
        
        return results


def demo_sim() -> None:
    """Run default formation flight demonstration."""
    params = Params()
    
    # Define formation configuration
    leader_config = ("Leader", "k")
    follower_specs = [
        ("Follower 1", "r", np.array([-120.0, -200.0, 100.0])),
        ("Follower 2", "g", np.array([-400.0, -150.0, -150.0])),
        ("Follower 3", "b", np.array([-600.0, -500.0, 0.0])),
        ("Follower 4", "orange", np.array([-600.0, -500.0, 0.0])),
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
