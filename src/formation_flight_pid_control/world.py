

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .airplane import Airplane6DoFLite, AirplaneState
from .controllers import PIDFollower
from .formation import build_formation
from .geometry import AIRCRAFT_GEOMETRY_BODY, rotate_body_to_earth, rotate_earth_to_body, _normalized_quaternion
from .params import Params
from .visualization import AircraftVisual


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
        # Leader control
        leader = self.formation[0]
        u_leader = PIDFollower.pilot_leader(self.t, leader.sim.state)
        leader.sim.step(u_leader, dt)
        leader.throttle_history.append(u_leader[0])

        # Follower control
        for follower, target in zip(self.formation[1:], self.formation[:-1]):
            if follower.pid is None:
                raise ValueError("Follower missing PID controller")
            u_cmd = follower.pid.pilot_follower(
                self.t, follower.sim.state, target.sim.state, dt
            )
            follower.sim.step(u_cmd, dt)
            follower.throttle_history.append(u_cmd[0])

        self.t += dt
    
    def get_formation(self) -> List[AircraftVisual]:
        """Get the formation for visualization."""
        return self.formation
    
    def get_time(self) -> float:
        """Get current simulation time."""
        return self.t