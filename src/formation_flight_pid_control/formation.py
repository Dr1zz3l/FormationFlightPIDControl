from __future__ import annotations

from dataclasses import replace
from typing import List

import numpy as np

from .controllers import PIDFollower
from .params import Params
from .simulation import Airplane
from .visualization import AircraftVisual


def build_formation(aircraft_params: Params) -> List[AircraftVisual]:
    """Build a formation of aircraft with leader and followers.
    
    Args:
        aircraft_params: Aircraft physical parameters
        
    Returns:
        List of AircraftVisual objects representing the formation
    """
    # Create leader aircraft
    leader = AircraftVisual(
        sim=Airplane(aircraft_params),
        label="Leader",
        color="k",
    )

    # Follower specifications: (name, color, position_offset)
    follower_specs = [
        ("Follower 1", "r", np.array([-120.0, -200.0, 100.0])),
        ("Follower 2", "g", np.array([-400.0, -150.0, -150.0])),
        ("Follower 3", "b", np.array([-600.0, -500.0, 0.0])),
        ("Follower 4", "orange", np.array([-600.0, -500.0, 0.0])),
    ]

    formation = [leader]
    
    for label, color, position_offset in follower_specs:
        # Create follower aircraft
        sim = Airplane(aircraft_params)
        
        # Set initial position offset from leader
        current_state = sim.state
        new_position = current_state.position + position_offset
        sim.state = replace(current_state, position=new_position)
        
        formation.append(
            AircraftVisual(sim=sim, label=label, color=color, pid=PIDFollower())
        )

    return formation
