from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .airplane import Airplane6DoFLite
from .controllers import PIDFollower
from .params import Params
from .visualization import AircraftVisual

# Type alias for follower specification: (label, color, offset_position)
FollowerSpec = Tuple[str, str, np.ndarray]


def build_formation(
    aircraft_params: Params, 
    leader_config: Optional[Tuple[str, str]] = None,
    follower_specs: Optional[List[FollowerSpec]] = None
) -> List[AircraftVisual]:
    """Build aircraft formation with configurable leader and followers.
    
    Args:
        aircraft_params: Aircraft physical parameters
        leader_config: (label, color) for leader aircraft. Defaults to ("Leader", "k")
        follower_specs: List of (label, color, offset_position) for followers.
                       Defaults to 2 followers if None provided.
    
    Returns:
        List of AircraftVisual objects representing the formation
    """
    # Default leader configuration
    if leader_config is None:
        leader_config = ("Leader", "k")

    # Default follower configuration - 2 followers
    if follower_specs is None:
        follower_specs = [
            ("Follower 1", "r", np.array([-120.0, -200.0, 100.0])),
            ("Follower 2", "g", np.array([-400.0, -150.0, -150.0])),
        ]
    
    leader_label, leader_color = leader_config
    leader = AircraftVisual(
        sim=Airplane6DoFLite(aircraft_params),
        label=leader_label,
        color=leader_color,
    )

    formation = [leader]
    for label, color, offset in follower_specs:
        sim = Airplane6DoFLite(aircraft_params)
        # Update the position using the new AirplaneState structure
        sim.state.position += offset
        formation.append(
            AircraftVisual(sim=sim, label=label, color=color, pid=PIDFollower())
        )

    return formation
