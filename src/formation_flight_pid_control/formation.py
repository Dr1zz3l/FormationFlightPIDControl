from __future__ import annotations

from typing import List

import numpy as np

from .controllers import PIDFollower
from .params import Params
from .simulation import Airplane6DoFLite
from .visualization import AircraftVisual


def build_formation(aircraft_params: Params) -> List[AircraftVisual]:
    leader = AircraftVisual(
        sim=Airplane6DoFLite(aircraft_params),
        label="Leader",
        color="k",
    )

    follower_specs = [
        ("Follower 1", "r", np.array([-120.0, -200.0, 100.0])),
        ("Follower 2", "g", np.array([-400.0, -150.0, -150.0])),
        ("Follower 3", "b", np.array([-600.0, -500.0, 0.0])),
        ("Follower 4", "orange", np.array([-600.0, -500.0, 0.0])),
    ]

    formation = [leader]
    for label, color, offset in follower_specs:
        sim = Airplane6DoFLite(aircraft_params)
        # Update the position using the new AirplaneState structure
        sim.state.position += offset
        formation.append(
            AircraftVisual(sim=sim, label=label, color=color, pid=PIDFollower())
        )

    return formation
