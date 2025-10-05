#!/usr/bin/env python3

from src.formation_flight_pid_control.formation import build_formation
from src.formation_flight_pid_control.params import Params
from src.formation_flight_pid_control.controllers import PIDFollower
from src.formation_flight_pid_control.geometry import rotate_earth_to_body
import numpy as np
import math

def debug_simulation():
    params = Params()
    formation = build_formation(params)
    
    dt = 0.001  # Same as main.py
    t = 0.0
    
    print('Stall angle:', math.degrees(params.alpha_stall), 'degrees')
    print(f'Formation size: {len(formation)} aircraft')
    print()
    
    for step in range(50):  # Just first 50 steps
        # Leader control
        leader = formation[0]
        u_leader = PIDFollower.pilot_leader(t, leader.sim.state)
        leader.sim.step(u_leader, dt)
        
        # Follower control
        for i, (follower, target) in enumerate(zip(formation[1:], formation[:-1])):
            u_cmd, force, moment = follower.pid.pilot_follower(
                t, follower.sim.state, target.sim.state, dt
            )
            follower.sim.step(u_cmd, dt, ext_F_body=force, ext_M_body=moment)
            
            # Check angle of attack
            quat_array = np.array([follower.sim.state.pose.w, follower.sim.state.pose.x, 
                                 follower.sim.state.pose.y, follower.sim.state.pose.z])
            vel_body = rotate_earth_to_body(quat_array, follower.sim.state.velocity)
            alpha = math.atan2(vel_body[2], vel_body[0])
            alpha_deg = math.degrees(alpha)
            
            if step % 10 == 0:  # Print every 10th step
                airspeed = np.linalg.norm(vel_body)
                print(f'Step {step:2d}, Follower {i+1}: AoA = {alpha_deg:7.2f}°, '
                      f'Airspeed = {airspeed:6.2f}, Pitch = {u_cmd[2]:6.3f}', end='')
                if alpha_deg > math.degrees(params.alpha_stall):
                    print(' [STALL]')
                else:
                    print()
        
        t += dt

if __name__ == "__main__":
    debug_simulation()