# Simple 6DOF Airplane Model with Visualization

This project implements a simple but thorough **6 Degrees of Freedom
(6DOF)** airplane dynamics model in discrete time with live 3D
visualization.

Author: **Peter Ryseck**\
Date: **09/27/2025**

Video: https://www.youtube.com/watch?v=voOmPTmaBiU

<img width="400" height="300" alt="Screenshot 2025-09-27 at 4 00 31 PM" src="https://github.com/user-attachments/assets/2ed3f0a1-880c-4ddb-a82e-12954a644c8f" />


------------------------------------------------------------------------

## Features

-   Full 6DOF dynamics: position, velocity, Euler angles, and body rates
-   Pilot inputs: throttle, roll, pitch, yaw
-   Physics:
    -   Gravity
    -   Thrust proportional to throttle
    -   Aerodynamic forces: lift, sideforce, drag
    -   Control torques: stick proportional + rate damping
-   Integration: Runge--Kutta 4 (RK4) fixed step
-   Visualization: Real-time 3D plotting using `matplotlib`

------------------------------------------------------------------------

## Dependencies

-   Python 3.8+
-   `numpy`
-   `matplotlib`

Install requirements with:

``` bash
pip install numpy matplotlib
```

------------------------------------------------------------------------

## Usage

Run the demo simulation:

``` bash
python airplane6dof.py
```

A live 3D visualization window will appear showing the airplane(s)
flying.

------------------------------------------------------------------------

## File Structure

-   `airplane6dof.py`: Main simulation script with airplane model,
    dynamics, and visualization
-   `Params`: Configuration class to adjust mass, wing area, thrust,
    aerodynamic coefficients, etc.

------------------------------------------------------------------------

## Controls

Inputs are normalized pilot commands: - `throttle` ∈ \[0, 1\] -
`roll_cmd` ∈ \[-1, 1\] - `pitch_cmd` ∈ \[-1, 1\] - `yaw_cmd` ∈ \[-1, 1\]

------------------------------------------------------------------------

## Notes

-   Suitable for experimenting with basic flight dynamics, formation
    control, and visualization.
-   To adjust behavior, modify the `Params` class.

------------------------------------------------------------------------

## License

MIT License
