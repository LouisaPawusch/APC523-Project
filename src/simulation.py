# Script to run the simulation of the 2D heat equation with advection and source terms, 
# using the problem definition and time-stepping methods defined in the other modules.

import os
import sys
import numpy as np

from .operators import build_laplacian_2d, build_advection_x, build_advection_y
from .explicit_timesteppers import forward_euler_step, runge_kutta_4_step
from .implicit_timesteppers import (backward_euler_step,
                                    crank_nicolson_step,
                                    implicit_midpoint_step)

def setup_operators(problem, advection_scheme="central"):
    """
    Build all sparse operators and boundary vectors once.
    """
    L, b_L = build_laplacian_2d(problem)
    Dx, b_x = build_advection_x(problem, scheme=advection_scheme)
    Dy, b_y = build_advection_y(problem, scheme=advection_scheme)

    return {"L": L,
        "b_L": b_L,
        "Dx": Dx,
        "b_x": b_x,
        "Dy": Dy,
        "b_y": b_y}

def apply_boundary_conditions(T, problem, t):
    """
    Explicitely apply boundary conditions to the temperature field T at time t. 
    The operators are built assuming certain BCs, but we still need to enforce them on the evolving T field at each time step.
    The BCs are applied in the following order: left, right, bottom, top.
    """
    dx = problem.get_dx()
    dy = problem.get_dy()

    # Left boundary
    if problem.bc_left_type == "dirichlet":
        #check whether the value is a function of time or a constant
        if callable(problem.bc_left_value):
            T[:, 0] = problem.bc_left_value(t)
        else:
            T[:, 0] = problem.bc_left_value
    elif problem.bc_left_type == "neumann":
        if callable(problem.bc_left_value):
            grad = problem.bc_left_value(t)
        else:
            grad = problem.bc_left_value
        # Apply 1st order finite difference for Neumann BC: T[0, j] = T[1, j] - grad * dx
        #TODO: this is a first-order approximation; for better accuracy, consider using a second-order scheme or ghost points.
        T[:, 0] = T[:, 1] - grad * dx

    # Right boundary
    if problem.bc_right_type == "dirichlet":
        if callable(problem.bc_right_value):
            T[:, -1] = problem.bc_right_value(t)
        else:
            T[:, -1] = problem.bc_right_value
    elif problem.bc_right_type == "neumann":
        if callable(problem.bc_right_value):
            grad = problem.bc_right_value(t)
        else:
            grad = problem.bc_right_value
        T[:, -1] = T[:, -2] + grad * dx

    # Bottom boundary
    if problem.bc_bottom_type == "dirichlet":
        if callable(problem.bc_bottom_value):
            T[0, :] = problem.bc_bottom_value(t)
        else:
            T[0, :] = problem.bc_bottom_value
    elif problem.bc_bottom_type == "neumann":
        if callable(problem.bc_bottom_value):
            grad = problem.bc_bottom_value(t)
        else:
            grad = problem.bc_bottom_value
        T[0, :] = T[1, :] - grad * dy

    # Top boundary
    if problem.bc_top_type == "dirichlet":
        if callable(problem.bc_top_value):
            T[-1, :] = problem.bc_top_value(t)
        else:
            T[-1, :] = problem.bc_top_value
    elif problem.bc_top_type == "neumann":
        if callable(problem.bc_top_value):
            grad = problem.bc_top_value(t)
        else:
            grad = problem.bc_top_value
        T[-1, :] = T[-2, :] + grad * dy


def run_simulation(problem, t_final, dt, save_every=1, advection_scheme="central",
                   timestepper="FE", linear_solver="direct", **solver_kwargs):
    """
    Main function to run the time-stepping simulation.
    """

    if dt <= 0:
        raise ValueError("dt must be positive.")
    if t_final <= 0:
        raise ValueError("t_final must be positive.")
    if save_every < 1:
        raise ValueError("save_every must be at least 1.")

    T = problem.get_initial_condition()
    t = 0.0

    apply_boundary_conditions(T, problem, t)

    operators = setup_operators(problem, advection_scheme=advection_scheme)

    times = [t]
    all_T = [T.copy()]

    n_steps = int(np.ceil(t_final / dt))

    for step in range(1, n_steps + 1):
        dt_step = min(dt, t_final - t)

        if dt_step <= 0:
            break

        T_old = T.copy()
        apply_boundary_conditions(T_old, problem, t)
        if timestepper == "FE":
            T = forward_euler_step(T_old, problem, operators, t, dt_step)
        elif timestepper == "RK4":
            T = runge_kutta_4_step(T_old, problem, operators, t, dt_step)
        elif timestepper == "BE":
            T = backward_euler_step(T_old, problem, operators, t, dt_step,
                                    linear_solver=linear_solver, **solver_kwargs)
        elif timestepper == "CN":
            T = crank_nicolson_step(T_old, problem, operators, t, dt_step,
                                    linear_solver=linear_solver, **solver_kwargs)
        elif timestepper == "IM":
            T = implicit_midpoint_step(T_old, problem, operators, t, dt_step,
                                       linear_solver=linear_solver, **solver_kwargs)
        else:
            raise ValueError(
                f"Unknown timestepper: '{timestepper}'. "
                "Supported options: 'FE', 'RK4', 'BE', 'CN', 'IM'."
            )

        apply_boundary_conditions(T, problem, t + dt_step)

        print(f"Step {step}/{n_steps}, Time: {t:.4f}/{t_final:.4f}, T min: {T.min():.4f}, T max: {T.max():.4f}")

        t += dt_step

        if step % save_every == 0 or np.isclose(t, t_final):
            times.append(t)
            all_T.append(T.copy())

    return np.array(times), all_T