# Script to run the simulation of the 2D heat equation with advection and source terms, 
# using the problem definition and time-stepping methods defined in the other modules.

import numpy as np

def apply_boundary_conditions(T, problem, t):
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

def compute_rhs(T, problem, t, advection_scheme="central"):

    dx = problem.get_dx()
    dy = problem.get_dy()

    alpha = problem.alpha
    vx_eff = problem.vx / problem.R_th
    vy_eff = problem.vy / problem.R_th

    rhs = np.zeros_like(T)

    source_eff = problem.get_source(t)

    # diffusion
    #TODO: replace this with formulation from operators.py for better accuracy and consistency with the sparse matrix implementation.
    T_xx = (T[1:-1, 2:] - 2.0 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dx**2
    T_yy = (T[2:, 1:-1] - 2.0 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dy**2

    # advection
    if advection_scheme == "central":
        T_x = (T[1:-1, 2:] - T[1:-1, :-2]) / (2.0 * dx)
        T_y = (T[2:, 1:-1] - T[:-2, 1:-1]) / (2.0 * dy)

    elif advection_scheme == "upwind":
        if vx_eff >= 0:
            T_x = (T[1:-1, 1:-1] - T[1:-1, :-2]) / dx
        else:
            T_x = (T[1:-1, 2:] - T[1:-1, 1:-1]) / dx

        if vy_eff >= 0:
            T_y = (T[1:-1, 1:-1] - T[:-2, 1:-1]) / dy
        else:
            T_y = (T[2:, 1:-1] - T[1:-1, 1:-1]) / dy

    else:
        raise ValueError("advection_scheme must be 'central' or 'upwind'.")

    rhs[1:-1, 1:-1] = (alpha * (T_xx + T_yy) - vx_eff * T_x - vy_eff * T_y + source_eff[1:-1, 1:-1])

    return rhs


def forward_euler_step(T, problem, t, dt, advection_scheme="central"):
    T_old = T.copy()
    apply_boundary_conditions(T_old, problem, t)

    rhs = compute_rhs(T_old, problem, t, advection_scheme=advection_scheme)
    T_new = T_old + dt * rhs

    apply_boundary_conditions(T_new, problem, t + dt)
    return T_new


def run_simulation(problem, t_final, dt, save_every=1, advection_scheme="central"):
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if t_final <= 0:
        raise ValueError("t_final must be positive.")
    if save_every < 1:
        raise ValueError("save_every must be at least 1.")

    T = problem.get_initial_condition()
    t = 0.0

    apply_boundary_conditions(T, problem, t)

    times = [t]
    all_T = [T.copy()]

    n_steps = int(np.ceil(t_final / dt))

    for step in range(1, n_steps + 1):
        dt_step = min(dt, t_final - t)

        if dt_step <= 0:
            break

        T = forward_euler_step(T, problem, t, dt_step, advection_scheme=advection_scheme)
        print(f"Step {step}/{n_steps}, Time: {t:.4f}/{t_final:.4f}, T min: {T.min():.4f}, T max: {T.max():.4f}")

        t += dt_step

        if step % save_every == 0 or np.isclose(t, t_final):
            times.append(t)
            all_T.append(T.copy())

    return np.array(times), all_T