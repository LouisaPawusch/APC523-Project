# Implicit time-stepping methods for the 2D heat transport PDE.
#
# All three methods exploit the fact that the spatial operator is LINEAR:
#
#     dT_flat/dt = M @ T_flat + b(t)
#
# where
#     M      = alpha * L  -  vx_eff * Dx  -  vy_eff * Dy   (sparse, time-independent)
#     b(t)   = alpha * b_L  -  vx_eff * b_x  -  vy_eff * b_y  +  source_flat(t)
#
# Each method reduces to solving a sparse linear system at every time step.
# The system matrix is built from the pre-computed operators dict (see simulation.py).
#
# Interface (identical to forward_euler_step / runge_kutta_4_step):
#
#   T_new = <method>_step(T, problem, operators, t, dt,
#                         linear_solver="direct", **solver_kwargs)
#
# linear_solver options: "direct", "jacobi", "gauss_seidel", "sor"
# solver_kwargs are forwarded to the chosen solver (e.g. tol, max_iter, omega).

import numpy as np
import scipy.sparse as sp

from .operators import flatten_field, unflatten_field
from .linear_solvers import solve_direct, solve_jacobi, solve_gauss_seidel, solve_sor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_spatial_operator(problem, operators):
    """
    Assemble the spatial operator M and the boundary-condition contribution b_bc:

        RHS_flat(T, t) = M @ T_flat + b_bc + source_flat(t)

    M is sparse (CSR) and time-independent for constant-coefficient problems.
    b_bc collects the boundary contributions from all three operator types.

    Handles both scalar vx/vy (uniform flow) and 2D array vx/vy (e.g. from
    Darcy). For arrays, diagonal sparse matrices diag(vx_flat) are used so
    that M = α·L - diag(vx)·Dx - diag(vy)·Dy remains sparse.
    """
    alpha  = problem.alpha
    vx_eff = problem.vx / problem.R_th
    vy_eff = problem.vy / problem.R_th

    if isinstance(vx_eff, np.ndarray):
        # Spatially varying velocity: build diagonal weight matrices
        Vx = sp.diags(vx_eff.ravel(), format="csr")
        Vy = sp.diags(vy_eff.ravel(), format="csr")
        M    = alpha * operators["L"] - Vx @ operators["Dx"] - Vy @ operators["Dy"]
        b_bc = (alpha * operators["b_L"]
                - vx_eff.ravel() * operators["b_x"]
                - vy_eff.ravel() * operators["b_y"])
    else:
        # Scalar velocity: simple scaling
        M    = alpha * operators["L"] - vx_eff * operators["Dx"] - vy_eff * operators["Dy"]
        b_bc = (alpha * operators["b_L"]
                - vx_eff * operators["b_x"]
                - vy_eff * operators["b_y"])
    return M, b_bc


def _source_flat(problem, t):
    """Return the flattened, rho_c_eff-normalised source field at time t."""
    return flatten_field(problem.get_source(t))


def _call_solver(linear_solver, A_sys, rhs, x0, **solver_kwargs):
    """
    Dispatch to the requested linear solver and return the solution vector.

    Parameters
    ----------
    linear_solver : str
        "direct"       — sparse LU via scipy (most robust baseline)
        "jacobi"       — Jacobi iteration (pass omega for relaxation)
        "gauss_seidel" — Gauss-Seidel iteration
        "sor"          — Successive Over-Relaxation (pass omega, default 1.5)
    A_sys         : sparse CSR matrix (N x N)
    rhs           : ndarray (N,)
    x0            : ndarray (N,) — initial guess for iterative solvers
    solver_kwargs : forwarded to the solver (tol, max_iter, omega, …)
    """
    if linear_solver == "direct":
        return solve_direct(A_sys, rhs)

    # Iterative solvers accept x0 as initial guess for warm-starting.
    # Filter kwargs to only those each solver recognises.
    base_kwargs = {k: v for k, v in solver_kwargs.items() if k in ("tol", "max_iter")}
    base_kwargs["x0"] = x0

    if linear_solver == "jacobi":
        omega = solver_kwargs.get("omega", 1.0)
        x, _, converged = solve_jacobi(A_sys, rhs, omega=omega, **base_kwargs)
    elif linear_solver == "gauss_seidel":
        # Gauss-Seidel has no relaxation parameter; ignore omega if supplied
        x, _, converged = solve_gauss_seidel(A_sys, rhs, **base_kwargs)
    elif linear_solver == "sor":
        omega = solver_kwargs.get("omega", 1.5)
        x, _, converged = solve_sor(A_sys, rhs, omega=omega, **base_kwargs)
    else:
        raise ValueError(
            f"Unknown linear_solver '{linear_solver}'. "
            "Choose from: 'direct', 'jacobi', 'gauss_seidel', 'sor'."
        )

    if not converged:
        import warnings
        warnings.warn(
            f"Iterative solver '{linear_solver}' did not converge within "
            f"{solver_kwargs.get('max_iter', 1000)} iterations.",
            RuntimeWarning, stacklevel=3
        )
    return x


# ---------------------------------------------------------------------------
# Backward Euler
# ---------------------------------------------------------------------------

def backward_euler_step(T, problem, operators, t, dt,
                        linear_solver="direct", **solver_kwargs):
    """
    Backward Euler (1st-order, A-stable implicit method).

    Discretisation:
        T_{n+1} = T_n + dt * RHS(T_{n+1}, t_{n+1})

    For the linear PDE this becomes the linear system:
        (I - dt * M) T_{n+1} = T_n + dt * b(t_{n+1})

    Parameters
    ----------
    T             : ndarray (Ny, Nx)  — current temperature field
    problem       : HeatTransportProblem
    operators     : dict of sparse matrices from setup_operators()
    t             : float — current time
    dt            : float — time step
    linear_solver : str   — see _call_solver docstring
    **solver_kwargs       — forwarded to linear solver (tol, max_iter, omega…)

    Returns
    -------
    T_new : ndarray (Ny, Nx)
    """
    M, b_bc = _build_spatial_operator(problem, operators)
    N       = M.shape[0]

    T_flat  = flatten_field(T)

    # System matrix:  A_sys = I - dt * M
    A_sys   = sp.eye(N, format="csr") - dt * M

    # RHS uses source evaluated at t_{n+1}
    b_np1   = b_bc + _source_flat(problem, t + dt)
    rhs     = T_flat + dt * b_np1

    T_new_flat = _call_solver(linear_solver, A_sys, rhs, x0=T_flat.copy(),
                              **solver_kwargs)
    return unflatten_field(T_new_flat, problem.Ny, problem.Nx)


# ---------------------------------------------------------------------------
# Crank-Nicolson
# ---------------------------------------------------------------------------

def crank_nicolson_step(T, problem, operators, t, dt,
                        linear_solver="direct", **solver_kwargs):
    """
    Crank-Nicolson (2nd-order, A-stable implicit method).

    Discretisation (trapezoid rule):
        T_{n+1} = T_n + dt/2 * [RHS(T_n, t_n) + RHS(T_{n+1}, t_{n+1})]

    For the linear PDE this becomes:
        (I - dt/2 * M) T_{n+1} = (I + dt/2 * M) T_n + dt/2 * (b(t_n) + b(t_{n+1}))

    Parameters
    ----------
    (same as backward_euler_step)

    Returns
    -------
    T_new : ndarray (Ny, Nx)
    """
    M, b_bc = _build_spatial_operator(problem, operators)
    N       = M.shape[0]

    T_flat  = flatten_field(T)
    half_dt = dt / 2.0

    # System matrix:  A_sys = I - dt/2 * M
    A_sys   = sp.eye(N, format="csr") - half_dt * M

    # Source at both endpoints
    b_n   = b_bc + _source_flat(problem, t)
    b_np1 = b_bc + _source_flat(problem, t + dt)

    # Explicit part of the RHS
    rhs = (sp.eye(N, format="csr") + half_dt * M) @ T_flat + half_dt * (b_n + b_np1)

    T_new_flat = _call_solver(linear_solver, A_sys, rhs, x0=T_flat.copy(),
                              **solver_kwargs)
    return unflatten_field(T_new_flat, problem.Ny, problem.Nx)


# ---------------------------------------------------------------------------
# Implicit Midpoint
# ---------------------------------------------------------------------------

def implicit_midpoint_step(T, problem, operators, t, dt,
                           linear_solver="direct", **solver_kwargs):
    """
    Implicit midpoint rule (2nd-order, symplectic, A-stable).

    Discretisation:
        T_{n+1} = T_n + dt * RHS(T_mid, t_mid)
        T_mid   = (T_n + T_{n+1}) / 2,   t_mid = t_n + dt/2

    For the linear PDE this becomes:
        (I - dt/2 * M) T_{n+1} = (I + dt/2 * M) T_n + dt * b(t_n + dt/2)

    Note: for time-independent sources this is identical to Crank-Nicolson.
    The difference appears when the source term varies in time:
      - CN averages source at t_n and t_{n+1}
      - Implicit midpoint evaluates source at the single midpoint t_n + dt/2

    Parameters
    ----------
    (same as backward_euler_step)

    Returns
    -------
    T_new : ndarray (Ny, Nx)
    """
    M, b_bc = _build_spatial_operator(problem, operators)
    N       = M.shape[0]

    T_flat  = flatten_field(T)
    half_dt = dt / 2.0

    # System matrix:  A_sys = I - dt/2 * M
    A_sys   = sp.eye(N, format="csr") - half_dt * M

    # Source evaluated at the single midpoint t_n + dt/2
    b_mid = b_bc + _source_flat(problem, t + half_dt)

    # Explicit part of the RHS
    rhs = (sp.eye(N, format="csr") + half_dt * M) @ T_flat + dt * b_mid

    T_new_flat = _call_solver(linear_solver, A_sys, rhs, x0=T_flat.copy(),
                              **solver_kwargs)
    return unflatten_field(T_new_flat, problem.Ny, problem.Nx)
