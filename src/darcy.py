# Darcy flow module: solves the steady-state groundwater flow equation
#
#     ∇·(K ∇h) = f
#
# where:
#   h  = hydraulic head [m]
#   K  = hydraulic conductivity [m/s] (scalar or 2D spatially varying field)
#   f  = source/sink term [1/s]  (positive = injection, negative = extraction)
#
# After solving for h, the Darcy flux and pore velocity are derived:
#
#   q  = -K ∇h          (Darcy flux / specific discharge [m/s])
#   v  = q / n           (pore velocity [m/s], n = porosity)
#
# The pore velocity (vx, vy) is a 2D field that can be passed directly
# into HeatTransportProblem.vx / .vy to couple the two physics.
#
# Coupling workflow in simulation.py:
#   1. Solve Darcy once (steady-state assumption)  →  (vx_field, vy_field)
#   2. Set problem.vx = vx_field, problem.vy = vy_field
#   3. Rebuild operators (advection now uses the 2D velocity field)
#   4. Time-step the heat equation as usual

import numpy as np
import scipy.sparse as sp

from .operators import get_flat_index, flatten_field, unflatten_field
from .linear_solvers import solve_direct, solve_jacobi, solve_gauss_seidel, solve_sor


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

class DarcyParams:
    """
    All parameters needed to define and solve the Darcy sub-problem.

    Parameters
    ----------
    K : float or ndarray (Ny, Nx)
        Hydraulic conductivity field [m/s].  A scalar means uniform K.
        Pass a 2D array for heterogeneous media (layers, channels, random fields, …).
    porosity : float
        Effective porosity n (dimensionless, 0 < n < 1).
        Converts Darcy flux to pore velocity: v = q / n.
    bc_head_*_type  : "dirichlet" or "neumann"
        Boundary condition type for hydraulic head on each side.
    bc_head_*_value : float
        - Dirichlet: prescribed head [m]
        - Neumann:   prescribed normal flux  qn = -K dh/dn  [m/s]
                     (positive = inflow, negative = outflow)
                     Use 0.0 for a no-flow (impermeable) boundary.
    source_fn : callable(X, Y) -> ndarray (Ny, Nx), optional
        Volumetric source/sink field [1/s] representing wells or recharge.
        Positive values = injection, negative = extraction.
    """

    def __init__(self, K, porosity,
                 bc_head_left_type="dirichlet",   bc_head_left_value=1.0,
                 bc_head_right_type="dirichlet",  bc_head_right_value=0.0,
                 bc_head_bottom_type="neumann",   bc_head_bottom_value=0.0,
                 bc_head_top_type="neumann",      bc_head_top_value=0.0,
                 source_fn=None):
        self.K         = K
        self.porosity  = porosity

        self.bc_head_left_type    = bc_head_left_type
        self.bc_head_left_value   = bc_head_left_value
        self.bc_head_right_type   = bc_head_right_type
        self.bc_head_right_value  = bc_head_right_value
        self.bc_head_bottom_type  = bc_head_bottom_type
        self.bc_head_bottom_value = bc_head_bottom_value
        self.bc_head_top_type     = bc_head_top_type
        self.bc_head_top_value    = bc_head_top_value

        self.source_fn = source_fn

        if not (0.0 < porosity < 1.0):
            raise ValueError(f"porosity must be in (0, 1); got {porosity}.")


# ---------------------------------------------------------------------------
# Helper: expand K to a 2D array
# ---------------------------------------------------------------------------

def get_K_field(darcy_params, Ny, Nx):
    """Return K as a 2D ndarray of shape (Ny, Nx)."""
    K = darcy_params.K
    if np.isscalar(K):
        return np.full((Ny, Nx), float(K))
    K = np.asarray(K, dtype=float)
    if K.shape != (Ny, Nx):
        raise ValueError(
            f"K must be scalar or shape (Ny={Ny}, Nx={Nx}); got {K.shape}."
        )
    return K


# ---------------------------------------------------------------------------
# Operator builder:  ∇·(K ∇h) = f
# ---------------------------------------------------------------------------

def build_darcy_operator(problem, darcy_params):
    """
    Build the sparse matrix A and RHS vector b such that:

        A @ h_flat = b

    represents the finite-difference discretisation of ∇·(K ∇h) = f.

    Boundary condition strategy
    ---------------------------
    Dirichlet nodes: identity row — A[k,k]=1, b[k]=bc_val.
        Neighbouring interior nodes see the known head value moved to the RHS
        with the appropriate (negative) sign:
            b[k] -= K_face / d**2 * bc_val
        This gives exact BC enforcement (h = bc_val at boundary nodes) and
        is the standard "eliminate boundary DOF" approach for linear systems.

    Neumann nodes: one-sided ghost-point formula.
        For inward flux qn (positive = inflow) at each face:
          left   (inward +x):  b[k] += qn / dx
          right  (inward -x):  b[k] -= qn / dx
          bottom (inward +y):  b[k] += qn / dy
          top    (inward -y):  b[k] -= qn / dy
        For no-flow (qn=0) these contribute nothing.

    Interfacial K uses arithmetic averaging:  K_{i+1/2} = (K_i + K_{i+1}) / 2

    Returns
    -------
    A : scipy.sparse.csr_matrix  (N x N),  N = Nx * Ny
    b : ndarray (N,)
    """
    Nx = problem.Nx
    Ny = problem.Ny
    dx = problem.get_dx()
    dy = problem.get_dy()

    K = get_K_field(darcy_params, Ny, Nx)

    N      = Nx * Ny
    rows   = []
    cols   = []
    values = []
    b      = np.zeros(N)

    # Pack BCs into a dict for convenience
    bcs = {
        "left":   (darcy_params.bc_head_left_type,   darcy_params.bc_head_left_value),
        "right":  (darcy_params.bc_head_right_type,  darcy_params.bc_head_right_value),
        "bottom": (darcy_params.bc_head_bottom_type, darcy_params.bc_head_bottom_value),
        "top":    (darcy_params.bc_head_top_type,    darcy_params.bc_head_top_value),
    }

    def add(r, c, v):
        rows.append(r); cols.append(c); values.append(v)

    def is_dirichlet(r, c):
        """True if node (r,c) lives on a Dirichlet boundary edge."""
        if c == 0    and bcs["left"][0]   == "dirichlet": return True
        if c == Nx-1 and bcs["right"][0]  == "dirichlet": return True
        if r == 0    and bcs["bottom"][0] == "dirichlet": return True
        if r == Ny-1 and bcs["top"][0]    == "dirichlet": return True
        return False

    def dirichlet_val(r, c):
        """Prescribed head for a Dirichlet boundary node."""
        if c == 0:    return bcs["left"][1]
        if c == Nx-1: return bcs["right"][1]
        if r == 0:    return bcs["bottom"][1]
        if r == Ny-1: return bcs["top"][1]
        raise ValueError(f"Node ({r},{c}) is not on a Dirichlet boundary.")

    for row in range(Ny):
        for col in range(Nx):
            k = get_flat_index(row, col, Nx)

            # ---- Dirichlet node: identity row, exact BC enforcement ----
            if is_dirichlet(row, col):
                add(k, k, 1.0)
                b[k] = dirichlet_val(row, col)
                continue

            # ---- Interior or Neumann-boundary node: full FD stencil ----
            # Interfacial conductivities (arithmetic mean at cell faces)
            K_e = (K[row, col] + K[row, min(col + 1, Nx - 1)]) / 2.0
            K_w = (K[row, col] + K[row, max(col - 1, 0)     ]) / 2.0
            K_n = (K[row, col] + K[min(row + 1, Ny - 1), col]) / 2.0
            K_s = (K[row, col] + K[max(row - 1, 0),       col]) / 2.0

            diag = 0.0

            # --- East neighbour (col + 1) ---
            if col == Nx - 1:
                # Right boundary: must be Neumann (Dirichlet handled above)
                qn = bcs["right"][1]
                b[k] -= qn / dx         # inward is -x at right face
            elif is_dirichlet(row, col + 1):
                # Dirichlet east: move known value to RHS
                b[k] -= K_e / dx**2 * dirichlet_val(row, col + 1)
                diag  -= K_e / dx**2
            else:
                add(k, get_flat_index(row, col + 1, Nx), K_e / dx**2)
                diag -= K_e / dx**2

            # --- West neighbour (col - 1) ---
            if col == 0:
                # Left boundary: must be Neumann
                qn = bcs["left"][1]
                b[k] += qn / dx         # inward is +x at left face
            elif is_dirichlet(row, col - 1):
                b[k] -= K_w / dx**2 * dirichlet_val(row, col - 1)
                diag  -= K_w / dx**2
            else:
                add(k, get_flat_index(row, col - 1, Nx), K_w / dx**2)
                diag -= K_w / dx**2

            # --- North neighbour (row + 1) ---
            if row == Ny - 1:
                # Top boundary: must be Neumann
                qn = bcs["top"][1]
                b[k] -= qn / dy         # inward is -y at top face
            elif is_dirichlet(row + 1, col):
                b[k] -= K_n / dy**2 * dirichlet_val(row + 1, col)
                diag  -= K_n / dy**2
            else:
                add(k, get_flat_index(row + 1, col, Nx), K_n / dy**2)
                diag -= K_n / dy**2

            # --- South neighbour (row - 1) ---
            if row == 0:
                # Bottom boundary: must be Neumann
                qn = bcs["bottom"][1]
                b[k] += qn / dy         # inward is +y at bottom face
            elif is_dirichlet(row - 1, col):
                b[k] -= K_s / dy**2 * dirichlet_val(row - 1, col)
                diag  -= K_s / dy**2
            else:
                add(k, get_flat_index(row - 1, col, Nx), K_s / dy**2)
                diag -= K_s / dy**2

            add(k, k, diag)

    A = sp.csr_matrix((values, (rows, cols)), shape=(N, N))
    return A, b


# ---------------------------------------------------------------------------
# Darcy solver
# ---------------------------------------------------------------------------

def solve_darcy(problem, darcy_params, linear_solver="direct", **solver_kwargs):
    """
    Solve the Darcy flow problem and return the pore velocity field.

    Calls build_darcy_operator, adds well/source terms, solves the linear
    system, then differentiates h to get flux and pore velocity.

    Parameters
    ----------
    problem       : HeatTransportProblem  (supplies Nx, Ny, dx, dy, mesh)
    darcy_params  : DarcyParams
    linear_solver : str  — "direct", "jacobi", "gauss_seidel", "sor"
    **solver_kwargs       — forwarded to the iterative solver

    Returns
    -------
    h  : ndarray (Ny, Nx)  hydraulic head [m]
    vx : ndarray (Ny, Nx)  pore velocity in x [m/s]
    vy : ndarray (Ny, Nx)  pore velocity in y [m/s]
    """
    Nx = problem.Nx
    Ny = problem.Ny
    dx = problem.get_dx()
    dy = problem.get_dy()

    A, b = build_darcy_operator(problem, darcy_params)

    # Add well / recharge source terms: ∇·(K∇h) = f  →  A h = b + f_flat
    if darcy_params.source_fn is not None:
        X, Y   = problem.get_mesh()
        f_flat = flatten_field(darcy_params.source_fn(X, Y))
        b      = b + f_flat

    # Solve the linear system
    if linear_solver == "direct":
        h_flat = solve_direct(A, b)
    elif linear_solver == "jacobi":
        h_flat, _, converged = solve_jacobi(A, b, **solver_kwargs)
        if not converged:
            import warnings
            warnings.warn("Darcy Jacobi solver did not converge.", RuntimeWarning)
    elif linear_solver == "gauss_seidel":
        h_flat, _, converged = solve_gauss_seidel(A, b, **solver_kwargs)
        if not converged:
            import warnings
            warnings.warn("Darcy Gauss-Seidel solver did not converge.", RuntimeWarning)
    elif linear_solver == "sor":
        h_flat, _, converged = solve_sor(A, b, **solver_kwargs)
        if not converged:
            import warnings
            warnings.warn("Darcy SOR solver did not converge.", RuntimeWarning)
    else:
        raise ValueError(f"Unknown linear_solver '{linear_solver}'. "
                         "Choose: 'direct', 'jacobi', 'gauss_seidel', 'sor'.")

    h = unflatten_field(h_flat, Ny, Nx)
    K = get_K_field(darcy_params, Ny, Nx)
    n = darcy_params.porosity

    # ---- Darcy flux q = -K ∇h ----
    # Central differences at interior; one-sided at boundaries.
    dhdx = np.empty_like(h)
    dhdy = np.empty_like(h)

    dhdx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) / (2.0 * dx)
    dhdx[:,  0  ] = (h[:,  1 ] - h[:,  0 ]) / dx
    dhdx[:, -1  ] = (h[:, -1 ] - h[:, -2 ]) / dx

    dhdy[1:-1, :] = (h[2:, :] - h[:-2, :]) / (2.0 * dy)
    dhdy[ 0,   :] = (h[ 1, :] - h[ 0,  :]) / dy
    dhdy[-1,   :] = (h[-1, :] - h[-2,  :]) / dy

    qx = -K * dhdx
    qy = -K * dhdy

    # ---- Pore velocity v = q / n ----
    vx = qx / n
    vy = qy / n

    return h, vx, vy


# ---------------------------------------------------------------------------
# Convenience: heterogeneous K field generators
# ---------------------------------------------------------------------------

def make_layered_K(Ny, Nx, K_values, layer_fractions):
    """
    Build a horizontally layered K field (layers run parallel to x-axis).

    Parameters
    ----------
    Ny, Nx         : grid dimensions
    K_values       : list of K [m/s] for each layer (top to bottom)
    layer_fractions: list of fractional thicknesses (must sum to 1)

    Example
    -------
    # Three-layer aquifer: high-K centre, low-K top and bottom
    K = make_layered_K(50, 50, [1e-5, 1e-3, 1e-5], [0.25, 0.50, 0.25])
    """
    assert abs(sum(layer_fractions) - 1.0) < 1e-10, "layer_fractions must sum to 1"
    K = np.zeros((Ny, Nx))
    boundaries = np.round(np.cumsum([0] + list(layer_fractions)) * Ny).astype(int)
    for i, K_val in enumerate(K_values):
        K[boundaries[i]: boundaries[i + 1], :] = K_val
    return K


def make_random_K(Ny, Nx, K_mean, K_std, log_normal=True, seed=None):
    """
    Generate a spatially random K field.

    Parameters
    ----------
    K_mean, K_std : mean and std of the distribution
    log_normal    : if True, sample ln(K) ~ Normal (realistic for geology)
    seed          : random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    if log_normal:
        mu    = np.log(K_mean**2 / np.sqrt(K_std**2 + K_mean**2))
        sigma = np.sqrt(np.log(1 + (K_std / K_mean)**2))
        K     = rng.lognormal(mean=mu, sigma=sigma, size=(Ny, Nx))
    else:
        K = np.clip(rng.normal(K_mean, K_std, size=(Ny, Nx)), 1e-10, None)
    return K


# ---------------------------------------------------------------------------
# Analytical solution: uniform K, linear head
# ---------------------------------------------------------------------------

def darcy_analytical_solution(problem, darcy_params):
    """
    Analytical head and pore velocity for uniform K with Dirichlet head BCs
    on left/right and no-flow Neumann BCs on top/bottom (no source term).

    Exact solution: h(x) = H_L + (H_R - H_L)*x/Lx  (linear, y-independent)
    Pore velocity:  vx = K*(H_L - H_R) / (n*Lx),  vy = 0

    Returns
    -------
    h_ana  : ndarray (Ny, Nx)
    vx_ana : float
    vy_ana : float  (always 0.0)
    """
    X, _ = problem.get_mesh()
    H_L  = darcy_params.bc_head_left_value
    H_R  = darcy_params.bc_head_right_value
    h_ana = H_L + (H_R - H_L) * X / problem.Lx
    K_val = darcy_params.K if np.isscalar(darcy_params.K) else float(np.mean(darcy_params.K))
    vx_ana = K_val * (H_L - H_R) / (darcy_params.porosity * problem.Lx)
    return h_ana, vx_ana, 0.0
