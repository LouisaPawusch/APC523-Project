# Linear iterative and direct solvers for sparse linear systems A @ x = b.
# Used by the implicit time-stepping methods (Backward Euler, Crank-Nicolson,
# Implicit Midpoint) and optionally for the Darcy sub-problem.
#
# All iterative solvers return (x, residuals, converged) where:
#   x         - solution vector
#   residuals - list of ||b - A @ x||_2 norms, one per iteration
#   converged - True if the tolerance was reached before max_iter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_direct(A, b):
    """
    Direct sparse solve via scipy.sparse.linalg.spsolve (sparse LU factorization).
    This is the most robust baseline; use it to validate iterative results.

    Parameters
    ----------
    A : scipy sparse matrix  (N x N)
    b : ndarray              (N,)

    Returns
    -------
    x : ndarray (N,)
    """
    return spla.spsolve(A.tocsr(), b)


def solve_jacobi(A, b, x0=None, tol=1e-8, max_iter=1000, omega=1.0):
    """
    Jacobi iterative solver (with optional under/over relaxation).

    Update rule (vectorised):
        x_new = (b - (A @ x - d * x)) / d          # standard Jacobi
        x     = (1 - omega) * x + omega * x_new     # relaxed Jacobi

    omega = 1.0 : standard Jacobi
    omega < 1.0 : under-relaxed Jacobi (improves stability for poorly conditioned A)
    omega > 1.0 : over-relaxed Jacobi (rarely useful; SOR is preferred)

    Parameters
    ----------
    A        : scipy sparse matrix (N x N)
    b        : ndarray (N,)
    x0       : ndarray (N,) or None — initial guess; defaults to zeros
    tol      : float — convergence tolerance on ||b - A @ x||_2
    max_iter : int
    omega    : float — relaxation parameter

    Returns
    -------
    x          : ndarray (N,)
    residuals  : list[float]
    converged  : bool
    """
    A_csr = A.tocsr()
    d = A_csr.diagonal()

    if np.any(d == 0.0):
        raise ValueError("Zero diagonal element found; Jacobi iteration requires a non-singular diagonal.")

    x = np.zeros(A_csr.shape[0]) if x0 is None else x0.copy()
    residuals = []

    for _ in range(max_iter):
        r = b - A_csr @ x
        residual = np.linalg.norm(r)
        residuals.append(residual)

        if residual < tol:
            return x, residuals, True

        # Jacobi update: x_new[i] = (b[i] - sum_{j != i} A[i,j]*x[j]) / A[i,i]
        # Equivalent to: x + r / d  (since r[i] = b[i] - A[i,:] @ x)
        x_new = x + r / d
        x = (1.0 - omega) * x + omega * x_new

    return x, residuals, False


def solve_gauss_seidel(A, b, x0=None, tol=1e-8, max_iter=1000):
    """
    Gauss-Seidel iterative solver.

    Each sweep solves the lower-triangular system (D + L) x^{k+1} = b - U x^k,
    which is equivalent to updating x[i] in order using already-updated values:

        x[i] = (b[i] - sum_{j < i} A[i,j] x_new[j] - sum_{j > i} A[i,j] x[j]) / A[i,i]

    Implemented via scipy sparse triangular solve for efficiency.

    Parameters
    ----------
    A        : scipy sparse matrix (N x N)
    b        : ndarray (N,)
    x0       : ndarray (N,) or None
    tol      : float
    max_iter : int

    Returns
    -------
    x         : ndarray (N,)
    residuals : list[float]
    converged : bool
    """
    A_csr = A.tocsr()
    # Split A = (D + L) + U where D+L is lower triangular (incl. diagonal)
    DL = sp.tril(A_csr, k=0).tocsr()   # D + L
    U  = sp.triu(A_csr, k=1).tocsr()   # strictly upper triangular

    x = np.zeros(A_csr.shape[0]) if x0 is None else x0.copy()
    residuals = []

    for _ in range(max_iter):
        r = b - A_csr @ x
        residual = np.linalg.norm(r)
        residuals.append(residual)

        if residual < tol:
            return x, residuals, True

        # Solve (D + L) x^{k+1} = b - U x^k
        rhs = b - U @ x
        x = spla.spsolve_triangular(DL, rhs, lower=True)

    return x, residuals, False


def solve_sor(A, b, omega=1.5, x0=None, tol=1e-8, max_iter=1000):
    """
    SOR (Successive Over-Relaxation) iterative solver.

    Each sweep solves:
        (D + omega * L) x^{k+1} = omega * b + ((1 - omega) * D - omega * U) x^k

    Per-element this is equivalent to:
        x[i] = (1 - omega) * x[i] + omega * GS_update[i]

    omega = 1.0          : reduces to Gauss-Seidel
    omega in (1.0, 2.0)  : over-relaxation — can accelerate convergence
    omega in (0.0, 1.0)  : under-relaxation — improves stability

    Parameters
    ----------
    A        : scipy sparse matrix (N x N)
    b        : ndarray (N,)
    omega    : float in (0, 2) — relaxation factor
    x0       : ndarray (N,) or None
    tol      : float
    max_iter : int

    Returns
    -------
    x         : ndarray (N,)
    residuals : list[float]
    converged : bool
    """
    if not (0.0 < omega < 2.0):
        raise ValueError(f"omega must be in (0, 2) for guaranteed convergence; got {omega}.")

    A_csr = A.tocsr()
    D  = sp.diags(A_csr.diagonal(), format="csr")
    L  = sp.tril(A_csr, k=-1).tocsr()   # strictly lower triangular
    U  = sp.triu(A_csr, k=1).tocsr()    # strictly upper triangular

    # LHS of SOR sweep: D + omega * L  (lower triangular)
    DL_sor = (D + omega * L).tocsr()
    # Precompute the static part of the RHS coefficient matrix
    rhs_mat = ((1.0 - omega) * D - omega * U).tocsr()

    x = np.zeros(A_csr.shape[0]) if x0 is None else x0.copy()
    residuals = []

    for _ in range(max_iter):
        r = b - A_csr @ x
        residual = np.linalg.norm(r)
        residuals.append(residual)

        if residual < tol:
            return x, residuals, True

        # Solve (D + omega*L) x^{k+1} = omega*b + ((1-omega)*D - omega*U) x^k
        rhs = omega * b + rhs_mat @ x
        x = spla.spsolve_triangular(DL_sor, rhs, lower=True)

    return x, residuals, False
