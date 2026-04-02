# Finite-difference operators for the 2D heat transport problem.
# These operators match the array shape used in simulation.py:
#
#     T.shape = (Ny, Nx)
#
# so:
#   - first index  = y-direction (row)
#   - second index = x-direction (column)
#
# The operators are built from the boundary-condition information stored
# in the problem object. For Dirichlet boundaries, nonzero boundary values
# are handled through an additional boundary vector.

import numpy as np
import scipy.sparse as sp


def get_flat_index(row, col, Nx):
    """
    Convert 2D array index (row, col) into a 1D flat index.

    Here:
        row = y-index, from 0 to Ny-1
        col = x-index, from 0 to Nx-1

    consistent with T.shape = (Ny, Nx).
    """
    return row * Nx + col


def flatten_field(T):
    """
    Flatten a 2D field T of shape (Ny, Nx) into a 1D vector.
    """
    return T.reshape(-1)


def unflatten_field(T_flat, Ny, Nx):
    """
    Reshape a 1D vector back into a 2D field of shape (Ny, Nx).
    """
    return T_flat.reshape(Ny, Nx)


def get_constant_bc_value(value):
    """
    Return a constant boundary value.
    """
    if callable(value):
        raise ValueError(
            "This operator builder currently assumes time-independent boundary values. "
            "Time-dependent BC functions should be handled separately during time stepping."
        )
    return value


def get_boundary_info(problem):
    """
    Read boundary types and values from the problem object.
    """
    bc = {
        "left": {
            "type": problem.bc_left_type,
            "value": get_constant_bc_value(problem.bc_left_value),
        },
        "right": {
            "type": problem.bc_right_type,
            "value": get_constant_bc_value(problem.bc_right_value),
        },
        "bottom": {
            "type": problem.bc_bottom_type,
            "value": get_constant_bc_value(problem.bc_bottom_value),
        },
        "top": {
            "type": problem.bc_top_type,
            "value": get_constant_bc_value(problem.bc_top_value),
        },
    }
    return bc


def build_laplacian_2d(problem):
    """
    Build the sparse matrix A and boundary vector b for the 2D Laplacian:

        T_xx + T_yy

    such that approximately

        laplacian(T)_flat = A @ T_flat + b

    where T has shape (Ny, Nx).

    Boundary conditions are read from the problem object.
    Supported types:
        - "dirichlet"
        - "neumann"

    """
    Nx = problem.Nx
    Ny = problem.Ny
    dx = problem.get_dx()
    dy = problem.get_dy()

    bc = get_boundary_info(problem)

    N = Nx * Ny
    rows = []
    cols = []
    values = []
    b = np.zeros(N)

    for row in range(Ny):
        for col in range(Nx):
            k = get_flat_index(row, col, Nx)

            # ---------- x-direction second derivative ----------
            # left neighbor / left boundary
            if col == 0:
                if bc["left"]["type"] == "dirichlet":
                    # T_{-1} replaced by boundary value contribution
                    rows.append(k)
                    cols.append(k)
                    values.append(-2.0 / dx**2)

                    rows.append(k)
                    cols.append(get_flat_index(row, col + 1, Nx))
                    values.append(1.0 / dx**2)

                    b[k] += bc["left"]["value"] / dx**2

                elif bc["left"]["type"] == "neumann":
                    # Use ghost-point relation:
                    # (T_1 - 2T_0 + T_-1)/dx^2 with T_-1 = T_0 - g*dx
                    rows.append(k)
                    cols.append(k)
                    values.append(-1.0 / dx**2)

                    rows.append(k)
                    cols.append(get_flat_index(row, col + 1, Nx))
                    values.append(1.0 / dx**2)

                    b[k] += -bc["left"]["value"] / dx

                else:
                    raise ValueError("Unknown left boundary type.")

            elif col == Nx - 1:
                if bc["right"]["type"] == "dirichlet":
                    rows.append(k)
                    cols.append(get_flat_index(row, col - 1, Nx))
                    values.append(1.0 / dx**2)

                    rows.append(k)
                    cols.append(k)
                    values.append(-2.0 / dx**2)

                    b[k] += bc["right"]["value"] / dx**2

                elif bc["right"]["type"] == "neumann":
                    # T_{Nx} = T_{Nx-1} + g*dx
                    rows.append(k)
                    cols.append(get_flat_index(row, col - 1, Nx))
                    values.append(1.0 / dx**2)

                    rows.append(k)
                    cols.append(k)
                    values.append(-1.0 / dx**2)

                    b[k] += bc["right"]["value"] / dx

                else:
                    raise ValueError("Unknown right boundary type.")

            else:
                rows.append(k)
                cols.append(get_flat_index(row, col - 1, Nx))
                values.append(1.0 / dx**2)

                rows.append(k)
                cols.append(k)
                values.append(-2.0 / dx**2)

                rows.append(k)
                cols.append(get_flat_index(row, col + 1, Nx))
                values.append(1.0 / dx**2)

            # ---------- y-direction second derivative ----------
            # bottom neighbor / bottom boundary
            if row == 0:
                if bc["bottom"]["type"] == "dirichlet":
                    rows.append(k)
                    cols.append(k)
                    values.append(-2.0 / dy**2)

                    rows.append(k)
                    cols.append(get_flat_index(row + 1, col, Nx))
                    values.append(1.0 / dy**2)

                    b[k] += bc["bottom"]["value"] / dy**2

                elif bc["bottom"]["type"] == "neumann":
                    rows.append(k)
                    cols.append(k)
                    values.append(-1.0 / dy**2)

                    rows.append(k)
                    cols.append(get_flat_index(row + 1, col, Nx))
                    values.append(1.0 / dy**2)

                    b[k] += -bc["bottom"]["value"] / dy

                else:
                    raise ValueError("Unknown bottom boundary type.")

            elif row == Ny - 1:
                if bc["top"]["type"] == "dirichlet":
                    rows.append(k)
                    cols.append(get_flat_index(row - 1, col, Nx))
                    values.append(1.0 / dy**2)

                    rows.append(k)
                    cols.append(k)
                    values.append(-2.0 / dy**2)

                    b[k] += bc["top"]["value"] / dy**2

                elif bc["top"]["type"] == "neumann":
                    rows.append(k)
                    cols.append(get_flat_index(row - 1, col, Nx))
                    values.append(1.0 / dy**2)

                    rows.append(k)
                    cols.append(k)
                    values.append(-1.0 / dy**2)

                    b[k] += bc["top"]["value"] / dy

                else:
                    raise ValueError("Unknown top boundary type.")

            else:
                rows.append(k)
                cols.append(get_flat_index(row - 1, col, Nx))
                values.append(1.0 / dy**2)

                rows.append(k)
                cols.append(k)
                values.append(-2.0 / dy**2)

                rows.append(k)
                cols.append(get_flat_index(row + 1, col, Nx))
                values.append(1.0 / dy**2)

    A = sp.csr_matrix((values, (rows, cols)), shape=(N, N))
    return A, b


def build_advection_x(problem, scheme="central"):
    """
    Build the sparse matrix A and boundary vector b for the x-derivative:

        T_x

    such that approximately

        (T_x)_flat = A @ T_flat + b

    Boundary conditions are read from the problem object.
    """
    Nx = problem.Nx
    Ny = problem.Ny
    dx = problem.get_dx()

    bc = get_boundary_info(problem)
    vx_eff = problem.vx / problem.R_th

    N = Nx * Ny
    rows = []
    cols = []
    values = []
    b = np.zeros(N)

    for row in range(Ny):
        for col in range(Nx):
            k = get_flat_index(row, col, Nx)

            if scheme == "central":
                if col == 0:
                    if bc["left"]["type"] == "dirichlet":
                        rows.append(k)
                        cols.append(get_flat_index(row, col + 1, Nx))
                        values.append(1.0 / (2.0 * dx))

                        b[k] += -bc["left"]["value"] / (2.0 * dx)

                    elif bc["left"]["type"] == "neumann":
                        # For central derivative at the boundary, use ghost value
                        rows.append(k)
                        cols.append(get_flat_index(row, col + 1, Nx))
                        values.append(1.0 / (2.0 * dx))

                        rows.append(k)
                        cols.append(k)
                        values.append(-1.0 / (2.0 * dx))

                        b[k] += -bc["left"]["value"] / 2.0

                    else:
                        raise ValueError("Unknown left boundary type.")

                elif col == Nx - 1:
                    if bc["right"]["type"] == "dirichlet":
                        rows.append(k)
                        cols.append(get_flat_index(row, col - 1, Nx))
                        values.append(-1.0 / (2.0 * dx))

                        b[k] += bc["right"]["value"] / (2.0 * dx)

                    elif bc["right"]["type"] == "neumann":
                        rows.append(k)
                        cols.append(k)
                        values.append(1.0 / (2.0 * dx))

                        rows.append(k)
                        cols.append(get_flat_index(row, col - 1, Nx))
                        values.append(-1.0 / (2.0 * dx))

                        b[k] += bc["right"]["value"] / 2.0

                    else:
                        raise ValueError("Unknown right boundary type.")

                else:
                    rows.append(k)
                    cols.append(get_flat_index(row, col + 1, Nx))
                    values.append(1.0 / (2.0 * dx))

                    rows.append(k)
                    cols.append(get_flat_index(row, col - 1, Nx))
                    values.append(-1.0 / (2.0 * dx))

            elif scheme == "upwind":
                if vx_eff >= 0:
                    if col == 0:
                        if bc["left"]["type"] == "dirichlet":
                            rows.append(k)
                            cols.append(k)
                            values.append(1.0 / dx)

                            b[k] += -bc["left"]["value"] / dx

                        elif bc["left"]["type"] == "neumann":
                            # backward derivative with ghost from Neumann BC
                            rows.append(k)
                            cols.append(k)
                            values.append(0.0)

                            b[k] += bc["left"]["value"]

                        else:
                            raise ValueError("Unknown left boundary type.")
                    else:
                        rows.append(k)
                        cols.append(k)
                        values.append(1.0 / dx)

                        rows.append(k)
                        cols.append(get_flat_index(row, col - 1, Nx))
                        values.append(-1.0 / dx)

                else:
                    if col == Nx - 1:
                        if bc["right"]["type"] == "dirichlet":
                            rows.append(k)
                            cols.append(k)
                            values.append(-1.0 / dx)

                            b[k] += bc["right"]["value"] / dx

                        elif bc["right"]["type"] == "neumann":
                            rows.append(k)
                            cols.append(k)
                            values.append(0.0)

                            b[k] += bc["right"]["value"]

                        else:
                            raise ValueError("Unknown right boundary type.")
                    else:
                        rows.append(k)
                        cols.append(get_flat_index(row, col + 1, Nx))
                        values.append(1.0 / dx)

                        rows.append(k)
                        cols.append(k)
                        values.append(-1.0 / dx)

            else:
                raise ValueError("scheme must be 'central' or 'upwind'")

    A = sp.csr_matrix((values, (rows, cols)), shape=(N, N))
    return A, b


def build_advection_y(problem, scheme="central"):
    """
    Build the sparse matrix A and boundary vector b for the y-derivative:

        T_y

    such that approximately

        (T_y)_flat = A @ T_flat + b

    Boundary conditions are read from the problem object.
    """
    Nx = problem.Nx
    Ny = problem.Ny
    dy = problem.get_dy()

    bc = get_boundary_info(problem)
    vy_eff = problem.vy / problem.R_th

    N = Nx * Ny
    rows = []
    cols = []
    values = []
    b = np.zeros(N)

    for row in range(Ny):
        for col in range(Nx):
            k = get_flat_index(row, col, Nx)

            if scheme == "central":
                if row == 0:
                    if bc["bottom"]["type"] == "dirichlet":
                        rows.append(k)
                        cols.append(get_flat_index(row + 1, col, Nx))
                        values.append(1.0 / (2.0 * dy))

                        b[k] += -bc["bottom"]["value"] / (2.0 * dy)

                    elif bc["bottom"]["type"] == "neumann":
                        rows.append(k)
                        cols.append(get_flat_index(row + 1, col, Nx))
                        values.append(1.0 / (2.0 * dy))

                        rows.append(k)
                        cols.append(k)
                        values.append(-1.0 / (2.0 * dy))

                        b[k] += -bc["bottom"]["value"] / 2.0

                    else:
                        raise ValueError("Unknown bottom boundary type.")

                elif row == Ny - 1:
                    if bc["top"]["type"] == "dirichlet":
                        rows.append(k)
                        cols.append(get_flat_index(row - 1, col, Nx))
                        values.append(-1.0 / (2.0 * dy))

                        b[k] += bc["top"]["value"] / (2.0 * dy)

                    elif bc["top"]["type"] == "neumann":
                        rows.append(k)
                        cols.append(k)
                        values.append(1.0 / (2.0 * dy))

                        rows.append(k)
                        cols.append(get_flat_index(row - 1, col, Nx))
                        values.append(-1.0 / (2.0 * dy))

                        b[k] += bc["top"]["value"] / 2.0

                    else:
                        raise ValueError("Unknown top boundary type.")

                else:
                    rows.append(k)
                    cols.append(get_flat_index(row + 1, col, Nx))
                    values.append(1.0 / (2.0 * dy))

                    rows.append(k)
                    cols.append(get_flat_index(row - 1, col, Nx))
                    values.append(-1.0 / (2.0 * dy))

            elif scheme == "upwind":
                if vy_eff >= 0:
                    if row == 0:
                        if bc["bottom"]["type"] == "dirichlet":
                            rows.append(k)
                            cols.append(k)
                            values.append(1.0 / dy)

                            b[k] += -bc["bottom"]["value"] / dy

                        elif bc["bottom"]["type"] == "neumann":
                            rows.append(k)
                            cols.append(k)
                            values.append(0.0)

                            b[k] += bc["bottom"]["value"]

                        else:
                            raise ValueError("Unknown bottom boundary type.")
                    else:
                        rows.append(k)
                        cols.append(k)
                        values.append(1.0 / dy)

                        rows.append(k)
                        cols.append(get_flat_index(row - 1, col, Nx))
                        values.append(-1.0 / dy)

                else:
                    if row == Ny - 1:
                        if bc["top"]["type"] == "dirichlet":
                            rows.append(k)
                            cols.append(k)
                            values.append(-1.0 / dy)

                            b[k] += bc["top"]["value"] / dy

                        elif bc["top"]["type"] == "neumann":
                            rows.append(k)
                            cols.append(k)
                            values.append(0.0)

                            b[k] += bc["top"]["value"]

                        else:
                            raise ValueError("Unknown top boundary type.")
                    else:
                        rows.append(k)
                        cols.append(get_flat_index(row + 1, col, Nx))
                        values.append(1.0 / dy)

                        rows.append(k)
                        cols.append(k)
                        values.append(-1.0 / dy)

            else:
                raise ValueError("scheme must be 'central' or 'upwind'")

    A = sp.csr_matrix((values, (rows, cols)), shape=(N, N))
    return A, b