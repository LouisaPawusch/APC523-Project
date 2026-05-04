# Define the PDE itself

import numpy as np

from .operators import flatten_field, unflatten_field



class HeatTransportProblem:
    """
    Class to define the heat transport problem with advection, diffusion, and source terms, 
    along with boundary conditions and initial conditions.
    """

    def __init__(self, Lx, Ly, Nx, Ny, alpha, R_th, rho_c_eff, vx=0.0, vy=0.0, source_fn=None,
        initial_condition_fn=None, bc_left_type="dirichlet", bc_left_value=1.0, bc_right_type="neumann",
        bc_right_value=0.0, bc_bottom_type="neumann", bc_bottom_value=0.0, bc_top_type="neumann", bc_top_value=0.0,
        gaussian_x0=0.0, gaussian_y0=0.0, gaussian_Q=1.0):

        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny

        self.alpha = alpha
        self.R_th = R_th
        self.rho_c_eff = rho_c_eff

        self.vx = vx
        self.vy = vy

        self.source_fn = source_fn
        self.initial_condition_fn = initial_condition_fn

        self.bc_left_type = bc_left_type
        self.bc_left_value = bc_left_value

        self.bc_right_type = bc_right_type
        self.bc_right_value = bc_right_value

        self.bc_bottom_type = bc_bottom_type
        self.bc_bottom_value = bc_bottom_value

        self.bc_top_type = bc_top_type
        self.bc_top_value = bc_top_value

        self.gaussian_x0 = gaussian_x0
        self.gaussian_y0 = gaussian_y0
        self.gaussian_Q = gaussian_Q

        if self.Nx < 3 or self.Ny < 3:
            raise ValueError("Nx and Ny must be at least 3.")

        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")

        if self.R_th <= 0:
            raise ValueError("R_th must be positive.")

        if self.rho_c_eff <= 0:
            raise ValueError("rho_c_eff must be positive.")

    def get_dx(self):
        """
        Get the grid spacing in the x and y directions based on the domain size and number of grid points.
        """
        return self.Lx / (self.Nx - 1)

    def get_dy(self):
        """
        Get the grid spacing in the y direction based on the domain size and number of grid points.
        """
        return self.Ly / (self.Ny - 1)

    def get_mesh(self):
        """
        Get the mesh grid for the problem based on the domain size and number of grid points.
        """
        x = np.linspace(0.0, self.Lx, self.Nx)
        y = np.linspace(0.0, self.Ly, self.Ny)
        X, Y = np.meshgrid(x, y, indexing="xy")
        return X, Y

    def get_initial_condition(self):
        """
        Get the initial condition for the problem based on the provided initial condition function.
        """
        X, Y = self.get_mesh()
        if self.initial_condition_fn is None:
            return np.zeros((self.Ny, self.Nx))

        return self.initial_condition_fn(X, Y)

    def get_source(self, t):
        """
        Get the source term for the problem based on the provided source function.
        """
        X, Y = self.get_mesh()

        if self.source_fn is None:
            return np.zeros((self.Ny, self.Nx))

        return self.source_fn(X, Y, t) / self.rho_c_eff
    

    def compute_rhs(self, T, operators, t):
        """
        Core function to compute the right-hand side of the PDE at time t, given the current temperature field T and the pre-built operators.
        This is where the actual PDE is evaluated.
        """

        alpha = self.alpha

        # vx and vy may be scalars (uniform flow) or 2D arrays (e.g. from Darcy).
        # Flatten to 1D for element-wise multiplication with the operator outputs.
        vx_eff = self.vx / self.R_th
        vy_eff = self.vy / self.R_th
        if isinstance(vx_eff, np.ndarray):
            vx_eff = vx_eff.ravel()
        if isinstance(vy_eff, np.ndarray):
            vy_eff = vy_eff.ravel()

        T_flat = flatten_field(T)
        source_flat = flatten_field(self.get_source(t))

        L = operators["L"]
        b_L = operators["b_L"]

        Dx = operators["Dx"]
        b_x = operators["b_x"]

        Dy = operators["Dy"]
        b_y = operators["b_y"]

        laplace_flat = L @ T_flat + b_L
        T_x_flat = Dx @ T_flat + b_x
        T_y_flat = Dy @ T_flat + b_y

        # Element-wise multiply handles both scalar and array velocity
        rhs_flat = alpha * laplace_flat - vx_eff * T_x_flat - vy_eff * T_y_flat + source_flat
        rhs = unflatten_field(rhs_flat, self.Ny, self.Nx)

        return rhs
    
    def analytical_solution(self, X, Y, t):
        """
        Gaussian benchmark solution for a point source of heat Q released at (gaussian_x0, gaussian_y0) at t=0.
        """
        if t <= 0:
            raise ValueError("Gaussian analytical solution is singular at t=0; provide t > 0.")

        ux = self.vx / self.R_th
        uy = self.vy / self.R_th

        xi = X - self.gaussian_x0 - ux * t
        eta = Y - self.gaussian_y0 - uy * t
        r2 = xi**2 + eta**2

        return self.gaussian_Q / (4 * np.pi * self.alpha * t) * np.exp(-r2 / (4 * self.alpha * t))
    
class GaussianIC:
    """
    Callable initial condition for a Gaussian point source of heat Q released at (x0, y0) at t=0,
    evaluated at time t0. Accounts for advection: the plume center at t0 is (x0 + ux*t0, y0 + uy*t0).
    Pass an instance as initial_condition_fn when constructing HeatTransportProblem.
    """

    def __init__(self, x0, y0, Q, alpha, t0, vx=0.0, vy=0.0, R_th=1.0):
        self.cx    = x0 + (vx / R_th) * t0   # advected center x at t0
        self.cy    = y0 + (vy / R_th) * t0   # advected center y at t0
        self.Q     = Q
        self.alpha = alpha
        self.t0    = t0

    def __call__(self, X, Y):
        r2 = (X - self.cx)**2 + (Y - self.cy)**2
        return self.Q / (4 * np.pi * self.alpha * self.t0) * np.exp(-r2 / (4 * self.alpha * self.t0))