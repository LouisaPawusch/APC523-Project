# Define the PDE itself

import numpy as np


class HeatTransportProblem:
    """
    Class to define the heat transport problem with advection, diffusion, and source terms, 
    along with boundary conditions and initial conditions.
    """

    def __init__(self, Lx, Ly, Nx, Ny, alpha, R_th, rho_c_eff, vx=0.0, vy=0.0, source_fn=None, 
        initial_condition_fn=None, bc_left_type="dirichlet", bc_left_value=1.0, bc_right_type="neumann", 
        bc_right_value=0.0, bc_bottom_type="neumann", bc_bottom_value=0.0, bc_top_type="neumann", bc_top_value=0.0,):

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