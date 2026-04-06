import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.problem import HeatTransportProblem
from src.simulation import run_simulation
from src.plotting import plot_initial_and_final

os.makedirs("results/plots", exist_ok=True)

def initial_temperature(X, Y):
    return np.zeros_like(X)

def main():
    problem = HeatTransportProblem(Lx=1.0, Ly=1.0, Nx=51, Ny=51, alpha=1e-3, R_th=2.0, rho_c_eff=1.0, vx=0.1, vy=0.0, 
                                   initial_condition_fn=initial_temperature, bc_left_type="dirichlet", bc_left_value=1.0, 
                                   bc_right_type="neumann", bc_right_value=0.0, bc_bottom_type="neumann", bc_bottom_value=0.0, 
                                   bc_top_type="neumann", bc_top_value=0.0)

    times, all_T = run_simulation(problem, t_final=10.0, dt=1e-3, save_every=1, advection_scheme="central", timestepper="RK4")

    plot_initial_and_final(problem, times, all_T, save_path="results/plots/basic_simulation_1_init_and_final.png", cmap="inferno")

    problem = HeatTransportProblem(Lx=1.0, Ly=1.0, Nx=51, Ny=51, alpha=1e-3, R_th=2.0, rho_c_eff=1.0, vx=0.1, vy=0.0, 
                                   initial_condition_fn=initial_temperature, bc_left_type="dirichlet", bc_left_value=1.0, 
                                   bc_right_type="neumann", bc_right_value=0.0, bc_bottom_type="neumann", bc_bottom_value=0.0, 
                                   bc_top_type="dirichlet", bc_top_value=1.0)

    times, all_T = run_simulation(problem, t_final=10.0, dt=1e-3, save_every=1, advection_scheme="central", timestepper="RK4")

    plot_initial_and_final(problem, times, all_T, save_path="results/plots/basic_simulation_2_init_and_final.png", cmap="inferno")

if __name__ == "__main__":
    main()