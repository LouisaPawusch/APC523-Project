import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.problem import HeatTransportProblem, GaussianIC
from src.simulation import run_simulation
from src.plotting import plot_gaussian_cuts, plot_initial_and_final

os.makedirs("results/plots", exist_ok=True)


def main():
    # --- Benchmark parameters ---
    Lx, Ly  = 2.0, 2.0
    Nx, Ny  = 101, 101
    alpha   = 1e-3
    vx, vy  = 0.05, 0.0
    R_th    = 1.0

    x0, y0  = 0.5, 1.0     # point source location
    Q       = 1.0

    t_start = 5.0           # physics time of IC; avoids the t=0 singularity
    t_end   = 15.0          # physics time at which we compare
    # FE stable timestep: dt < h^2 / (4*alpha) = 0.02^2 / (4*1e-3) = 0.1
    dt      = 0.05

    # Build IC as Gaussian evaluated at t_start (outside constructor to avoid forward reference)
    ic_fn = GaussianIC(x0, y0, Q, alpha, t_start, vx=vx, vy=vy, R_th=R_th)

    problem = HeatTransportProblem(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, alpha=alpha, R_th=R_th, rho_c_eff=1.0,
        vx=vx, vy=vy, initial_condition_fn=ic_fn, 
        bc_left_type="neumann",   bc_left_value=0.0, bc_right_type="neumann",  bc_right_value=0.0,
        bc_bottom_type="neumann", bc_bottom_value=0.0, bc_top_type="neumann",    bc_top_value=0.0,
        gaussian_x0=x0, gaussian_y0=y0, gaussian_Q=Q)

    t_sim = t_end - t_start     # simulation duration (physics elapsed time)

    print("Running Forward Euler...")
    times_FE, all_T_FE = run_simulation( problem, t_final=t_sim, dt=dt, save_every=1, advection_scheme="central", timestepper="FE")

    print("Running RK4...")
    times_RK4, all_T_RK4 = run_simulation( problem, t_final=t_sim, dt=dt, save_every=1, advection_scheme="central", timestepper="RK4")

    X, Y = problem.get_mesh()
    T_analytical = problem.analytical_solution(X, Y, t_end)
    T_FE = all_T_FE[-1]
    T_RK4 = all_T_RK4[-1]

    print(f"\nMax |T_analytical|:  {np.max(np.abs(T_analytical)):.6f}")
    print(f"Max |error FE|:      {np.max(np.abs(T_FE  - T_analytical)):.6f}")
    print(f"Max |error RK4|:     {np.max(np.abs(T_RK4 - T_analytical)):.6f}")

    plot_initial_and_final(problem, times_FE, all_T_FE, save_path="results/plots/gaussian_FE_init_and_final.png", cmap="inferno")
    plot_initial_and_final(problem, times_RK4, all_T_RK4, save_path="results/plots/gaussian_RK4_init_and_final.png", cmap="inferno")    

    plot_gaussian_cuts(problem, T_analytical, T_FE, t_end, numerical_label="FE", save_path="results/plots/gaussian_FE_cuts.png")
    plot_gaussian_cuts(problem, T_analytical, T_RK4, t_end, numerical_label="RK4", save_path="results/plots/gaussian_RK4_cuts.png")


if __name__ == "__main__":
    main()
