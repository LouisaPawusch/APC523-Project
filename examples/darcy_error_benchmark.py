"""
Darcy error benchmark: analytical solution comparison.

Part 1 — Darcy head convergence (manufactured solution):
  Uses h_exact = sin(pi*x/Lx)*sin(pi*y/Ly) on a unit square with
  K=1, all-Dirichlet h=0 BCs, and the matching source term.
  Expected: 2nd-order convergence in L2 for both h and vx.

Part 2 — Heat transport L2 error vs time (three coupling strategies):
  (a) No Darcy:            vx = vy = 0      — advection ignored
  (b) Exact Darcy velocity: vx = K*DH/(n*Lx) — analytical uniform flow
  (c) Numerical Darcy:     vx from solve_darcy — full coupled solve
  All three are compared against the Gaussian analytical plume advected
  at the exact Darcy pore velocity. This shows both the physical error
  from ignoring flow and the numerical accuracy of the Darcy coupling.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.problem import HeatTransportProblem, GaussianIC
from src.darcy import DarcyParams, solve_darcy, darcy_analytical_solution
from src.simulation import run_simulation

os.makedirs("results/plots", exist_ok=True)


# ---------------------------------------------------------------------------
# Shared heat-transport parameters
# ---------------------------------------------------------------------------

LX_HEAT, LY_HEAT = 2.0, 1.0
ALPHA    = 5e-5
R_TH     = 1.0
RHO_C    = 1.0
K_UNIF   = 1e-4       # [m/s]
POROSITY = 0.3
H_LEFT   = 1.0        # [m]
H_RIGHT  = 0.0        # [m]
X0, Y0   = 0.3, 0.5  # Gaussian source location
Q        = 1.0
T_PHYS_0 = 20.0       # physics time at IC (avoids t=0 singularity)
T_SIM    = 300.0      # simulation duration [s]

# Exact pore velocity for uniform K with linear head gradient
VX_EXACT = K_UNIF * (H_LEFT - H_RIGHT) / (POROSITY * LX_HEAT)
VY_EXACT = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def l2_error(T_num, T_ref):
    return np.sqrt(np.mean((T_num - T_ref) ** 2))


def gaussian_analytical(X, Y, t_phys):
    """Analytical Gaussian plume at physics time t_phys using the exact Darcy vx."""
    ux  = VX_EXACT / R_TH
    xi  = X - X0 - ux * t_phys
    eta = Y - Y0
    return Q / (4 * np.pi * ALPHA * t_phys) * np.exp(-(xi**2 + eta**2) / (4 * ALPHA * t_phys))


def _dummy_problem(Nx, Ny, Lx=1.0, Ly=1.0):
    """Minimal HeatTransportProblem used only to supply grid info to Darcy solver."""
    return HeatTransportProblem(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        alpha=1e-3, R_th=1.0, rho_c_eff=1.0,
    )


def make_heat_problem(Nx, Ny, vx=0.0, vy=0.0):
    ic_fn = GaussianIC(X0, Y0, Q, ALPHA, T_PHYS_0, vx=VX_EXACT, vy=VY_EXACT, R_th=R_TH)
    return HeatTransportProblem(
        Lx=LX_HEAT, Ly=LY_HEAT, Nx=Nx, Ny=Ny,
        alpha=ALPHA, R_th=R_TH, rho_c_eff=RHO_C,
        vx=vx, vy=vy,
        initial_condition_fn=ic_fn,
        bc_left_type="neumann",   bc_left_value=0.0,
        bc_right_type="neumann",  bc_right_value=0.0,
        bc_bottom_type="neumann", bc_bottom_value=0.0,
        bc_top_type="neumann",    bc_top_value=0.0,
        gaussian_x0=X0, gaussian_y0=Y0, gaussian_Q=Q,
    )


def make_heat_darcy_params():
    return DarcyParams(
        K=K_UNIF, porosity=POROSITY,
        bc_head_left_type="dirichlet",   bc_head_left_value=H_LEFT,
        bc_head_right_type="dirichlet",  bc_head_right_value=H_RIGHT,
        bc_head_bottom_type="neumann",   bc_head_bottom_value=0.0,
        bc_head_top_type="neumann",      bc_head_top_value=0.0,
    )


# ---------------------------------------------------------------------------
# Part 1: Darcy head convergence — manufactured sin solution
# ---------------------------------------------------------------------------

def darcy_head_convergence():
    """
    Manufactured solution:  h_exact = sin(pi*x/Lx) * sin(pi*y/Ly)
    on [0,Lx]x[0,Ly] with K=1, all-Dirichlet h=0 BCs, and source
      f = -(pi/Lx)^2 - (pi/Ly)^2) * h_exact  = -pi^2*(1/Lx^2+1/Ly^2)*h_exact
    Expected: O(dx^2) convergence in L2.
    """
    print("=" * 60)
    print("Part 1: Darcy head convergence (manufactured solution)")
    print("=" * 60)

    Lx, Ly = 1.0, 1.0
    K_val  = 1.0
    coeff  = np.pi**2 * (1.0 / Lx**2 + 1.0 / Ly**2)

    source_fn = lambda X, Y: -K_val * coeff * np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)

    darcy_params = DarcyParams(
        K=K_val, porosity=0.3,
        bc_head_left_type="dirichlet",   bc_head_left_value=0.0,
        bc_head_right_type="dirichlet",  bc_head_right_value=0.0,
        bc_head_bottom_type="dirichlet", bc_head_bottom_value=0.0,
        bc_head_top_type="dirichlet",    bc_head_top_value=0.0,
        source_fn=source_fn,
    )

    grid_sizes = [8, 16, 32, 64, 128]
    dx_list, l2_h_list, l2_vx_list = [], [], []

    print(f"\n  {'N':>5}  {'dx':>8}  {'L2(h)':>12}  {'L2(vx)':>12}")
    print("  " + "-" * 44)

    for N in grid_sizes:
        problem = _dummy_problem(N + 1, N + 1, Lx=Lx, Ly=Ly)
        dx      = problem.get_dx()
        X, Y    = problem.get_mesh()

        h_num, vx_num, _ = solve_darcy(problem, darcy_params)

        h_exact  = np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
        # exact pore velocity vx = -K/n * dh/dx (solve_darcy returns pore velocity)
        vx_exact = (-K_val / darcy_params.porosity) * (np.pi / Lx) * np.cos(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)

        err_h  = l2_error(h_num, h_exact)
        err_vx = l2_error(vx_num, vx_exact)

        dx_list.append(dx)
        l2_h_list.append(err_h)
        l2_vx_list.append(err_vx)
        print(f"  {N:>5}  {dx:>8.5f}  {err_h:>12.4e}  {err_vx:>12.4e}")

    dx_arr = np.array(dx_list)
    h_arr  = np.array(l2_h_list)
    vx_arr = np.array(l2_vx_list)

    # Convergence order from log-log slope (exclude last point, near machine eps)
    p_h  = np.polyfit(np.log(dx_arr[:-1]), np.log(h_arr[:-1]),  1)[0]
    p_vx = np.polyfit(np.log(dx_arr[:-1]), np.log(vx_arr[:-1]), 1)[0]
    print(f"\n  Convergence order:  h -> {p_h:.2f},  vx -> {p_vx:.2f}  (expected ~2)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(dx_arr, h_arr,  "o-",  label=f"L2(h),  slope={p_h:.2f}")
    ax.loglog(dx_arr, vx_arr, "s--", label=f"L2(vx), slope={p_vx:.2f}")
    ref = h_arr[0] * (dx_arr / dx_arr[0]) ** 2
    ax.loglog(dx_arr, ref, "k:", lw=1, label="O(dx^2) reference")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("L2 error")
    ax.set_title("Darcy FD convergence (manufactured sin solution)")
    ax.legend()
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    path = "results/plots/darcy_head_convergence.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Part 2: Heat transport L2 error vs time — with/without Darcy coupling
# ---------------------------------------------------------------------------

def heat_error_comparison():
    print("\n" + "=" * 60)
    print("Part 2: Heat L2 error vs time — Darcy coupling comparison")
    print("=" * 60)

    NX, NY   = 60, 30
    DT       = 20.0

    darcy_params = make_heat_darcy_params()

    cases = [
        ("Prescribed $v_x$",  make_heat_problem(NX, NY, vx=VX_EXACT, vy=0.0), None),
        ("Numerical Darcy",   make_heat_problem(NX, NY, vx=0.0,     vy=0.0), darcy_params),
    ]

    error_results = {}   # label -> (times_phys, errors)
    final_fields  = {}   # label -> T_final array

    for label, problem, dp in cases:
        print(f"\n  Running: {label}  (VX_EXACT = {VX_EXACT:.4e} m/s)")
        times_sim, all_T = run_simulation(
            problem, T_SIM, DT,
            save_every=1,
            advection_scheme="central",
            timestepper="CN",
            linear_solver="direct",
            darcy_params=dp,
            verbose=False,
        )

        X, Y       = problem.get_mesh()
        times_phys = times_sim + T_PHYS_0
        errors     = [l2_error(T, gaussian_analytical(X, Y, tp))
                      for T, tp in zip(all_T, times_phys)]

        error_results[label] = (times_phys, errors)
        final_fields[label]  = all_T[-1]
        print(f"    Initial L2 = {errors[0]:.4e},  Final L2 = {errors[-1]:.4e}")

    # --- Plot 1: L2 error vs physics time ---
    fig, ax = plt.subplots(figsize=(7, 4))
    styles = {
        "Prescribed $v_x$": ("tab:green", "-",  2.0),
        "Numerical Darcy":  ("tab:blue",  "-.", 2.0),
    }
    for label, (t_arr, err_arr) in error_results.items():
        col, ls, lw = styles[label]
        ax.semilogy(t_arr, err_arr, color=col, ls=ls, lw=lw, label=label)

    ax.set_xlabel("Physics time [s]")
    ax.set_ylabel("L2 error vs analytical Gaussian")
    ax.set_title("Heat transport error: prescribed $v_x$ vs numerical Darcy (CN, uniform K)")
    ax.legend()
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    path = "results/plots/darcy_heat_error_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved: {path}")

    # --- Plot 2: final temperature fields side by side ---
    X_ref, Y_ref = cases[0][1].get_mesh()
    T_ana_final  = gaussian_analytical(X_ref, Y_ref, T_PHYS_0 + T_SIM)
    final_fields["Analytical"] = T_ana_final

    n     = len(final_fields)
    vmax  = np.percentile(T_ana_final, 99.5)
    fig2, axes = plt.subplots(1, n, figsize=(5 * n, 3.5), constrained_layout=True)
    for ax2, (label, T) in zip(axes, final_fields.items()):
        im = ax2.pcolormesh(X_ref, Y_ref, T, shading="auto",
                            cmap="inferno", vmin=0, vmax=vmax)
        ax2.set_title(label, fontsize=9)
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("y [m]")
        plt.colorbar(im, ax=ax2)
    fig2.suptitle(f"Final temperature field (t = {T_PHYS_0 + T_SIM:.0f} s)", fontweight="bold")
    path2 = "results/plots/darcy_heat_error_final_fields.png"
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"  Saved: {path2}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    darcy_head_convergence()
    heat_error_comparison()
    print("\nDone. Results in results/plots/")


if __name__ == "__main__":
    main()
