"""
Darcy-coupled heat transport benchmark.

Physics
-------
Groundwater flows through a heterogeneous aquifer driven by a head gradient
(high head on the left, low on the right).  A Gaussian heat plume is
released near the inlet and is advected and diffused by the computed flow
field.

The key demo: with a high-K channel in the centre of the domain, the
pore velocity is much higher inside the channel than outside.  This
makes the CFL stability limit for explicit methods very restrictive
(tiny dt needed everywhere).  Implicit methods (BE, CN) handle large dt
without stability issues.

Three scenarios are shown:
  1. Uniform K          — baseline, uniform flow
  2. Layered K          — high-K central channel, slow top/bottom layers
  3. Random K field     — heterogeneous permeability, complex streamlines
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.problem import HeatTransportProblem, GaussianIC
from src.darcy import DarcyParams, solve_darcy, make_layered_K, make_random_K
from src.simulation import run_simulation

os.makedirs("results/plots", exist_ok=True)


# ---------------------------------------------------------------------------
# Shared grid and problem parameters
# ---------------------------------------------------------------------------

LX, LY  = 2.0, 1.0
NX, NY  = 60, 30          # coarse enough to run quickly
ALPHA   = 5e-5            # thermal diffusivity [m²/s]
R_TH    = 1.0
RHO_C   = 1.0
T_SIM   = 500.0           # total simulation time [s]
T_START = 20.0            # time at which IC Gaussian is evaluated
X0, Y0  = 0.2, 0.5       # heat source location
Q       = 1.0             # total heat released [J/m]

# Head BCs: prescribed head drives flow left → right
H_LEFT  = 1.0             # [m]
H_RIGHT = 0.0             # [m]
POROSITY = 0.3


def make_base_problem(vx=0.0, vy=0.0):
    """Return a HeatTransportProblem with placeholder zero velocity."""
    ic_fn = GaussianIC(X0, Y0, Q, ALPHA, T_START, vx=vx, vy=vy, R_th=R_TH)
    return HeatTransportProblem(
        Lx=LX, Ly=LY, Nx=NX, Ny=NY,
        alpha=ALPHA, R_th=R_TH, rho_c_eff=RHO_C,
        vx=vx, vy=vy,
        initial_condition_fn=ic_fn,
        bc_left_type="neumann",   bc_left_value=0.0,
        bc_right_type="neumann",  bc_right_value=0.0,
        bc_bottom_type="neumann", bc_bottom_value=0.0,
        bc_top_type="neumann",    bc_top_value=0.0,
        gaussian_x0=X0, gaussian_y0=Y0, gaussian_Q=Q,
    )


def make_darcy_params(K):
    """DarcyParams with head gradient driving flow left → right, no-flow top/bottom."""
    return DarcyParams(
        K=K, porosity=POROSITY,
        bc_head_left_type="dirichlet",   bc_head_left_value=H_LEFT,
        bc_head_right_type="dirichlet",  bc_head_right_value=H_RIGHT,
        bc_head_bottom_type="neumann",   bc_head_bottom_value=0.0,
        bc_head_top_type="neumann",      bc_head_top_value=0.0,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_darcy_field(problem, h, vx, vy, K, title, save_path):
    """Plot hydraulic head, K field, and velocity streamlines."""
    X, Y = problem.get_mesh()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)

    # Left: K field + head contours
    ax = axes[0]
    if np.ndim(K) == 0:
        K_plot = np.full((NY, NX), float(K))
    else:
        K_plot = K
    im = ax.pcolormesh(X, Y, np.log10(K_plot), shading="auto", cmap="YlOrRd")
    ax.contour(X, Y, h, levels=10, colors="black", linewidths=0.7)
    plt.colorbar(im, ax=ax, label="log₁₀(K)  [m/s]")
    ax.set_title("Hydraulic conductivity + head contours")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    # Right: velocity magnitude + streamlines
    ax = axes[1]
    speed = np.sqrt(vx**2 + vy**2)
    im2 = ax.pcolormesh(X, Y, speed, shading="auto", cmap="Blues")
    ax.streamplot(X[0, :], Y[:, 0], vx, vy, density=1.2,
                  color="black", linewidth=0.6, arrowsize=0.8)
    plt.colorbar(im2, ax=ax, label="|v|  [m/s]")
    ax.set_title("Pore velocity magnitude + streamlines")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_heat_snapshots(problem, times, all_T, title, save_path):
    """Plot heat plume at three snapshots."""
    X, Y = problem.get_mesh()
    n_snaps = min(3, len(all_T))
    indices = np.linspace(0, len(all_T) - 1, n_snaps, dtype=int)

    T_all = np.concatenate([t.ravel() for t in all_T])
    vmax  = np.percentile(T_all, 99)

    fig, axes = plt.subplots(1, n_snaps, figsize=(5 * n_snaps, 4),
                             constrained_layout=True)
    if n_snaps == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        im = ax.pcolormesh(X, Y, all_T[idx], shading="auto",
                           cmap="inferno", vmin=0, vmax=vmax)
        ax.set_title(f"t = {times[idx]:.0f} s")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        plt.colorbar(im, ax=ax, label="T")

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison(problem, results_dict, t_final, save_path):
    """
    Side-by-side final temperature fields for multiple (label, T) pairs.
    Used to compare methods or dt choices on the same K field.
    """
    X, Y    = problem.get_mesh()
    labels  = list(results_dict.keys())
    T_list  = list(results_dict.values())
    n       = len(labels)

    all_vals = np.concatenate([T.ravel() for T in T_list if np.isfinite(T).all()])
    vmax     = np.percentile(all_vals, 99) if len(all_vals) > 0 else 1.0

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, label, T in zip(axes, labels, T_list):
        if not np.isfinite(T).all():
            ax.text(0.5, 0.5, "DIVERGED", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, color="red")
        else:
            im = ax.pcolormesh(X, Y, T, shading="auto",
                               cmap="inferno", vmin=0, vmax=vmax)
            plt.colorbar(im, ax=ax, label="T")
        ax.set_title(label)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    fig.suptitle(f"Final temperature field (t = {t_final:.0f} s)", fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(label, K, dt_implicit, dt_explicit=None, timestepper_implicit="CN"):
    """
    Run a full Darcy + heat scenario and plot results.

    1. Solve Darcy → velocity field
    2. Run implicit method (CN or BE) at dt_implicit
    3. If dt_explicit given, also run FE at dt_explicit and compare
    """
    print(f"\n{'='*60}")
    print(f"Scenario: {label}")
    print(f"{'='*60}")

    darcy_params = make_darcy_params(K)
    problem      = make_base_problem()   # vx=0 placeholder, will be overwritten

    # --- Darcy solve (standalone, for plotting)
    h, vx, vy = solve_darcy(problem, darcy_params)
    v_max    = max(np.abs(vx).max(), np.abs(vy).max())
    dx_min   = min(problem.get_dx(), problem.get_dy())
    cfl_lim  = dx_min / v_max if v_max > 0 else float("inf")
    print(f"  max pore velocity = {v_max:.4e} m/s")
    print(f"  explicit CFL limit ≈ {cfl_lim:.4e} s")

    plot_darcy_field(problem, h, vx, vy, K,
                     title=f"Darcy flow — {label}",
                     save_path=f"results/plots/darcy_{label.replace(' ','_')}_flow.png")

    # --- Implicit run at large dt
    print(f"\n  Running {timestepper_implicit} at dt={dt_implicit:.2f} s ...")
    t0 = time.perf_counter()
    problem_impl = make_base_problem()
    times_impl, all_T_impl = run_simulation(
        problem_impl, T_SIM, dt_implicit,
        save_every=max(1, int(T_SIM / dt_implicit / 3)),
        advection_scheme="central",
        timestepper=timestepper_implicit,
        linear_solver="direct",
        darcy_params=darcy_params,
    )
    elapsed_impl = time.perf_counter() - t0
    print(f"  Done in {elapsed_impl:.1f} s  ({len(times_impl)-1} saved snapshots)")

    plot_heat_snapshots(problem_impl, times_impl + T_START, all_T_impl,
                        title=f"{label} — {timestepper_implicit}, dt={dt_implicit} s",
                        save_path=f"results/plots/darcy_{label.replace(' ','_')}_heat_{timestepper_implicit}.png")

    # --- Stability comparison: FE at different dt
    if dt_explicit is not None:
        results = {f"{timestepper_implicit} dt={dt_implicit}": all_T_impl[-1]}

        for ts, dt in [("FE", cfl_lim * 0.8), ("FE", dt_implicit)]:
            label_run = f"{ts} dt={dt:.2e}"
            print(f"\n  Running {label_run} ...")
            problem_expl = make_base_problem()
            try:
                _, all_T_expl = run_simulation(
                    problem_expl, T_SIM, dt,
                    save_every=max(1, int(T_SIM / dt)),
                    advection_scheme="central",
                    timestepper=ts,
                    darcy_params=darcy_params,
                )
                T_fin = all_T_expl[-1]
                if not np.isfinite(T_fin).all():
                    print(f"  → DIVERGED")
                    T_fin = np.full_like(T_fin, np.nan)
                else:
                    print(f"  → OK  (T_max={T_fin.max():.4f})")
            except Exception as e:
                print(f"  → ERROR: {e}")
                T_fin = np.full((NY, NX), np.nan)
            results[label_run] = T_fin

        plot_comparison(problem, results, T_SIM,
                        save_path=f"results/plots/darcy_{label.replace(' ','_')}_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Uniform K — baseline
    K_uniform = 1e-4   # [m/s]
    run_scenario("uniform K",
                 K=K_uniform,
                 dt_implicit=50.0,
                 dt_explicit=True)

    # 2. Layered K: slow–fast–slow (high-K channel in the centre)
    K_layered = make_layered_K(NY, NX,
                               K_values=[1e-6, 1e-3, 1e-6],
                               layer_fractions=[0.3, 0.4, 0.3])
    run_scenario("layered K (high-K channel)",
                 K=K_layered,
                 dt_implicit=50.0,
                 dt_explicit=True)

    # 3. Random log-normal K field
    K_random = make_random_K(NY, NX, K_mean=1e-4, K_std=5e-5, log_normal=True, seed=42)
    run_scenario("random K field",
                 K=K_random,
                 dt_implicit=50.0,
                 dt_explicit=True)

    print("\nAll scenarios complete. Results in results/plots/")


if __name__ == "__main__":
    main()
