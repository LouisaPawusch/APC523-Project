"""
Convergence experiments for the 2D heat transport PDE.

Two sweeps, each run for all five time-steppers × two advection schemes:

  1. dt sweep   — error vs. time-step size at a fixed representative grid.
  2. Nx sweep   — error vs. grid size at a fixed representative dt.

Each sweep produces a 5×2 figure (rows = methods, columns = advection schemes)
with L2 error on a log-log axis and a reference slope line.
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.problem import GaussianIC, HeatTransportProblem
from src.simulation import run_simulation

os.makedirs("results/plots", exist_ok=True)


# ---------------------------------------------------------------------------
# Shared problem parameters  (matches gaussian_benchmark.py)
# ---------------------------------------------------------------------------
Lx, Ly  = 2.0, 2.0
alpha   = 1e-3
vx, vy  = 0.05, 0.0
R_th    = 1.0
x0, y0  = 0.5, 1.0
Q       = 1.0
t_start = 5.0
t_end   = 15.0
t_sim   = t_end - t_start

METHODS = ["FE", "RK4", "BE", "CN", "IM"]
SCHEMES = ["central", "upwind"]

METHOD_LABELS = {
    "FE":  "Forward Euler",
    "RK4": "Runge-Kutta 4",
    "BE":  "Backward Euler",
    "CN":  "Crank-Nicolson",
    "IM":  "Implicit Midpoint",
}

# Convergence order of each method
METHOD_ORDER = {"FE": 1, "RK4": 4, "BE": 1, "CN": 2, "IM": 2}


def make_problem(Nx, Ny=None):
    if Ny is None:
        Ny = Nx
    ic_fct = GaussianIC(x0, y0, Q, alpha, t_start, vx=vx, vy=vy, R_th=R_th)
    return HeatTransportProblem(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        alpha=alpha, R_th=R_th, rho_c_eff=1.0,
        vx=vx, vy=vy, initial_condition_fn=ic_fct,
        bc_left_type="neumann",   bc_left_value=0.0,
        bc_right_type="neumann",  bc_right_value=0.0,
        bc_bottom_type="neumann", bc_bottom_value=0.0,
        bc_top_type="neumann",    bc_top_value=0.0,
        gaussian_x0=x0, gaussian_y0=y0, gaussian_Q=Q)


def run_one(problem, dt, timestepper, scheme):
    """
    Run one simulation and return (T_final, elapsed_seconds).
    Returns (None, nan) when the run diverges or raises an exception.
    """
    n_steps = max(1, int(t_sim / dt))
    t0 = time.perf_counter()
    try:
        _, all_T = run_simulation(
            problem, t_final=t_sim, dt=dt,
            save_every=n_steps,
            advection_scheme=scheme,
            timestepper=timestepper,
            linear_solver="direct",
            verbose=False)
        sim_time = time.perf_counter() - t0
        T_final = all_T[-1]
        if not np.isfinite(T_final).all():
            return None, float("nan")
        return T_final, sim_time
    except Exception:
        return None, float("nan")


def l2_error(T_num, T_ana):
    return np.sqrt(np.mean((T_num - T_ana) ** 2))

def linf_error(T_num, T_ana):
    return np.max(np.abs(T_num - T_ana))

# ---------------------------------------------------------------------------
# dt convergence sweep
# ---------------------------------------------------------------------------
# Representative grid: dx = 2/70 ≈ 0.029, FE diffusion stability limit ≈ 0.20.
# DT_LIST spans well below and above that limit so explicit divergence is visible.
REP_NX  = 71
DT_LIST = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

def dt_convergence():
    print("=" * 65)
    print(f"dt convergence sweep  (Nx = Ny = {REP_NX})")
    print("=" * 65)

    problem = make_problem(REP_NX)
    X, Y  = problem.get_mesh()
    T_ref = problem.analytical_solution(X, Y, t_end)

    # results[method][scheme] = list of (dt, l2, linf)  (nan = diverged)
    results = {m: {s: [] for s in SCHEMES} for m in METHODS}

    for timestepper in METHODS:
        for scheme in SCHEMES:
            print(f"  {timestepper:4s}  {scheme:8s}", end="  ", flush=True)
            for dt in DT_LIST:
                T_num, runtime = run_one(problem, dt, timestepper, scheme)
                if T_num is None:
                    results[timestepper][scheme].append((dt, np.nan, np.nan, np.nan))
                    print("D", end="", flush=True)
                else:
                    results[timestepper][scheme].append(
                        (dt, l2_error(T_num, T_ref), linf_error(T_num, T_ref), runtime))
                    print(".", end="", flush=True)
            print()

    return results


# ---------------------------------------------------------------------------
# Nx convergence sweep
# ---------------------------------------------------------------------------
# Representative dt = 0.04 is below the FE diffusion stability limit for all
# grids up to Nx = 101  (limit at Nx=101: dx=0.02 → dt_max = 0.1 > 0.04).
REP_DT  = 0.04
NX_LIST = [11, 21, 31, 41, 51, 71, 101]


def nx_convergence():
    print("\n" + "=" * 65)
    print(f"Nx convergence sweep  (dt = {REP_DT:.3f})")
    print("=" * 65)

    # results[method][scheme] = list of (Nx, l2, linf)
    results = {m: {s: [] for s in SCHEMES} for m in METHODS}

    for timestepper in METHODS:
        for scheme in SCHEMES:
            print(f"  {timestepper:4s}  {scheme:8s}", end="  ", flush=True)
            for Nx in NX_LIST:
                problem = make_problem(Nx)
                X, Y  = problem.get_mesh()
                T_ref = problem.analytical_solution(X, Y, t_end)
                T_num, runtime = run_one(problem, REP_DT, timestepper, scheme)
                if T_num is None:
                    results[timestepper][scheme].append((Nx, np.nan, np.nan, np.nan))
                    print("D", end="", flush=True)
                else:
                    results[timestepper][scheme].append(
                        (Nx, l2_error(T_num, T_ref), linf_error(T_num, T_ref), runtime))
                    print(".", end="", flush=True)
            print()

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _ref_slope_dt(ax, xs, ys, order):
    """Draw a dashed reference line of slope `order` on a dt-error log-log plot."""
    valid = [(x, y) for x, y in zip(xs, ys) if np.isfinite(y)]
    if len(valid) < 2:
        return
    x0r, y0r = valid[0]
    xa = np.array([v[0] for v in valid])
    ax.plot(xa, y0r * (xa / x0r) ** order, "k--", linewidth=0.8,
            label=f"$O(\\Delta t^{{{order}}})$")


def _ref_slope_nx(ax, xs, ys, order):
    """Draw a dashed reference line of slope -`order` on an Nx-error log-log plot."""
    valid = [(x, y) for x, y in zip(xs, ys) if np.isfinite(y)]
    if len(valid) < 2:
        return
    x0r, y0r = valid[0]
    xa = np.array([v[0] for v in valid])
    ax.plot(xa, y0r * (xa / x0r) ** (-order), "k--", linewidth=0.8,
            label=f"$O(h^{{{order}}})$")

# ---------------------------------------------------------------------------
# dt convergence plot  (5 × 2 subplots)
# ---------------------------------------------------------------------------

def plot_dt_convergence(results, save_path="results/plots/convergence_dt.png"):
    fig, axes = plt.subplots(5, 2, figsize=(12, 18), constrained_layout=True)
    fig.suptitle(f"Error vs. time-step size  (Nx = Ny = {REP_NX})", fontsize=13)

    for row, timestepper in enumerate(METHODS):
        for col, scheme in enumerate(SCHEMES):
            ax = axes[row][col]
            data     = results[timestepper][scheme]
            dts      = [d[0] for d in data]
            l2s      = [d[1] for d in data]
            linfs    = [d[2] for d in data]

            converged = [np.isfinite(e) for e in l2s]
            dts_ok   = [dts[i]   for i in range(len(dts)) if converged[i]]
            l2s_ok   = [l2s[i]   for i in range(len(dts)) if converged[i]]
            linfs_ok = [linfs[i] for i in range(len(dts)) if converged[i]]
            dts_div  = [dts[i]   for i in range(len(dts)) if not converged[i]]

            if dts_ok:
                ax.loglog(dts_ok, l2s_ok,   "o-", label="$L_2$ error")
                ax.loglog(dts_ok, linfs_ok, "s--", label="$L_\\infty$ error")
                _ref_slope_dt(ax, dts_ok, l2s_ok, METHOD_ORDER[timestepper])

            if dts_div:
                ax.axvline(min(dts_div), color="red", linestyle=":",
                           linewidth=1.2, label=f"diverged (dt ≥ {min(dts_div)})")

            ax.set_title(f"{METHOD_LABELS[timestepper]}  |  {scheme}", fontsize=9)
            ax.set_xlabel("$\\Delta t$")
            ax.set_ylabel("error")
            ax.legend(fontsize=7)
            ax.grid(True, which="both", alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    # plt.show()


# ---------------------------------------------------------------------------
# Nx convergence plot  (5 × 2 subplots)
# ---------------------------------------------------------------------------

def plot_nx_convergence(results, save_path="results/plots/convergence_nx.png"):
    fig, axes = plt.subplots(5, 2, figsize=(12, 18), constrained_layout=True)
    fig.suptitle(f"Error vs. grid size  (dt = {REP_DT:.3f})", fontsize=13)

    for row, timestepper in enumerate(METHODS):
        for col, scheme in enumerate(SCHEMES):
            ax = axes[row][col]
            data     = results[timestepper][scheme]
            nxs      = [d[0] for d in data]
            l2s      = [d[1] for d in data]
            linfs    = [d[2] for d in data]

            converged = [np.isfinite(e) for e in l2s]
            nxs_ok   = [nxs[i]   for i in range(len(nxs)) if converged[i]]
            l2s_ok   = [l2s[i]   for i in range(len(nxs)) if converged[i]]
            linfs_ok = [linfs[i] for i in range(len(nxs)) if converged[i]]
            nxs_div  = [nxs[i]   for i in range(len(nxs)) if not converged[i]]

            if nxs_ok:
                ax.loglog(nxs_ok, l2s_ok,   "o-",  label="$L_2$ error")
                ax.loglog(nxs_ok, linfs_ok, "s--", label="$L_\\infty$ error")
                spatial_order = 1 if scheme == "upwind" else 2
                _ref_slope_nx(ax, nxs_ok, l2s_ok, spatial_order)

            if nxs_div:
                for nx in nxs_div:
                    ax.axvline(nx, color="red", linestyle=":", linewidth=1.2)
                ax.plot([], [], color="red", linestyle=":", linewidth=1.2,
                        label="diverged")

            ax.set_title(f"{METHOD_LABELS[timestepper]}  |  {scheme}", fontsize=9)
            ax.set_xlabel("$N_x = N_y$")
            ax.set_ylabel("error")
            ax.legend(fontsize=7)
            ax.grid(True, which="both", alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    #plt.show()


# ---------------------------------------------------------------------------
# Runtime plots  (1 × 2 subplots: one per advection scheme, all methods overlaid)
# ---------------------------------------------------------------------------

METHOD_COLORS = {"FE": "C0", "RK4": "C1", "BE": "C2", "CN": "C3", "IM": "C4"}
METHODS_LINESTYLES = {"FE": "solid", "RK4": "dashed", "BE": "solid", "CN": "dotted", "IM": "dashed"}
METHODS_MARKERS = {"FE": "o", "RK4": "s", "BE": "*", "CN": "^", "IM": "v"}

def plot_dt_runtime(results, save_path="results/plots/runtime_dt.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    fig.suptitle(f"Wall-clock runtime vs. time-step size  (Nx = Ny = {REP_NX})", fontsize=13)

    for col, scheme in enumerate(SCHEMES):
        ax = axes[col]
        for timestepper in METHODS:
            data     = results[timestepper][scheme]
            dts      = [d[0] for d in data]
            runtimes = [d[3] for d in data]

            converged   = [np.isfinite(r) for r in runtimes]
            dts_ok      = [dts[i]      for i in range(len(dts)) if converged[i]]
            runtimes_ok = [runtimes[i] for i in range(len(dts)) if converged[i]]

            if dts_ok:
                ax.semilogy(dts_ok, runtimes_ok,
                            color=METHOD_COLORS[timestepper],
                            label=METHOD_LABELS[timestepper],
                            linestyle=METHODS_LINESTYLES[timestepper],
                            marker=METHODS_MARKERS[timestepper])

        ax.set_title(scheme)
        ax.set_xlabel("$\\Delta t$")
        ax.set_ylabel("runtime (s)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")


def plot_nx_runtime(results, save_path="results/plots/runtime_nx.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    fig.suptitle(f"Wall-clock runtime vs. grid size  (dt = {REP_DT:.3f})", fontsize=13)

    for col, scheme in enumerate(SCHEMES):
        ax = axes[col]
        for timestepper in METHODS:
            data     = results[timestepper][scheme]
            nxs      = [d[0] for d in data]
            runtimes = [d[3] for d in data]

            converged   = [np.isfinite(r) for r in runtimes]
            nxs_ok      = [nxs[i]      for i in range(len(nxs)) if converged[i]]
            runtimes_ok = [runtimes[i] for i in range(len(nxs)) if converged[i]]

            if nxs_ok:
                ax.semilogy(nxs_ok, runtimes_ok,
                            color=METHOD_COLORS[timestepper],
                            label=METHOD_LABELS[timestepper],
                            linestyle=METHODS_LINESTYLES[timestepper],
                            marker=METHODS_MARKERS[timestepper])

        ax.set_title(scheme)
        ax.set_xlabel("$N_x = N_y$")
        ax.set_ylabel("runtime (s)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    dt_results = dt_convergence()
    nx_results = nx_convergence()

    plot_dt_convergence(dt_results)
    plot_nx_convergence(nx_results)
    plot_dt_runtime(dt_results)
    plot_nx_runtime(nx_results)
    print("\nDone. Plots saved to results/plots/")


if __name__ == "__main__":
    main()
