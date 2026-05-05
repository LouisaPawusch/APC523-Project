"""
Spatial convergence study for Darcy-coupled heat transport.

Mirrors convergence_experiments.py but replaces the prescribed uniform
velocity (vx=0.05) with a velocity field computed from the Darcy solve.
K is chosen so that VX_EXACT = K*(H_L-H_R)/(n*Lx) = 0.05 m/s, making
the two experiments directly comparable.

Only the three implicit time-steppers (BE, CN, IM) are tested:
  - Explicit methods (FE, RK4) behave identically to the non-Darcy case
    because for uniform K the Darcy velocity is spatially constant.
  - The implicit methods are the main result of this project.

Plots produced (3 x 2 subplots, methods x advection schemes):
  darcy_convergence_nx.png  — L2 and L-inf error vs Nx = Ny
"""

import matplotlib
matplotlib.use("Agg")

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.problem import GaussianIC, HeatTransportProblem
from src.darcy import DarcyParams
from src.simulation import run_simulation

os.makedirs("results/plots", exist_ok=True)


# ---------------------------------------------------------------------------
# Parameters — chosen to match convergence_experiments.py exactly
# ---------------------------------------------------------------------------

Lx, Ly   = 2.0, 2.0
alpha    = 1e-3
R_th     = 1.0
x0, y0   = 0.5, 1.0
Q        = 1.0
t_start  = 5.0
t_end    = 15.0
t_sim    = t_end - t_start

# Darcy parameters: K chosen so VX_EXACT = 0.05 m/s (= vx in convergence_experiments.py)
#   VX_EXACT = K * (H_L - H_R) / (porosity * Lx) = K / (0.3 * 2) = 0.05  →  K = 0.03
POROSITY = 0.3
H_LEFT   = 1.0
H_RIGHT  = 0.0
K_UNIF   = 0.03          # [m/s] — gives VX_EXACT = 0.05 m/s
VX_EXACT = K_UNIF * (H_LEFT - H_RIGHT) / (POROSITY * Lx)   # = 0.05

METHODS = ["BE", "CN", "IM"]
SCHEMES = ["central", "upwind"]

METHOD_LABELS = {
    "BE": "Backward Euler",
    "CN": "Crank-Nicolson",
    "IM": "Implicit Midpoint",
}

# Spatial order of the advection discretisation (time order doesn't matter
# for the Nx sweep when dt is small enough that temporal error is subdominant)
SPATIAL_ORDER = {"central": 2, "upwind": 1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_darcy_params():
    return DarcyParams(
        K=K_UNIF, porosity=POROSITY,
        bc_head_left_type="dirichlet",   bc_head_left_value=H_LEFT,
        bc_head_right_type="dirichlet",  bc_head_right_value=H_RIGHT,
        bc_head_bottom_type="neumann",   bc_head_bottom_value=0.0,
        bc_head_top_type="neumann",      bc_head_top_value=0.0,
    )


def make_problem(Nx, Ny=None):
    if Ny is None:
        Ny = Nx
    ic_fn = GaussianIC(x0, y0, Q, alpha, t_start, vx=VX_EXACT, vy=0.0, R_th=R_th)
    return HeatTransportProblem(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        alpha=alpha, R_th=R_th, rho_c_eff=1.0,
        vx=0.0, vy=0.0,          # placeholder; overwritten by Darcy solve
        initial_condition_fn=ic_fn,
        bc_left_type="neumann",   bc_left_value=0.0,
        bc_right_type="neumann",  bc_right_value=0.0,
        bc_bottom_type="neumann", bc_bottom_value=0.0,
        bc_top_type="neumann",    bc_top_value=0.0,
        gaussian_x0=x0, gaussian_y0=y0, gaussian_Q=Q,
    )


def l2_error(T_num, T_ana):
    return np.sqrt(np.mean((T_num - T_ana) ** 2))

def linf_error(T_num, T_ana):
    return np.max(np.abs(T_num - T_ana))


def run_one(problem, dt, timestepper, scheme, darcy_params):
    """Run one simulation and return (T_final, elapsed_s). Returns (None, nan) on failure."""
    n_steps = max(1, int(t_sim / dt))
    t0 = time.perf_counter()
    try:
        _, all_T = run_simulation(
            problem, t_final=t_sim, dt=dt,
            save_every=n_steps,
            advection_scheme=scheme,
            timestepper=timestepper,
            linear_solver="direct",
            darcy_params=darcy_params,
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
        T_final = all_T[-1]
        if not np.isfinite(T_final).all():
            return None, float("nan")
        return T_final, elapsed
    except Exception:
        return None, float("nan")


# ---------------------------------------------------------------------------
# Nx convergence sweep
# ---------------------------------------------------------------------------

REP_DT  = 0.04
NX_LIST = [11, 21, 31, 41, 51, 71, 101]


def nx_convergence():
    print("=" * 65)
    print(f"Darcy Nx convergence sweep  (dt = {REP_DT:.3f})")
    print(f"  VX_EXACT = {VX_EXACT:.4f} m/s  (K = {K_UNIF:.3f} m/s)")
    print("=" * 65)

    darcy_params = make_darcy_params()

    # results[method][scheme] = list of (Nx, l2, linf, runtime)
    results = {m: {s: [] for s in SCHEMES} for m in METHODS}

    for timestepper in METHODS:
        for scheme in SCHEMES:
            print(f"  {timestepper:4s}  {scheme:8s}", end="  ", flush=True)
            for Nx in NX_LIST:
                problem = make_problem(Nx)
                X, Y    = problem.get_mesh()
                T_ref   = (Q / (4 * np.pi * alpha * t_end) *
                           np.exp(-((X - x0 - VX_EXACT / R_th * t_end)**2 +
                                    (Y - y0)**2) / (4 * alpha * t_end)))
                T_num, runtime = run_one(problem, REP_DT, timestepper, scheme, darcy_params)
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
# Plotting
# ---------------------------------------------------------------------------

def _ref_slope(ax, xs, ys, order):
    valid = [(x, y) for x, y in zip(xs, ys) if np.isfinite(y)]
    if len(valid) < 2:
        return
    x0r, y0r = valid[0]
    xa = np.array([v[0] for v in valid])
    ax.plot(xa, y0r * (xa / x0r) ** (-order), "k--", linewidth=0.8,
            label=f"$O(h^{{{order}}})$")


def plot_nx_convergence(results, save_path="results/plots/darcy_convergence_nx.png"):
    fig, axes = plt.subplots(len(METHODS), len(SCHEMES),
                             figsize=(12, 4 * len(METHODS)),
                             constrained_layout=True)
    fig.suptitle(
        f"Darcy-coupled heat transport: error vs. grid size  (dt = {REP_DT:.3f})\n"
        f"VX = {VX_EXACT:.4f} m/s from Darcy solve  (K = {K_UNIF} m/s)",
        fontsize=12,
    )

    for row, timestepper in enumerate(METHODS):
        for col, scheme in enumerate(SCHEMES):
            ax   = axes[row][col]
            data = results[timestepper][scheme]
            nxs      = [d[0] for d in data]
            l2s      = [d[1] for d in data]
            linfs    = [d[2] for d in data]

            ok       = [np.isfinite(e) for e in l2s]
            nxs_ok   = [nxs[i]   for i in range(len(nxs))  if ok[i]]
            l2s_ok   = [l2s[i]   for i in range(len(l2s))  if ok[i]]
            linfs_ok = [linfs[i] for i in range(len(linfs)) if ok[i]]
            nxs_div  = [nxs[i]   for i in range(len(nxs))  if not ok[i]]

            if nxs_ok:
                ax.loglog(nxs_ok, l2s_ok,   "o-",  label="$L_2$ error")
                ax.loglog(nxs_ok, linfs_ok, "s--", label="$L_\\infty$ error")
                _ref_slope(ax, nxs_ok, l2s_ok, SPATIAL_ORDER[scheme])

            if nxs_div:
                for nx in nxs_div:
                    ax.axvline(nx, color="red", linestyle=":", linewidth=1.2)
                ax.plot([], [], color="red", linestyle=":", linewidth=1.2, label="diverged")

            ax.set_title(f"{METHOD_LABELS[timestepper]}  |  {scheme}", fontsize=9)
            ax.set_xlabel("$N_x = N_y$")
            ax.set_ylabel("error")
            ax.legend(fontsize=7)
            ax.grid(True, which="both", alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    nx_results = nx_convergence()
    plot_nx_convergence(nx_results)
    print("\nDone. Plot saved to results/plots/darcy_convergence_nx.png")


if __name__ == "__main__":
    main()
