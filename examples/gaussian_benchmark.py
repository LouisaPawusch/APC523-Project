"""
Gaussian benchmark: compare all five time-stepping methods against the analytical solution.

The analytical solution is a Gaussian plume advected by a uniform flow:
    T(x, y, t) = Q / (4 pi alpha t) * exp(-((x - x0 - ux*t)^2 + (y - y0 - uy*t)^2) / (4*alpha*t))

Two comparisons are made:
  1. Same dt for all methods — accuracy comparison on equal footing.
  2. Large dt (above FE/RK4 stability limit) — stability advantage of implicit methods.
"""

import numpy as np
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.problem import HeatTransportProblem, GaussianIC
from src.simulation import run_simulation
from src.plotting import plot_gaussian_cuts, plot_initial_and_final

os.makedirs("results/plots", exist_ok=True)


# ---------------------------------------------------------------------------
# Shared problem setup
# ---------------------------------------------------------------------------

def make_problem(Nx=101, Ny=101):
    Lx, Ly = 2.0, 2.0
    alpha   = 1e-3
    vx, vy  = 0.05, 0.0
    R_th    = 1.0
    x0, y0  = 0.5, 1.0
    Q       = 1.0
    t_start = 5.0        # IC evaluated at this physics time (avoids t=0 singularity)

    ic_fn = GaussianIC(x0, y0, Q, alpha, t_start, vx=vx, vy=vy, R_th=R_th)

    return HeatTransportProblem(
        Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny,
        alpha=alpha, R_th=R_th, rho_c_eff=1.0,
        vx=vx, vy=vy,
        initial_condition_fn=ic_fn,
        bc_left_type="neumann",   bc_left_value=0.0,
        bc_right_type="neumann",  bc_right_value=0.0,
        bc_bottom_type="neumann", bc_bottom_value=0.0,
        bc_top_type="neumann",    bc_top_value=0.0,
        gaussian_x0=x0, gaussian_y0=y0, gaussian_Q=Q,
    ), t_start


def compute_errors(T_num, T_ana):
    """Return (L2, Linf) global errors between numerical and analytical fields."""
    diff = T_num - T_ana
    return np.sqrt(np.mean(diff**2)), np.max(np.abs(diff))


def run_method(problem, t_sim, dt, timestepper, linear_solver="direct", **kw):
    """Run one method and return (T_final, elapsed_seconds)."""
    t0 = time.perf_counter()
    _, all_T = run_simulation(
        problem, t_final=t_sim, dt=dt,
        save_every=max(1, int(t_sim / dt)),   # save only final snapshot
        advection_scheme="central",
        timestepper=timestepper,
        linear_solver=linear_solver,
        **kw,
    )
    elapsed = time.perf_counter() - t0
    return all_T[-1], elapsed


# ---------------------------------------------------------------------------
# Part 1 — accuracy comparison at equal dt
# ---------------------------------------------------------------------------

def accuracy_comparison():
    print("=" * 65)
    print("Part 1: Accuracy comparison at equal dt")
    print("=" * 65)

    problem, t_start = make_problem(Nx=101, Ny=101)
    t_end  = 15.0
    t_sim  = t_end - t_start

    # FE stability limit: dt < dx^2 / (4*alpha) ≈ 0.1; use dt = 0.05 for FE/RK4
    dt = 0.05

    X, Y         = problem.get_mesh()
    T_analytical = problem.analytical_solution(X, Y, t_end)

    methods = [
        ("FE",  "FE",  "direct", {}),
        ("RK4", "RK4", "direct", {}),
        ("BE",  "BE",  "direct", {}),
        ("CN",  "CN",  "direct", {}),
        ("IM",  "IM",  "direct", {}),
    ]

    results = {}
    print(f"\n{'Method':<6}  {'L2 error':>12}  {'Linf error':>12}  {'Wall time (s)':>14}")
    print("-" * 50)

    for label, ts, ls, kw in methods:
        print(f"  Running {label}...", flush=True)
        T_num, elapsed = run_method(problem, t_sim, dt, ts, ls, **kw)
        l2, linf = compute_errors(T_num, T_analytical)
        results[label] = T_num
        print(f"{label:<6}  {l2:>12.4e}  {linf:>12.4e}  {elapsed:>14.2f}")

        plot_gaussian_cuts(
            problem, T_analytical, T_num, t_end,
            numerical_label=label,
            save_path=f"results/plots/gaussian_{label}_cuts.png",
        )

    # Side-by-side initial / final for BE and CN as representative examples
    for label, ts, ls, kw in [("BE", "BE", "direct", {}), ("CN", "CN", "direct", {})]:
        _, all_T_full = run_simulation(
            problem, t_final=t_sim, dt=dt,
            save_every=1,
            advection_scheme="central",
            timestepper=ts, linear_solver=ls, **kw,
        )
        times_full = np.linspace(t_start, t_end, len(all_T_full))
        plot_initial_and_final(
            problem, times_full, all_T_full,
            save_path=f"results/plots/gaussian_{label}_init_and_final.png",
            cmap="inferno",
        )


# ---------------------------------------------------------------------------
# Part 2 — stability demo: large dt that makes explicit methods diverge
# ---------------------------------------------------------------------------

def stability_demo():
    print("\n" + "=" * 65)
    print("Part 2: Stability demo at dt >> FE/RK4 stability limit")
    print("=" * 65)

    # Use a coarser grid so it runs fast; dx = 0.04, FE limit ≈ 0.4 s
    problem, t_start = make_problem(Nx=51, Ny=51)
    t_end = 15.0
    t_sim = t_end - t_start

    # FE stability limit for this grid: dx^2/(4*alpha) = (0.04)^2/(4*1e-3) = 0.4
    dt_stable   = 0.2     # within limit (reference)
    dt_large    = 1.0     # 2.5x above limit — explicit methods diverge

    X, Y         = problem.get_mesh()
    T_analytical = problem.analytical_solution(X, Y, t_end)

    print(f"\n  Grid: {problem.Nx}x{problem.Ny},  FE stability limit ≈ 0.4 s")
    print(f"  dt_stable = {dt_stable},  dt_large = {dt_large}\n")

    print(f"{'Method':<18}  {'dt':>6}  {'L2 error':>12}  {'Linf error':>12}  {'Wall time (s)':>14}")
    print("-" * 70)

    # Explicit reference at small dt
    for ts in ("FE", "RK4"):
        T_num, elapsed = run_method(problem, t_sim, dt_stable, ts)
        l2, linf = compute_errors(T_num, T_analytical)
        print(f"{ts + ' (stable dt)':<18}  {dt_stable:>6.2f}  {l2:>12.4e}  {linf:>12.4e}  {elapsed:>14.2f}")

    # Explicit at large dt — expect divergence
    for ts in ("FE", "RK4"):
        try:
            T_num, elapsed = run_method(problem, t_sim, dt_large, ts)
            if not np.isfinite(T_num).all():
                print(f"{ts + ' (large dt)':<18}  {dt_large:>6.2f}  {'DIVERGED (NaN/Inf)':>40}")
            else:
                l2, linf = compute_errors(T_num, T_analytical)
                print(f"{ts + ' (large dt)':<18}  {dt_large:>6.2f}  {l2:>12.4e}  {linf:>12.4e}  {elapsed:>14.2f}")
        except Exception as e:
            print(f"{ts + ' (large dt)':<18}  {dt_large:>6.2f}  {'ERROR: ' + str(e)[:30]:>40}")

    # Implicit methods at large dt — should remain stable
    for ts in ("BE", "CN", "IM"):
        T_num, elapsed = run_method(problem, t_sim, dt_large, ts, linear_solver="direct")
        l2, linf = compute_errors(T_num, T_analytical)
        print(f"{ts + ' (large dt)':<18}  {dt_large:>6.2f}  {l2:>12.4e}  {linf:>12.4e}  {elapsed:>14.2f}")

    # Implicit at small dt for comparison
    for ts in ("BE", "CN", "IM"):
        T_num, elapsed = run_method(problem, t_sim, dt_stable, ts, linear_solver="direct")
        l2, linf = compute_errors(T_num, T_analytical)
        print(f"{ts + ' (stable dt)':<18}  {dt_stable:>6.2f}  {l2:>12.4e}  {linf:>12.4e}  {elapsed:>14.2f}")


# ---------------------------------------------------------------------------
# Part 3 — iterative solver comparison (BE with different linear solvers)
# ---------------------------------------------------------------------------

def iterative_solver_comparison():
    print("\n" + "=" * 65)
    print("Part 3: Iterative solver comparison (Backward Euler)")
    print("=" * 65)

    problem, t_start = make_problem(Nx=51, Ny=51)
    t_end = 15.0
    t_sim = t_end - t_start
    dt    = 0.2

    X, Y         = problem.get_mesh()
    T_analytical = problem.analytical_solution(X, Y, t_end)

    solvers = [
        ("direct",       {}),
        ("jacobi",       {"tol": 1e-8, "max_iter": 2000}),
        ("jacobi (ω=0.8)", {"tol": 1e-8, "max_iter": 2000, "omega": 0.8}),
        ("gauss_seidel", {"tol": 1e-8, "max_iter": 2000}),
        ("sor (ω=1.5)",  {"tol": 1e-8, "max_iter": 2000, "omega": 1.5}),
    ]

    print(f"\n{'Solver':<22}  {'L2 error':>12}  {'Linf error':>12}  {'Wall time (s)':>14}")
    print("-" * 65)

    for label, kw in solvers:
        # map display label to actual solver string
        ls = label.split()[0]   # e.g. "jacobi (ω=0.8)" -> "jacobi"
        T_num, elapsed = run_method(problem, t_sim, dt, "BE", linear_solver=ls, **kw)
        l2, linf = compute_errors(T_num, T_analytical)
        print(f"{label:<22}  {l2:>12.4e}  {linf:>12.4e}  {elapsed:>14.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    accuracy_comparison()
    stability_demo()
    iterative_solver_comparison()
    print("\nDone. Plots saved to results/plots/")


if __name__ == "__main__":
    main()
