# Script to define plotting functions for visualizing the results

import matplotlib.pyplot as plt
import numpy as np


def plot_initial_and_final(problem, times, all_T, save_path=None, cmap="inferno"):
    """
    Plot the temperature field at the initial and final saved timestep.
    """
    X, Y = problem.get_mesh()

    T_initial = all_T[0]
    T_final = all_T[-1]

    vmin = min(np.min(T_initial), np.min(T_final))
    vmax = max(np.max(T_initial), np.max(T_final))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].pcolormesh(X, Y, T_initial, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Initial condition (t = {times[0]:.3f})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    im1 = axes[1].pcolormesh(X, Y, T_final, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Final condition (t = {times[-1]:.3f})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    cbar = fig.colorbar(im1, ax=axes, shrink=0.9)
    cbar.set_label("Temperature")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_gaussian_cuts(problem, T_analytical, T_numerical, t, numerical_label="Numerical", save_path=None):
    """
    Plot 1D cuts through the temperature field at x=gaussian_x0 and y=gaussian_y0,
    comparing the analytical solution against a numerical solution.
    """
    X, Y = problem.get_mesh()
    x = X[0, :]
    y = Y[:, 0]

    cx = problem.gaussian_x0 + (problem.vx / problem.R_th) * t
    cy = problem.gaussian_y0 + (problem.vy / problem.R_th) * t

    ix = np.argmin(np.abs(x - cx))
    iy = np.argmin(np.abs(y - cy))

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    err_x = T_numerical[iy, :] - T_analytical[iy, :]
    err_y = T_numerical[:, ix] - T_analytical[:, ix]

    l2_x   = np.sqrt(np.sum(err_x**2) * dx)
    linf_x = np.max(np.abs(err_x))
    l2_y   = np.sqrt(np.sum(err_y**2) * dy)
    linf_y = np.max(np.abs(err_y))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Horizontal cut at y = cy: T vs x
    axes[0].plot(x, T_analytical[iy, :], label="Analytical", color="black", linewidth=2)
    axes[0].plot(x, T_numerical[iy, :],  label=numerical_label, linestyle="--")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Temperature")
    axes[0].set_title(f"Cut at y = {y[iy]:.2f}  (t = {t:.2f})")

    axes[0].set_ylim(0, axes[0].get_ylim()[1])

    ax0b = axes[0].twinx()
    ax0b.plot(x, np.abs(err_x), color="red", linewidth=1, linestyle=":", label=f"|error|  L2={l2_x:.2e}  L∞={linf_x:.2e}")
    ax0b.set_ylim(0, 2 * linf_x)
    ax0b.set_ylabel("|error|", color="red")
    ax0b.tick_params(axis="y", labelcolor="red")

    lines0, labels0 = axes[0].get_legend_handles_labels()
    lines0b, labels0b = ax0b.get_legend_handles_labels()
    axes[0].legend(lines0 + lines0b, labels0 + labels0b, fontsize=8)

    # Vertical cut at x = cx: T vs y
    axes[1].plot(y, T_analytical[:, ix], label="Analytical", color="black", linewidth=2)
    axes[1].plot(y, T_numerical[:, ix],  label=numerical_label, linestyle="--")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("Temperature")
    axes[1].set_title(f"Cut at x = {x[ix]:.2f}  (t = {t:.2f})")

    axes[1].set_ylim(0, axes[1].get_ylim()[1])

    ax1b = axes[1].twinx()
    ax1b.plot(y, np.abs(err_y), color="red", linewidth=1, linestyle=":", label=f"|error|  L2={l2_y:.2e}  L∞={linf_y:.2e}")
    ax1b.set_ylim(0, 2 * linf_y)
    ax1b.set_ylabel("|error|", color="red")
    ax1b.tick_params(axis="y", labelcolor="red")

    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    axes[1].legend(lines1 + lines1b, labels1 + labels1b, fontsize=8)

    # Sync upper limits across both subplots
    T_ymax   = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    err_ymax = max(ax0b.get_ylim()[1],    ax1b.get_ylim()[1])
    axes[0].set_ylim(0, T_ymax)
    axes[1].set_ylim(0, T_ymax)
    ax0b.set_ylim(0, err_ymax)
    ax1b.set_ylim(0, err_ymax)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_snapshot(problem, T, t, save_path=None, cmap="inferno"):
    """
    Plot a single snapshot of the temperature field at time t.
    """
    X, Y = problem.get_mesh()

    plt.figure(figsize=(6, 5))
    mesh = plt.pcolormesh(X, Y, T, shading="auto", cmap=cmap)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Temperature at t = {t:.3f}")
    plt.colorbar(mesh, label="Temperature")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()