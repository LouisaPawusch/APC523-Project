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