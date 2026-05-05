# APC523-Project
Authors: Luis Eguiguren, Louisa Pawusch
Course: APC523 - Spring 2026

## Content
Heat transport in fluid-saturated porous media governs a wide range of
geoscientific phenomena, from hydrothermal circulation in mid-ocean ridges to
the thermal evolution of serpentinizing peridotite. In these systems, heat is
carried both through the bulk medium and with the
flowing pore fluid, and the thermal inertia of the solid matrix retards the
propagation of thermal fronts relative to fluid flow. Understanding and
accurately simulating this coupled process requires solving a two-dimensional
advection-diffusion equation over a porous domain, with an optional flow field
computed from Darcy's Law.

To solve this model numerically, we will compare and contrast a variety of numerical algorithms.

We consider a two-dimensional heat-transport model in a porous medium, discretized in space with finite differences. This reduces the governing PDE to a large system of ODEs in time, which we will solve using a range of explicit and implicit time-stepping methods. For the diffusion term we will use central differences, while for the advection term we will compare central and upwind discretizations. If needed, we will also compute the flow field from a steady elliptic Darcy subproblem for the hydraulic head.
To solve these equations, we will leverage the methods discussed in the lecture and compare the performance of different approaches.

We will discretize the space first using finite differences, which will yield an ODE system in time, which we will solve for the temperature.
We will explore a central differences scheme for the diffusion term and a central or an upwind scheme for the advection term.

This reduces the governing PDE to a large system of ODEs in time. 
Therefore, we will investigate various different time-stepping schemes to integrate this system, such as explicit methods (e.g. forward Euler, Predictor-Corrector, and/or fourth-order Runge-Kutta (RK4)), and implicit methods (e.g. Backward Euler, implicit Midpoint, and/or Crank-Nicolson).
For the implicit scheme and the optional Darcy flow subproblem, we compare iterative linear
solvers (e.g. Jacobi, under-relaxed Jacobi, Gauss-Seidel, SOR) against the direct approach of a sparse LU as a baseline.

All spatial and temporal schemes can be validated in a simplified setting with uniform flow and an instantaneous point heat source released at the origin, using a Gaussian benchmark solution.
This will allow us to investigate various different error metrics and compare our numerical approaches with regards to speed and accuracy.

## Installation

To run the project on Adroit, install dependencies using 
```bash
# From the repository root
pip install -e .
```
## Run example simulations
```bash
# From the repository root or from the examples folder
python run_basic_simulation.py
python gaussian_benchmark.py
# Darcy-coupled heat transport
python darcy_benchmark.py
python darcy_error_benchmark.py
python darcy_convergence_experiments.py
```


To use the slurm script, add activation of the environment at the bottom and run
```bash
# From the examples folder
sbatch convergence_experiments.slurm
```
