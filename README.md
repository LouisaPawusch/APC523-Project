# APC523-Project

### TODO:

  1. src/timesteppers.py — only forward_euler_step exists. Still needed for explicit methods:
  - one 2nd order explicit method (eg RK2, Predictor Corrector, explicit midpoint(?))
    - RK4                                                                                                                                                                                 

    For implicit methods - requires implicit formulation of the problem:
    - Backward Euler, Crank-Nicolson, implicit midpoint (implicit — require linear solves)            
                                                                                                                         
  2. src/linear_solvers.py — completely empty. Needed for implicit schemes:             

    - Jacobi, under-relaxed Jacobi, Gauss-Seidel, SOR

    - Direct solve via a Sparse LU baseline (scipy.sparse.linalg.spsolve), to compare implicit solvers to                                

  3. test/ — empty, no tests at all - start with testing operator.py   
  - constant solution remains constant
  - zero source + zero velocity + zero-flux BC behaves sensibly
  - Dirichlet BC is actually enforced
  - operator-based RHS matches slice-based RHS for a simple case
  - Forward Euler gives the same result before/after refactor
  - maybe a manufactured solution or simple diffusion test                                                         

  4. Benchmarking/comparison tasts 
    - with the analytical solution — the Gaussian benchmark (analytic solution for eg point heat source + uniform flow) for convergence/error analysis
    - add comparison tools to compare analytical v numerical solution (plotting, error metrics, eg L2 or L_infty, also runtime, memory)
    - error vs timestep size, error vs grid size, runtime vs error, explicit vs implicit comparison, central vs upwind comparison for advection
    - plot of 1D slicing through the solution, comparison plot across time-steppers, comparison plot across advection schemes, maybe stability-failure plots for too-large Δt
    - calculate stability restrictions for FE, compare with experimental stability

  5. Optional: Darcy flow module for a computed velocity field 
  - implement hydraulic head solve
  - compute Darcy flux
  - convert to pore velocity
  - plug that velocity field into the heat equation solver

Priority order

- finish and stabilize the current Forward Euler + operator-based solver
- add RK4
- add analytical benchmark solution
- add error metrics and comparison plots
- add one implicit method, probably Backward Euler
- add spsolve for the implicit linear system
- add one or two iterative solvers
- only then decide whether to do Darcy as an extension