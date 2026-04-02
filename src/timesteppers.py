#RK4, Backward Euler, and Forward Euler time-stepping methods

from .problem import HeatTransportProblem

def forward_euler_step(T, problem, operators, t, dt):

    rhs = problem.compute_rhs(T, operators, t)
    T_new = T + dt * rhs

    return T_new