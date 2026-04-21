#RK4, Backward Euler, and Forward Euler time-stepping methods

def forward_euler_step(T, problem, operators, t, dt):

    rhs = problem.compute_rhs(T, operators, t)
    T_new = T + dt * rhs

    return T_new

def runge_kutta_4_step(T, problem, operators, t, dt):

    k1 = problem.compute_rhs(T, operators, t)
    k2 = problem.compute_rhs(T + 0.5 * dt * k1, operators, t + 0.5 * dt)
    k3 = problem.compute_rhs(T + 0.5 * dt * k2, operators, t + 0.5 * dt)
    k4 = problem.compute_rhs(T + dt * k3, operators, t + dt)
    T_new = T + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return T_new