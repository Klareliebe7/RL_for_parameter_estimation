import numpy as np
from EM_parameter_estimation.em_algo import Expectation
from EM_parameter_estimation.em_algo import Maximization

def em_estimation(state_his,F_t,M_t,I,invI,mass,dt):
    steps = len(state_his)
    Y = np.zeros((steps, 12))
    for k in range(steps):
        y = state_his[k]
        Y[k] = y



    m = Y[0].reshape(12,1)
    Q = np.eye(12) * np.sqrt(dt)  # process noise
    R = np.eye(12) * np.sqrt(dt/100)   # observation noise
    P0 = np.eye(12)
    P = P0

    # Expectation (E)-step
    Xs, Ps = Expectation(Y, m, P, I, invI, mass, F_t, M_t, Q, R, dt)
    # Maximization (M)-step
    Inertial, invI,mass = Maximization(Xs, dt, F_t, M_t,I)


    return Inertial,invI,mass
