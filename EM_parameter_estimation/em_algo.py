import numpy as np
from numpy.linalg import  solve
from math import cos,sin,tan
import model.params as params
from utils.utilities import RPYToRot_ZYX as RPYToRot
from utils.utilities import state_dot_

def jakobF(m,dt):
    # jacobian matrix , m is observation
    phi, theta, psi = m[6:9]
    wx,wy,wz = m[9:12]
    dphi6 = 1 + dt * (cos(phi) * tan(theta) * wy - sin(phi) * tan(theta) * wz)
    dphi7 = dt * (sin(theta) * wy / np.square(1 / cos(theta)) + cos(phi) * wz * np.square(1 / cos(theta)))
    dphi10 = dt * sin(phi) * tan(theta)
    dphi11 = dt * cos(phi) * tan(theta)
    dtheta6 = dt * (-wy * sin(phi) - cos(phi) * wz)
    dtheta10 = dt * cos(phi)
    dtheta11 = -dt * sin(phi)
    dpsi6 = dt * ((cos(phi) / cos(theta)) * wy - (sin(phi) / cos(theta)) * wz)
    dpsi7 = dt * (sin(phi) * (sin(theta) / np.square(cos(theta))) * wy + cos(phi) * (
            sin(theta) / np.square(cos(theta))) * wz)
    dpsi10 = dt * (sin(phi) / cos(theta))
    dpsi11 = dt * (cos(phi) / cos(theta))
    F = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, dt, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, dphi6, dphi7, 0, dt, dphi10, dphi11],
                  [0, 0, 0, 0, 0, 0, dtheta6, 1, 0, 0, dtheta10, dtheta11],
                  [0, 0, 0, 0, 0, 0, dpsi6, dpsi7, 1, 0, dpsi10, dpsi11],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    return F


def Expectation(Y , m , P , inertial,invI, mass, F_t , M_t , Q , R, dt ):
    """Expectation maximization algorithm---E step"""
    m_ = m.shape[0]
    y_ = len(Y)
    p_1, p_2 = P.shape
    kf_m = np.zeros((y_, 12, 1))
    kf_P = np.zeros((y_, 12, 12))
    kf_m[0] = m
    kf_P[0] = P

    H = np.eye(12)
    for k in range(y_ - 1):
        # prediction
        F = jakobF(m.reshape(1, 12)[0],dt)
        m = m + dt * state_dot_(m.reshape(1, 12)[0], F_t[k], M_t[k], inertial,invI, mass).reshape(12, 1)
        # correction
        P = np.dot(np.dot(np.dot(F,P),F.T) + Q, H.T)
        S = np.dot(np.dot(H,P),H.T)  + R
        # EKF gain matrix
        K = solve(S.T, P.T).T
        # update
        m += K @ (Y[k + 1].reshape(12, 1) - m)
        P -= K @ P
        # saving
        kf_m[k + 1] = m
        kf_P[k + 1] = P

    # RTS smoother
    ms = m # initial value
    Ps = P # initial value
    rts_m = np.zeros((y_, m_, 1))
    rts_P = np.zeros((y_, p_1, p_2))
    rts_m[-1] = ms
    rts_P[-1] = Ps
    for k in range(y_ - 2, -1, -1):
        # prediction
        mp = kf_m[k] + dt * state_dot_(kf_m[k].reshape(1, 12)[0], F_t[k], M_t[k], inertial,invI,mass).reshape(12, 1)
        Pp = kf_P[k] + Q
        # correction
        Ck = solve(Pp.T, kf_P[k].T).T
        # update
        ms = kf_m[k] + Ck @ (ms - mp)
        Ps = kf_P[k] + Ck @ (Ps - Pp) @ Ck.T
        # saving
        rts_m[k] = ms
        rts_P[k] = Ps
    return rts_m, rts_P

def Maximization(rts_m, dt, F_t, M_t, inertial):
    """Expectation maximization algorithm---M step"""
    y_ = len(rts_m)
    sumA0 = 0
    sumB0 = 0

    sumA1 = 0
    sumB1 = 0

    sumA2 = 0
    sumB2 = 0

    sumA = 0
    sumB = 0
    sumC = 0
    sumD = 0
    r_list = []
    for k in range(y_):
        x, y, z, xdot, ydot, zdot, phi, theta, psi, wx, wy, wz = rts_m[k]
        rot = RPYToRot(phi,theta,psi) #from body to world rotation matrix
        r = rot[2][2] # last element of rotation matrix
        r_list.append(r) # saving it to list


    for k in range(y_-1):
        # inertia matrix maximazation
        sumA0 += (np.square(M_t[k][0]) + 2 * M_t[k][0] * (inertial[1][1] - inertial[2][2]) * rts_m[k][10] * rts_m[k][11] + np.square(inertial[1][1] - inertial[2][2])*np.square(rts_m[k][10])*np.square(rts_m[k][11]) ) * dt
        sumB0 +=(rts_m[k+1][9] - rts_m[k][9]) * (M_t[k][0] + (inertial[1][1] - inertial[2][2]) * rts_m[k][10]*rts_m[k][11])
        sumA1 += (np.square(M_t[k][1]) + 2 * M_t[k][1] * (inertial[2][2] - inertial[0][0]) * rts_m[k][9] * rts_m[k][11] + np.square(inertial[2][2] - inertial[0][0])*np.square(rts_m[k][9])*np.square(rts_m[k][11]) ) * dt
        sumB1 += (rts_m[k+1][10] - rts_m[k][10]) * (M_t[k][1] + (inertial[2][2] - inertial[0][0]) * rts_m[k][9]*rts_m[k][11])

        sumA2 += (np.square(M_t[k][2]) + 2 * M_t[k][2] * (inertial[0][0] - inertial[1][1]) * rts_m[k][9] * rts_m[k][10] + np.square(inertial[0][0] - inertial[1][1])*np.square(rts_m[k][9])*np.square(rts_m[k][10]) ) * dt
        sumB2 += (rts_m[k+1][11] - rts_m[k][11]) * (M_t[k][2] + (inertial[0][0] - inertial[1][1]) * rts_m[k][9]*rts_m[k][10])
        #mass maximazation
        sumA += r_list[k] * F_t[k]
        sumB +=np.square(r_list[k]) * np.square(F_t[k])
        sumC +=r_list[k] * F_t[k] * rts_m[k+1][5]
        sumD +=r_list[k] * F_t[k] * rts_m[k][5]

    # inertia matrix maximazation
    Ixx = sumA0 / sumB0
    Iyy = sumA1 / sumB1
    Izz = sumA2 / sumB2
    Inertial = np.array([[Ixx[0], 0, 0],
                        [0, Iyy[0], 0],
                        [0, 0, Izz[0]]])
    invI = np.array([[1/Ixx[0], 0, 0],
                        [0, 1/Iyy[0], 0],
                        [0, 0, 1/Izz[0]]])
    # mass maximazation
    mass = (dt * sumB)/(params.g * dt * sumA + sumC - sumD)

    return Inertial,invI,mass





