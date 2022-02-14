import numpy as np
from math import sin, cos,tan
import model.params as params

def RotToRPY_ZYX(R):
    """
        Euler angle convention is ZYX, which means first apply
        rotaion of psi-degrees around Z axis, then rotation of
        theta-degrees around new Y axis, and then rotation of
        phi-degrees around new X axis.
        ** The rotation R received should be from body to world frame. **
    """
    theta = np.arcsin(-1.0 * R[2, 0])
    phi = np.arctan2(R[2, 1] / np.cos(theta), R[2, 2] / np.cos(theta))
    psi = np.arctan2(R[1, 0] / np.cos(theta), R[0, 0] / np.cos(theta))
    return np.array([[phi], [theta], [psi]])

def RPYToRot_ZYX(phi, theta, psi):
    """
    The euler angle convention used is ZYX. This means: first a rotation of psi-degrees
    around Z axis, then rotation of theta-degrees around Y axis, and finally rotation of
    phi-degress around X axis
    """
    return np.array([[cos(theta)*cos(psi), cos(theta)*sin(psi), -sin(theta)],
                     [-cos(phi)*sin(psi) + sin(phi)*sin(theta)*cos(psi), cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi), sin(phi)*cos(theta)],
                     [sin(phi)*sin(psi) + cos(phi)*sin(theta)*cos(psi), -sin(phi)*cos(psi) + cos(phi)*sin(theta)*sin(psi), cos(phi)*cos(theta)]])

def Eulerangleratesmatrix(phi,theta,psi):
    """Compute inverse euler angle rates matrix"""
    peta = np.array([
        [1.0, sin(phi) * tan(theta), cos(phi) * tan(theta)],
        [0.0, cos(phi), -1.0 * sin(phi)],
        [0.0, sin(phi) / cos(theta), cos(phi) / cos(theta)]])
    return peta

def state_dot_(state,F, M,inertial,invI,mass):
    x, y, z, xdot, ydot, zdot, phi, theta, psi, wx, wy, wz = state
    # acceleration - Newton's second law of motion
    wRb = RPYToRot_ZYX(phi, theta, psi)  # from body to world
    accel = -params.g * params.e3 + (F / mass) * np.dot(wRb, params.e3)
    omega = np.array([wx, wy, wz])
    # euler angle rates
    eulerdot = np.dot(Eulerangleratesmatrix(phi, theta, psi), omega.reshape(3, 1))
    # Angular acceleration
    omegadot = invI.dot(M - np.cross(omega,inertial.dot(omega)))
    state_dot = np.zeros(12)
    state_dot[0] = xdot
    state_dot[1] = ydot
    state_dot[2] = zdot
    state_dot[3] = accel[0][0]
    state_dot[4] = accel[1][0]
    state_dot[5] = accel[2][0]
    state_dot[6] = eulerdot[0]
    state_dot[7] = eulerdot[1]
    state_dot[8] = eulerdot[2]
    state_dot[9] = omegadot[0]
    state_dot[10] = omegadot[1]
    state_dot[11] = omegadot[2]
    state_dot = state_dot.reshape(1,12)[0]
    return state_dot

def get_yc(sigma4):
    # calculate y coordinate axis of the coordinate system C
    y_c = np.array([[-1.0 * np.sin(sigma4)], [np.cos(sigma4)], [0.0]])
    return y_c