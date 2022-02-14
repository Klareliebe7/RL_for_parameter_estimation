"""linear–quadratic–Gaussian controller based on differential flatness """
import numpy as np
import model.params as params
from utils.utilities import RotToRPY_ZYX as RotToRPY
from utils.utilities import get_yc

K = np.array([[2.9580, 0, 0, 3.1090, 0, 0], [0, 2.9580, 0, 0, 3.1090, 0],[0, 0, 2.9580, 0, 0, 3.1090]])
gainmatrix = np.array([[1.5811, 0, 0 , 15.9111 , 0 , 0],[0,1.5811,0,0 , 15.9111, 0],[0,0 ,1.5811,0,0,15.9111]])
Kr = 40

def controller(quad, extendkalmanflter,sensor,I,mass,endpoints,dyaw,dt):
    observe = sensor.update(quad,dt)
    f = observe[0:3]
    w = observe[3:6]
    # get quadcopter estimated state
    state_est = extendkalmanflter.update(f,w,sensor.Y_obs3(),dt)
    pos_ref = endpoints.reshape(3, 1)
    v_ref = np.array([[0], [0], [0]])

    pos_est = state_est[0:3]
    v_est = state_est[3:6]
    or_est = state_est[6:9].reshape(3, 1)
    w_est = state_est[9:12].reshape(3, 1)

    #  Compute thrust
    Z_E = np.array([[0.0], [0.0], [1.0]])  # the Z axis of Earth frame expressed in body frame is equal to Z_b...

    ref_state = np.array([[pos_ref[0][0]], [pos_ref[1][0]],[pos_ref[2][0]], [v_ref[0][0]],[v_ref[1][0]], [v_ref[2][0]]])
    est_state = np.array([[pos_est[0]], [pos_est[1]],[pos_est[2]], [v_est[0]],[v_est[1]], [v_est[2]]])
    ua = -np.dot(K,est_state - ref_state)
    F = (((ua[2][0]) + params.g) * mass) / (np.cos(state_est[6]) * np.cos(state_est[7]))
    #  Compute torque
    #  Compute desired orientation
    Z_B = (ua + params.g * Z_E) / np.linalg.norm(ua + params.g * Z_E)
    Y_C = np.array(get_yc(dyaw[0]))  # transform to np.array 'cause comes as np.matrix
    X_B = np.cross(Y_C, Z_B, axis=0)
    X_B = X_B / np.linalg.norm(X_B)
    Y_B = np.cross(Z_B, X_B, axis=0)
    Rbw_ref = np.concatenate((X_B, Y_B, Z_B), axis=1) # from body to world rotation matrix
    or_ref = RotToRPY(Rbw_ref)  # get desired roll, pitch, yaw angles
    #  Compute Reference angular velocity
    w_ref = euler_angular_velocity_des(or_est, or_ref.reshape(3,1), Kr)
    e = np.vstack([or_ref - or_est, w_ref - w_est])
    ub_e = np.dot(gainmatrix, e)
    M = np.dot(I, ub_e) + np.cross(w_est, np.dot(I, w_est), axis=0)
    return F, M



def euler_angular_velocity_des(euler, euler_ref, gain):
    """
    Control law is of the form: u = K*(euler_ref - euler)
    """
    gain_matrix = np.diag([gain, gain, gain])
    euler_error = euler - euler_ref
    u = -1.0 * np.dot(gain_matrix, euler_error)
    euler_dot = u
    phi = euler[0][0]
    theta = euler[1][0]
    # compute inverse euler angle rates matrix
    K = np.array([[1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                  [0.0, np.cos(phi), -1.0 * np.sin(phi)],
                  [0.0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]])

    Kinv = np.linalg.inv(K)

    w_ref = np.dot(Kinv, euler_dot)

    return w_ref





