import numpy as np
from math import cos,sin,tan
from numpy.linalg import solve
from utils.utilities import Eulerangleratesmatrix as Eulermatrix
class extendKalmanFilter:
    ''' filter out noise and restore the true state from noisy observations.
    state  - 1 dimensional vector but used as 12 x 1. [x, y, z, vx, vy, vz, phi, theta, psi, wx, wy, wz]'''
    def __init__(self):
        # initial state value
        state0 = np.array([0.5, 1, 0, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0])
        self.state_est = np.zeros(12)
        self.state_est[0] = state0[0]
        self.state_est[1] = state0[1]
        self.state_est[2] = state0[2]
        self.state_est[3] = state0[3]
        self.state_est[4] = state0[4]
        self.state_est[5] = state0[5]
        self.state_est[6] = state0[6]
        self.state_est[7] = state0[7]
        self.state_est[8] = state0[8]
        self.state_est[9] = state0[9]
        self.state_est[10] = state0[10]
        self.state_est[11] = state0[11]

        # covariance
        self.P_est = np.eye(12)

    def update(self,f, w , Y_mea,dt):
        f = np.array([[f[0]],[f[1]],[f[2]]])
        # prediction model
        x_p= self.state_est[0:3].reshape(3,1) + (dt * self.state_est[3:6].reshape(3,1))
        v_p= self.state_est[3:6].reshape(3,1) + (dt * f)
        wx, wy, wz = self.state_est[9:12]
        eulerdot = np.dot(Eulermatrix(self.state_est[6],self.state_est[7],self.state_est[8]),self.state_est[9:12])
        euler_pre = self.state_est[6:9] + eulerdot * dt

        state_P = np.zeros(12) # Prediction storage vector
        w_p = w
        phi,theta,psi = self.state_est[6:9]
        state_P[0:3] = x_p.T[0]
        state_P[3:6] = v_p.T[0]
        state_P[6:9] = euler_pre
        state_P[9:12] = w_p
        # components of jacobian matrix Phi
        dphi6 = 1+ dt*(cos(phi) * tan(theta) * wy - sin(phi)*tan(theta)*wz)
        dphi7 = dt*(sin(theta)* wy / np.square(1/cos(theta)) + cos(phi) * wz * np.square(1/cos(theta)) )
        dphi10 = dt * sin(phi) * tan(theta)
        dphi11 = dt * cos(phi)*tan(theta)
        dtheta6 = dt*(-wy* sin(phi) - cos(phi) * wz)
        dtheta10 =  dt * cos(phi)
        dtheta11 = -dt * sin(phi)
        dpsi6 = dt * ((cos(phi)/cos(theta)) * wy - (sin(phi)/cos(theta))*wz)
        dpsi7 = dt*(sin(phi)*(sin(theta)/np.square(cos(theta)))*wy + cos(phi)*(sin(theta)/np.square(cos(theta)))*wz)
        dpsi10 = dt*(sin(phi)/cos(theta))
        dpsi11 = dt * (cos(phi)/cos(theta))
        # jacobian matrix Phi
        Phi = np.array([[1,0,0,dt,0,0,0,0,0,0,0,0],
                      [0,1,0,0,dt,0,0,0,0,0,0,0],
                      [0,0,1,0,0,dt,0,0,0,0,0,0],
                      [0,0,0,1,0,0,0,0,0,0,0,0],
                      [0,0,0,0,1,0,0,0,0,0,0,0],
                      [0,0,0,0,0,1,0,0,0,0,0,0],
                      [0, 0, 0, 0, 0, 0, dphi6,dphi7, 0, dt,dphi10 ,dphi11 ],
                      [0, 0, 0, 0, 0, 0, dtheta6, 1, 0, 0, dtheta10 , dtheta11],
                      [0, 0, 0, 0, 0, 0, dpsi6, dpsi7, 1, 0, dpsi10, dpsi11],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        # observation matrix
        H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        R = np.eye(12) * np.sqrt(dt/100) # observation noise
        Q = np.eye(12) * np.sqrt(dt)*dt  # process noise
        P_p = np.dot(np.dot(Phi,self.P_est),Phi.T) + Q# time update equation P_k_ = AP_k-1 A ＋Q
        #EKF gain matrix
        K = solve(np.dot(np.dot(H,P_p),(H.T)).T + R , np.dot(P_p,(H.T)).T).T #state update euqation K_k ＝P_k_－ H ^T （ HP_k_－ H^T ＋R ）^ －2

        self.P_est = np.dot((np.eye((12))-np.dot(K,H)),P_p)#state update euqation P_k ＝ （ I－ K_k H ） P_k_
        # correction
        correct = np.dot(K,(Y_mea - np.dot(H,state_P.reshape(12,1))))
        #####print(f"correct={correct} correct.t = {correct.T}")
        self.state_est = state_P + correct.T[0]#  state update euqation X赞 k ＝X赞 k － ＋K k （ Z k －HX赞 k － ）
        #print(f"state_est = {self.state_est}")
        return self.state_est
    def position_est(self):
        return self.state_est[0:3]

    def velocity_est(self):
        return self.state_est[3:6]

    def attitude_est(self):
        return self.state_est[6:9]

    def omega_est(self):
        return self.state_est[10:12]










