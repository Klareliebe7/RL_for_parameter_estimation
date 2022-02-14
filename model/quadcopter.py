import numpy as np
from utils.utilities import RPYToRot_ZYX as RPYToRot
from utils.utilities import Eulerangleratesmatrix as Eulermatrix
import model.params as params

class Quadcopter:
    """ Quadcopter class

    state  - 1 dimensional vector but used as 12 x 1. [x, y, z, vx, vy, vz, phi, theta, psi, wx, wy, wz]
             where [phi, theta, psi] is Euler angle and [wx, wy, wz] are angular velocity.
    F      - 1 x 1, thrust output from controller
    M      - 3 x 1, moments output from controller
    params - system parameters struct, g, mass, etc.
    """

    def __init__(self, pos, attitude):
        """ pos = [x,y,z] attitude = [phi, theta, psi]
            """
        self.state = np.array([1, 0.5, 0, 0.1, 0.1, 0.1, 0, 0, 0, 0.1,0.1,0.1])
        phi, theta, psi = attitude
        self.state[0] = pos[0]
        self.state[1] = pos[1]
        self.state[2] = pos[2]
        self.state[6] = phi
        self.state[7] = theta
        self.state[8] = psi
        self.statedot = np.zeros(12)

    def world_frame(self):
        """ position returns a 3x6 matrix
            where row is [x, y, z] column is m1 m2 m3 m4 origin h
            """
        origin = self.state[0:3]
        phi, theta, psi = self.state[6:9]
        rot = RPYToRot( phi, theta, psi)

        wHb = np.r_[np.c_[rot,origin], np.array([[0, 0, 0, 1]])]
        quadBodyFrame = params.body_frame.T
        quadWorldFrame = wHb.dot(quadBodyFrame)
        world_frame = quadWorldFrame[0:3]
        return world_frame

    def position(self):
        return self.state[0:3]

    def velocity(self):
        return self.state[3:6]

    def attitude(self):
        return self.state[6:9]

    def omega(self):
        return self.state[10:12]




    def state_dot(self, F, M):

        x, y, z, xdot, ydot, zdot, phi, theta, psi, wx, wy, wz = self.state

        # acceleration - Newton's second law of motion
        wRb = RPYToRot(phi, theta, psi) #from body to world rotation matrix
        accel = -params.g*params.e3 + (F/params.mass)*np.dot(wRb.T,params.e3)
        omega = np.array([wx, wy, wz])
        # euler angle rates
        eulerdot = np.dot(Eulermatrix(phi, theta, psi),omega.reshape(3,1))
        # Angular acceleration
        omegadot = params.invI.dot( M.flatten() - np.cross(omega, params.I.dot(omega)) )

        state_dot = np.zeros(12)
        state_dot[0]  = xdot
        state_dot[1]  = ydot
        state_dot[2]  = zdot
        state_dot[3]  = accel[0][0]
        state_dot[4]  = accel[1][0]
        state_dot[5]  = accel[2][0]
        state_dot[6]  = eulerdot[0]
        state_dot[7]  = eulerdot[1]
        state_dot[8]  = eulerdot[2]
        state_dot[9] = omegadot[0]
        state_dot[10] = omegadot[1]
        state_dot[11] = omegadot[2]
        self.statedot = state_dot

        return state_dot


    def update(self, dt, F, M):
        '''SDE update model
        The form is dx = f(x,u)dt + Gdw'''
        x, y, z, xdot, ydot, zdot, phi, theta, psi, wx, wy, wz = self.state
        wRb = RPYToRot(phi, theta, psi) # body to wold rotation matrix
        # random noise
        W =  np.random.normal(0, np.sqrt(dt)*dt, 4).reshape(4,1)

        Rm = (1.0 / params.mass) * np.dot(wRb, params.e3)

        G = np.array([[0, 0, 0, 0],
                      [0 ,0 ,0, 0],
                      [0, 0, 0, 0],
                      [Rm[0][0], 0, 0, 0],
                      [0, Rm[1][0], 0, 0 ],
                      [0, 0, Rm[2][0], 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, params.invI[0][0], 0, 0],
                      [0, 0, params.invI[1][1], 0],
                      [0, 0, 0, params.invI[2][2]],
                      ])


        self.state = self.state + dt * self.state_dot(F,M) +  G.dot(W).T[0]
