import gym
from gym import spaces
import numpy as np
from display.GUI_quadcopter import plot_quad_3d
from control import LQG
from model.quadcopter import Quadcopter
from estimator.extendKalmanFilter import extendKalmanFilter
from model.sensor import Sensor
from EM_parameter_estimation.em_est import em_estimation
from sim.result_plot import Plotresult
from utils.check_distance import distance
import json
 
class Quadrotor_Env(gym.Env):
    """
    ## Description
    The inverted pendulum swingup problem is a classic problem in the control literature. In this
    version of the problem, the pendulum starts in a random position, and the goal is to swing it up so
    it stays upright.
    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.
    ![Pendulum Coordinate System](./diagrams/pendulum.png)
    - `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta`: angle in radians.
    - `tau`: torque in `N * m`. Defined as positive _counter-clockwise_.
    ## Action Space
    The action is the torque applied to the pendulum.
    | Num | Action |   Min   |   Max    |
    |------- |----------|-------- |--------- |
    | 0   |    m   |  0.001  |  0.3     |
    | 1   |    a0  |   0     |  0.0005  |	
    | 1   |    a1  |   0     |  0.0005  |	
    | 1   |    a2  |   0     |  0.0005  |	
    ## Observation Space
    The observations correspond to the x-y coordinate of the pendulum's end, and its angular velocity.
    | Num | Observation      |  Min   |  Max  |
    |-----|------------------|--------|-------|
    |   1   | x_o      |   0    |   7   |
    |   2   | y_o      |   0    |   7   |
    |   3   | z_o      |   0    |   7   |
    |   4   | Vx_o     |  -1.5  |   3   |
    |   5   | Vy_o     |  -1.5  |   3   |
    |   6   | Vz_o     |  -1.5  |   3   |
    |   7   | phi_o    |  -1.0  |   1   |
    |   8   | theta_o  |  -1.0  |   1   |
    |    9   | psi_o    |  -1.0  |   1   |
    | 10  | Wx_o     |  -20   |  20   |
    | 11  | Wy_o     |  -20   |  20   |
    | 12  | Wz_o     |  -20   |  20   |
    | 13  | m1          |   0    |   7   |
    | 14  | m2       |   0    |   7   |
    | 15  | m3       |   0    |   7   |
    | 16  | F            |   0    |   7   |
    #| 13  | Xr(target x)     |   0    |   7   |
   # | 14  | Yr(target y)     |   0    |   7   |
    #| 15  | Zr(target z)     |   0    |   7   |
 
    ## Rewards
    The reward is defined as:
    ```
    r = -log((x - Xr)^2+(y - Yr)^2+(z - Zr)^2 )-1
    ```
    
    ## Starting State
    The starting state is a random angle in `[-pi, pi]` and a random angular velocity in `[-1,1]`.
    ## Episode Termination
    An episode terminates after 200 steps. There's no other criteria for termination.
    ## Arguments
    - `g`: acceleration of gravity measured in `(m/s^2)` used to calculate the pendulum dynamics. The default is
    `g=10.0`.
    ```
    gym.make('CartPole-v1', g=9.81)
    ```
    ## Version History
    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release (1.0.0)
    """

    #metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}
    def __init__(self):

    	# initial parameters
        self.max_steps = 1000
        self.step_ctr = 0
        self.animation_frequency = 50
        self.control_frequency = 200
        self.control_iterations = self.control_frequency /  self.animation_frequency
        self.dt = 1.0 /  self.control_frequency
        self.time = [0.0]
        self.k = [0]
        #self.waypoints = np.array([[0.5, 1, 0], [4, 3, 3], [3, 5, 4], [6, 4, 5],[4,3,4],[2,1,5]])
        self.yaw = np.array([[0], [0], [0], [0],[0],[0]])
        self.targetpoint =  np.random.rand(3)*10# 
        self.startpoint = np.array([5, 5, 5])#
        self.dyaw =  self.yaw[1]
        self.pos =  self.startpoint
        self.attitude = (0, 0, 0)
        #self.extendkalmanfilter = extendKalmanFilter()
        self.quadcopter = Quadcopter( self.pos,  self.attitude)
        self.sensor = Sensor()
        #self.ii = 1
        self.a0 = 0.0001
        self.a1 = 0.0002
        self.a2 = 0.0001
        self.I = np.array([[ self.a0, 0, 0],
                                            [0,  self.a1, 0],
                                            [0, 0,  self.a2]])
        self.invI = np.array([[1 /  self.a0, 0, 0],
                                                [0, 1 /  self.a1, 0],
                                                [0, 0, 1 /  self.a2]])
        self.mass = [0.001]#np.random.normal(0.18, np.sqrt(0.5))
        self.viewer = None
        self.system_input = np.array([0,0,0,0])


# variables to plot
        self.F_t = list()  # Thrust
        self.M_t = list()  # Torque
        self.t_s = list()  # simulation time
        self.k_s = list()

        self.state_his = list() # observation
        self.Mass_his = list()
        self.Ixx_his = list()
        self.Iyy_his = list()
        self.Izz_his = list()



	# continious action space
        mass_max = 0.3
        mass_min = 0.001
        self.action_space = spaces.Box(
            low=np.array([0.001,0.0,0.0,0.0], dtype=np.float32), high=np.array([0.3, 0.0005, 0.0005, 0.0005], dtype=np.float32), shape=(4,), dtype=np.float32)
    
# continious observation space
        obs_max = np.array([10.0,10.0,10.0,
                                                    3.0,3.0,3.0,
                                                    1.0,1.0,1.0,
                                                    20.0,20.0,20.0,
                                                    10.0,10.0,10.0,
                                                    10], dtype=np.float32)
        obs_min = np.array([0.0,0.0,0.0,
                                                    -1.5,-1.5,-1.5,
                                                    -1.0,-1.0,-1.0,
                                                    -20.0,-20.0,-20.0,
                                                    0.0,0.0,0.0,
                                                    0.0], dtype=np.float32)   
        self.observation_space = spaces.Box(low=obs_min, high=obs_max, dtype=np.float32)
        self.state = np.array([0.0,0.0,0.0,
                                                    0.0,0.0,0.0,
                                                    0.0,0.0,0.0,
                                                    0.0,0.0,0.0,
                                                    0.0,0.0,0.0,
                                                    0.0], dtype=np.float32)
	
    def step(self, estimation):
        """estimations are m, a0, a1, a2"""
        done = False
        m, a0, a1, a2 = estimation[0],estimation[1],estimation[2],estimation[3]
        reward = 0
        self.I = np.array([[a0, 0, 0],
                                        [0, a1, 0],
                                        [0, 0, a2]])
        self.invI = np.array([[1 / a0, 0, 0],
	     [0, 1 / a1, 0],
	     [0, 0, 1 / a2]])
        self.mass[0] = m
        # save the I and m
        self.Ixx_his.append(a0)
        self.Iyy_his.append(a1)
        self.Izz_his.append(a2)
        self.Mass_his.append(m)
        if self.step_ctr%10 == 0:
            print("filter(lambda position: position <-0.5 and postion> 10.5, self.quadcopter.position())")
            print(self.quadcopter.position())
            print(list(filter(lambda position: position <-0.5 or position> 10.5, self.quadcopter.position())))
        
     	# went out of boundary
        if list(filter(lambda position: position <-0.5 or position> 10.5, self.quadcopter.position())) != []:
            reward+= -1000000
            done = True
        # reached a target point
        if  self.step_ctr > self.max_steps:
            done = True
            reward-=1000000
        if distance(self.quadcopter.position(), self.targetpoint) < 0.1:
            #self.ii = self.ii + 1
            reward += 1000000
            done = True
            #if self.ii < len(self.waypoints):
               # self.targetpoint = self.waypoints[self.ii]
                #self.dyaw = self.yaw[self.ii]
                #done = False
            #else: 
                #self.ii = self.ii -1#?????
                #done = True
        #else:
            #self.targetpoint = self.targetpoint
            #done = False
        for _ in range(4):
            F,M = attitudeControl(self.quadcopter, self.sensor, self.time, self.k, self.targetpoint, self.dyaw, self.I, self.mass[0])
        #x_o, y_o,z_o,Vx_o,Vy_o,Vz_o,phi_o,theta_o,psi_o,Wx_o,Wy_o,Wz_o = self.sensor.Y_obs2().reshape(12,)
        #m1, m2, m3 = M
        #print(M)
        m1 = M[0][0]
        m2 = M[1][0]
        m3 = M[2][0]
        reward += -np.log(distance(self.sensor.position_noisy(), self.targetpoint)) - 1
        self.state = self.sensor.Y_obs2().reshape(12,)
        self.step_ctr += 1
        self.system_input = np.array([m1, m2, m3, F])
        if m == float('nan'):
            done = True
        # placeholder for information
        info = {}
        #print(f"self.state{np.hstack((self.state[0], self.targetpoint))}")
        #print(f"self.targetpoint{self.targetpoint}")
        #sprint(f"done=============================> {done}")
        #print(f"reward{reward}")
        #print(f"info{info}")
        
        return np.hstack((self.state , self.system_input )), reward, done, info
    
    
    
    
    def reset(self):
        self.step_ctr = 0
        self.reward = 0
        self.state = self.state = np.array([5, 5, 5,
		    0.0001,0.0001,0.0001,
		    0.0001,0.0001,0.0001,
		    0.0001,0.0001,0.0001], dtype=np.float32)
        self.time = [0.0]
        self.k = [0]
        #self.waypoints = np.array([[0.5, 1, 0], [4, 3, 3], [3, 5, 4], [6, 4, 5],[4,3,4],[2,1,5]])
        self.yaw = np.array([[0], [0], [0], [0],[0],[0]])
        self.targetpoint =  np.random.rand(3)*10
        self.startpoint = np.array([5, 5, 5])#
        self.dyaw = self.yaw[1]
        self.pos = self.startpoint
        self.attitude = (0, 0, 0)

        self.quadcopter = Quadcopter(self.pos, self.attitude)
        self.sensor = Sensor()
        #self.ii = 1
        self.a0 = 0.0001
        self.a1 = 0.0002
        self.a2 = 0.0001
        self.I = np.array([[self.a0, 0, 0],
                                            [0, self.a1, 0],
                                            [0, 0, self.a2]])
        self.invI = np.array([[1 / self.a0, 0, 0],
	     [0, 1 / self.a1, 0],
	     [0, 0, 1 / self.a2]])
        self.mass = [0.001]#np.random.normal(0.18, np.sqrt(0.5))
        self.system_input = np.array([0,0,0,0])
        self.viewer = None
        return np.hstack((self.state, self.system_input))


    def render(self):
        pass
        
	
def attitudeControl(quad,sensor, time, k, targetpoint,dyaw,I,mass):
    control_frequency = 200
    dt = 1.0 /   control_frequency
    F, M = LQG.controller(quad,  sensor, I,mass,targetpoint,dyaw,dt)
    #q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11 = quad.state
    #state_his.append(list(sensor.Y_obs3().reshape(1, 12)[0])) 暂时关闭
    quad.update(dt, F, M)
    time[0] += dt
    k[0] +=1
    #print("k",k)
    # save variables to graph later

    #m1, m2, m3 = M
    
    #k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11 = extendkalmanfilter.state_est
    #extendkalmanstate.append((k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11))暂时关闭
    #truestate.append((q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11))暂时关闭
    #print("time",time[0])
    return F,M
   
   



