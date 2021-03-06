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
 
class Quadrotor_Env_discrete(gym.Env):
    """
    ## Description
    Parameter estimation is not a frequently used field for reinforcement learning, it will be tested in this enviorment. It is asssumpted, that the centroid will not change as the mass of the quadrotor changes. And here only the mass are going to be estimated, and the inertial matrix will change correspondingly.
 
    ## Action Space
    The action is the torque applied to the pendulum.
    | Num | Action |    
    |------- |----------| 
    | 1  |    Hold   |  
    | 2   |    ascend  | 
    | 3   |    descend  |   
    
    ## Observation Space
    The observations correspond to the x-y coordinate of the pendulum's end, and its angular velocity.
    | Num | Observation      |  Min   |  Max  |
    |-----|------------------|--------|-------|

    |   1   | x_o     |   0    |   7   |
    |   2   | y_o     |   0    |   7   |
    |   3   | z_o     |   0    |   7   |
    |   4   | Vx_o     |  -1.5  |   3   |
    |   5   | Vy_o     |  -1.5  |   3   |
    |   6   | Vz_o     |  -1.5  |   3   |
    |   7   | phi_o    |  -1.0  |   1   |
    |   8   | theta_o  |  -1.0  |   1   |
    |   9   | psi_o    |  -1.0  |   1   |
    | 10  | Wx_o     |  -20   |  20   |
    | 11  | Wy_o     |  -20   |  20   |
    | 12  | Wz_o     |  -20   |  20   |
    | 13  | m1          |   0    |   7   |
    | 14  | m2       |   0    |   7   |
    | 15  | m3       |   0    |   7   |
    | 16  | F            |   0    |   7   |
    |   17   | Xr(target x)-x_o      |   0    |   7   | target direction to quad
    |   18   | Yr(target y)-y_o      |   0    |   7   |
    |   19   | Zr(target z)-z_o      |   0    |   7   |
    |   20   | Xr(target x)-Xs      |   0    |   7   | target direction to start point
    |   21   | Yr(target y)-Ys      |   0    |   7   |
    |   22   | Zr(target z)-Ys      |   0    |   7   | 

 
    ## Rewards
    The reward is defined as:
    ```
    r = -log((x - Xr)^2+(y - Yr)^2+(z - Zr)^2 )-1
    ```
    
    ## Starting State
     
    ## Episode Termination
    An episode terminates after 1000 steps. 
    or other 
    
    ## Version History
    * v1:  
    * v0:   (1.0.0)
    """
 
    def __init__(self):

    	# initial parameters
    	
        self.mission = "hovering"
        if self.mission == "hovering":
            self.max_steps = 1000
            self.reward_list = []
        elif self.mission == "approching":
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
        
        self.startpoint = np.array([5, 5, 5])#
        if self.mission == "hovering":
            self.targetpoint =  self.startpoint  
        elif self.mission == "approching":
            self.targetpoint =  np.array([6, 6, 9])#np.random.rand(3)*10# 
        
        self.dyaw =  self.yaw[1]
        self.pos =  self.startpoint
        self.attitude = (0, 0, 0)
        #self.extendkalmanfilter = extendKalmanFilter()
        ###
        self.mass_estimate = 0.36
    	###
        self.mass_variant_param = np.random.rand()*2+1
        self.real_mass = 0.18*self.mass_variant_param
        self.real_I = np.array([(0.00025*self.mass_variant_param, 0, 0),(0, 0.00031*self.mass_variant_param, 0),(0, 0, 0.00020*self.mass_variant_param)])
        self.quadcopter = Quadcopter(self.pos, self.attitude,self.real_mass,self.real_I )
        self.sensor = Sensor()
        #self.ii = 1
        self.a0 = 0.0002
        self.a1 = 0.0003	
        self.a2 = 0.0002
        self.I = np.array([[ self.a0, 0, 0],
                                            [0,  self.a1, 0],
                                            [0, 0,  self.a2]], dtype=np.float32)
        self.invI = np.array([[1 /  self.a0, 0, 0],
                                                [0, 1 /  self.a1, 0],
                                                [0, 0, 1 /  self.a2]], dtype=np.float32)
        self.mass = [0.2]#np.random.normal(0.18, np.sqrt(0.5))
        self.viewer = None
        self.system_input = np.array([0,0,0,0])
        self.target_startpoint_vector =  self.targetpoint  -self.startpoint

# variables to plot
        self.F_t = list()  # Thrust
        self.M_t = list()  # Torque
        self.t_s = list()  # simulation time
        self.k_s = list()
            
        self.state_his = list() # observation
        self.Mass_his = list() # estimation
        self.Ixx_his = list()
        self.Iyy_his = list()
        self.Izz_his = list()



	# discrete action space
        self.action_space = spaces.Discrete(3)
# continious observation space
        obs_max = np.array([10.0,10.0,10.0,
                                                    3.0,3.0,3.0,
                                                    1.0,1.0,1.0,
                                                    20.0,20.0,20.0,
                                                    10.0,10.0,10.0,
                                                    10,10.0,10.0,10.0,
                                                    10.0,10.0,10.0], dtype=np.float32)
        obs_min = np.array([0.0,0.0,0.0,
                                                    -1.5,-1.5,-1.5,
                                                    -1.0,-1.0,-1.0,
                                                    -20.0,-20.0,-20.0,
                                                    0.0,0.0,0.0,
                                                    0.0,0.0,0.0,0.0,
                                                    0.0,0.0,0.0], dtype=np.float32)   
        self.observation_space = spaces.Box(low=obs_min, high=obs_max, dtype=np.float32)
        self.state = np.array([0.0,0.0,0.0,
                                                    0.0,0.0,0.0,
                                                    0.0,0.0,0.0,
                                                    0.0,0.0,0.0], dtype=np.float32)
	
    def step(self, action):
        """action is one of hold ascend descend."""
        
        done = False
        reward = 0
        self.mass_estimate += 0.01*(action-1)
        self.Mass_his.append(self.mass_estimate )       
        ratio = self.mass_estimate/0.18
        I = np.array([(0.00025, 0, 0),
             (0, 0.00031, 0),
             (0, 0, 0.00020)], dtype=np.float32)
        self.I = I*ratio
        invI = np.linalg.inv(I)
        self.invI = invI/ratio
        self.mass[0] =self.mass_estimate

        info = {}#f"mass error: {100*(sum(Mass_his)/len(Mass_his)-self.real_mass)/self.real_mass:.3f}%"
        info["msg"] = ""
        info["result"] = False
        info["real_mass"] = self.real_mass
        info["estimated_mass_ave"] = 0
        self.target_quad_vector = self.targetpoint - self.sensor.position_noisy()
        two_directionvectors = np.hstack((self.target_quad_vector  , self.target_startpoint_vector ))
         
        

        """
            for mission I :
            approching target point.
        """
        if self.mission == "approching":
            #calculate error                                                                                         
            if self.mass_estimate  < -0.1:# or a0<0 or a1<0 or a2<0:
                reward+= -10000
                done = True
                info["msg"] = f"calculation error"
                info["result"] = False
                info["real_mass"] = self.real_mass
                info["estimated_mass_ave"] = sum(self.Mass_his)/len(self.Mass_his)
                states_and_input = np.hstack((self.state , self.system_input ))             
                observation = np.hstack((states_and_input , two_directionvectors)) 
                return observation, reward, done, info

            # went out of boundary
            if list(filter(lambda position: position <-0.5 or position> 10.5, self.quadcopter.position())) != []:
                reward+= -10000
                done = True
                info["msg"]= f"went out of boundary{self.quadcopter.position()}"
                info["result"] = False
                info["real_mass"] = self.real_mass
                info["estimated_mass_ave"] = sum(self.Mass_his)/len(self.Mass_his)
                states_and_input = np.hstack((self.state , self.system_input ))             
                observation = np.hstack((states_and_input , two_directionvectors)) 
                return observation, reward, done, info
            #too much steps
            if  self.step_ctr > self.max_steps:
                done = True
                reward-=10000
                info["msg"]= f"too much steps"
                info["result"] = False
                info["real_mass"] = self.real_mass
                info["estimated_mass_ave"] = sum(self.Mass_his)/len(self.Mass_his)
                states_and_input = np.hstack((self.state , self.system_input ))             
                observation = np.hstack((states_and_input , two_directionvectors)) 
                return observation, reward, done, info
            #reached a target point
            if distance(self.quadcopter.position(), self.targetpoint) < 0.1:
                done = True
                info["msg"]= f"reached a target point!"
                info["result"] = True
                info["real_mass"] = self.real_mass
                info["estimated_mass_ave"] = sum(self.Mass_his)/len(self.Mass_his)
                states_and_input = np.hstack((self.state , self.system_input ))             
                observation = np.hstack((states_and_input , two_directionvectors)) 
                return observation, reward, done, info
        """
        for mission II:
        hovering
        """
        if self.mission == "hovering":
            #calculate error
            if self.mass_estimate < 0: 
                reward+= -10000
                done = True
                info["msg"] = f"calculation error "
                info["result"] = False
                info["real_mass"] = self.real_mass
                info["estimated_mass_ave"] = sum(self.Mass_his)/len(self.Mass_his)
                states_and_input = np.hstack((self.state , self.system_input ))          
                observation = np.hstack((states_and_input , two_directionvectors)) 
                return observation, reward, done, info

            # went out of boundary
            if list(filter(lambda position: position <-0.5 or position> 10.5, self.quadcopter.position())) != []:
                reward+= -10000
                done = True
                info["msg"]= f"went out of boundary{self.quadcopter.position()}"
                info["result"] = False
                info["real_mass"] = self.real_mass
                info["estimated_mass_ave"] = sum(self.Mass_his)/len(self.Mass_his)
                states_and_input = np.hstack((self.state , self.system_input ))             
                observation = np.hstack((states_and_input , two_directionvectors)) 
                return observation, reward, done, info
            # finished iteration
            if  self.step_ctr > self.max_steps:
                done = True
                reward+=10000
                info["msg"]= f"hovering iteration stopping"
                info["result"] = True
                info["real_mass"] = self.real_mass
                info["estimated_mass_ave"] = sum(self.Mass_his)/len(self.Mass_his)
                states_and_input = np.hstack((self.state , self.system_input ))             
                observation = np.hstack((states_and_input , two_directionvectors)) 
                return observation, reward, done, info
            
        for _ in range(4):
            F,M = attitudeControl(self.quadcopter, self.sensor, self.time, self.k, self.targetpoint, self.dyaw, self.I, self.mass[0])
        m1 = M[0][0]
        m2 = M[1][0]
        m3 = M[2][0]
        if self.mission == "hovering":
            reward +=  - distance(self.sensor.position_noisy(), self.targetpoint) **3
            self.reward_list.append(reward)
        elif self.mission == "approching":
            reward += -1
        #reward += -np.log(distance(self.sensor.position_noisy(), self.targetpoint)) - 1#-distance(self.sensor.position_noisy(), self.targetpoint) -1 
        # how many steps 
        self.state = self.sensor.Y_obs2().reshape(12,)
        self.step_ctr += 1
        self.system_input = np.array([m1, m2, m3, F])
        self.target_quad_vector = self.targetpoint - self.sensor.position_noisy()
        states_and_input = np.hstack((self.state , self.system_input ))
        two_directionvectors = np.hstack((self.target_quad_vector  , self.target_startpoint_vector ))
        observation = np.hstack((states_and_input , two_directionvectors))  
        
        print(f"real m:{self.real_mass:.3f} estimated m: {self.mass_estimate:.3f}, act:{action-1} ,ave r:{sum(self.reward_list)/len(self.reward_list):.2f},score:{}")
        return observation, reward, done, info
    
    
    
    
    def reset(self):
        if self.mission == "hovering":
 
            self.reward_list = []
        elif self.mission == "approching":
            pass
        self.mass_estimate = 0.36
        self.step_ctr = 0
        self.reward = 0
        self.state = np.array([5, 5, 5,
		    0.0001,0.0001,0.0001,
		    0.0001,0.0001,0.0001,
		    0.0001,0.0001,0.0001], dtype=np.float32)
        self.time = [0.0]
        self.k = [0]
        #self.waypoints = np.array([[0.5, 1, 0], [4, 3, 3], [3, 5, 4], [6, 4, 5],[4,3,4],[2,1,5]])
        self.yaw = np.array([[0], [0], [0], [0],[0],[0]])
      
        self.startpoint = np.array([5, 5, 5])#
        if self.mission == "hovering":
            self.targetpoint =  self.startpoint 
        elif self.mission == "approching":
            self.targetpoint =  np.array([6, 6, 9])#np.random.rand(3)*10
        self.dyaw = self.yaw[1]
        self.pos = self.startpoint
        self.attitude = (0, 0, 0)
        self.mass_variant_param = np.random.rand()*2+1
        self.real_mass = 0.18*self.mass_variant_param
        self.real_I = np.array([(0.00025*self.mass_variant_param, 0, 0),(0, 0.00031*self.mass_variant_param, 0),(0, 0, 0.00020*self.mass_variant_param)], dtype=np.float32)
        self.quadcopter = Quadcopter(self.pos, self.attitude,self.real_mass,self.real_I )
        self.sensor = Sensor()
        #self.ii = 1
        self.a0 = 0.0001
        self.a1 = 0.0002
        self.a2 = 0.0001
        self.I = np.array([[self.a0, 0, 0],
                                            [0, self.a1, 0],
                                            [0, 0, self.a2]], dtype=np.float32)
        self.invI = np.array([[1 / self.a0, 0, 0],
	     [0, 1 / self.a1, 0],
	     [0, 0, 1 / self.a2]], dtype=np.float32)
        self.mass = [0.001]#np.random.normal(0.18, np.sqrt(0.5))
        self.system_input = np.array([0,0,0,0], dtype=np.float32)
        self.viewer = None
        self.Mass_his = list()
        self.target_startpoint_vector =  self.targetpoint  -self.startpoint
        self.target_quad_vector = self.targetpoint - self.sensor.position_noisy()
        two_directionvectors = np.hstack((self.target_quad_vector  , self.target_startpoint_vector ))
        states_and_input = np.hstack((self.state , self.system_input ))             
        observation = np.hstack((states_and_input , two_directionvectors)) 
        return observation 


    def render(self):
        pass
        
	
def attitudeControl(quad,sensor, time, k, targetpoint,dyaw,I,mass):
    control_frequency = 200
    dt = 1.0 /   control_frequency
    F, M = LQG.controller(quad,  sensor, I,mass,targetpoint,dyaw,dt)
    #q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11 = quad.state
    #state_his.append(list(sensor.Y_obs3().reshape(1, 12)[0])) ????????????
    quad.update(dt, F, M)
    time[0] += dt
    k[0] +=1

    return F,M
   
   



