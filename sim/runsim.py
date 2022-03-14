from display.GUI_quadcopter import plot_quad_3d
from control import LQG
from model.quadcopter import Quadcopter
import numpy as np
from estimator.extendKalmanFilter import extendKalmanFilter
from model.sensor import Sensor
from EM_parameter_estimation.em_est import em_estimation
from sim.result_plot import Plotresult
from utils.check_distance import distance
import json


animation_frequency = 50
control_frequency = 200 #200  # Hz for attitude control loop
control_iterations = control_frequency / animation_frequency
dt = 1.0 / control_frequency
time = [0.0]
k = [0]
# variables to plot
F_t = list()  # Thrust
M_t = list()  # Torque
t_s = list()  # simulation time
k_s = list()

state_his = list() # observation
Mass_his = list()
Ixx_his = list()
Iyy_his = list()
Izz_his = list()

extendkalmanstate = list()
truestate = list()

def attitudeControl(quad, extendkalmanfilter, sensor, time, k, targetpoints,dyaw,I,mass):
    F, M = LQG.controller(quad, extendkalmanfilter, sensor, I,mass,targetpoints,dyaw,dt)
    q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11 = quad.state
    state_his.append(list(sensor.Y_obs3().reshape(1, 12)[0]))
    quad.update(dt, F, M)
    time[0] += dt
    k[0] +=1
    print("k",k)
    # save variables to graph later
    m1, m2, m3 = M
    k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11 = extendkalmanfilter.state_est
    extendkalmanstate.append((k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11))
    truestate.append((q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11))
    print("time",time[0])
    F_t.append(F)
    M_t.append((m1[0], m2[0], m3[0]))
    t_s.append(time[0])
    k_s.append(k[0])




def main():
    waypoints = np.array([[0.5, 1, 0], [4, 3, 3], [3, 5, 4], [6, 4, 5],[4,3,4],[2,1,5]])
    yaw = np.array([[0], [0], [0], [0],[0],[0]])
    targetpoints = waypoints[1]
    dyaw = yaw[1]
    pos = waypoints[0]
    attitude = (0, 0, 0)
    extendkalmanfilter = extendKalmanFilter()
    quadcopter = Quadcopter(pos, attitude)
    sensor = Sensor()
    ii = 1

    a0 = 0.0001
    a1 = 0.0002
    a2 = 0.0001
    I = np.array([[a0, 0, 0],
                  [0, a1, 0],
                  [0, 0, a2]])
    invI = np.array([[1 / a0, 0, 0],
                     [0, 1 / a1, 0],
                     [0, 0, 1 / a2]])
    mass = [0.001]#np.random.normal(0.18, np.sqrt(0.5))
    def loop(i):
        nonlocal I
        nonlocal invI
        nonlocal mass
        nonlocal ii
        nonlocal targetpoints
        nonlocal dyaw
        if distance(quadcopter.position(), targetpoints) < 0.1:
            ii = ii + 1
            if ii < len(waypoints):
                targetpoints = waypoints[ii]
                dyaw = yaw[ii]
            else: ii = ii -1

        else:
            targetpoints = targetpoints

        for _ in range(4):
           attitudeControl(quadcopter, extendkalmanfilter, sensor, time, k, targetpoints, dyaw, I,mass[0])
        Ixx_his.append(I[0][0])
        Iyy_his.append(I[1][1])
        Izz_his.append(I[2][2])
        Mass_his.append(mass[0])
        I,invI,mass = em_estimation(state_his[-800:], F_t[-800:], M_t[-800:], I, invI,mass[0],dt)
        print("mass",mass[0])
        print("I",I)
        return quadcopter.world_frame()

    plot_quad_3d(waypoints, loop)
    Plotresult(Mass_his,Ixx_his,Iyy_his,Izz_his,extendkalmanstate,truestate,t_s,k_s)
    if (True):  # save inputs and states graphs
        print("Saving figures...")
        with open('Ixx_his.txt', 'w') as outfile:
            data = {}
            data = Ixx_his
            json.dump(data, outfile)
        with open('Iyy_his.txt', 'w') as outfile:
            data = {}
            data = Iyy_his
            json.dump(data, outfile)
        with open('Izz_his.txt', 'w') as outfile:
            data = {}
            data = Izz_his
            json.dump(data, outfile)
        with open('Mass_his.txt', 'w') as outfile:
            data = {}
            data = Mass_his
            json.dump(data, outfile)
        # plt.savefig('test')

    print("Closing.")


if __name__ == "__main__":

    main()
