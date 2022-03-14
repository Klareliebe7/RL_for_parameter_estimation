import numpy as np

import matplotlib.pyplot as plt
def Plotresult(Mass,Ixx,Iyy,Izz,extendkalmanstate,realstate,t,ks):
    '''Plot the results from simulation. '''
    steps = len(Mass)
    Mass_his = np.zeros((steps, 1))
    Ixx_his = np.zeros((steps, 1))
    Iyy_his = np.zeros((steps, 1))
    Izz_his = np.zeros((steps, 1))

    for k in range(steps):
        mass = Mass[k]
        Ixx_his[k] = Ixx[k]
        Iyy_his[k] = Iyy[k]
        Izz_his[k] = Izz[k]
        Mass_his[k] = mass

    Mass_real = [0.18]*len(Mass_his)
    Ixx_real = [0.00025]*len(Ixx_his)
    Iyy_real = [0.00031]*len(Iyy_his)
    Izz_real = [0.00020]*len(Izz_his)
    # Plot Estimated Mass result
    x = np.arange(steps)
    plt.figure()
    plt.plot(x, Mass_his, color="r", linestyle="--", linewidth=2.0, label='Estimated mass')
    plt.plot(x, Mass_real, color="r", linestyle="-", linewidth=2.0, label='Real mass')
    plt.grid(linestyle='-.')
    plt.legend(loc='best')
    plt.title('Estimated mass',fontsize=25,verticalalignment='bottom')
    plt.xlabel('Iterations',fontsize=20)
    plt.ylabel('mass (kg)',fontsize=20)
    plt.tick_params(labelsize=20)
    show_min = '[' + str(steps-2) + ' , ' + str(Mass_his[-1]) + ']'
    plt.annotate(show_min, xytext=(steps - 250, Mass_his[-1] + 0.003), xy=(steps, Mass_his[-1]),fontsize=20)
    plt.plot(steps, Mass_his[-1], 'gs')
    # Plot Estimated Inertia Matrix result
    plt.figure()
    plt.plot(x, Ixx_his, color="blue", linestyle="--", linewidth=2.0, label='Estimated Ixx')
    plt.plot(x, Ixx_real, color="blue", linestyle="-", linewidth=2.0, label='Real Ixx')
    plt.grid(linestyle='-.')
    show_min = '[' + str(steps-2) + ' , ' + str(Ixx_his[-1]) + ']'
    plt.annotate(show_min, xytext=(steps - 250, Ixx_his[-1] + 0.00003), xy=(steps, Ixx_his[-1]),fontsize=20)
    plt.plot(steps, Ixx_his[-1], 'gs')
    plt.plot(x, Iyy_his, color="black", linestyle="--", linewidth=2.0, label='Estimated Iyy')
    plt.plot(x, Iyy_real, color="black", linestyle="-", linewidth=2.0, label='Real Iyy')
    show_min = '[' + str(steps - 2) + ' , ' + str(Iyy_his[-1]) + ']'
    plt.annotate(show_min, xytext=(steps - 250, Iyy_his[-1] + 0.00003), xy=(steps, Iyy_his[-1]),fontsize=20)
    plt.plot(steps, Iyy_his[-1], 'bs')
    plt.plot(x, Izz_his, color="red", linestyle="--", linewidth=2.0, label='Estimated Izz')
    plt.plot(x, Izz_real, color="red", linestyle="-", linewidth=2.0, label='Real Izz')
    plt.title('Estimated Inertia matrix',fontsize=25)
    plt.xlabel('Iterations',fontsize=20)
    plt.ylabel('I ( kg·m²)',fontsize=20)
    show_min = '[' + str(steps - 2) + ' , ' + str(Izz_his[-1]) + ']'
    plt.annotate(show_min, xytext=(steps - 250, Izz_his[-1] - 0.00004), xy=(steps, Izz_his[-1]),fontsize=20)
    plt.plot(steps, Izz_his[-1], 'rs')
    plt.legend(loc='best')
    plt.tick_params(labelsize=20)

# plot result of EKF

    # the error between kalmanstate and realstate
    err0 = []
    err1 = []
    err2 = []
    err3 = []
    err4 = []
    err5 = []
    err6 = []
    err7 = []
    err8 = []
    err9 = []
    err10 = []
    err11 = []
    # kalmanstate
    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []
    phi = []
    theta = []
    psi = []
    wx = []
    wy = []
    wz = []
    # realstate
    rx = []
    ry = []
    rz = []
    rvx = []
    rvy = []
    rvz = []
    rphi = []
    rtheta = []
    rpsi = []
    rwx = []
    rwy = []
    rwz = []

    Y = len(extendkalmanstate)

    for k in range(Y):
        err0.append(np.abs(extendkalmanstate[k][0] - realstate[k][0]))
        err1.append(np.abs(extendkalmanstate[k][1] - realstate[k][1]))
        err2.append(np.abs(extendkalmanstate[k][2] - realstate[k][2]))
        err3.append(np.abs(extendkalmanstate[k][3] - realstate[k][3]))
        err4.append(np.abs(extendkalmanstate[k][4] - realstate[k][4]))
        err5.append(np.abs(extendkalmanstate[k][5] - realstate[k][5]))
        err6.append(np.abs(extendkalmanstate[k][6] - realstate[k][6]))
        err7.append(np.abs(extendkalmanstate[k][7] - realstate[k][7]))
        err8.append(np.abs(extendkalmanstate[k][8] - realstate[k][8]))
        err9.append(np.abs(extendkalmanstate[k][9] - realstate[k][9]))
        err10.append(np.abs(extendkalmanstate[k][10] - realstate[k][10]))
        err11.append(np.abs(extendkalmanstate[k][11] - realstate[k][11]))

        x.append(extendkalmanstate[k][0])
        y.append(extendkalmanstate[k][1])
        z.append(extendkalmanstate[k][2])
        vx.append(extendkalmanstate[k][3])
        vy.append(extendkalmanstate[k][4])
        vz.append(extendkalmanstate[k][5])
        phi.append(extendkalmanstate[k][6])
        theta.append(extendkalmanstate[k][7])
        psi.append(extendkalmanstate[k][8])
        wx.append(extendkalmanstate[k][9])
        wy.append(extendkalmanstate[k][10])
        wz.append(extendkalmanstate[k][11])
        rx.append(realstate[k][0])
        ry.append(realstate[k][1])
        rz.append(realstate[k][2])
        rvx.append(realstate[k][3])
        rvy.append(realstate[k][4])
        rvz.append(realstate[k][5])
        rphi.append(realstate[k][6])
        rtheta.append(realstate[k][7])
        rpsi.append(realstate[k][8])
        rwx.append(realstate[k][9])
        rwy.append(realstate[k][10])
        rwz.append(realstate[k][11])
    # Plot Estimated state from EKF
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.title('Estimated state from EKF', fontsize=25)
    plt.plot(ks, x, color="black", linestyle="-", linewidth=2.0, label='x')
    plt.plot(ks, y, color="green", linestyle="-", linewidth=2.0, label='y')
    plt.plot(ks, z, color="blue", linestyle="-", linewidth=2.0, label='z')
    plt.grid(linestyle='-.')
    plt.legend(['x', 'y', 'z'], loc='upper right')
    plt.ylabel('Pos(m)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplot(4, 1, 2)
    plt.plot(ks, vx, color="black", linestyle="-", linewidth=2.0, label='vx')
    plt.plot(ks, vy, color="green", linestyle="-", linewidth=2.0, label='vy')
    plt.plot(ks, vz, color="blue", linestyle="-", linewidth=2.0, label='vz')
    plt.grid(linestyle='-.')
    plt.legend(['Vx', 'Vy', 'Vz'], loc='upper right')
    plt.ylabel('Vel (m/s)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplot(4, 1, 3)
    plt.plot(ks, phi, color="black", linestyle="-", linewidth=2.0)
    plt.plot(ks, theta, color="green", linestyle="-", linewidth=2.0)
    plt.plot(ks, psi, color="blue", linestyle="-", linewidth=2.0)
    plt.grid(linestyle='-.')
    plt.legend(['phi', 'theta', 'psi'], loc='upper right')
    plt.ylabel('Euler angle(rad)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.subplot(4, 1, 4)
    plt.plot(ks, wx, color="black", linestyle="-", linewidth=2.0)
    plt.plot(ks, wy, color="green", linestyle="-", linewidth=2.0)
    plt.plot(ks, wz, color="blue", linestyle="-", linewidth=2.0)
    plt.grid(linestyle='-.')
    plt.legend(['wx', 'wy', 'wz'], loc='upper right')
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('w(rad/s)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # Plot True state from quadcopter
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.title('True state from quadcopter', fontsize=20)

    plt.plot(ks, rx, color="black", linestyle="-", linewidth=2.0, label='x')
    plt.plot(ks, ry, color="green", linestyle="-", linewidth=2.0, label='y')
    plt.plot(ks, rz, color="blue", linestyle="-", linewidth=2.0, label='z')
    plt.grid(linestyle='-.')
    plt.legend(['x', 'y', 'z'], loc='upper right')

    plt.ylabel('Pos(m)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(4, 1, 2)

    plt.plot(ks, rvx, color="black", linestyle="-", linewidth=2.0, label='vx')
    plt.plot(ks, rvy, color="green", linestyle="-", linewidth=2.0, label='vy')
    plt.plot(ks, rvz, color="blue", linestyle="-", linewidth=2.0, label='vz')
    plt.grid(linestyle='-.')
    plt.legend(['Vx', 'Vy', 'Vz'], loc='upper right')

    plt.ylabel('Vel (m/s)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(4, 1, 3)

    plt.plot(ks, rphi, color="black", linestyle="-", linewidth=2.0)
    plt.plot(ks, rtheta, color="green", linestyle="-", linewidth=2.0)
    plt.plot(ks, rpsi, color="blue", linestyle="-", linewidth=2.0)
    plt.grid(linestyle='-.')
    plt.legend(['phi', 'theta', 'psi'], loc='upper right')

    plt.ylabel('Euler angle(rad)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(4, 1, 4)

    plt.plot(ks, rwx, color="black", linestyle="-", linewidth=2.0)
    plt.plot(ks, rwy, color="green", linestyle="-", linewidth=2.0)
    plt.plot(ks, rwz, color="blue", linestyle="-", linewidth=2.0)
    plt.grid(linestyle='-.')
    plt.legend(['wx', 'wy', 'wz'], loc='upper right')
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('w(rad/s)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # Plot Error between estimated state and true state
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.title('Error between estimated state and true state', fontsize=25)

    plt.plot(ks, err0, color="black", linestyle="-", linewidth=2.0, label='x')
    plt.plot(ks, err1, color="green", linestyle="-", linewidth=2.0, label='y')
    plt.plot(ks, err2, color="blue", linestyle="-", linewidth=2.0, label='z')
    plt.grid(linestyle='-.')
    plt.legend(['x', 'y', 'z'], loc='upper right')

    plt.ylabel('Pos(m)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(4, 1, 2)

    plt.plot(ks, err3, color="black", linestyle="-", linewidth=2.0, label='vx')
    plt.plot(ks, err4, color="green", linestyle="-", linewidth=2.0, label='vy')
    plt.plot(ks, err5, color="blue", linestyle="-", linewidth=2.0, label='vz')
    plt.grid(linestyle='-.')
    plt.legend(['Vx', 'Vy', 'Vz'], loc='upper right')

    plt.ylabel('Vel (m/s)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(4, 1, 3)

    plt.plot(ks, err6, color="black", linestyle="-", linewidth=2.0)
    plt.plot(ks, err7, color="green", linestyle="-", linewidth=2.0)
    plt.plot(ks, err8, color="blue", linestyle="-", linewidth=2.0)
    plt.grid(linestyle='-.')
    plt.legend(['phi', 'theta', 'psi'], loc='upper right')

    plt.ylabel('Euler angle(rad)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.subplot(4, 1, 4)

    plt.plot(ks, err9, color="black", linestyle="-", linewidth=2.0)
    plt.plot(ks, err10, color="green", linestyle="-", linewidth=2.0)
    plt.plot(ks, err11, color="blue", linestyle="-", linewidth=2.0)
    plt.grid(linestyle='-.')
    plt.legend(['wx', 'wy', 'wz'], loc='upper right')
    plt.xlabel('Steps', fontsize=20)
    plt.ylabel('w(rad/s)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show()


