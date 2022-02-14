from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
history = np.zeros((3000,3))
count = 0
def plot_quad_3d(waypoints, get_world_frame):
    """
    get_world_frame is a function which return the "next" world frame to be drawn
    """
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.plot([], [], [], '-', c='g',markersize=30)[0]
    ax.plot([], [], [], '-', c='brown',markersize=22)[0]
    ax.plot([], [], [], 'o', c='blue', marker='o', markevery=2)[0]
    ax.plot([], [], [], '*', c='red', markersize=8)[0]
    ax.plot([], [], [], '.', c='blue', markersize=1)[0]
    set_limit((0, 7), (-1, 6), (-1, 7))
    plot_waypoints(waypoints)
    ax.set_xlabel('X(m)',fontsize=20)
    ax.set_ylabel('Y(m)',fontsize=20)
    ax.set_zlabel('Z(m)',fontsize=20)
    ax.set_title('Quadcopter',fontsize=25,verticalalignment='bottom')
    plt.tick_params(labelsize=20)
    an = animation.FuncAnimation(fig,
                                 anim_callback,
                                 fargs=(get_world_frame,),
                                 init_func=None,
                                 frames=3000, interval=10, blit=False)



    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        print('Saving gif')
        an.save('df3_airdrag.gif', dpi=80, writer='imagemagick', fps=60)
    else:
        plt.show()

def plot_waypoints(waypoints):
    ax = plt.gca()
    lines = ax.get_lines()
    lines[-2].set_data(waypoints[:,0], waypoints[:,1])
    lines[-2].set_3d_properties(waypoints[:,2])

def set_limit(x, y, z):
    # Limit the range of the coordinate axis.
    ax = plt.gca()
    ax.set_xlim(x)
    ax.set_ylim(y)
    ax.set_zlim(z)

def anim_callback(i, get_world_frame):
    frame = get_world_frame(i)
    set_frame(frame)

def set_frame(frame):
    # convert 3x6 world_frame matrix into three line_data objects which is 3x2 (row:point index, column:x,y,z)
    lines_data = [frame[:,[0,2]], frame[:,[1,3]], frame[:,[4,5]]]
    ax = plt.gca()
    lines = ax.get_lines()
    for line, line_data in zip(lines[:3], lines_data):
        x, y, z = line_data
        line.set_data(x, y)
        line.set_3d_properties(z)

    global history, count
    # plot history trajectory
    history[count] = frame[:,4]
    if count < np.size(history, 0) - 1:
        count += 1
    zline = history[:count,-1]
    xline = history[:count,0]
    yline = history[:count,1]
    lines[-1].set_data(xline, yline)
    lines[-1].set_3d_properties(zline)
    ax.plot3D(xline, yline, zline, 'brown')
