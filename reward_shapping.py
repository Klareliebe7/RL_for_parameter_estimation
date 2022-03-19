import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
reward = np.zeros((23 ,23 )) 
x_axis_labels = [] # labels for x-axis
y_axis_labels = [] # labels for y-axis

for x in range(23 ):
     
    x_value = (x-11 )/2 
    print(x_value)
    x_axis_labels.append(round(x_value,1))
    for v in range(23 ):
        v_value = (v-11 )/5.5
        #z_dist_reward = 1-np.absolute(x_value /5)**0.4
        #z_speed_reward = 1-np.absolute(v_value/2)**0.4
        #reward[v][x] = z_dist_reward*2 + z_speed_reward
        
        dist = np.sqrt((x_value/5)**2 +(v_value/5)**2)
        # if dist < 0.1:
            # dist = 0.1
        reward[v][x] =5*( 1-dist)
        if np.absolute(x_value)> 5:
            reward[v][x] = -5
for v in range(23):
    v_value = (v-11 )/5.5
    y_axis_labels.append(round(v_value,1))
    
ax = plt.axes()
 


rfig=sns.heatmap(reward,xticklabels=x_axis_labels, yticklabels=y_axis_labels,ax = ax,cmap="coolwarm")
 

plt.title('Shaped Reward', fontsize = 15) # title with fontsize 20
plt.xlabel('Distance to hovering point', fontsize = 10) # x-axis label with fontsize 15
plt.ylabel('Velocity', fontsize = 10) # y-axis label with fontsize 15
figure = rfig.get_figure()    
figure.savefig('rewardshapping.png', dpi=400)