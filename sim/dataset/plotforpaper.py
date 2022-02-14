import matplotlib.pyplot as plt
import numpy as np
import json
Ixx_his1 = []
Ixx_his2 = []
Ixx_his3 = []
Ixx_his4 = []
Ixx_his5 = []
Ixx_his6 = []
Ixx_his7 = []
Ixx_his8 = []
Ixx_his9 = []
Ixx_his10 = []
Ixx_his11 = []
Ixx_his12 = []
Ixx_his13 = []
Ixx_his14 = []
Ixx_his15 = []
Ixx_his16 = []
Ixx_his17 = []
Ixx_his18 = []
Ixx_his19 = []
Ixx_his20 = []


Iyy_his1 = []
Iyy_his2 = []
Iyy_his3 = []
Iyy_his4 = []
Iyy_his5 = []
Iyy_his6 = []
Iyy_his7 = []
Iyy_his8 = []
Iyy_his9 = []
Iyy_his10 = []
Iyy_his11 = []
Iyy_his12 = []
Iyy_his13 = []
Iyy_his14 = []
Iyy_his15 = []
Iyy_his16 = []
Iyy_his17 = []
Iyy_his18 = []
Iyy_his19 = []
Iyy_his20 = []

Izz_his1 = []
Izz_his2 = []
Izz_his3 = []
Izz_his4 = []
Izz_his5 = []
Izz_his6 = []
Izz_his7 = []
Izz_his8 = []
Izz_his9 = []
Izz_his10 = []
Izz_his11 = []
Izz_his12 = []
Izz_his13 = []
Izz_his14 = []
Izz_his15 = []
Izz_his16 = []
Izz_his17 = []
Izz_his18 = []
Izz_his19 = []
Izz_his20 = []

Mass_his1 = []
Mass_his2 = []
Mass_his3 = []
Mass_his4 = []
Mass_his5 = []
Mass_his6 = []
Mass_his7 = []
Mass_his8 = []
Mass_his9 = []
Mass_his10 = []
Mass_his11 = []
Mass_his12 = []
Mass_his13 = []
Mass_his14 = []
Mass_his15 = []
Mass_his16 = []
Mass_his17 = []
Mass_his18 = []
Mass_his19 = []
Mass_his20 = []

# extract estimated parameter from simulations
with open('Ixx_his1.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his1.append(np.array(p))
with open('Ixx_his2.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his2.append(np.array(p))
with open('Ixx_his3.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his3.append(np.array(p))

with open('Ixx_his4.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his4.append(np.array(p))
with open('Ixx_his5.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his5.append(np.array(p))
with open('Ixx_his6.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his6.append(np.array(p))
with open('Ixx_his7.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his7.append(np.array(p))
with open('Ixx_his8.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his8.append(np.array(p))

with open('Ixx_his9.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his9.append(np.array(p))
with open('Ixx_his10.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his10.append(np.array(p))
with open('Ixx_his11.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his11.append(np.array(p))
with open('Ixx_his12.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his12.append(np.array(p))
with open('Ixx_his13.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his13.append(np.array(p))
with open('Ixx_his14.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his14.append(np.array(p))
with open('Ixx_his15.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his15.append(np.array(p))

with open('Ixx_his16.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his16.append(np.array(p))
with open('Ixx_his17.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his17.append(np.array(p))
with open('Ixx_his18.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his18.append(np.array(p))
with open('Ixx_his19.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his19.append(np.array(p))
with open('Ixx_his20.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Ixx_his20.append(np.array(p))








with open('Iyy_his1.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his1.append(np.array(p))
with open('Iyy_his2.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his2.append(np.array(p))
with open('Iyy_his3.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his3.append(np.array(p))
with open('Iyy_his4.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his4.append(np.array(p))
with open('Iyy_his5.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his5.append(np.array(p))
with open('Iyy_his6.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his6.append(np.array(p))
with open('Iyy_his7.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his7.append(np.array(p))
with open('Iyy_his8.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his8.append(np.array(p))
with open('Iyy_his9.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his9.append(np.array(p))
with open('Iyy_his10.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his10.append(np.array(p))
with open('Iyy_his11.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his11.append(np.array(p))
with open('Iyy_his12.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his12.append(np.array(p))
with open('Iyy_his13.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his13.append(np.array(p))
with open('Iyy_his14.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his14.append(np.array(p))
with open('Iyy_his15.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his15.append(np.array(p))

with open('Iyy_his16.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his16.append(np.array(p))
with open('Iyy_his17.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his17.append(np.array(p))
with open('Iyy_his18.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his18.append(np.array(p))
with open('Iyy_his19.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his19.append(np.array(p))
with open('Iyy_his20.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Iyy_his20.append(np.array(p))



with open('Izz_his1.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his1.append(np.array(p))
with open('Izz_his2.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his2.append(np.array(p))
with open('Izz_his3.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his3.append(np.array(p))
with open('Izz_his4.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his4.append(np.array(p))
with open('Izz_his5.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his5.append(np.array(p))
with open('Izz_his6.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his6.append(np.array(p))
with open('Izz_his7.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his7.append(np.array(p))
with open('Izz_his8.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his8.append(np.array(p))
with open('Izz_his9.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his9.append(np.array(p))
with open('Izz_his10.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his10.append(np.array(p))
with open('Izz_his11.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his11.append(np.array(p))
with open('Izz_his12.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his12.append(np.array(p))
with open('Izz_his13.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his13.append(np.array(p))
with open('Izz_his14.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his14.append(np.array(p))
with open('Izz_his15.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his15.append(np.array(p))

with open('Izz_his16.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his16.append(np.array(p))
with open('Izz_his17.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his17.append(np.array(p))
with open('Izz_his18.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his18.append(np.array(p))
with open('Izz_his19.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his19.append(np.array(p))
with open('Izz_his20.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Izz_his20.append(np.array(p))




with open('Mass_his1.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his1.append(np.array(p))
with open('Mass_his2.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his2.append(np.array(p))
with open('Mass_his3.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his3.append(np.array(p))
with open('Mass_his4.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his4.append(np.array(p))


with open('Mass_his5.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his5.append(np.array(p))

with open('Mass_his6.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his6.append(np.array(p))
with open('Mass_his7.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his7.append(np.array(p))
with open('Mass_his8.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his8.append(np.array(p))
with open('Mass_his9.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his9.append(np.array(p))
with open('Mass_his10.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his10.append(np.array(p))
with open('Mass_his11.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his11.append(np.array(p))
with open('Mass_his12.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his12.append(np.array(p))
with open('Mass_his13.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his13.append(np.array(p))
with open('Mass_his14.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his14.append(np.array(p))
with open('Mass_his15.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his15.append(np.array(p))

with open('Mass_his16.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his16.append(np.array(p))
with open('Mass_his17.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his17.append(np.array(p))
with open('Mass_his18.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his18.append(np.array(p))
with open('Mass_his19.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his19.append(np.array(p))
with open('Mass_his20.txt') as json_file:
    data = json.load(json_file)
    for p in data:
        Mass_his20.append(np.array(p))



steps1 = range(len(Ixx_his1))
steps2 = range(len(Ixx_his2))
steps3 = range(len(Ixx_his3))
steps4 = range(len(Ixx_his4))
steps5 = range(len(Ixx_his5))
steps6 = range(len(Ixx_his6))
steps7 = range(len(Ixx_his7))
steps8 = range(len(Ixx_his8))
steps9 = range(len(Ixx_his9))
steps10 = range(len(Ixx_his10))
steps11 = range(len(Ixx_his11))
steps12 = range(len(Ixx_his12))
steps13 = range(len(Ixx_his13))
steps14 = range(len(Ixx_his14))
steps15 = range(len(Ixx_his15))
steps16 = range(len(Ixx_his16))
steps17 = range(len(Ixx_his17))
steps18 = range(len(Ixx_his18))
steps19 = range(len(Ixx_his19))
steps20 = range(len(Ixx_his20))

steps = range(1530)
mass_real = [0.18]*len(steps)
Ixx_real = [0.00025]*len(steps)
Iyy_real = [0.00031]*len(steps)
Izz_real = [0.00020]*len(steps)
# plot estimated inertia matrix
plt.figure()
plt.plot(steps1, Ixx_his1, color="blue", linestyle="--", linewidth=2.0, label='Simulation 1')
plt.plot(steps2, Ixx_his2, color="darkorange", linestyle="--", linewidth=2.0, label='Simulation 2')
plt.plot(steps3[:1530], Ixx_his3[:1530], color="black", linestyle="--", linewidth=2.0, label='Simulation 3')
plt.plot(steps4, Ixx_his4, color="green", linestyle="--", linewidth=2.0, label='Simulation 4')
plt.plot(steps5, Ixx_his5, color="brown", linestyle="--", linewidth=2.0, label='Simulation 5')
plt.plot(steps6, Ixx_his6, color="yellow", linestyle="--", linewidth=2.0, label='Simulation 6')
plt.plot(steps7, Ixx_his7, color="gray", linestyle="--", linewidth=2.0, label='Simulation 7')
plt.plot(steps8, Ixx_his8, color="chocolate", linestyle="--", linewidth=2.0, label='Simulation 8')
plt.plot(steps9, Ixx_his9, color="darkcyan", linestyle="--", linewidth=2.0, label='Simulation 9')
plt.plot(steps10, Ixx_his10, color="darkgray", linestyle="--", linewidth=2.0, label='Simulation 10')
plt.plot(steps11, Ixx_his11, color="cyan", linestyle="--", linewidth=2.0, label='Simulation 11')
plt.plot(steps12, Ixx_his12, color="blueviolet", linestyle="--", linewidth=2.0, label='Simulation 12')
plt.plot(steps13[:1530], Ixx_his13[:1530], color="hotpink", linestyle="--", linewidth=2.0, label='Simulation 13')
plt.plot(steps14[:1530], Ixx_his14[:1530], color="slateblue", linestyle="--", linewidth=2.0, label='Simulation 14')
plt.plot(steps15[:1530], Ixx_his15[:1530], color="tan", linestyle="--", linewidth=2.0, label='Simulation 15')
plt.plot(steps16[:1530], Ixx_his16[:1530], color="maroon", linestyle="--", linewidth=2.0, label='Simulation 16')
plt.plot(steps17[:1530], Ixx_his17[:1530], color="olive", linestyle="--", linewidth=2.0, label='Simulation 17')
plt.plot(steps18[:1530], Ixx_his18[:1530], color="seagreen", linestyle="--", linewidth=2.0, label='Simulation 18')
plt.plot(steps19[:1530], Ixx_his19[:1530], color="dodgerblue", linestyle="--", linewidth=2.0, label='Simulation 19')
plt.plot(steps20[:1530], Ixx_his20[:1530], color="rebeccapurple", linestyle="--", linewidth=2.0, label='Simulation 20')
plt.plot(steps, Ixx_real, color="red", linestyle="-", linewidth=2.0, label='True Ixx')
plt.title('Estimated Ixx for multi time simulation',fontsize=25,verticalalignment='bottom')
plt.xlabel('Iterations',fontsize=20)
plt.ylabel('I ( kg·m²)',fontsize=20)
plt.tick_params(labelsize=20)
plt.grid(linestyle='-.')
plt.legend(loc='best')
plt.ylim(0, 0.0005)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

plt.figure()
plt.plot(steps1, Iyy_his1, color="blue", linestyle="--", linewidth=2.0, label='Simulation 1')
plt.plot(steps2, Iyy_his2, color="darkorange", linestyle="--", linewidth=2.0, label='Simulation 2')
plt.plot(steps3[:1530], Iyy_his3[:1530], color="black", linestyle="--", linewidth=2.0, label='Simulation 3')
plt.plot(steps4, Iyy_his4, color="green", linestyle="--", linewidth=2.0, label='Simulation 4')
plt.plot(steps5, Iyy_his5, color="brown", linestyle="--", linewidth=2.0, label='Simulation 5')
plt.plot(steps6, Iyy_his6, color="yellow", linestyle="--", linewidth=2.0, label='Simulation 6')
plt.plot(steps7, Iyy_his7, color="gray", linestyle="--", linewidth=2.0, label='Simulation 7')
plt.plot(steps8, Iyy_his8, color="chocolate", linestyle="--", linewidth=2.0, label='Simulation 8')
plt.plot(steps9, Iyy_his9, color="darkcyan", linestyle="--", linewidth=2.0, label='Simulation 9')
plt.plot(steps10, Iyy_his10, color="darkgray", linestyle="--", linewidth=2.0, label='Simulation 10')
plt.plot(steps11, Iyy_his11, color="cyan", linestyle="--", linewidth=2.0, label='Simulation 11')
plt.plot(steps12, Iyy_his12, color="blueviolet", linestyle="--", linewidth=2.0, label='Simulation 12')
plt.plot(steps13[:1530], Iyy_his13[:1530], color="hotpink", linestyle="--", linewidth=2.0, label='Simulation 13')
plt.plot(steps14[:1530], Iyy_his14[:1530], color="slateblue", linestyle="--", linewidth=2.0, label='Simulation 14')
plt.plot(steps15[:1530], Iyy_his15[:1530], color="tan", linestyle="--", linewidth=2.0, label='Simulation 15')
plt.plot(steps16[:1530], Iyy_his16[:1530], color="maroon", linestyle="--", linewidth=2.0, label='Simulation 16')
plt.plot(steps17[:1530], Iyy_his17[:1530], color="olive", linestyle="--", linewidth=2.0, label='Simulation 17')
plt.plot(steps18[:1530], Iyy_his18[:1530], color="seagreen", linestyle="--", linewidth=2.0, label='Simulation 18')
plt.plot(steps19[:1530], Iyy_his19[:1530], color="dodgerblue", linestyle="--", linewidth=2.0, label='Simulation 19')
plt.plot(steps20[:1530], Iyy_his20[:1530], color="rebeccapurple", linestyle="--", linewidth=2.0, label='Simulation 20')
plt.plot(steps, Iyy_real, color="red", linestyle="-", linewidth=2.0, label='True Iyy')
plt.title('Estimated Iyy for multi time simulation',fontsize=25,verticalalignment='bottom')
plt.xlabel('Iterations',fontsize=20)
plt.ylabel('I ( kg·m²)',fontsize=20)
plt.tick_params(labelsize=20)
plt.grid(linestyle='-.')
plt.legend(loc='best')
plt.ylim(0.00015, 0.0008)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

plt.figure()
plt.plot(steps1, Izz_his1, color="blue", linestyle="--", linewidth=2.0, label='Simulation 1')
plt.plot(steps2, Izz_his2, color="darkorange", linestyle="--", linewidth=2.0, label='Simulation 2')
plt.plot(steps3[:1530], Izz_his3[:1530], color="black", linestyle="--", linewidth=2.0, label='Simulation 3')
plt.plot(steps4, Izz_his4, color="green", linestyle="--", linewidth=2.0, label='Simulation 4')
plt.plot(steps5, Izz_his5, color="brown", linestyle="--", linewidth=2.0, label='Simulation 5')
plt.plot(steps6, Izz_his6, color="yellow", linestyle="--", linewidth=2.0, label='Simulation 6')
plt.plot(steps7, Izz_his7, color="gray", linestyle="--", linewidth=2.0, label='Simulation 7')
plt.plot(steps8, Izz_his8, color="chocolate", linestyle="--", linewidth=2.0, label='Simulation 8')
plt.plot(steps9, Izz_his9, color="darkcyan", linestyle="--", linewidth=2.0, label='Simulation 9')
plt.plot(steps10, Izz_his10, color="darkgray", linestyle="--", linewidth=2.0, label='Simulation 10')
plt.plot(steps11, Izz_his11, color="cyan", linestyle="--", linewidth=2.0, label='Simulation 11')
plt.plot(steps12, Izz_his12, color="blueviolet", linestyle="--", linewidth=2.0, label='Simulation 12')
plt.plot(steps13[:1530], Izz_his13[:1530], color="hotpink", linestyle="--", linewidth=2.0, label='Simulation 13')
plt.plot(steps14[:1530], Izz_his14[:1530], color="slateblue", linestyle="--", linewidth=2.0, label='Simulation 14')
plt.plot(steps15[:1530], Izz_his15[:1530], color="tan", linestyle="--", linewidth=2.0, label='Simulation 15')
plt.plot(steps16[:1530], Izz_his16[:1530], color="maroon", linestyle="--", linewidth=2.0, label='Simulation 16')
plt.plot(steps17[:1530], Izz_his17[:1530], color="olive", linestyle="--", linewidth=2.0, label='Simulation 17')
plt.plot(steps18[:1530], Izz_his18[:1530], color="seagreen", linestyle="--", linewidth=2.0, label='Simulation 18')
plt.plot(steps19[:1530], Izz_his19[:1530], color="dodgerblue", linestyle="--", linewidth=2.0, label='Simulation 19')
plt.plot(steps20[:1530], Izz_his20[:1530], color="rebeccapurple", linestyle="--", linewidth=2.0, label='Simulation 20')
plt.plot(steps, Izz_real, color="red", linestyle="-", linewidth=2.0, label='True Izz')
plt.title('Estimated Izz for multi time simulation',fontsize=25,verticalalignment='bottom')
plt.xlabel('Iterations',fontsize=20)
plt.ylabel('I ( kg·m²)',fontsize=20)
plt.tick_params(labelsize=20)
plt.grid(linestyle='-.')
plt.legend(loc='best')
plt.ylim(0, 0.0005)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# plot estimated mass
plt.figure()
plt.plot(steps1, Mass_his1, color="blue", linestyle="--", linewidth=2.0, label='Simulation 1')
plt.plot(steps2, Mass_his2, color="darkorange", linestyle="--", linewidth=2.0, label='Simulation 2')
plt.plot(steps3[:1530], Mass_his3[:1530], color="black", linestyle="--", linewidth=2.0, label='Simulation 3')
plt.plot(steps4, Mass_his4, color="green", linestyle="--", linewidth=2.0, label='Simulation 4')
plt.plot(steps5, Mass_his5, color="brown", linestyle="--", linewidth=2.0, label='Simulation 5')
plt.plot(steps6, Mass_his6, color="blue", linestyle="--", linewidth=2.0, label='Simulation 6')
plt.plot(steps7, Mass_his7, color="darkorange", linestyle="--", linewidth=2.0, label='Simulation 7')
plt.plot(steps8, Mass_his8, color="chocolate", linestyle="--", linewidth=2.0, label='Simulation 8')
plt.plot(steps9, Mass_his9, color="darkcyan", linestyle="--", linewidth=2.0, label='Simulation 9')
plt.plot(steps10, Mass_his10, color="darkgray", linestyle="--", linewidth=2.0, label='Simulation 10')
plt.plot(steps11, Mass_his11, color="cyan", linestyle="--", linewidth=2.0, label='Simulation 11')
plt.plot(steps12, Mass_his12, color="blueviolet", linestyle="--", linewidth=2.0, label='Simulation 12')
plt.plot(steps13[:1530], Mass_his13[:1530], color="hotpink", linestyle="--", linewidth=2.0, label='Simulation 13')
plt.plot(steps14[:1530], Mass_his14[:1530], color="slateblue", linestyle="--", linewidth=2.0, label='Simulation 14')
plt.plot(steps15[:1530], Mass_his15[:1530], color="tan", linestyle="--", linewidth=2.0, label='Simulation 15')
plt.plot(steps16[:1530], Mass_his16[:1530], color="maroon", linestyle="--", linewidth=2.0, label='Simulation 16')
plt.plot(steps17[:1530], Mass_his17[:1530], color="olive", linestyle="--", linewidth=2.0, label='Simulation 17')
plt.plot(steps18[:1530], Mass_his18[:1530], color="seagreen", linestyle="--", linewidth=2.0, label='Simulation 18')
plt.plot(steps19[:1530], Mass_his19[:1530], color="dodgerblue", linestyle="--", linewidth=2.0, label='Simulation 19')
plt.plot(steps20[:1530], Mass_his20[:1530], color="rebeccapurple", linestyle="--", linewidth=2.0, label='Simulation 20')
plt.plot(steps, mass_real, color="red", linestyle="-", linewidth=2.0, label='True mass')

plt.title('Estimated mass for multi time simulation',fontsize=25,verticalalignment='bottom')
plt.xlabel('Iterations',fontsize=20)
plt.ylabel('mass (kg)',fontsize=20)
plt.tick_params(labelsize=20)
plt.grid(linestyle='-.')
plt.legend(loc='best')
plt.ylim(0.15, 0.25)


plt.show()