import numpy as np
class Sensor:
    '''There are three types sensors which are sensor A ,sensor B and sensor C.'''
    def __init__(self):
        self.angularMea = np.zeros(3)
        self.accMea = np.zeros(3)
        self.YMea = np.zeros(12)
    def observe(self):
        return  np.hstack([self.accMea, self.angularMea])


    def update(self,quad,dt):
        noise = np.random.normal(0, np.sqrt(dt/100), 12)
        self.YMea = quad.state[0:12] + noise
        self.angularMea = self.YMea[9:12]
        # accelerator
        acc_world = quad.statedot[3:6]

        noise_acc = np.random.normal(0, np.sqrt(dt), 3)

        self.accMea = acc_world + noise_acc #+ self.accBias
        return self.observe()

    def Y_obs1(self):
        # observations based on sensor A
        y = self.YMea
        y[3:9] = np.array([0, 0, 0, 0, 0, 0])
        return y.reshape(12, 1)

    def Y_obs2(self):
        # observations based on sensor B
        y = self.YMea
        return y.reshape(12,1)
    def Y_obs3(self):
        # observations based on sensor C
        y = self.YMea
        y[6:9] = np.array([0, 0, 0])
        return y.reshape(12,1)




