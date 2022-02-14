import numpy as np

mass = 0.18 # kg

g = 9.81 # m/s/s
I = np.array([(0.00025, 0, 0),
              (0, 0.00031, 0),
              (0, 0, 0.00020)])

invI = np.linalg.inv(I)

arm_length = 0.086 # meter
height = 0.05

L = arm_length
H = height


body_frame = np.array([(L, 0, 0, 1),
                       (0, L, 0, 1),
                       (-L, 0, 0, 1),
                       (0, -L, 0, 1),
                       (0, 0, 0, 1),
                       (0, 0, H, 1)])

# R3 basis vectors
e1 = np.array([[1.0],[0.0],[0.0]])
e2 = np.array([[0.0],[1.0],[0.0]])
e3 = np.array([[0.0],[0.0],[1.0]])


