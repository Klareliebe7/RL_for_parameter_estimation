import numpy as np
def distance(quad_pos, waypoint):
    '''Checking distance between two points in space.'''
    x,y,z = quad_pos
    x_ref,y_ref,z_ref = waypoint
    d1 = np.sqrt(np.square(x - x_ref) + np.square(y - y_ref) + np.square(z - z_ref))
    return d1