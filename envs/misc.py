from numpy import pi
import numpy as np
import os

def angle_normalize(x):
    return (((x+pi) % (2*pi)) - pi)


class Trajectory(object):
    def __init__(self, n, order, dt):
        pf = os.path.join(os.getcwd(), os.path.dirname(__file__), 'traj.npy')
        assert os.path.exists(pf)
        self.n = n
        with open(pf, 'rb') as f:
            traj = np.load(f)
        self.N = len(traj)
        assert self.N / n == int(self.N / n)

        self.states = np.zeros((self.N, (order+1) * 2))
        self.states[:, :2] = traj
        for i in range(1, order+1):
            l1, r1 = i*2, i*2+2
            l2, r2 = (i-1)*2, (i-1)*2+2
            self.states[i:, l1:r1] = np.diff(self.states[i-1:, l2:r2], axis=0) / dt
