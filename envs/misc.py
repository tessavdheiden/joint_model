from numpy import pi
import numpy as np
import os

def angle_normalize(x):
    return (((x+pi) % (2*pi)) - pi)


class Trajectory(object):
    def __init__(self, order, dt):
        pf = os.path.join(os.getcwd(), os.path.dirname(__file__), 'traj.npy')
        assert os.path.exists(pf)
        with open(pf, 'rb') as f:
            traj = np.load(f)
        self.N = len(traj)
        self.dt = dt
        self.t = np.arange(0, self.N) * self.dt

        self.states = np.zeros((self.N, (order+1) * 2))
        self.states[:, :2] = traj
        for i in range(1, order+1):
            l1, r1 = i*2, i*2+2
            l2, r2 = (i-1)*2, (i-1)*2+2
            self.states[i:, l1:r1] = np.diff(self.states[i-1:, l2:r2], axis=0) / dt

def rk4(derivs, y0, t, *args, **kwargs):
    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout