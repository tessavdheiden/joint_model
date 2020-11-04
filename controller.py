import numpy as np
from gym import spaces


class Controller(object):
    P, I, D = .1, .1, .1
    desired_mask = np.array([1, 1, 1])

    def __init__(self):
        self.action_space = spaces.Box(
            low=-1.,
            high=1., shape=(3,),
            dtype=np.float32
        )

    def compute(self, e_terms):
        error, integral, derivative = e_terms
        pid = -np.dot(self.P * error + self.I * integral + self.D * derivative, self.desired_mask)
        return pid

    def update(self, gains):
        p, i, d = gains
        self.P = p
        self.I = i
        self.D = d


