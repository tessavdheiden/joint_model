import numpy as np
from gym import spaces


class Controller(object):
    def __init__(self, env):
        action_dim = env.action_space.shape[0]
        high = np.ones((action_dim, 3)) * .1
        self.action_space = spaces.Box(
            low=-high,
            high=high, shape=(action_dim, 3),
            dtype=np.float32
        )

        self.P, self.I, self.D = .1, .1, .1

    def compute(self, e_terms):
        error, integral, derivative = e_terms
        pid = -np.dot(self.PID[:, 0] * error + self.PID[:, 1] * integral + self.PID[:, 2] * derivative)
        return pid

    def update(self, gains):
        p, i, d = gains
        self.P = p
        self.I = i
        self.D = d


