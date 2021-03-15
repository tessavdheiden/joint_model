import math
import numpy as np
from gym.envs.classic_control import rendering
from pyglet.gl import *

class Tentacle(object):
    def __init__(self, x, y):
        self.n = 2
        self.segments = [None]*self.n
        self.L = 1
        self.length = self.L / self.n
        self.base = (x, y)
        base = self.base
        for i in range(self.n):
            self.segments[i] = Segment(*base, self.length)
            base = self.segments[i].end


class Segment(rendering.Line):
    def __init__(self, x, y, length):
        super(Segment, self).__init__()
        self.start = np.array([x, y])
        self.length = length
        self.theta = 0
        self.calc_end(self.theta, self.start)

    def calc_end(self, angle_xy, start):
        x, y = start
        self.end = (x - self.length * math.cos(angle_xy), y - self.length * math.sin(angle_xy))

    def set_start(self, start):
        self.start = start
        self.calc_end(self.theta, self.start)

    def follow(self, tx, ty):
        target = np.array([tx, ty])
        dir = self.start - target
        rev_theta = np.arctan2(dir[1], dir[0])

        self.start = target + np.array([self.length * math.cos(rev_theta), self.length * math.sin(rev_theta)])
        self.calc_end(rev_theta, self.start)
        self.theta = rev_theta

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()