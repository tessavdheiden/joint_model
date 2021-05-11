import numpy as np
from numpy import cos, sin, pi

class Shape(object):
    def __init__(self):
        raise NotImplementedError

    @property
    def points(self):
        raise NotImplementedError

    def get_next(self):
        p = self.points[self.it]
        self.it = (self.it + 1) % len(self.points)
        return p


class Rectangle(Shape):
    def __init__(self, center, w, h, N):
        l = center[0] - w/2
        t = center[1] + h/2
        b = t - h
        r = l + w
        n = int(N / (2*h + 2*w) * w)
        m = int(N / (2*h + 2*w) * h)
        top = np.stack([np.linspace(l, r, n + 1), np.full(n + 1, t)], axis=1)[:-1]
        left = np.stack([np.full(m + 1, l), np.linspace(t, b, m + 1)], axis=1)[:-1]
        right = left.copy()
        right[:, 0] += w
        bottom = top.copy()
        bottom[:, 1] -= h

        self.top, self.left, self.right, self.bottom = top, left[::-1], right, bottom[::-1]
        self.it = 0

    @property
    def points(self):
        return np.concatenate([self.top, self.right, self.bottom, self.left])


class Circle(Shape):
    def __init__(self, center, w, h, N):
        self.circle = np.array([cos(np.linspace(-pi, pi, N)), sin(np.linspace(-pi, pi, N))]).transpose()[::-1]

    @property
    def points(self):
        return self.circle

class Compound(Shape):
    def __init__(self):
        self.a = []
        self.it = 0

    def add(self, arr):
        if len(self.a) > 0:
            self.a = np.concatenate((self.a, arr.points), axis=0)
        else:
            self.a = arr.points

    @property
    def points(self):
        return self.a


class CircleToSquare(Shape):
    def __init__(self, center, w, h, N, r=.2):
        t, l = w / 2, 0
        assert r * 100 % 2 == 0
        n = N // 4
        n_arc = int(r / w * n)
        arc = self.create_corner(center, w, n_arc, r)[::-1]
        n_line = (n - n_arc) // 2
        top = np.stack([np.linspace(l, w / 2 - r, n_line), np.full(n_line, t)], axis=1)
        right = np.stack([np.full(n_line, w / 2), np.linspace(w / 2 - r, 0, n_line)], axis=1)

        self.corner = np.concatenate([top, arc, right])
        self.corner_rb = np.array([1, -1] * self.corner[::-1])
        self.corner_lb = np.array([-1, -1] * self.corner)
        self.corner_ru = np.array([-1, 1] * self.corner[::-1])
        self.p = np.concatenate([self.corner, self.corner_rb, self.corner_lb, self.corner_ru])

        self.it = 0

    def create_corner(self, center, w, n, r):
        start = np.array([w / 2 - r, w / 2 - r])
        return center + start + r * np.stack([cos(np.linspace(0, pi/2, n)), sin(np.linspace(0, pi/2, n))], axis=1)

    @property
    def points(self):
        return self.p