import numpy as np
from gym import spaces
from numpy import sin, cos, pi
import torch
import torch.nn as nn
import torch.optim as optim


from envs.env_abs import AbsEnv
from envs.misc import angle_normalize

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


class ArmFollowShapeEnv(AbsEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    dt = .1

    LINK_LENGTH_1 = 1
    LINK_LENGTH_2 = 1

    u_low = np.array([-1., -1.])
    u_high = np.array([1., 1.])
    action_space = spaces.Box(
        low=u_low,
        high=u_high, shape=(2,),
        dtype=np.float32
    )
    s_min = -1
    s_max = 1
    observation_space = spaces.Box(
        low=-s_max,
        high=s_max, shape=(6,),
        dtype=np.float32
    )

    action_dim = 2

    point_l = .5
    grab_counter = 0

    def __init__(self):
        self.name = 'Arm'
        self.state = np.zeros(2)
        self.target_location = np.zeros(2)
        # self.shape = CircleToSquare((0, 0), 2, 3, 500, r=.75)
        self.shape = Compound()
        for radius in [.08, .2, .4, .6]:
            self.shape.add(CircleToSquare((0, 0), 2, 2, 800, r=radius))

        self.viewer = None

        self.u_low_torch = torch.from_numpy(self.u_low).float()
        self.u_high_torch = torch.from_numpy(self.u_high).float()

    def step(self, u):
        self.target = self.shape.get_next()
        u = np.clip(u, self.u_low, self.u_high)

        self.state += u * self.dt

        obs = self._get_obs()
        r = self._r_func(obs[4:])
        return obs, r, self.get_point, {}

    def _reset_arm(self):
        self.state = np.array([0, -pi])

    def _reset_target(self):
        self.target = self.shape.get_next()

    def reset(self):
        self.get_point = False
        self.grab_counter = 0

        self._reset_target()
        self._reset_arm()
        return self._get_obs()

    def _get_obs(self):
        arm1dx_dy = np.array([self.LINK_LENGTH_1 * cos(self.state[0]), self.LINK_LENGTH_1 * sin(self.state[0])])
        arm2dx_dy = np.array([self.LINK_LENGTH_2 * cos(self.state[0] + self.state[1]), self.LINK_LENGTH_2 * sin(self.state[0] + self.state[1])])
        xy2 = arm1dx_dy + arm2dx_dy  # (x2, y2)
        delta = np.ravel(xy2 - self.target)

        return np.hstack([cos(self.state[0]), sin(self.state[0]),
                          cos(self.state[1]), sin(self.state[1]),
                          delta[0], delta[1]])

    def _r_func(self, distance):
        t = 50
        abs_distance = np.sqrt(np.sum(np.square(distance)))
        r = -abs_distance
        if abs_distance < self.point_l and (not self.get_point):
            r += 1.
            self.grab_counter += 1
            if self.grab_counter > t:
                r += 10.
                self.get_point = True
        elif abs_distance > self.point_l:
            self.grab_counter = 0
            self.get_point = False
        return r

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)
            # draw target trajectory
            self.traj = self.viewer.draw_polyline(self.shape.points, filled=False)
            circ = rendering.make_circle(.05)
            self.ctransform = rendering.Transform()
            circ.add_attr(self.ctransform)
            self.viewer.add_geom(circ)

        if s is None: return None

        p1 = [self.LINK_LENGTH_1 * cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [p1[0] + self.LINK_LENGTH_2 * cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1])]

        xys = np.array([[0, 0], p1, p2])#[:, ::-1]
        thetas = [s[0], s[0] + s[1]]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, .8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        self.ctransform.set_translation(*self.target)

        self.viewer.add_onetime(self.traj)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_benchmark_data(self, data={}):
        if len(data) == 0:
            names = ['θ1', 'θ2', 'Δx', 'Δy']
            data = {name: [] for name in names}

        names = list(data.keys())
        for i, name in enumerate(names[:2]):
            data[name].append(self.state[i])

        obs = self._get_obs()
        data['Δx'].append(obs[-2])
        data['Δy'].append(obs[-1])

        return data

    def do_benchmark(self, data):
        names = list(data.keys())
        # convert to numpy
        for i, name in enumerate(names):
            data[name] = np.array(data[name])

        return data


def solve_trajectory():
    traj = np.zeros((len(env.shape.points), 2))
    learning_rate = .0005

    for j, target_xy in enumerate(env.shape.points):
        thetas = torch.autograd.Variable(torch.zeros(1, 2), requires_grad=True)
        if j > 0:
            thetas.data = torch.from_numpy(traj[j-1, :2]).unsqueeze(0)
        else:
            thetas.data = torch.tensor([pi, -pi/2]).unsqueeze(0)
        env.target = target_xy
        target_xy = torch.from_numpy(target_xy).unsqueeze(0).float()
        loss = 1
        while loss > 0.0001:
            xy_ = torch.cat((env.LINK_LENGTH_1 * torch.cos(thetas[:, :1]) + env.LINK_LENGTH_2 * torch.cos(thetas[:, :1] + thetas[:, 1:]),
                             env.LINK_LENGTH_1 * torch.sin(thetas[:, :1]) + env.LINK_LENGTH_2 * torch.sin(thetas[:, :1] + thetas[:, 1:])), dim=1)

            error = ((target_xy - xy_) ** 2)
            loss = error.sum()
            loss.backward()
            thetas.data -= learning_rate * thetas.grad.data
            thetas.grad.data.zero_()

        env.state = thetas.detach().numpy().squeeze(0)
        env.render()
        traj[j, :2] = thetas.detach().numpy().squeeze(0)

    return traj

if __name__ == '__main__':
    from viz import *
    v = Video()

    env = ArmFollowShapeEnv()
    env.seed()
    traj = solve_trajectory()
    with open('traj.npy', 'wb') as f:
        np.save(f, traj)

    with open('traj.npy', 'rb') as f:
        traj = np.load(f)

    for j, txy in enumerate(env.shape.points):
        env.target = txy
        env.state = traj[j]
        v.add(env.render(mode='rgb_array'))
    v.save("test.gif")

