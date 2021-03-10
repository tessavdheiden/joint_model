import numpy as np
from numpy import pi, cos, sin
from gym import spaces
import os
import torch

from envs.env_abs import AbsEnv
from envs.misc import angle_normalize


class Trajectory(object):
    def __init__(self):
        assert os.path.exists('traj.npy')
        with open('traj.npy', 'rb') as f:
            traj = np.load(f)
        self.states = np.zeros((len(traj), 4))
        self.states[:, :2] = traj
        self.states[1:, 2:4] = np.diff(traj, axis=0)
        self.N = len(self.states)
        self.it = 0

    def get_next(self, n=1):
        if self.it + n <= self.N:
            res = self.states[self.it:self.it + n]
        else:
            res = np.vstack((self.states[self.it:], self.states[:n - (self.N - self.it)]))
        self.it = (self.it + n) % len(self.states)
        return res


class ControlledArmEnv(AbsEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    dt = .1

    LINK_LENGTH_1 = 1
    LINK_LENGTH_2 = 1

    MAX_VEL = 9 * pi
    GAIN_P = 8.
    GAIN_D = 1.
    n = 20
    action_dim = 4 * n
    u_high = np.ones(action_dim) * .0001
    u_low = -u_high
    action_space = spaces.Box(
        low=u_low,
        high=u_high, shape=(action_dim,),
        dtype=np.float32
    )
    s_min = -1
    s_max = 1
    observation_space = spaces.Box(
        low=-s_max,
        high=s_max, shape=(8,),
        dtype=np.float32
    )

    point_l = .5
    grab_counter = 0

    def __init__(self):
        self.name = 'ControlledArm'
        self.state = np.zeros(4)
        self.states = np.zeros((self.n, 4))
        self.p = np.ones(2) * self.GAIN_P
        self.d = np.ones(2) * self.GAIN_D
        self.traj = Trajectory()
        self.state_names = ['θ1', 'θ2', 'dotθ1', 'dotθ2', 'barθ1r', 'barθ2r']
        self.viewer = None

    def step(self, a):
        dt = self.dt

        self.reference = self.traj.get_next(self.n)
        a = a.reshape(self.n, 4)

        self.states = np.zeros((self.n, 4))
        for t in range(self.n):
            self.target = self.reference[t] + a[t]

            delta_p = angle_normalize(self.target[:2] - self.state[:2])
            delta_v = self.target[2:4] - self.state[2:4]

            alpha = self.p * delta_p + self.d * delta_v

            self.state[2:4] += alpha * dt
            self.state[:2] += self.state[2:4] * dt
            self.state[:2] = angle_normalize(self.state[:2])
            self.states[t, :2] = self.state[:2]
            self.states[t, 2:4] = self.state[2:4]

        obs = self._get_obs()

        return obs, None, [], {}

    def _get_obs(self):
        l1, l2 = self.LINK_LENGTH_1, self.LINK_LENGTH_2
        obs = np.zeros((self.n, 6))
        for t in range(self.n):
            t1, t2 = self.states[t, 0], self.states[t, 1]
            x1, y1 = l1*cos(t1), l1*sin(t1)
            x2, y2 = x1 + l2*cos(t1+t2), y1 + l2*sin(t1+t2)
            obs[t] = np.hstack([x1, y1, x2, y2, self.states[t, 2], self.states[t, 3]])
        return np.hstack((np.ravel(obs), np.ravel(self.reference)))

    def step_batch(self, x, u):
        th = torch.cat(torch.atan2(x[:, 1:2], x[:, 0:1]), torch.atan2(x[:, 3:4], x[:, 2:3]), dim=1)
        thdot = x[:, 4:6]

        delta_p = angle_normalize(u[:, :2] - th)
        delta_v = u[:, 2:4] - thdot

        alpha = self.p * delta_p + self.d * delta_v

        newthdot = thdot + alpha * self.dt
        newth = th + newthdot * self.dt
        newth = angle_normalize(newth)

        reference = x[:, 6:]
        return self._get_obs_from_state(newth, newthdot, reference)

    def _get_obs_from_state(self, ang, vel, reference):
        return torch.cat((torch.cos(ang[:, 0:1]), torch.sin(ang[:, 0:1]),
                          torch.cos(ang[:, 1:2]), torch.sin(ang[:, 1:2]),
                          vel[:, 0:1], vel[:, 1:2],
                          reference[:, :]), dim=1)

    def reset(self):
        self.reference = None#self.traj.states[:self.n]
        self.state[:4] = self.traj.states[0]
        self.states = np.zeros((self.n, 4))

    def draw(self, s, alpha=1.):
        from gym.envs.classic_control import rendering
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
            link._color.vec4 = (0, .8, .8, alpha)
            circ = self.viewer.draw_circle(.1)
            circ._color.vec4 = (.8, .8, 0, alpha)
            circ.add_attr(jtransform)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound, bound, -bound, bound)
            self.pts = [None] * self.n
            for i in range(self.n):
                p = rendering.make_circle(.01)
                self.pts[i] = rendering.Transform()
                p.add_attr(self.pts[i])
                self.viewer.add_geom(p)

        self.draw(self.state[:2])
        self.draw(self.target, .2)

        for i, p in enumerate(self.reference):
            s = p
            p1 = [self.LINK_LENGTH_1 * cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

            p2 = [p1[0] + self.LINK_LENGTH_2 * cos(s[0] + s[1]),
                  p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1])]
            self.pts[i].set_translation(*p2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_benchmark_data(self, data={}):
        if len(data) == 0:
            names = ['θ1', 'θ2', 'dotθ1', 'dotθ2' ,'ddotθ1', 'ddotθ2']
            data = {name: [] for name in names}

        data['θ1'].append(self.state[0])
        data['θ2'].append(self.state[1])
        data['dotθ1'].append(self.state[2])
        data['dotθ2'].append(self.state[3])

        return data

    def do_benchmark(self, data):
        names = list(data.keys())
        # convert to numpy
        for i, name in enumerate(names[:4]):
            data[name] = np.array(data[name])

        data['ddotθ1'] = np.diff(data['dotθ1']) / self.dt
        data['ddotθ2'] = np.diff(data['dotθ2']) / self.dt

        return data


def make_video():
    from viz.video import Video
    v = Video()
    env = ControlledArmEnv()
    env.seed()
    env.reset()
    for _ in range(1000):
        a = env.action_space.sample()
        env.step(a)
        v.add(env.render(mode='rgb_array'))
    v.save(f'../img/p={env.p}_d={env.d}.gif')
    env.close()


def make_plot():
    from viz.benchmark_plot import BenchmarkPlot
    b = BenchmarkPlot()
    env = ControlledArmEnv()
    env.seed()
    env.reset()
    data = env.get_benchmark_data()
    for _ in range(5):
        a = env.action_space.sample()
        env.step(a)
        for t in range(env.states.shape[0]):
            env.state = env.states[t]
            data = env.get_benchmark_data(data)
            env.render()
    env.do_benchmark(data)
    b.add(data)
    b.plot("../img/derivatives.png")
    env.close()


if __name__ == '__main__':
    make_plot()
