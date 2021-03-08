import numpy as np
from numpy import pi, cos, sin
from gym import spaces
import os

from envs.env_abs import AbsEnv
from envs.misc import angle_normalize


class Trajectory(object):
    def __init__(self):
        assert os.path.exists('traj.npy')
        with open('traj.npy', 'rb') as f:
            traj = np.load(f)
        self.points = traj[10:] # first
        self.it = 0

    def get_next(self, n=1):
        res = self.points[self.it:self.it+n]
        self.it = (self.it + 1) % len(self.points)
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

    u_low = np.array([0., 0.])
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
        high=s_max, shape=(8,),
        dtype=np.float32
    )

    action_dim = 2

    point_l = .5
    grab_counter = 0

    def __init__(self):
        self.name = 'ControlledArm'
        self.state = np.zeros(4)
        self.p = np.ones(2) * self.GAIN_P
        self.d = np.ones(2) * self.GAIN_D
        self.traj = Trajectory()
        self.n = 20

        self.viewer = None

    def step(self, a):
        dt = self.dt

        delta_p = angle_normalize(self.target - self.state[:2])
        delta_v = -self.state[2:]

        u = self.p * delta_p + self.d * delta_v

        self.state[2:] += u * dt
        self.state[:2] += self.state[2:] * dt

        obs = self._get_obs()

        self.points = self.traj.get_next(self.n)
        self.target = self.points[0]
        return obs, None, [], {}

    def _get_obs(self):
        delta = self.target - self.state[:2]
        return np.hstack([cos(self.state[0]), sin(self.state[0]),
                          cos(self.state[1]), sin(self.state[1]),
                          self.state[2], self.state[3],
                          delta[0], delta[1]])

    def reset(self):
        self.points = self.traj.get_next(self.n)
        self.target = self.points[0]
        self.state[:2] = self.target
        self.state[2:] = np.zeros(2)

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

        for i, p in enumerate(self.points):
            s = p
            p1 = [self.LINK_LENGTH_1 * cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

            p2 = [p1[0] + self.LINK_LENGTH_2 * cos(s[0] + s[1]),
                  p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1])]
            self.pts[i].set_translation(*p2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_benchmark_data(self, data={}):
        if len(data) == 0:
            names = ['dotθ1', 'dotθ2', 'Δx', 'Δy' ,'ddotθ1', 'ddotθ2']
            data = {name: [] for name in names}

        obs = self._get_obs()
        data['dotθ1'].append(obs[4])
        data['dotθ2'].append(obs[5])
        data['Δx'].append(obs[-2])
        data['Δy'].append(obs[-1])

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
        v.add(env.render(mode='rgb_array'))
        env.step(None)
    v.save(f'../img/p={env.p}_d={env.d}.gif')
    env.close()


def make_plot():
    from viz.benchmark_plot import BenchmarkPlot
    b = BenchmarkPlot()
    env = ControlledArmEnv()
    env.seed()
    env.reset()
    data = env.get_benchmark_data()
    for _ in range(1000):
        env.step(None)
        data = env.get_benchmark_data(data)
    env.do_benchmark(data)
    b.add(data)
    b.plot("../img/derivatives.png")
    env.close()


if __name__ == '__main__':
    make_video()