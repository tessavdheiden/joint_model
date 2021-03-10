import numpy as np
from numpy import pi, cos, sin
from gym import spaces
import os
import torch

from envs.env_abs import AbsEnv
from envs.misc import angle_normalize


class Trajectory(object):
    observation_space = spaces.Box(
        low=-1,
        high=1, shape=(4,),
        dtype=np.float32
    )
    def __init__(self):
        pf = os.path.join(os.getcwd(), os.path.dirname(__file__), 'traj.npy')
        assert os.path.exists(pf)
        with open(pf, 'rb') as f:
            traj = np.load(f)
        self.states = np.zeros((len(traj), 4))
        self.states[:, :2] = traj
        self.states[:-1, 2:4] = np.diff(traj, axis=0)
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
    GAIN_P = 18.
    GAIN_D = 1.
    n = 100
    action_dim = 4 * n
    state_dim = 4
    m = Trajectory.observation_space.shape[0]

    u_high = np.ones(action_dim) * .0001
    u_low = -u_high
    action_space = spaces.Box(
        low=u_low,
        high=u_high, shape=(action_dim,),
        dtype=np.float32
    )
    s_min = -1
    s_max = 1

    obs_dim = state_dim + 2  # +2 for cos and sin
    observation_space = spaces.Box(
        low=-s_max,
        high=s_max, shape=(n * (obs_dim + m),),
        dtype=np.float32
    )

    point_l = .5
    grab_counter = 0

    def __init__(self):
        self.name = 'ControlledArm'
        self.state = np.zeros(self.state_dim)
        self.states = np.zeros((self.n, self.state_dim))
        self.p = np.ones(2) * self.GAIN_P
        self.d = np.ones(2) * self.GAIN_D
        self.ptorch = torch.from_numpy(self.p)
        self.dtorch = torch.from_numpy(self.d)
        self.traj = Trajectory()
        self.state_names = ['θ1', 'θ2', 'dotθ1', 'dotθ2', 'barθ1r', 'barθ2r']
        self.viewer = None

    def step(self, a):
        dt = self.dt

        a = a.reshape(self.n, self.m)

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
        self.reference = self.traj.get_next(self.n)

        return obs, None, [], {}

    def _get_obs(self):
        l1, l2 = self.LINK_LENGTH_1, self.LINK_LENGTH_2
        obs = np.zeros((self.n, self.obs_dim))
        for t in range(self.n):
            t1, t2 = self.states[t, 0], self.states[t, 1]
            x1, y1 = l1*cos(t1), l1*sin(t1)
            x2, y2 = x1 + l2*cos(t1+t2), y1 + l2*sin(t1+t2)
            obs[t] = np.hstack([x1, y1, x2, y2, self.states[t, 2], self.states[t, 3]])
        return np.hstack((np.ravel(obs), np.ravel(self.reference)))

    def step_batch(self, x, u):
        batch_size = x.shape[0]
        reference = x[:, -self.n * self.m:].view(batch_size, self.n, -1)  # end of array contains reference
        x = x[:, :-self.n * self.m].view(batch_size, self.n, -1)     # everything else
        actions = u.view(batch_size, self.n, -1)
        actions = torch.clamp(actions, -.01, .01)

        newx = torch.zeros_like(x)
        for t in range(self.n):
            target = reference[:, t] + actions[:, t]
            obs = x[:, t]

            th = torch.cat((torch.atan2(obs[:, 1:2], obs[:, 0:1]), torch.atan2(obs[:, 3:4], obs[:, 2:3])), dim=1)
            thdot = obs[:, 4:6]

            delta_p = angle_normalize(target[:, :2] - th)
            delta_v = target[:, 2:4] - thdot

            alpha = self.ptorch * delta_p + self.dtorch * delta_v

            newthdot = thdot + alpha * self.dt
            newth = th + newthdot * self.dt
            newth = angle_normalize(newth)
            nobs = self._get_obs_from_state(newth, newthdot)
            newx[:, t] = nobs
        newx = newx.view(batch_size, -1)
        reference = reference.view(batch_size, -1)
        return torch.cat((newx, reference), dim=1)

    def _get_obs_from_state(self, ang, vel):
        return torch.cat((torch.cos(ang[:, 0:1]), torch.sin(ang[:, 0:1]),
                          torch.cos(ang[:, 1:2]), torch.sin(ang[:, 1:2]),
                          vel[:, 0:1], vel[:, 1:2]), dim=1)

    def reset(self):
        self.reference = self.traj.get_next(self.n)
        self.states = self.reference
        self.state = self.states[0]
        self.target = self.reference[0]
        return self._get_obs()

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

        for i, p in enumerate(self.traj.states[:self.n, :2]):
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

    def get_state_from_obs(self, obs):
        batch_size = obs.shape[0]
        reference = obs[:, -self.n * self.m:].view(batch_size, self.n, -1)  # end of array contains reference
        # obs = obs[:, :-self.n * self.m].view(batch_size, self.n, -1)  # everything else
        reference_xy = torch.zeros((batch_size, self.n, 2))
        for t in range(self.n):
            reference_th = torch.cat((torch.atan2(reference[:, t, 1:2], reference[:, t, 0:1]), torch.atan2(reference[:, t, 3:4], reference[:, t, 2:3])), dim=1)
            th1 = reference_th[:, 0:1]
            th2 = reference_th[:, 1:2]
            x1 = self.LINK_LENGTH_1 * torch.cos(th1)
            y1 = self.LINK_LENGTH_1 * torch.sin(th1)
            x2 = x1 + self.LINK_LENGTH_2 * torch.cos(th1 + th2)
            y2 = y1 + self.LINK_LENGTH_2 * torch.sin(th1 + th2)
            reference_xy[:, t] = torch.cat((x2, y2), dim=1)

        return reference_xy.view(batch_size, self.n, 2)


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
    env.render()
    for _ in range(10):
        a = env.action_space.sample()
        env.step(a)
        for t in range(env.n):
            env.state = env.states[t]
            data = env.get_benchmark_data(data)
            env.render()
    env.do_benchmark(data)
    b.add(data)
    b.plot("../img/derivatives.png")
    env.close()


if __name__ == '__main__':
    make_plot()
