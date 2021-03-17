import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
from gym import spaces
from numpy import pi, cos, sin


from envs.env_abs import AbsEnv
from envs.misc import Trajectory, rk4, angle_normalize



class ReacherControlledEnv(nn.Module, AbsEnv):
    dt = .1

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2

    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi

    MAX_TORQUE = 1.

    action_dim = 4
    u_high = np.ones(action_dim)
    u_low = -u_high
    action_space = spaces.Box(
        low=u_low,
        high=u_high, shape=(action_dim,),
        dtype=np.float32
    )

    # cos/sin of 2 angles, 2 angular vel, 2 angle errors, 2 vel errors, 2 p's, 2 d's
    high = LINK_LENGTH_1 + LINK_LENGTH_2
    obs_dim = 12  # +2 for cos and sin
    observation_space = spaces.Box(
        low=-high,
        high=high, shape=(12,),
        dtype=np.float32
    )
    viewer = None
    target = None

    point_l = .5
    grab_counter = 0

    def __init__(self):
        super(ReacherControlledEnv, self).__init__()
        self.name = 'controlled_reacher'
        self.state = np.zeros(8)
        self.t0 = nn.Parameter(torch.tensor([0.0]))
        self.n = 4
        self.traj = Trajectory(800, 3, self.dt)
        self.state_names = ['θ1', 'θ2', 'dotθ1', 'dotθ2', 'θ1r', 'θ2r', 'dotθ1r', 'dotθ2r']
        self.obs_names = ['x1', 'y1', 'x2', 'y2', 'ω1', 'ω2', 'x1r', 'y1r', 'x2r', 'y2r', 'ω1r', 'ω2r']

    def step(self, u):
        x, dotx, t, dott = self.state[:2], self.state[2:4], self.state[4:6], self.state[6:8]
        dx, dv = u[0:2], u[2:4]

        for i in range(self.n):
            t = self.traj.states[self.it + i, :2]
            dott = self.traj.states[self.it + i, 2:4]
            self.states[i] = x
            self.targets[i] = t
            delta_x, delta_v = t - x, dott - dotx
            torque = delta_x * dx + delta_v * dv

            x_ = rk4(self._dsdt, np.hstack([x, dotx, torque]), [0, self.dt])[-1]
            x = x_[:2]
            dotx = x_[2:4]
            # t = t + dott * self.dt

        self.state = np.concatenate([x, dotx, t, dott])
        obs = self._get_obs()
        return obs, None, [], {}

    def _get_obs(self):
        l1, l2 = self.LINK_LENGTH_1, self.LINK_LENGTH_2
        x = self.state[:2]
        xy1 = np.array([l1*cos(x[0]), l1*sin(x[0])])
        xy2 = xy1 + np.array([l2*cos(x[0]+x[1]), l2*sin(x[0]+x[1])])

        t = self.state[4:6]
        txy1 = np.array([l1 * cos(t[0]), l1 * sin(t[0])])
        txy2 = txy1 + np.array([l2 * cos(t[0] + t[1]), l2 * sin(t[0] + t[1])])

        return np.concatenate([xy1, xy2, self.state[2:4], txy1, txy2, self.state[6:8]])

    def get_state_from_obs(self, obs):
        # s = np.random.rand(2) * 2 * pi - pi
        # obs = env._get_obs_from_state(torch.from_numpy(s).unsqueeze(0))
        # assert all(s == env.get_state_from_obs(obs).squeeze(0).numpy())
        xy1, xy2, dotx, txy1, txy2, dott = obs[:, 0:2], obs[:, 2:4], obs[:, 4:6], obs[:, 6:8], obs[:, 8:10], obs[:, 10:12]
        th1 = torch.atan2(xy1[:, 1:2], xy1[:, 0:1])
        th2 = torch.atan2(xy2[:, 1:2] - xy1[:, 1:2], xy2[:, 0:1] - xy1[:, 0:1]) - th1
        th = torch.cat((th1, th2), dim=1)

        tth1 = torch.atan2(txy1[:, 1:2], txy1[:, 0:1])
        tth2 = torch.atan2(txy2[:, 1:2] - txy1[:, 1:2], txy2[:, 0:1] - txy1[:, 0:1]) - tth1
        tth = torch.cat((tth1, tth2), dim=1)
        return torch.cat((angle_normalize(th), dotx, angle_normalize(tth), dott), dim=1)

    def _get_obs_from_state(self, x):
        l1, l2 = self.LINK_LENGTH_1, self.LINK_LENGTH_2
        xy1 = torch.cat((l1 * torch.cos(x[:, 0:1]), l1 * torch.sin(x[:, 0:1])), dim=1)
        xy2 = xy1 + torch.cat([l2 * torch.cos(x[:, 0:1] + x[:, 1:2]), l2 * torch.sin(x[:, 0:1] + x[:, 1:2])], dim=1)

        t = x[:, 4:6]
        txy1 = torch.cat([l1 * torch.cos(t[:, 0:1]), l1 * torch.sin(t[:, 0:1])], dim=1)
        txy2 = txy1 + torch.cat([l2 * torch.cos(t[:, 0:1] + t[:, 1:2]), l2 * torch.sin(t[:, 0:1] + t[:, 1:2])], dim=1)
        return torch.cat((xy1, xy2, x[:, 2:4], txy1, txy2, x[:, 6:8]), dim=1)

    def reset(self):
        i = np.random.choice(len(self.traj.states)-self.n - 1)
        self.it = i
        reference = self.traj.states[i].copy()
        x, t = reference[:2], reference[:2]
        dotx, dott = reference[2:4], reference[2:4]
        self.state[:2], self.state[2:4], self.state[4:6], self.state[6:8] = x, dotx, t, dott
        self.states = np.zeros((self.n, 2))
        self.targets = np.zeros((self.n, 2))

        return self._get_obs()

    def draw(self, s, alpha=1.):
        from gym.envs.classic_control import rendering
        if s is None: return None

        l1, l2 = self.LINK_LENGTH_1, self.LINK_LENGTH_2
        p1 = [l1 * cos(s[0]), l1 * sin(s[0])]

        p2 = [p1[0] + l2 * cos(s[0] + s[1]),
              p1[1] + l2 * sin(s[0] + s[1])]

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
            self.reference_transform = [None] * self.n
            for i in range(self.n):
                p = rendering.make_circle(.01)
                self.reference_transform[i] = rendering.Transform()
                p.add_attr(self.reference_transform[i])
                self.viewer.add_geom(p)

        l1, l2 = self.LINK_LENGTH_1, self.LINK_LENGTH_2
        for i in range(self.n):
            t = self.targets[i]
            txy1 = np.array([l1 * cos(t[0]), l1 * sin(t[0])])
            txy2 = txy1 + np.array([l2 * cos(t[0] + t[1]), l2 * sin(t[0] + t[1])])
            self.reference_transform[i].set_translation(*txy2)

        images = []
        for i in range(self.n):
            self.draw(self.states[i])
            self.draw(self.targets[i], .2)

            images.append(self.viewer.render(return_rgb_array=mode == 'rgb_array'))
        return images

    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        L1 = self.LINK_LENGTH_1
        L2 = self.LINK_LENGTH_2
        a = s_augmented[-2:]
        s = s_augmented[:-2]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        tau1 = a[0]
        tau2 = a[1]

        # run equations_of_motion() to compute these
        ddtheta1 = (-L1 * np.cos(theta2) - L2) * (-L1 * L2 * dtheta1 ** 2 * m2 * np.sin(theta2) + tau2) / (
                    L1 ** 2 * L2 * m1 - L1 ** 2 * L2 * m2 * np.cos(theta2) ** 2 + L1 ** 2 * L2 * m2) + (
                               L1 * L2 * m2 * (2 * dtheta1 * dtheta2 + dtheta2 ** 2) * np.sin(theta2) + tau1) / (
                               L1 ** 2 * m1 - L1 ** 2 * m2 * np.cos(theta2) ** 2 + L1 ** 2 * m2)
        ddtheta2 = (-L1 * np.cos(theta2) - L2) * (
                    L1 * L2 * m2 * (2 * dtheta1 * dtheta2 + dtheta2 ** 2) * np.sin(theta2) + tau1) / (
                               L1 ** 2 * L2 * m1 - L1 ** 2 * L2 * m2 * np.cos(theta2) ** 2 + L1 ** 2 * L2 * m2) + (
                               -L1 * L2 * dtheta1 ** 2 * m2 * np.sin(theta2) + tau2) * (
                               L1 ** 2 * m1 + L1 ** 2 * m2 + 2 * L1 * L2 * m2 * np.cos(theta2) + L2 ** 2 * m2) / (
                               L1 ** 2 * L2 ** 2 * m1 * m2 - L1 ** 2 * L2 ** 2 * m2 ** 2 * np.cos(
                           theta2) ** 2 + L1 ** 2 * L2 ** 2 * m2 ** 2)

        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0., 0.)

    def forward(self, t, s):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        L1 = self.LINK_LENGTH_1
        L2 = self.LINK_LENGTH_2

        s, u = s

        tau1, tau2 = u[:, 0:1], u[:, 1:2]
        theta1, theta2 = s[:, 0:1], s[:, 1:2]
        dtheta1, dtheta2 = s[:, 2:3], s[:, 3:4]

        # run equations_of_motion() to compute these
        ddtheta1 = (-L1 * torch.cos(theta2) - L2) * (-L1 * L2 * dtheta1 ** 2 * m2 * torch.sin(theta2) + tau2) / (
                    L1 ** 2 * L2 * m1 - L1 ** 2 * L2 * m2 * torch.cos(theta2) ** 2 + L1 ** 2 * L2 * m2) + (
                               L1 * L2 * m2 * (2 * dtheta1 * dtheta2 + dtheta2 ** 2) * torch.sin(theta2) + tau1) / (
                               L1 ** 2 * m1 - L1 ** 2 * m2 * torch.cos(theta2) ** 2 + L1 ** 2 * m2)
        ddtheta2 = (-L1 * torch.cos(theta2) - L2) * (
                    L1 * L2 * m2 * (2 * dtheta1 * dtheta2 + dtheta2 ** 2) * torch.sin(theta2) + tau1) / (
                               L1 ** 2 * L2 * m1 - L1 ** 2 * L2 * m2 * torch.cos(theta2) ** 2 + L1 ** 2 * L2 * m2) + (
                               -L1 * L2 * dtheta1 ** 2 * m2 * torch.sin(theta2) + tau2) * (
                               L1 ** 2 * m1 + L1 ** 2 * m2 + 2 * L1 * L2 * m2 * torch.cos(theta2) + L2 ** 2 * m2) / (
                               L1 ** 2 * L2 ** 2 * m1 * m2 - L1 ** 2 * L2 ** 2 * m2 ** 2 * torch.cos(
                           theta2) ** 2 + L1 ** 2 * L2 ** 2 * m2 ** 2)

        return (dtheta1, dtheta2, ddtheta1, ddtheta2, torch.zeros_like(tau1), torch.zeros_like(tau2))

    # def get_initial_obs_action(self):
    #     state = torch.from_numpy(self.state).float().unsqueeze(0)
    #     target = torch.from_numpy(self.target).float().unsqueeze(0)
    #     p = torch.from_numpy(self.p).float().unsqueeze(0)
    #     d = torch.from_numpy(self.d).float().unsqueeze(0)
    #     s = self._get_obs_from_state(state, target, p, d)
    #
    #     self.init_s = nn.Parameter(s).squeeze(0)
    #     self.init_u = nn.Parameter(torch.tensor([.0, .0, .0, .0]))
    #     return self.t0, (self.init_s, self.init_u)

    def step_batch(self, x, u, update=False):
        state = self.get_state_from_obs(x)
        x, dotx, t, dott = state[:, :2], state[:, 2:4], state[:, 4:6], state[:, 6:8]
        dx, dv = u[:, 0:2], u[:, 2:4]

        for i in range(self.n):
            delta_x, delta_v = t - x, dott - dotx
            ddotx = delta_x * dx + delta_v * dv
            dotx = dotx + ddotx * self.dt
            t = t + dott * self.dt

            s_aug = (torch.cat((x, dotx), dim=1), ddotx)
            x_ = odeint(self, s_aug, torch.tensor([0, self.dt]), method='rk4')[0] # leave out action
            x_ = x_[-1] # last time step

            t = t + dott * self.dt
            dotx = x_[:, 2:4]
            x = torch.where(torch.abs(dotx[:, 1:2]) < 0.1, x, x + dotx * self.dt)

        state = torch.cat((x, dotx, t, dott), dim=1)
        obs = self._get_obs_from_state(state)
        return obs

def make_video():
    from viz.video import Video
    v = Video()
    env = ReacherControlledEnv()
    env.seed()
    env.reset()

    for _ in range(1000):
        env.reset()
        a = env.action_space.sample() * 0 + np.array([8., 8., 1., 1.])
        env.step(a)
        images = env.render(mode='rgb_array')
        for image in images:
            v.add(image)
    v.save(f'../img/vid.gif')
    env.close()


def make_plot():
    from viz.benchmark_plot import BenchmarkPlot
    env = ReacherControlledEnv()
    env.reset()
    set(env, params='good', task="turn quarter circle")

    b = BenchmarkPlot()
    data = env.get_benchmark_data()
    for _ in range(200):
        env.render(mode='rgb_array')
        a = env.action_space.sample() * 0
        env.step(a)

        data = env.get_benchmark_data(data)
    env.do_benchmark(data)
    b.add(data)
    b.plot("../img/derivatives.png")
    env.close()


def use_torchdiffeq():
    from itertools import permutations, product
    from viz.benchmark_plot import BenchmarkPlot
    from viz.video import Video
    env = ReacherControlledEnv()

    l = [-pi, 0, pi]
    print(l)
    for j in range(len(l)):
        v = Video()
        b = BenchmarkPlot()
        env.reset()
        env.state = np.array([0, l[j], 0, 0])
        env.target = env.state[:2]
        env.p = np.ones_like(env.p) * env.MAX_GAIN_P
        env.d = np.ones_like(env.p) * env.MAX_GAIN_D
        t0, obs_action = env.get_initial_obs_action()
        obs = obs_action[0].unsqueeze(0)
        data = env.get_benchmark_data()
        for i in range(100):
            a = np.zeros_like(env.action_space.sample())
            obs = env.step_batch(x=obs, u=torch.from_numpy(a).float().unsqueeze(0), update=True)
            state = env.get_state_from_obs(obs)[:, :4]  # for rendering
            env.state = state.squeeze(0).detach().numpy()
            v.add(env.render(mode='rgb_array'))
            data = env.get_benchmark_data(data)
        env.do_benchmark(data)
        b.add(data)
        b.plot(f"../img/p={env.p}_d={env.d}.png")
        v.save(f'../img/p={env.p}_d={env.d}_θ2={l[j]:.2f}.gif')

    env.close()



def make_landscape_plot():
    from viz.landscape_plot import LandscapePlot
    import pandas as pd
    lp = LandscapePlot()

    env = ReacherControlledEnv()
    env.seed()
    env.reset()
    obsv = []
    for _ in range(10000):
        env.reset()
        a = env.action_space.sample() * 0 + np.array([1., 1., 1., 1.])
        obs = env.step(a)[0]
        o = torch.from_numpy(obs).unsqueeze(0)
        obsv.append(o)
        #images = env.render(mode='rgb_array')

    obsv = torch.cat(obsv, dim=0).numpy()
    import matplotlib.pyplot as plt
    plt.scatter(obsv[:, 2], obsv[:, 3])
    plt.savefig('../img/test')
    lp.add(xy=pd.DataFrame(obsv[:, 2:4], index=np.arange(len(obsv)), columns=['x2', 'y2']), z=np.abs(obsv[:, 5:6]))
    lp.plot(f'../img/test')
    env.close()

if __name__ == '__main__':
    make_video()



