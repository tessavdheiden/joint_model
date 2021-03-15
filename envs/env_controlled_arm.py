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

    def __init__(self, n, order, dt):
        pf = os.path.join(os.getcwd(), os.path.dirname(__file__), 'traj.npy')
        assert os.path.exists(pf)
        self.n = n
        with open(pf, 'rb') as f:
            traj = np.load(f)
        self.N = len(traj)
        assert self.N / n == int(self.N / n)

        self.states = np.zeros((self.N, (order+1) * 2))
        self.states[:, :2] = traj
        for i in range(1, order+1):
            l1, r1 = i*2, i*2+2
            l2, r2 = (i-1)*2, (i-1)*2+2
            self.states[i:, l1:r1] = np.diff(self.states[i-1:, l2:r2], axis=0) / dt

        self.it = 0

    def get_next(self):
        res = self.states[self.it:self.it + self.n]
        self.it = (self.it + self.n) % len(self.states)
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
    n = 50
    action_dim = 4
    state_dim = 8
    m = Trajectory.observation_space.shape[0]

    u_high = np.ones(action_dim)
    u_low = -u_high
    action_space = spaces.Box(
        low=u_low,
        high=u_high, shape=(action_dim,),
        dtype=np.float32
    )

    s_max = LINK_LENGTH_1 + LINK_LENGTH_2
    s_min = -s_max
    obs_dim = 12  # +2 for cos and sin
    observation_space = spaces.Box(
        low=-s_max,
        high=s_max, shape=(obs_dim,),
        dtype=np.float32
    )

    point_l = .5
    grab_counter = 0

    def __init__(self):
        self.name = 'ControlledArm'
        self.traj = Trajectory(self.n, 2, self.dt)
        self.state_names = ['θ1', 'θ2', 'dotθ1', 'dotθ2', 'θ1r', 'θ2r', 'dotθ1r', 'dotθ2r']
        self.obs_names = ['x1', 'y1', 'x2', 'y2', 'ω1', 'ω2', 'x1r', 'y1r', 'x2r', 'y2r', 'ω1r', 'ω2r']
        self.viewer = None

    def step(self, u):
        x, dotx, t, dott = self.state[:2], self.state[2:4], self.state[4:6], self.state[6:8]
        dx, dv = u[0:2], u[2:4]
        for i in range(self.n):
            self.states[i] = x
            self.targets[i] = t

            delta_x, delta_v = t - x, dott - dotx
            ddotx = delta_x * dx + delta_v * dv
            dotx = dotx + ddotx * self.dt
            x = x + dotx * self.dt

            t = t + dott * self.dt

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

    def step_batch(self, x, u):
        state = self.get_state_from_obs(x)
        x, dotx, t, dott = state[:, :2], state[:, 2:4], state[:, 4:6], state[:, 6:8]

        dx, dv = u[:, 0:2], u[:, 2:4]
        for i in range(self.n):
            delta_x, delta_v = t - x, dott - dotx
            ddotx = delta_x * dx + delta_v * dv
            dotx = dotx + ddotx * self.dt
            x = x + dotx * self.dt
            t = t + dott * self.dt

        state = torch.cat((x, dotx, t, dott), dim=1)
        obs = self._get_obs_from_state(state)
        return obs

    def reset(self):
        i = np.random.choice(len(self.traj.states)-self.n)
        reference = self.traj.states[i].copy()
        self.state = np.zeros(8)
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
        for t in range(self.n):
            self.draw(self.states[t])
            self.draw(self.targets[t], .2)

            images.append(self.viewer.render(return_rgb_array=mode == 'rgb_array'))
        return images

    def get_benchmark_data(self, data={}):
        if len(data) == 0:
            names = ['θ1', 'θ2', 'dotθ1', 'dotθ2' ,'ddotθ1', 'ddotθ2']
            data = {name: [] for name in names}

        data['θ1'].append(self.states[:, 0].copy())
        data['θ2'].append(self.states[:, 1].copy())
        data['dotθ1'].append(self.states[:, 2].copy())
        data['dotθ2'].append(self.states[:, 3].copy())

        return data

    def do_benchmark(self, data):
        names = list(data.keys())
        # convert to numpy
        for i, name in enumerate(names[:4]):
            data[name] = np.array(data[name])

        data['ddotθ1'] = np.diff(data['dotθ1'], axis=1) / self.dt
        data['ddotθ2'] = np.diff(data['dotθ2'], axis=1) / self.dt

        # take max for each trajectories
        for i, name in enumerate(names):
            data[name] = np.max(data[name], axis=1)

        return data


def make_video():
    from viz.video import Video
    v = Video()
    env = ControlledArmEnv()
    env.seed()
    env.reset()
    for _ in range(4):
        a = env.action_space.sample() * 0 + np.array([1, 1, 8, 8])
        env.step(a)
        images = env.render(mode='rgb_array')
        for image in images:
            v.add(image)
    v.save(f'../img/vid.gif')
    env.close()


def test_selection_plot():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='fraction of data to be witheld in validation set')
    parser.add_argument('--seq_length', type=int, default=2, help='sequence length for training')
    parser.add_argument('--batch_size', type=int, default=32, help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=2001, help='number of epochs')
    parser.add_argument('--n_trials', type=int, default=64,
                        help='number of data sequences to collect in each episode')
    parser.add_argument('--trial_len', type=int, default=2, help='number of steps in each trial')
    parser.add_argument('--n_subseq', type=int, default=1,
                        help='number of subsequences to divide each sequence into')
    parser.add_argument('--env', type=str, default='controlled_arm',
                        help='pendulum, ball_in_box, tanh2d, arm, reacher, pendulum')
    parser.add_argument('--filter_type', type=int, default=1,
                        help='0=bayes filter, 1=bayes filter fully connected')
    parser.add_argument('--use_filter', type=int, default=0, help='0=env, 1=filter')
    parser.add_argument('--n_step', type=int, default=1, help='empowerment calculated over n actions')
    args = parser.parse_args()
    from memory.replay_memory import ReplayMemory

    from viz.selection_plot import SelectionPlot
    sp = SelectionPlot()
    env = ControlledArmEnv()
    env.seed()
    replay_memory = ReplayMemory(args, env)
    replay_memory.reset_batchptr_train()

    xy_list, z_list = [], []
    for b in range(replay_memory.n_batches_train):
        batch_dict = replay_memory.next_batch_train()
        x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])
        obs = x.view(args.batch_size * args.seq_length, -1)  # take first batch
        reference_xy = env.get_state_from_obs(obs).numpy()
        reference_ddotxy = np.sum(np.square(np.diff(np.diff(reference_xy, axis=1))), axis=2)
        z = []
        for ref_ddotxy, ref_xy in zip(reference_ddotxy, reference_xy):
            z.append(ref_ddotxy.max(0))
        xy_list.append(reference_xy)
        z_list.append(np.array(z))

    xy = np.array(xy_list).reshape(-1, env.n, 2)
    z = np.array(z_list).reshape(-1)
    sp.add(xy, z)
    sp.plot(f"../img/selection", env.s_min, env.s_max, env.s_min, env.s_max)


def make_plot():
    from viz.benchmark_plot import BenchmarkPlot
    b = BenchmarkPlot()

    env = ControlledArmEnv()
    env.seed()
    env.reset()
    data = env.get_benchmark_data()
    env.render()

    for i in range(4):
        a = env.action_space.sample() * 0
        env.step(a)
        data = env.get_benchmark_data(data)
        env.render()

    env.do_benchmark(data)
    b.add(data)
    b.plot("../img/derivatives.png")
    env.close()


if __name__ == '__main__':
    make_video()

