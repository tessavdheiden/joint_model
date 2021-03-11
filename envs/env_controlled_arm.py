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

    def __init__(self, n):
        pf = os.path.join(os.getcwd(), os.path.dirname(__file__), 'traj.npy')
        assert os.path.exists(pf)
        self.n = n
        with open(pf, 'rb') as f:
            traj = np.load(f)
        self.N = len(traj)
        assert self.N / n == int(self.N / n)

        self.states = np.zeros((self.N, 4))
        self.states[:, :2] = traj
        self.states[:-1, 2:4] = np.diff(traj, axis=0)
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
    n = 200
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

    s_max = LINK_LENGTH_1 + LINK_LENGTH_2
    s_min = -s_max
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
        # self.state = np.zeros(self.state_dim)
        self.states = np.zeros((self.n, self.state_dim))
        self.p = np.ones(2) * self.GAIN_P
        self.d = np.ones(2) * self.GAIN_D
        self.ptorch = torch.from_numpy(self.p)
        self.dtorch = torch.from_numpy(self.d)
        self.traj = Trajectory(self.n)
        self.state_names = ['θ1', 'θ2', 'dotθ1', 'dotθ2', 'barθ1r', 'barθ2r']
        self.viewer = None

    def single_step(self, s, a, t):
        target = self.reference[t] + a[t]
        th, thdot = s[:2], s[2:4]

        delta_p = angle_normalize(target[:2] - th)
        delta_v = target[2:4] - thdot

        alpha = self.p * delta_p + self.d * delta_v

        newthdot = thdot + alpha * self.dt
        newth = angle_normalize(th + newthdot * self.dt)
        state = np.hstack([newth, newthdot])
        return state

    def step(self, a):
        a = a.reshape(self.n, self.m)
        self.reference = self.traj.get_next()
        state = self.reference[0]
        for t in range(self.n):
            state = self.single_step(state, a, t)
            self.states[t] = state

        obs = self._get_obs()
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
        reference = x[:, -self.n * self.m:].view(batch_size, self.n, self.m)  # end of array contains reference
        x = x[:, :-self.n * self.m].view(batch_size, self.n, self.obs_dim)     # everything else
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
        self.reference = self.traj.get_next()
        self.states = self.reference.copy() # important, otherwise states can change reference
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
            self.reference_transform = [None] * self.n
            for i in range(self.n):
                p = rendering.make_circle(.01)
                self.reference_transform[i] = rendering.Transform()
                p.add_attr(self.reference_transform[i])
                self.viewer.add_geom(p)

        for i, p in enumerate(self.reference):
            s = p
            p1 = [self.LINK_LENGTH_1 * cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

            p2 = [p1[0] + self.LINK_LENGTH_2 * cos(s[0] + s[1]),
                  p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1])]
            self.reference_transform[i].set_translation(*p2)

        images = []
        for t in range(self.n):
            self.draw(self.states[t, :2])
            self.draw(self.reference[t, :2], .2)

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

    def get_state_from_obs(self, obs):
        batch_size = obs.shape[0]
        reference = obs[:, -self.n * self.m:].view(batch_size, self.n, self.m)  # end of array contains reference
        reference_xy = torch.zeros((batch_size, self.n, 2))
        for t in range(self.n):
            th1 = reference[:, t, 0:1]
            th2 = reference[:, t, 1:2]
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
    images = env.render(mode='rgb_array')
    for image in images:
        v.add(image)
    for _ in range(11):
        a = env.action_space.sample()
        env.step(a)
        images = env.render(mode='rgb_array')
        for image in images:
            v.add(image)
    v.save(f'../img/p={env.p}_d={env.d}.gif')
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

    xy, z = [], []
    for b in range(replay_memory.n_batches_train):
        batch_dict = replay_memory.next_batch_train()
        x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])
        obs = x.view(args.batch_size * args.seq_length, -1)  # take first batch
        reference_xy = env.get_state_from_obs(obs).numpy()
        xy.append(reference_xy)
        z.append(np.random.rand(len(reference_xy)))

    xy = np.array(xy).reshape(-1, env.n, 2)
    z = np.array(z).reshape(-1)
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

    for i in range(11):
        a = env.action_space.sample() * 0
        env.step(a)
        data = env.get_benchmark_data(data)
        env.render()

    env.do_benchmark(data)
    b.add(data)
    b.plot("../img/derivatives.png")
    env.close()


if __name__ == '__main__':
    make_plot()
