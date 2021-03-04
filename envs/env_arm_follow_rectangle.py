import numpy as np
from gym import spaces
from numpy import sin, cos, pi
import torch
import torch.nn as nn
import torch.optim as optim


from envs.env_abs import AbsEnv


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Rectange(object):
    def __init__(self, center, w, h, n):
        l = center[0] - w/2
        t = center[1] + h/2
        b = t - h
        r = l + w
        top = np.stack([np.linspace(l, r, n // 4 + 1), np.full(n // 4 + 1, t)], axis=1)[:-1]
        left = np.stack([np.full(n // 4 + 1, l), np.linspace(t, b, n // 4 + 1)], axis=1)[:-1]
        right = left.copy()
        right[:, 0] += w
        bottom = top.copy()
        bottom[:, 1] -= h
        self.points = np.concatenate([top, right, bottom[::-1], left[::-1]])

        self.corners = (l, b), (l, t), (r, t), (r, b)
        self.it = 0

    def get_next(self):
        p = self.points[self.it]
        self.it = (self.it + 1) % len(self.points)
        return p


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        STATE_DIM = 2
        H_DIM = 100

        self.fc = nn.Sequential(nn.Linear(STATE_DIM, H_DIM),
                                nn.ReLU(), nn.BatchNorm1d(H_DIM),
                                nn.Linear(H_DIM, H_DIM),
                                nn.ReLU(), nn.BatchNorm1d(H_DIM),
                                nn.Linear(H_DIM, STATE_DIM))

        self.optimizer = optim.RMSprop(self.fc.parameters(), lr=.0001)

    def forward(self, state):
        return self.fc(state)

    def update(self, error):
        self.optimizer.zero_grad()
        error.backward()
        self.optimizer.step()

    @property
    def networks(self):
        return [self.fc]

    def prepare_update(self):
        if DEVICE == 'cuda':
            self.cast = lambda x: x.cuda()
        else:
            self.cast = lambda x: x.cpu()

        for network in self.networks:
            network = self.cast(network)
            network.train()

    def prepare_eval(self):
        self.cast = lambda x: x.cpu()
        for network in self.networks:
            network = self.cast(network)
            network.eval()


class ArmFollowRectangleEnv(AbsEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    dt = .2

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
        self.rect = Rectange((0, 0), 2, 3, 1000)

        self.viewer = None

        self.u_low_torch = torch.from_numpy(self.u_low).float()
        self.u_high_torch = torch.from_numpy(self.u_high).float()

    def step(self, u):
        self.target = self.rect.get_next()
        u = np.clip(u, self.u_low, self.u_high)

        self.state += u * self.dt

        obs = self._get_obs()
        r = self._r_func(obs[4:])
        return obs, r, self.get_point, {}

    def _reset_arm(self):
        self.state = np.array([0, -pi])

    def _reset_target(self):
        self.target = self.rect.get_next()

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
            (l, b), (l, t), (r, t), (r, b) = self.rect.corners
            self.traj = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)], filled=False)
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

def angle_normalize(x):
    return (((x+pi) % (2*pi)) - pi)


def solve_trajectory():
    N_ITER = 10
    net = Net()
    traj = np.zeros((len(env.rect.points), 2))

    for j, txy in enumerate(env.rect.points):
        env.target = txy
        txy = torch.from_numpy(txy).unsqueeze(0).float()
        for i in range(N_ITER):
            net.prepare_eval()
            thetas = net(txy)

            xy_ = torch.cat((env.LINK_LENGTH_1 * torch.cos(thetas[:, :1]) + env.LINK_LENGTH_2 * torch.cos(
                thetas[:, :1] + thetas[:, 1:]),
                             env.LINK_LENGTH_1 * torch.sin(thetas[:, :1]) + env.LINK_LENGTH_2 * torch.sin(
                                 thetas[:, :1] + thetas[:, 1:])), dim=1)

            error = ((txy - xy_) ** 2).sum()
            net.prepare_update()
            net.update(error)
            net.prepare_eval()

        traj[j] = thetas.detach().numpy().squeeze(0)
    return traj


if __name__ == '__main__':
    from viz import *
    v = Video()

    env = ArmFollowRectangleEnv()
    env.seed()

    traj = solve_trajectory()
    with open('traj.npy', 'wb') as f:
        np.save(f, traj)

    # rendering
    with open('traj.npy', 'rb') as f:
        traj = np.load(f)

    for j, txy in enumerate(env.rect.points):
        env.target = txy
        env.state = traj[j]
        v.add(env.render(mode='rgb_array'))
    v.save("test.gif")

