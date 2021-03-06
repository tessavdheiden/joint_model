import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch

from os import path

MASS = 2.
DELTA_T = .05
MAX_TORQUE = 1.

from envs.env_abs import AbsEnv


class PendulumEnv(AbsEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    u_low = np.array([-1])
    u_high = np.array([1])
    action_dim = 1
    action_space = spaces.Box(
        low=u_low,
        high=u_high, shape=(1,),
        dtype=np.float32
    )
    max_speed = 8
    high = np.array([1., 1., max_speed], dtype=np.float32)
    observation_space = spaces.Box(
        low=-high,
        high=high,
        dtype=np.float32
    )
    state_names = ['θ', 'dotθ']

    def __init__(self, g=10.0):
        self.name = 'Pendulum'

        self.u_max = MAX_TORQUE
        self.dt = DELTA_T
        self.g = g
        self.m = MASS
        self.l = 1.
        self.viewer = None
        self.seed()
        self.obs_name = ['cos(θ)', 'sin(θ)', 'dotθ']

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, self.u_low, self.u_high)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, self.max_speed])
        self.state = self.np_random.uniform(low=-high, high=high)
        #self.state[0] = np.pi / 2
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(gym.__file__[:-12], "envs/classic_control/assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def step_batch(self, x, u):
        th, thdot = torch.atan2(x[:, 1], x[:, 0]).view(-1, 1), x[:, 2].view(-1, 1)  # th := theta

        max_speed = 8
        dt = DELTA_T
        g = 10.0
        m = MASS
        l = 1.

        up = torch.from_numpy(self.u_high).float()
        lo = torch.from_numpy(self.u_low).float()
        u = torch.max(torch.min(u, up), lo)

        newthdot = thdot + (-3 * g / (2 * l) * torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -max_speed, max_speed)

        x = torch.stack((torch.cos(newth), torch.sin(newth), newthdot), dim=-1).squeeze(1)
        return x

    def get_state_from_obs(self, obs):
        pos, vel = obs[:, :2], obs[:, 2:]
        angle = torch.atan2(pos[:, 1:2], pos[:, 0:1])
        return torch.cat((angle, vel), dim=1)

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


if __name__ == '__main__':
    env = PendulumEnv()
    env.seed()
    env.reset()
    for _ in range(1000):
        env.render()
        a = env.action_space.sample()
        env.step(a)
    env.close()