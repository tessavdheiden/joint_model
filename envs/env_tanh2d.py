import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch

from envs.env_abs import AbsEnv


class Tanh2DEnv(AbsEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    dt = .1
    u_low = np.array([-1., -1.])
    u_high = np.array([1., 1.])

    action_space = spaces.Box(
        low=u_low,
        high=u_high, shape=(2,),
        dtype=np.float32
    )
    s_max = 1
    s_min = -1
    observation_space = spaces.Box(
        low=s_min,
        high=s_max, shape=(2,),
        dtype=np.float32
    )
    state_names =['x', 'y']

    def __init__(self):
        self.name = 'Sigmoid2D'
        self.viewer = None

    def step(self, u):
        x = self.state
        u = np.clip(u, self.u_low, self.u_high)
        x += u * self.dt
        self.state = np.tanh(x)
        return self.state, None, False, {}

    def reset(self):
        self.state = self.np_random.uniform(low=self.s_min, high=self.s_max, size=(2, ))
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)

            self.viewer.set_bounds(left=self.s_min, right=self.s_max, bottom=self.s_min, top=self.s_max)
            ball = rendering.make_circle(.1)
            ball.set_color(0, 0, 0)
            self.ball_transform = rendering.Transform()
            ball.add_attr(self.ball_transform)
            self.viewer.add_geom(ball)
            self.imgtrans = rendering.Transform()
            # self.img.add_attr(self.imgtrans)

        # self.viewer.add_onetime(self.img)
        self.ball_transform.set_translation(self.state[0], self.state[1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def step_batch(self, x, u):
        up = torch.from_numpy(self.u_high).float()
        lo = torch.from_numpy(self.u_low).float()
        u = torch.max(torch.min(u, up), lo)
        return torch.tanh(x + u * self.dt)

    def get_state_from_obs(self, obs):
        return obs


if __name__ == '__main__':
    env = Tanh2DEnv()
    env.seed()

    for _ in range(100):
        env.reset()
        for _ in range(32):
            env.render()
            a = env.action_space.sample()
            env.step(a)
    env.close()