import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch
from envs.env_abs import AbsEnv


class BallBoxEnv(AbsEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    dt = .5
    u_low = np.array([-1., -1.])
    u_high = np.array([1., 1.])
    action_dim = 2
    action_space = spaces.Box(
        low=u_low,
        high=u_high, shape=(action_dim,),
        dtype=np.float32
    )
    s_max = 1
    observation_space = spaces.Box(
        low=-s_max,
        high=s_max, shape=(2,),
        dtype=np.float32
    )

    def __init__(self):
        self.seed()
        self.name = 'BallInBox'
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        x = self.state
        u = np.clip(u, self.u_low, self.u_high)
        x += u * self.dt
        self.state = np.clip(x, -self.s_max, self.s_max)
        return self.state, cost, False, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-self.s_max, high=self.s_max, size=(2, ))
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
            self.viewer.set_bounds(-1, 1, -1, 1)
            ball = rendering.make_circle(.1)
            ball.set_color(0, 0, 0)
            self.ball_transform = rendering.Transform()
            ball.add_attr(self.ball_transform)
            self.viewer.add_geom(ball)

        self.ball_transform.set_translation(self.state[0], self.state[1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def step_batch(self, x, u):
        up = torch.from_numpy(self.u_high).float()
        lo = torch.from_numpy(self.u_low).float()
        u = torch.max(torch.min(u, up), lo)
        return torch.clamp(x + u * self.dt, -self.s_max, self.s_max)


if __name__ == '__main__':
    env = BallBoxEnv()
    env.seed()
    env.reset()
    for _ in range(100):
        env.render()
        a = env.action_space.sample()
        env.step(a)
    env.close()