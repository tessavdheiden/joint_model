import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class BallBoxEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.m = .1
        self.w = .25
        self.dt = .05
        self.viewer = None
        self.damping = 0.25
        self.max_speed = 10
        self.max_dim = 1

        self.action_space = spaces.Box(
            low=-self.max_speed,
            high=self.max_speed, shape=(2,),
            dtype=np.float32
        )
        high = np.array([self.max_dim, self.max_dim], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        x, y = self.state
        p = np.array([x, y])

        p += u * self.dt
        self.state = np.clip(p, -self.max_dim + self.w / 2, self.max_dim - self.w / 2)
        return self.state, None, False, {}

    def reset(self):
        high = np.array([self.max_dim - self.w / 2, self.max_dim - self.w / 2])
        self.state = self.np_random.uniform(low=-high, high=high)
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
            self.viewer.set_bounds(-self.max_dim, self.max_dim, -self.max_dim, self.max_dim)
            ball = rendering.make_circle(self.w)
            ball.set_color(0, 0, 0)
            self.ball_transform = rendering.Transform()
            ball.add_attr(self.ball_transform)
            self.viewer.add_geom(ball)
            self.imgtrans = rendering.Transform()
            # self.img.add_attr(self.imgtrans)

        # self.viewer.add_onetime(self.img)
        self.ball_transform.set_translation(self.state[0], self.state[1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')