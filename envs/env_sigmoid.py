import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


def sigmoid(X):
   return 1/(1+np.exp(-X))


class SigmoidEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.name = 'Sigmoid'
        self.viewer = None
        self.u_max = 1
        self.action_space = spaces.Box(
            low=-self.u_max,
            high=self.u_max, shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0,
            high=1, shape=(1,),
            dtype=np.float32
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        x = self.state

        x += u
        self.state = sigmoid(x)
        return self.state, None, False, {}

    def reset(self):
        self.state = self.np_random.uniform(low=0., high=1.)
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
            self.viewer.set_bounds(0, 1, -.5, .5)
            ball = rendering.make_circle(.01)
            ball.set_color(0, 0, 0)
            self.ball_transform = rendering.Transform()
            ball.add_attr(self.ball_transform)
            self.viewer.add_geom(ball)
            self.imgtrans = rendering.Transform()
            # self.img.add_attr(self.imgtrans)

        # self.viewer.add_onetime(self.img)
        self.ball_transform.set_translation(self.state, 0)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')