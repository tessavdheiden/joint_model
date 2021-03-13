import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch
from envs.env_abs import AbsEnv


class BallBoxForceEnv(AbsEnv):
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

    max_speed = 1.
    max_pos = 1.
    max_acc = 1.
    damping = 0.25
    mass = 1.
    s_max = np.array([max_pos, max_pos, max_speed, max_speed, max_acc, max_acc, 2*max_pos, 2*max_pos])
    s_min = -s_max
    observation_space = spaces.Box(
        low=-s_max,
        high=s_min, shape=(8,),
        dtype=np.float32
    )

    state_names = ['x', 'y', 'dotx', 'doty', 'ddotx', 'ddoty', 'tx', 'ty']

    def __init__(self):
        self.seed()
        self.name = 'BallInBoxForce'
        self.viewer = None
        self.state = np.zeros(8)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        x, dotx, ddotx, t = self.state[:2], self.state[2:4], self.state[4:6], self.state[6:8]

        noise = np.random.random(x.shape) - .5
        delta_x = (t - x) #+ noise
        ddotx = np.clip(ddotx + u / self.mass * delta_x * self.dt, -self.max_acc, self.max_acc)
        dotx = dotx * (1-self.damping)
        dotx = dotx + ddotx * self.dt

        x = x + dotx * self.dt
        self.state = np.concatenate([x, dotx, ddotx, t])
        cost = np.sqrt(np.sum(np.square(delta_x)))

        return self._get_obs(), -cost, False, {}

    def step_batch(self,o, u):
        x, dotx, ddotx, t = o[:, :2], o[:, 2:4], o[:, 4:6], o[:, 6:8]

        noise = torch.rand(x.shape) - .5
        delta_x = (t - x)# + noise
        ddotx = torch.clamp(ddotx + u / self.mass * delta_x * self.dt, -self.max_acc, self.max_acc)
        dotx = dotx * (1 - self.damping)
        dotx = dotx + ddotx * self.dt

        x = x + dotx * self.dt
        return torch.cat((x, dotx, ddotx, t), dim=1)

    def reset(self):
        self.state[:2] = np.random.rand(2) * 2 * self.max_pos - self.max_pos
        self.state[2:4] = np.random.rand(2) * 2 * self.max_speed - self.max_speed
        self.state[4:6] = np.random.rand(2) * 2 * self.max_acc - self.max_acc
        self.state[6:8] = np.random.rand(2) * 2 * self.max_pos - self.max_pos
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        return self.state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_state_from_obs(self, obs):
        return obs

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-1, 1, -1, 1)
            target = rendering.make_circle(.1)
            target.set_color(0, 0, 0)
            self.target_transform = rendering.Transform()
            target.add_attr(self.target_transform)
            self.viewer.add_geom(target)

            ball = rendering.make_circle(.1)
            ball.set_color(1., 0, 0)
            self.ball_transform = rendering.Transform()
            ball.add_attr(self.ball_transform)
            self.viewer.add_geom(ball)


        self.target_transform.set_translation(self.state[-2], self.state[-1])
        self.ball_transform.set_translation(self.state[0], self.state[1])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


if __name__ == '__main__':
    env = BallBoxForceEnv()
    env.seed()
    env.reset()
    for _ in range(100):
        env.render()
        a = env.action_space.sample()
        env.step(a)
    env.close()