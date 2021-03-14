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

    n = 3
    action_dim = 2 * 3
    u_high = np.ones(action_dim)
    u_low = -u_high
    action_space = spaces.Box(
        low=u_low,
        high=u_high, shape=(action_dim,),
        dtype=np.float32
    )

    max_speed = 1.
    max_pos = 1.
    max_acc = 1.
    mass = 1.5
    damping = 0.25
    s_max = np.array([max_pos, max_pos, max_speed, max_speed, max_acc, max_acc, 2*max_pos, 2*max_pos, 2*max_speed, 2*max_speed, 2*max_acc, 2*max_acc, 2*max_acc, 2*max_acc])
    s_min = -s_max
    observation_space = spaces.Box(
        low=-s_max,
        high=s_min, shape=(14,),
        dtype=np.float32
    )

    state_names = ['x', 'y', 'dotx', 'doty', 'ddotx', 'ddoty', 'dddotx', 'dddoty', 'Δx', 'Δy', 'dotΔx', 'dotΔy', 'ddotx_r', 'ddoty_r']

    def __init__(self):
        self.seed()
        self.name = 'BallInBoxForce'
        self.viewer = None
        self.state = np.zeros(14)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        x, dotx, ddotx, dddotx, delta_x, delta_v, ddott = self.state[:2], self.state[2:4], self.state[4:6], self.state[6:8], self.state[8:10], self.state[10:12], self.state[12:14]

        cost = 0
        t, dott = delta_x + x, delta_v + dotx
        for i in range(self.n):
            delta_x, delta_v, delta_a = t - x, dott - dotx, ddott - ddotx
            # dddotx = (delta_x * u[i * 6:i * 6 + 2] + delta_v * u[i * 6 + 2:i * 6 + 4] + delta_a * u[i * 6 + 4:i * 6 + 6])
            dddotx = delta_x * u[0:2] + delta_v * u[2:4] + delta_a * u[4:6]
            ddotx = ddotx / self.mass
            ddotx = (ddotx + dddotx * self.dt)
            # dotx = dotx * (1-self.damping)
            dotx = dotx + ddotx * self.dt
            x = x + dotx * self.dt
            #ddott = ddott / self.mass
            dott = dott + ddott * self.dt
            t = t + dott * self.dt
            # cost += np.sqrt(np.sum(np.square(delta_v)))

        self.state = np.concatenate([x, dotx, ddotx, dddotx, delta_x, delta_v, ddott])

        return self._get_obs(), -cost, False, {}

    def step_batch(self, o, u):
        x, dotx, ddotx, dddotx, delta_x, delta_v, ddott = o[:, :2], o[:, 2:4], o[:, 4:6], o[:, 6:8], o[:, 8:10], o[:, 10:12], o[:, 12:14]

        t, dott = delta_x + x, delta_v + dotx
        for i in range(self.n):
            delta_x, delta_v, delta_a = t - x, dott - dotx, ddott - ddotx
            dddotx = delta_x * u[:, 0:2] + delta_v * u[:, 2:4] + delta_a * u[:, 4:6]
            # dddotx = (delta_x * u[:, i * 6:i * 6 + 2] + delta_v * u[:, i * 6 + 2:i * 6 + 4] + delta_a * u[:, i * 6 + 4:i * 6 + 6])
            ddotx = ddotx / self.mass
            ddotx = (ddotx + dddotx * self.dt)
            # dotx = dotx * (1 - self.damping)
            dotx = dotx + ddotx * self.dt
            x = x + dotx * self.dt
            #ddott = ddott / self.mass
            dott = dott + ddott * self.dt
            t = t + dott * self.dt

        return torch.cat((x, dotx, ddotx, dddotx, delta_x, delta_v, ddott), dim=1)

    def reset(self):
        self.state[:2] = np.random.rand(2) * 2 * self.max_pos - self.max_pos
        self.state[2:4] = np.random.rand(2) * 2 * self.max_speed - self.max_speed
        self.state[4:6] = np.random.rand(2) * 2 * self.max_acc - self.max_acc
        self.state[6:8] = np.random.rand(2) * 2 * self.max_acc - self.max_acc
        t = np.random.rand(2) * 2 * self.max_pos - self.max_pos
        dott = np.random.rand(2) * 2 * self.max_speed - self.max_speed
        ddott = np.random.rand(2) * 2 * self.max_acc - self.max_acc

        self.state[8:10] = t - self.state[:2]
        self.state[10:12] = dott - self.state[2:4]
        self.state[12:14] = ddott

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
            self.viewer.set_bounds(-2, 2, -2, 2)

            target1 = rendering.make_circle(.1)
            target1.set_color(0.5, 0.5, 0.5)
            self.target_transform1 = rendering.Transform()
            target1.add_attr(self.target_transform1)
            self.viewer.add_geom(target1)

            target2 = rendering.make_circle(.1)
            target2.set_color(0, 0, 0)
            self.target_transform2 = rendering.Transform()
            target2.add_attr(self.target_transform2)
            self.viewer.add_geom(target2)

            ball = rendering.make_circle(.1)
            ball.set_color(1., 0, 0)
            self.ball_transform = rendering.Transform()
            ball.add_attr(self.ball_transform)
            self.viewer.add_geom(ball)

        self.target_transform1.set_translation(self.state[-4], self.state[-3])
        self.target_transform2.set_translation(self.state[-2], self.state[-1])
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