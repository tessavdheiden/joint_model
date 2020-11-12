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
        self.w = .1
        self.dt = .05
        self.viewer = None
        self.max_force = 5
        self.damping = 0.25
        self.max_speed = 1
        self.max_dim = 1

        self.action_space = spaces.Box(
            low=-self.max_force,
            high=self.max_force, shape=(2,),
            dtype=np.float32
        )
        high = np.array([self.max_dim, self.max_dim, self.max_speed, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        x, y, vx, vy = self.state
        v = np.array([vx, vy])
        p = np.array([x, y])

        m = self.m
        dt = self.dt

        u = np.clip(u, -self.max_force, self.max_force)

        vnew = v * (1 - self.damping)
        vnew = vnew + (u / self.m) * self.dt
        speed = np.sqrt(np.square(vnew[0]) + np.square(vnew[1]))
        if speed > self.max_speed:
            vnew = vnew / np.sqrt(np.square(vnew[0]) + np.square(vnew[1])) * self.max_speed

        p += vnew * self.dt
        newx, newy = p[0], p[1]
        hit_wall = np.any(newx > self.max_dim - self.w / 2 or newx < - self.max_dim + self.w / 2 or
                          newy > self.max_dim - self.w / 2 or newy < - self.max_dim + self.w / 2)

        if not hit_wall:
            x, y = newx, newy
        self.state = np.array([x, y, vnew[0], vnew[1]])
        return self.state, None, False, {}

    def reset(self):
        high = np.array([self.max_dim - self.w / 2, self.max_dim - self.w / 2, 0, 0])
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