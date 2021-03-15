import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import torch
from envs.env_abs import AbsEnv
from viz.tentacle import Tentacle




class BallBoxForceEnv(AbsEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    dt = .1

    n = 3
    action_dim = 2 * 3
    u_high = np.ones(action_dim)
    u_low = -u_high
    action_space = spaces.Box(
        low=u_low,
        high=u_high, shape=(action_dim,),
        dtype=np.float32
    )
    state_dim = 14
    max_pos = 1.
    max_speed = 1.
    max_acc = 1.
    max_jerk = 1.
    mass = 1.5
    damping = 0.25
    s_max = np.ones(state_dim)
    s_min = -s_max
    observation_space = spaces.Box(
        low=-s_max,
        high=s_min, shape=(state_dim,),
        dtype=np.float32
    )

    state_names = ['x', 'y', 'dotx', 'doty', 'ddotx', 'ddoty', 'dddotx', 'dddoty', 'x_r', 'y_r', 'dotx_r', 'doty_r', 'ddotx_r', 'ddoty_r']

    def __init__(self):
        self.seed()
        self.name = 'BallInBoxForce'
        self.viewer = None
        self.state = np.zeros(self.state_dim)
        self.tentacle = Tentacle(0, 0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        x, dotx, ddotx, dddotx, t, dott, ddott = self.state[:2], self.state[2:4], self.state[4:6], self.state[6:8], self.state[8:10], self.state[10:12], self.state[12:14]
        dx, dv, da = u[0:2], u[2:4], u[4:6]

        cost = 0
        for i in range(self.n):
            delta_x, delta_v, delta_a = t - x, dott - dotx, ddott - ddotx
            dddotx = delta_x * dx + delta_v * dv + delta_a * da
            ddotx = ddotx / self.mass
            ddotx = ddotx + dddotx * self.dt
            #dotx = dotx * (1 - self.damping)
            dotx = dotx + ddotx * self.dt
            newx = x + dotx * self.dt

            dist = np.sqrt(newx[0]**2 + newx[1]**2)
            if dist < 2 * self.max_pos:
                x = newx

            dott = dott + ddott * self.dt
            t = t + dott * self.dt
            self.states[i] = x
            self.targets[i] = t

        self.state = np.concatenate([x, dotx, ddotx, dddotx, t, dott, ddott])

        return self._get_obs(), -cost, False, {}

    def step_batch(self, o, u):
        x, dotx, ddotx, dddotx, t, dott, ddott = o[:, :2], o[:, 2:4], o[:, 4:6], o[:, 6:8], o[:, 8:10], o[:, 10:12], o[:, 12:14]
        dx, dv, da = u[:, 0:2], u[:, 2:4], u[:, 4:6]

        for _ in range(self.n):
            delta_x, delta_v, delta_a = t - x, dott - dotx, ddott - ddotx
            dddotx = delta_x * dx + delta_v * dv + delta_a * da
            ddotx = ddotx / self.mass
            ddotx = ddotx + dddotx * self.dt
            #dotx = dotx * (1 - self.damping)
            dotx = dotx + ddotx * self.dt
            newx = x + dotx * self.dt

            dist = torch.sqrt(newx[:, 0]**2 + newx[:, 1]**2).unsqueeze(1).repeat(1, x.shape[1])
            x = torch.where(dist < 2 * self.max_pos, x, newx)

            dott = dott + ddott * self.dt
            t = t + dott * self.dt

        return torch.cat((x, dotx, ddotx, dddotx, t, dott, ddott), dim=1)

    def reset(self):
        self.state[:2] = np.random.rand(2) * 2 * self.max_pos - self.max_pos
        self.state[2:4] = np.random.rand(2) * 2 * self.max_speed - self.max_speed
        self.state[4:6] = np.random.rand(2) * 2 * self.max_acc - self.max_acc
        self.state[6:8] = np.random.rand(2) * 2 * self.max_jerk - self.max_jerk

        self.state[8:10] = self.state[:2].copy()    #np.random.rand(2) * 2 * self.max_pos - self.max_pos
        self.state[10:12] = self.state[2:4].copy()  #np.random.rand(2) * 2 * self.max_speed - self.max_speed
        self.state[12:14] = np.random.rand(2) * 2 * self.max_acc - self.max_acc
        self.states = np.zeros((self.n, 2))
        self.targets = np.zeros((self.n, 2))

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
            self.viewer.set_bounds(-self.max_pos, self.max_pos, -self.max_pos, self.max_pos)
            self.target_transform = []
            self.ball_transform = []
            for i in range(self.n):
                target = rendering.make_circle(.02)
                target.set_color(0, 0, 0)
                self.target_transform.append(rendering.Transform())
                target.add_attr(self.target_transform[i])
                self.viewer.add_geom(target)

                ball = rendering.make_circle(.02)
                ball.set_color(1., 0, 0)
                self.ball_transform.append(rendering.Transform())
                ball.add_attr(self.ball_transform[i])
                self.viewer.add_geom(ball)

                t = self.tentacle
                for segment in t.segments:
                    self.viewer.add_geom(segment)

        target_ = self.targets[0]
        t = self.tentacle
        for segment in reversed(t.segments):
            segment.follow(*target_)
            target_ = segment.start
        target_ = [0, 0]
        for segment in t.segments:
            segment.set_start(target_)
            target_ = segment.end
        images = []
        for i in range(self.n):
            self.ball_transform[i].set_translation(self.states[i][0], self.states[i][1])
            self.target_transform[i].set_translation(self.targets[i][0], self.targets[i][1])
            images.append(self.viewer.render(return_rgb_array=mode == 'rgb_array'))

        return images




if __name__ == '__main__':
    from viz import Video
    v = Video()
    env = BallBoxForceEnv()
    env.seed()

    for _ in range(100):
        env.reset()
        env.state[12:14] = np.zeros(2)
        for _ in range(2):
            a = env.action_space.sample() * 0 + np.array([10, 10, 10, 10, 10, 10]) / 10
            env.step(a)
            for image in env.render(mode='rgb_array'):
                v.add(image)

    env.close()
    v.save('../img/vid.gif')