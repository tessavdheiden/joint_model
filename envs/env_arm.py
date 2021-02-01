import numpy as np
from gym import spaces
from numpy import sin, cos, pi
import torch


from envs.env_abs import AbsEnv

MODE = "clamp"
OBS = "sin/cos"

class ArmEnv(AbsEnv):
    '''
    [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]
    '''
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
    s_min = -1
    s_max = 1
    if OBS == "angles":
        observation_space = spaces.Box(
            low=-s_max,
            high=s_max, shape=(2,),
            dtype=np.float32
        )
    elif OBS == "sin/cos":
        observation_space = spaces.Box(
            low=-s_max,
            high=s_max, shape=(4,),
            dtype=np.float32
        )
    action_dim = 2

    LINK_LENGTH_1 = 1
    LINK_LENGTH_2 = 1

    point_l = .5
    grab_counter = 0

    def __init__(self):
        # node1 (d_rad, x, y),
        # node2 (d_rad, x, y)
        self.name = 'Arm'
        self.state = np.zeros(self.observation_space.shape[0])
        self.point_info = np.array([1, 1])
        self.point_info_init = self.point_info.copy()
        self.center_coord = np.array([0, 0])

        self.last_u = np.array([None, None])
        self.viewer = None

        self.u_low_torch = torch.from_numpy(self.u_low).float()
        self.u_high_torch = torch.from_numpy(self.u_high).float()

    def step(self, u):
        # action = (node1 angular v, node2 angular v)
        u = np.clip(u, self.u_low, self.u_high)

        if MODE == "clamp":
            self.state[0] = np.clip(self.state[0] + u[0] * self.dt, -pi, pi)
            self.state[1] = np.clip(self.state[1] + u[1] * self.dt, -pi, pi)
        elif MODE == "wrap":
            self.state[0] = wrap(self.state[0] + u[0] * self.dt, -pi, pi)
            self.state[1] = wrap(self.state[1] + u[1] * self.dt, -pi, pi)

        assert any(self.state <= pi) & any(self.state >= -pi)
        s, arm2_distance = self._get_obs()
        r = self._r_func(arm2_distance)

        return s, r, self.get_point, {}

    def _reset_arm(self):

        arm1rad, arm2rad = np.random.rand(2) * 2 * pi - pi
        self.state[0] = arm1rad
        self.state[1] = arm2rad

    def reset(self):
        self.get_point = False
        self.grab_counter = 0

        angle = np.random.rand(1)[0] * pi * 2 - pi
        radius = np.random.rand(1)[0] * (self.LINK_LENGTH_1 + self.LINK_LENGTH_2)
        self.point_info = radius * np.array([cos(angle), sin(angle)])
        self._reset_arm()
        return self._get_obs()[0]

    def _get_obs(self):
        arm1rad, arm2rad = self.state[0], self.state[1]
        arm1dx_dy = np.array([self.LINK_LENGTH_1 * cos(arm1rad), self.LINK_LENGTH_1 * sin(arm1rad)])
        arm2dx_dy = np.array([self.LINK_LENGTH_2 * cos(arm1rad + arm2rad), self.LINK_LENGTH_2 * sin(arm1rad + arm2rad)])
        xy1 = self.center_coord + arm1dx_dy  # (x1, y1)
        xy2 = xy1 + arm2dx_dy  # (x2, y2)

        # return the distance (dx, dy) between arm finger point with blue point
        delta1 = np.ravel(xy1 - self.point_info)
        delta2 = np.ravel(xy2 - self.point_info)

        if OBS == "sin/cos":
            return np.hstack([cos(self.state[0]), sin(self.state[0]),
                              cos(self.state[1]), sin(self.state[1])]), delta2
        elif OBS == "angles":
            return np.hstack([self.state[0], self.state[1]]), delta2

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

        if s is None: return None

        p1 = [self.LINK_LENGTH_1 * cos(s[0]), self.LINK_LENGTH_1 * sin(s[0])]

        p2 = [p1[0] + self.LINK_LENGTH_2 * cos(s[0] + s[1]),
              p1[1] + self.LINK_LENGTH_2 * sin(s[0] + s[1])]

        xys = np.array([[0, 0], p1, p2])#[:, ::-1]
        thetas = [s[0], s[0] + s[1]]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        target = self.viewer.draw_circle(.1)
        target.set_color(8, 0, 0)
        ttransform = rendering.Transform(translation=(self.point_info[0], self.point_info[1]))
        target.add_attr(ttransform)

        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, .8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def step_batch(self, x, u):
        u = torch.max(torch.min(u, self.u_high_torch), self.u_low_torch)

        if OBS == "angles":
            theta1 = x[:, 0:1]
            theta2 = x[:, 1:2]
        elif OBS == "sin/cos":
            # -π <= atan2() <= π
            c1, s1, c2, s2 = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
            theta1 = torch.atan2(s1, c1)
            theta2 = torch.atan2(s2, c2)

        if MODE == "clamp":
            theta1 = torch.clamp(theta1 + u[:, :1] * self.dt, min=-pi, max=pi)
            theta2 = torch.clamp(theta2 + u[:, 1:] * self.dt, min=-pi, max=pi)
        elif MODE == "wrap":
            theta1 = angle_normalize(theta1 + u[:, :1] * self.dt)
            theta2 = angle_normalize(theta2 + u[:, 1:] * self.dt)

        if OBS == "angles":
            state = torch.cat((theta1, theta2), dim=1)
        elif OBS == "sin/cos":
            state = torch.cat((torch.cos(theta1), torch.sin(theta1), torch.cos(theta2), torch.sin(theta2)), dim=1)

        return state

    def simple_test_case(self):
        self.point_info = np.array([1, 1])
        self.state = np.array([np.pi, -np.pi * (3/8)])

    def benchmark_data(self, data={}):
        if len(data) == 0:
            data = {'arm2_distance': [],
                    'velocity_arm2_distance': [],
                    'acceleration_arm2_distance': [],
                    'jerk_arm2_distance': []}

        arm2_distance = self._get_obs()[1]
        data['arm2_distance'].append(arm2_distance)

        if len(data['arm2_distance']) >= 2:
            data['velocity_arm2_distance'].append(data['arm2_distance'][-1] - data['arm2_distance'][-2])

        if len(data['velocity_arm2_distance']) >= 3:
            data['acceleration_arm2_distance'].append(data['velocity_arm2_distance'][-1] - data['velocity_arm2_distance'][-2])

        if len(data['acceleration_arm2_distance']) >= 4:
            data['jerk_arm2_distance'].append(data['acceleration_arm2_distance'][-1] - data['acceleration_arm2_distance'][-2])

        return data


def angle_normalize(x):
    return (((x+pi) % (2*pi)) - pi)


def wrap(x, m, M):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range
    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

if __name__ == '__main__':
    env = ArmEnv()
    env.seed()

    for _ in range(32):
        env.reset()
        for _ in range(100):
            env.render()
            a = env.action_space.sample()
            env.step(a)
    env.close()