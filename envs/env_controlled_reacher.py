import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
from gym import spaces
from numpy import pi


from envs.env_abs import AbsEnv


class Env(AbsEnv):
    dt = .1

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2

    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi

    MAX_GAIN = 1.
    MAX_GAIN_CHANGE = 1.

    MAX_TORQUE = 1.

    action_dim = 4
    u_high = np.array([1., 1., 1., 1.])
    u_low = -u_high
    action_space = spaces.Box(
        low=u_low,
        high=u_high, shape=(action_dim,),
        dtype=np.float32
    )
    # 2 angles, 2 angular vel, 2 angle errors, 2 vel errors, 2 p's, 2 d's
    high = np.array([pi, pi, MAX_VEL_1, MAX_VEL_2, pi, pi, MAX_VEL_1, MAX_VEL_2, 1., 1., 1., 1.], dtype=np.float32)
    observation_space = spaces.Box(
        low=-high,
        high=high, shape=(12,),
        dtype=np.float32
    )
    viewer = None
    target = None

    point_l = .5
    grab_counter = 0

    def __init__(self):
        self.state = np.zeros(4)
        self.p = np.zeros(2)
        self.d = np.zeros(2)
        self.target = np.zeros(4)

    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        L1 = self.LINK_LENGTH_1
        L2 = self.LINK_LENGTH_2
        a = s_augmented[-2:]
        s = s_augmented[:-2]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        tau1 = a[0]
        tau2 = a[1]

        # run equations_of_motion() to compute these
        ddtheta1 = (-L1 * np.cos(theta2) - L2) * (-L1 * L2 * dtheta1 ** 2 * m2 * np.sin(theta2) + tau2) / (
                    L1 ** 2 * L2 * m1 - L1 ** 2 * L2 * m2 * np.cos(theta2) ** 2 + L1 ** 2 * L2 * m2) + (
                               L1 * L2 * m2 * (2 * dtheta1 * dtheta2 + dtheta2 ** 2) * np.sin(theta2) + tau1) / (
                               L1 ** 2 * m1 - L1 ** 2 * m2 * np.cos(theta2) ** 2 + L1 ** 2 * m2)
        ddtheta2 = (-L1 * np.cos(theta2) - L2) * (
                    L1 * L2 * m2 * (2 * dtheta1 * dtheta2 + dtheta2 ** 2) * np.sin(theta2) + tau1) / (
                               L1 ** 2 * L2 * m1 - L1 ** 2 * L2 * m2 * np.cos(theta2) ** 2 + L1 ** 2 * L2 * m2) + (
                               -L1 * L2 * dtheta1 ** 2 * m2 * np.sin(theta2) + tau2) * (
                               L1 ** 2 * m1 + L1 ** 2 * m2 + 2 * L1 * L2 * m2 * np.cos(theta2) + L2 ** 2 * m2) / (
                               L1 ** 2 * L2 ** 2 * m1 * m2 - L1 ** 2 * L2 ** 2 * m2 ** 2 * np.cos(
                           theta2) ** 2 + L1 ** 2 * L2 ** 2 * m2 ** 2)

        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0., 0.)

    def step(self, a):
        a = np.clip(a, -self.MAX_GAIN_CHANGE, self.MAX_GAIN_CHANGE)

        deltaP = self.target - self.state[:2]
        deltaV = self.state[2:]

        dp1, dp2, dd1, dd2 = a[0], a[1], a[2], a[3]

        self.p[0] += dp1 * self.dt
        self.p[1] += dp2 * self.dt
        self.d[0] += dd1 * self.dt
        self.d[1] += dd2 * self.dt

        tau1 = self.p[0] * deltaP[0] + self.d[0] * deltaV[0]
        tau2 = self.p[1] * deltaP[1] + self.d[1] * deltaV[1]
        torque = np.array([tau1, tau2]).squeeze()
        torque = np.clip(torque, -self.MAX_TORQUE, self.MAX_TORQUE)
        s_augmented = np.hstack((self.state, torque))

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action

        # new angle
        # ns[0] = angle_normalize(ns[0])
        # ns[1] = angle_normalize(ns[1])
        ns[2] = np.clip(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = np.clip(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns

        obs, deltaP, deltaV = self._get_obs()
        r = self._r_func(deltaP) - np.sqrt(np.sum(np.square(deltaV)))
        return obs, r, self.get_point, {}

    def _get_obs(self):
        deltaP = np.ravel(self.target - self.state[:2])
        deltaV = np.ravel(- self.state[2:])
        return np.hstack([self.state[0], self.state[1], self.state[2], self.state[3],
                          deltaP[0], deltaP[1], deltaV[0], deltaV[1],
                          self.p[0], self.p[1], self.d[0], self.d[1]]), deltaP, deltaV

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

    def _reset_target(self):
        self.target = np.random.rand(2) * 2 * pi - pi

    def _reset_state(self):
        theta = np.random.rand(2) * pi * 2 - pi
        # dtheta1 = np.random.rand(1) * self.MAX_VEL_1 * 2 - self.MAX_VEL_1
        # dtheta2 = np.random.rand(1) * self.MAX_VEL_2 * 2 - self.MAX_VEL_2
        dtheta1 = np.random.rand(1) * .2 - .1
        dtheta2 = np.random.rand(1) * .2 - .1
        self.state = np.array([theta[0], theta[1], dtheta1[0], dtheta2[0]])

    def reset(self):
        self.get_point = False
        self.grab_counter = 0
        self.p = np.random.rand(2) * 2 - 1
        self.d = np.random.rand(2) * 2 - 1
        self._reset_target()
        self._reset_state()
        return self._get_obs()[0]

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

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        from numpy import cos, sin
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

        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, .8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        pt = self.target
        pt1 = [self.LINK_LENGTH_1 * cos(pt[0]), self.LINK_LENGTH_1 * sin(pt[0])]

        pt2 = [p1[0] + self.LINK_LENGTH_2 * cos(pt[0] + pt[1]),
               p1[1] + self.LINK_LENGTH_2 * sin(pt[0] + pt[1])]

        xys = np.array([[0, 0], pt1, pt2])  # [:, ::-1]
        thetas = [pt[0], pt[0] + pt[1]]
        link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link._color.vec4 = (0, .8, .8, .2)
            circ = self.viewer.draw_circle(.1)
            circ._color.vec4 = (.8, .8, 0, .2)
            circ.add_attr(jtransform)


        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def step_batch(self, x, u):
        pass


class ReacherControlledEnv(nn.Module, Env):
    def __init__(self):
        super().__init__()
        self.t0 = nn.Parameter(torch.tensor([0.0]))
        self.init_s = nn.Parameter(torch.tensor([.0, .0, .0, .0]))
        self.init_u = nn.Parameter(torch.tensor([.0, .0]))
        self.u_low_torch = torch.from_numpy(-self.u_high).float()
        self.u_high_torch = torch.from_numpy(self.u_high).float()
        # self.seed(1)

    def get_initial_state_action(self):
        state = (self.init_s, self.init_u)
        return self.t0, state

    def forward(self, t, s):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        L1 = self.LINK_LENGTH_1
        L2 = self.LINK_LENGTH_2

        s, u = s

        tau1, tau2 = torch.split(u, 1, dim=1)
        theta1, theta2 = torch.split(s[:, :2], 1, dim=1)
        dtheta1, dtheta2 = torch.split(s[:, 2:], 1, dim=1)

        # run equations_of_motion() to compute these
        ddtheta1 = (-L1 * torch.cos(theta2) - L2) * (-L1 * L2 * dtheta1 ** 2 * m2 * torch.sin(theta2) + tau2) / (
                    L1 ** 2 * L2 * m1 - L1 ** 2 * L2 * m2 * torch.cos(theta2) ** 2 + L1 ** 2 * L2 * m2) + (
                               L1 * L2 * m2 * (2 * dtheta1 * dtheta2 + dtheta2 ** 2) * torch.sin(theta2) + tau1) / (
                               L1 ** 2 * m1 - L1 ** 2 * m2 * torch.cos(theta2) ** 2 + L1 ** 2 * m2)
        ddtheta2 = (-L1 * torch.cos(theta2) - L2) * (
                    L1 * L2 * m2 * (2 * dtheta1 * dtheta2 + dtheta2 ** 2) * torch.sin(theta2) + tau1) / (
                               L1 ** 2 * L2 * m1 - L1 ** 2 * L2 * m2 * torch.cos(theta2) ** 2 + L1 ** 2 * L2 * m2) + (
                               -L1 * L2 * dtheta1 ** 2 * m2 * torch.sin(theta2) + tau2) * (
                               L1 ** 2 * m1 + L1 ** 2 * m2 + 2 * L1 * L2 * m2 * torch.cos(theta2) + L2 ** 2 * m2) / (
                               L1 ** 2 * L2 ** 2 * m1 * m2 - L1 ** 2 * L2 ** 2 * m2 ** 2 * torch.cos(
                           theta2) ** 2 + L1 ** 2 * L2 ** 2 * m2 ** 2)

        return (dtheta1, dtheta2, ddtheta1, ddtheta2, torch.zeros_like(tau1), torch.zeros_like(tau2))

    def step_batch(self, x, u):
        u = torch.max(torch.min(u, self.u_high_torch), self.u_low_torch)

        if OBS == "sin/cos":
            pos, vel = x[:, :4], x[:, 4:]
            pos = torch.cat((torch.atan2(pos[:, 1:2], pos[:, 0:1]), torch.atan2(pos[:, 3:4], pos[:, 2:3])), dim=1)
            x = torch.cat((pos, vel), dim=1)
        elif OBS == "angles":
            pos, vel = x[:, :2], x[:, 2:]
        s_aug = (x, u)

        solution = odeint(self, s_aug, torch.tensor([0, self.dt]), method='rk4')
        state, action = solution
        state = state[-1] # last time step

        npos, nvel = state[:, :2], state[:, 2:]
        if MODE == "clamp":
            vel = torch.cat([nvel[:, :1].clamp(max=self.MAX_VEL_1, min=-self.MAX_VEL_1),
                             nvel[:, 1:].clamp(max=self.MAX_VEL_2, min=-self.MAX_VEL_2)], dim=1)
            pos = npos.clamp(max=pi, min=-pi)

        elif MODE == "reset":
            mask_l1 = (nvel[:, :1] < self.MAX_VEL_1) & (nvel[:, :1] > -self.MAX_VEL_1) & (npos[:, :1] < pi) & (npos[:, :1] > -pi)
            mask_l2 = (nvel[:, 1:] < self.MAX_VEL_2) & (nvel[:, 1:] > -self.MAX_VEL_2) & (npos[:, 1:] < pi) & (npos[:, 1:] > -pi)
            mask_l1 = mask_l1.squeeze(1)
            mask_l2 = mask_l2.squeeze(1)

            pos = pos.clone()
            vel = vel.clone()
            pos[mask_l1, :1] = npos[mask_l1, :1]
            pos[mask_l2, 1:] = npos[mask_l2, 1:]
            vel[mask_l1, :1] = nvel[mask_l1, :1]
            vel[mask_l2, 1:] = nvel[mask_l2, 1:]

        if OBS == "angles":
            state = torch.cat((pos, vel), dim=1)
        elif OBS == "sin/cos":
            state = torch.cat((torch.cos(pos[:, 0:1]), torch.sin(pos[:, 0:1]),
                               torch.cos(pos[:, 1:2]), torch.sin(pos[:, 1:2]),
                               vel[:, 0:1], vel[:, 1:2]), dim=1)

        return state


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


def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    Args:
        x: scalar
    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t, *args, **kwargs):
    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout


def angle_normalize(x):
    return (((x + pi) % (2 * pi)) - pi)


if __name__ == '__main__':
    env = ReacherControlledEnv()
    env.seed()

    for _ in range(32):
        env.reset()
        for _ in range(100):
            env.render()
            a = env.action_space.sample()
            env.step(a)
    env.close()
