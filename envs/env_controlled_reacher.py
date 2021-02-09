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

    MAX_GAIN_P = 1.
    MAX_GAIN_D = 6.
    MAX_GAIN_CHANGE = 1.

    MAX_TORQUE = 1.

    action_dim = 4
    u_high = np.array([MAX_GAIN_CHANGE, MAX_GAIN_CHANGE, MAX_GAIN_CHANGE, MAX_GAIN_CHANGE])
    u_low = -u_high
    action_space = spaces.Box(
        low=u_low,
        high=u_high, shape=(action_dim,),
        dtype=np.float32
    )

    # cos/sin of 2 angles, 2 angular vel, 2 angle errors, 2 vel errors, 2 p's, 2 d's
    high = np.array([1, 1, 1, 1, MAX_VEL_1, MAX_VEL_2, pi, pi, MAX_GAIN_P, MAX_GAIN_P, MAX_GAIN_D, MAX_GAIN_D], dtype=np.float32)
    low = np.array([-1, -1, -1, -1, -MAX_VEL_1, -MAX_VEL_2, -pi, -pi, 0, 0, 0, 0], dtype=np.float32)
    state_names = ['$\\theta_1$', '$\\theta_2$', '$\\dot{\\theta}_1$', '$\\dot{\\theta}_2$', '$\Delta\\theta_1$',
                   '$\Delta\\theta_2$', '$P_1$', '$P_2$', '$D_1$', '$D_2$']
    observation_space = spaces.Box(
        low=low,
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

        deltaP = angle_normalize(self.target - self.state[:2])
        deltaV = -self.state[2:]

        dp1, dp2, dd1, dd2 = a[0], a[1], a[2], a[3]

        self.p[0] = np.clip(self.p[0] + dp1 * self.dt, 0, self.MAX_GAIN_P)
        self.p[1] = np.clip(self.p[1] + dp2 * self.dt, 0, self.MAX_GAIN_P)
        self.d[0] = np.clip(self.d[0] + dd1 * self.dt, 0, self.MAX_GAIN_D)
        self.d[1] = np.clip(self.d[1] + dd2 * self.dt, 0, self.MAX_GAIN_D)

        tau1 = self.p[0] * deltaP[0] + self.d[0] * deltaV[0]
        tau2 = self.p[1] * deltaP[1] + self.d[1] * deltaV[1]
        torque = np.array([tau1, tau2])
        torque = np.clip(torque, -self.MAX_TORQUE, self.MAX_TORQUE)
        s_augmented = np.hstack((self.state, torque))

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action

        # new angle
        ns[2] = np.clip(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = np.clip(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns

        obs, deltaP, deltaV = self._get_obs()
        r = self._r_func(deltaP) - np.sqrt(np.sum(np.square(deltaV)))
        return obs, r, self.get_point, {}

    def _get_obs(self):
        deltaP = self.target - self.state[:2]
        deltaV = -self.state[2:]
        return np.hstack([np.cos(self.state[0]), np.sin(self.state[0]),
                          np.cos(self.state[1]), np.sin(self.state[1]),
                          self.state[2], self.state[3],
                          angle_normalize(self.target[0]-self.state[0]), angle_normalize(self.target[1]-self.state[1]),
                          #self.target[0], self.target[1],
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
        dtheta = np.zeros(2)
        self.state = np.array([theta[0], theta[1], dtheta[0], dtheta[1]])

    def reset(self):
        self.get_point = False
        self.grab_counter = 0
        self.p = np.random.rand(2) * self.MAX_GAIN_P
        self.d = np.random.rand(2) * self.MAX_GAIN_D
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
        self.name = 'controlled_reacher'
        self.t0 = nn.Parameter(torch.tensor([0.0]))

        self.u_low_torch = torch.from_numpy(-self.u_high).float()
        self.u_high_torch = torch.from_numpy(self.u_high).float()

    def get_initial_obs_action(self):
        state = torch.from_numpy(self.state).float().unsqueeze(0)
        target = torch.from_numpy(self.target).float().unsqueeze(0)
        p = torch.from_numpy(self.p).float().unsqueeze(0)
        d = torch.from_numpy(self.d).float().unsqueeze(0)
        s = self._get_obs_from_state(state, target, p, d)

        self.init_s = nn.Parameter(s).squeeze(0)
        self.init_u = nn.Parameter(torch.tensor([.0, .0, .0, .0]))
        return self.t0, (self.init_s, self.init_u)

    def forward(self, t, s):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        L1 = self.LINK_LENGTH_1
        L2 = self.LINK_LENGTH_2

        s, u = s

        tau1, tau2 = u[:, 0:1], u[:, 1:2]
        theta1, theta2 = s[:, 0:1], s[:, 1:2]
        dtheta1, dtheta2 = s[:, 2:3], s[:, 3:4]

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

        pos, vel = x[:, :4], x[:, 4:6]
        angle = torch.cat((torch.atan2(pos[:, 1:2], pos[:, 0:1]), torch.atan2(pos[:, 3:4], pos[:, 2:3])), dim=1)

        deltaP = x[:, 6:8]
        target = deltaP + angle
        #deltaP = angle_normalize(target - angle)
        deltaV = -vel
        p = x[:, 8:10]
        d = x[:, 10:12]

        dp1, dp2, dd1, dd2 = u[:, 0:1], u[:, 1:2], u[:, 2:3], u[:, 3:4]

        p_, d_ = p.clone(), d.clone()
        p_[:, 0:1] = torch.clamp(p[:, 0:1] + dp1 * self.dt, 0, self.MAX_GAIN_P)
        p_[:, 1:2] = torch.clamp(p[:, 1:2] + dp2 * self.dt, 0, self.MAX_GAIN_P)
        d_[:, 0:1] = torch.clamp(d[:, 0:1] + dd1 * self.dt, 0, self.MAX_GAIN_D)
        d_[:, 1:2] = torch.clamp(d[:, 1:2] + dd2 * self.dt, 0, self.MAX_GAIN_D)

        tau1 = p[:, 0:1] * deltaP[:, 0:1] + d[:, 0:1] * deltaV[:, 0:1]
        tau2 = p[:, 1:2] * deltaP[:, 1:2] + d[:, 1:2] * deltaV[:, 1:2]
        torque = torch.cat((tau1, tau2), dim=1)
        torque = torch.clamp(torque, -self.MAX_TORQUE, self.MAX_TORQUE)

        s_aug = (torch.cat((angle, vel), dim=1), torque)

        solution = odeint(self, s_aug, torch.tensor([0, self.dt]), method='rk4')
        state_, action = solution
        state_ = state_[-1] # last time step

        return self._get_obs_from_state(state_, target, p_, d_)

    def _get_obs_from_state(self, state, target, p, d):
        ang, vel = state[:, :2], state[:, 2:]

        vel = torch.cat([vel[:, :1].clamp(max=self.MAX_VEL_1, min=-self.MAX_VEL_1),
                          vel[:, 1:].clamp(max=self.MAX_VEL_2, min=-self.MAX_VEL_2)], dim=1)
        deltaP = angle_normalize(target - ang)

        return torch.cat((torch.cos(ang[:, 0:1]), torch.sin(ang[:, 0:1]),
                          torch.cos(ang[:, 1:2]), torch.sin(ang[:, 1:2]),
                           vel[:, 0:1], vel[:, 1:2],
                           deltaP[:, 0:1], deltaP[:, 1:2],
                          # target[:, 0:1], target[:, 1:2],
                           p[:, 0:1], p[:, 1:2], d[:, 0:1], d[:, 1:2]), dim=1)

    def get_state_from_obs(self, obs):
        pos, vel = obs[:, :4], obs[:, 4:6]
        angle = torch.cat((torch.atan2(pos[:, 1:2], pos[:, 0:1]), torch.atan2(pos[:, 3:4], pos[:, 2:3])), dim=1)
        return torch.cat((angle, vel), dim=1)


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
    import imageio
    env = ReacherControlledEnv()
    env.seed()

    frames=[]

    env.reset()
    env.state = np.array([0, 0, 0, 0])
    # env.MAX_GAIN_P = .5
    # env.MAX_GAIN_D = 3
    env.p = np.ones(2) * env.MAX_GAIN_P / 2
    env.d = np.ones(2) * env.MAX_GAIN_D / 2
    env.target = np.array([pi, 0])

    t0, obs_action = env.get_initial_obs_action()
    obs = obs_action[0].unsqueeze(0)
    for _ in range(200):
        a = env.action_space.sample()
        obs = env.step_batch(x=obs, u=torch.from_numpy(a).float().unsqueeze(0))
        state = env.get_state_from_obs(obs)     # for rendering
        env.state = state.squeeze(0).detach().numpy()

        f = env.render(mode='rgb_array')
        frames.append(f)

    # for _ in range(100):
    #     f = env.render(mode='rgb_array')
    #     frames.append(f)
    #     a = env.action_space.sample()*0
    #     env.step(a)
    env.close()
    imageio.mimsave(f'../img/video_p={env.MAX_GAIN_P}_d={env.MAX_GAIN_D}.gif', frames)