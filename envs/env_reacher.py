import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_adjoint
import numpy as np
from gym import spaces
from numpy import pi


from envs.env_abs import AbsEnv


MODE = "clamp"
OBS = "angles"


class Env(AbsEnv):
    dt = .1

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2

    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi

    action_dim = 2
    u_low = np.array([-1., -1.])
    u_high = np.array([1., 1.])
    action_space = spaces.Box(
        low=u_low,
        high=u_high, shape=(action_dim,),
        dtype=np.float32
    )

    if OBS == "angles":
        high = np.array([pi, pi, MAX_VEL_1, MAX_VEL_2], dtype=np.float32)
        observation_space = spaces.Box(
            low=-high,
            high=high, shape=(4,),
            dtype=np.float32
        )
    elif OBS == "sin/cos":
        high = np.array([1, 1, 1, 1, MAX_VEL_1, MAX_VEL_2], dtype=np.float32)
        observation_space = spaces.Box(
            low=-high,
            high=high, shape=(6,),
            dtype=np.float32
        )

    viewer = None
    target = None

    def __init__(self):
        pass

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
        s = self.state

        a = np.clip(a, self.u_low, self.u_high)
        s_augmented = np.append(s, a)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action
        if MODE == "reset":
            if -pi < ns[0] < pi and -pi < ns[1] < pi and -self.MAX_VEL_1 < ns[2] < self.MAX_VEL_1 and -self.MAX_VEL_2 < ns[3] < self.MAX_VEL_2:
                self.state = ns

        elif MODE == "clamp":
            ns[0:2] = np.clip(ns[0:2], -pi, pi)
            ns[2] = np.clip(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
            ns[3] = np.clip(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
            self.state = ns

        terminal = self._terminal()
        reward = -1. if not terminal else 0.
        return (self._get_ob(), reward, terminal, {})

    def _terminal(self):
        s = self.state
        xy1 = np.array([self.LINK_LENGTH_1 * np.cos(s[0]), self.LINK_LENGTH_1 * np.sin(s[0])])
        xy2 = xy1 + np.array([self.LINK_LENGTH_2 * np.cos(s[0] + s[1]), self.LINK_LENGTH_2 * np.sin(s[0] + s[1])])
        dist = np.sqrt(np.sum(np.square(xy2 - self.target)))

        return bool(dist < .1)

    def _get_ob(self):
        s = self.state
        t = self.target
        if OBS == "sin/cos":
            return np.array([np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]])
        elif OBS == "angles":
            return np.array([s[0], s[1], s[2], s[3]])

    def _reset_target(self):
        a = np.random.rand(1)[0] * pi * 2
        r = np.random.rand(1)[0] * (self.LINK_LENGTH_1 + self.LINK_LENGTH_2)
        self.target = r * np.array([np.cos(a), np.sin(a)])

    def _reset_state(self):
        theta = np.random.rand(2) * pi * 2 - pi
        dtheta1 = np.random.rand(1) * self.MAX_VEL_1 * 2 - self.MAX_VEL_1
        dtheta2 = np.random.rand(1) * self.MAX_VEL_2 * 2 - self.MAX_VEL_2
        self.state = np.array([theta[0], theta[1], dtheta1[0], dtheta2[0]])

    def reset(self):
        self._reset_target()
        self._reset_state()
        return self._get_ob()

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

        if self.target:
            target = self.viewer.draw_circle(.1)
            target.set_color(8, 0, 0)
            ttransform = rendering.Transform(translation=(self.target[0], self.target[1]))
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
        pass


class ReacherEnv(nn.Module, Env):
    def __init__(self):
        super().__init__()
        self.t0 = nn.Parameter(torch.tensor([0.0]))
        self.init_s = nn.Parameter(torch.tensor([.0, .0, .0, .0]))
        self.init_u = nn.Parameter(torch.tensor([.0, .0]))
        self.u_low_torch = torch.from_numpy(self.u_low).float()
        self.u_high_torch = torch.from_numpy(self.u_high).float()
        self.seed(1)

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


def equations_of_motion():
    import sympy as sym
    from sympy import cos, sin
    sym.init_printing()

    theta1, theta2 = sym.symbols('theta1 theta2')
    dtheta1, dtheta2 = sym.symbols('dtheta1 dtheta2')
    m1, m2, L1, L2 = sym.symbols('m1 m2 L1 L2')
    M = sym.Matrix([[m1*L1**2 + m2*(L1**2 + 2*L1*L2*cos(theta2) + L2**2), m2*(L1*L2*cos(theta2) + L2**2)],
                    [m2*(L1*L2*cos(theta2) + L2**2), m2*L2**2]])
    c = sym.Matrix([[-m2*L1*L2*sin(theta2)*(2*dtheta1*dtheta2 + dtheta2**2)],
                    [m2*L1*L2*dtheta1**2*sin(theta2)]])
    M_inv = M.inv()
    tau1, tau2 = sym.symbols('tau1 tau2')
    tau = sym.Matrix([[tau1],
                       [tau2]])

    dtheta = M_inv * (tau - c)

    print(f"dtheta_1={dtheta[0]}")
    print(f"dtheta_2={dtheta[1]}")


def visualize(t, solution):
    t = t.detach().cpu().numpy()
    state, action = solution
    theta = state[:, 0, :2].detach().cpu().numpy()
    dtheta = state[:, 0, 2:].detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 4), facecolor='white')
    ax_theta = fig.add_subplot(121, frameon=False)
    ax_ttheta = fig.add_subplot(122, frameon=False)

    ax_theta.plot(t, theta[:, 0], color="C1", alpha=0.7, linestyle="--", linewidth=2.0, label="$\\theta_1$")
    ax_theta.plot(t, theta[:, 1], color="C0", linewidth=2.0, label="$\\theta_2$")
    ax_theta.set_ylabel("$\\theta$", fontsize=16)
    ax_theta.legend(fontsize=16)
    ax_theta.hlines(0, 0, 100)
    ax_theta.set_xlim([t[0], t[-1]])
    ax_theta.set_xlabel("Time", fontsize=13)

    ax_ttheta.plot(t, dtheta[:, 0], color="C1", alpha=0.7, linestyle="--", linewidth=2.0, label="$\\dot{\\theta}_1$")
    ax_ttheta.plot(t, dtheta[:, 1], color="C0", linewidth=2.0, label="$\\dot{\\theta}_2$")
    ax_ttheta.set_ylabel("$\\dot{\\theta}$", fontsize=16)
    ax_ttheta.legend(fontsize=16)
    ax_ttheta.hlines(0, 0, 100)
    ax_ttheta.set_xlim([t[0], t[-1]])
    ax_ttheta.set_xlabel("Time", fontsize=13)

    plt.tight_layout()
    plt.savefig("reacher.png")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import imageio

    system = ReacherEnv()
    frames = []

    for _ in range(32):
        t0, state_action = system.get_initial_state_action()
        system.state = state_action[0].detach().numpy()
        for _ in range(100):
            a = system.action_space.sample()

            state = system.step_batch(x=torch.from_numpy(system.state).float().unsqueeze(0),
                                      u=torch.from_numpy(a).float().unsqueeze(0))
            system.state = state.squeeze(0).detach().numpy()   # only need last time state

            f = system.render(mode='rgb_array')
            frames.append(f)
        break
    imageio.mimsave('../img/video.gif', frames)

    t0, state_action = system.get_initial_state_action()
    t = torch.linspace(0., 25., 10)

    state, action = state_action
    state_action = (state.unsqueeze(0), action.unsqueeze(0))
    solution = odeint(system, state_action, t, atol=1e-8, rtol=1e-8)
    visualize(t, solution)


