import torch
import torch.nn as nn
from torchdiffeq import odeint


class Reacher(nn.Module):
    dt = .2

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2

    def __init__(self):
        super().__init__()
        self.t0 = nn.Parameter(torch.tensor([0.0]))
        self.init_theta = nn.Parameter(torch.tensor([.0, .0]))
        self.init_dtheta = nn.Parameter(torch.tensor([.0, .1]))
        self.init_tau = nn.Parameter(torch.tensor([.0, .0]))

    def get_initial_state_action(self):
        state = (self.init_theta, self.init_dtheta, self.init_tau)
        return self.t0, state

    def forward(self, t, s):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        L1 = self.LINK_LENGTH_1
        L2 = self.LINK_LENGTH_2

        theta, dtheta, tau = s

        tau1, tau2 = torch.split(tau, 1, dim=0)
        theta1, theta2 = torch.split(theta, 1, dim=0)
        dtheta1, dtheta2 = torch.split(dtheta, 1, dim=0)

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

        return (dtheta1, dtheta2, ddtheta1, ddtheta2, tau1, tau2)


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    system = Reacher()

    t0, state = system.get_initial_state_action()
    t = torch.linspace(0., 25., 10)

    solution = odeint(system, state, t, atol=1e-8, rtol=1e-8)

    t = t.detach().cpu().numpy()
    theta = solution[0].detach().cpu().numpy()
    dtheta = solution[1].detach().cpu().numpy()
    tau = solution[2].detach().cpu().numpy()

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_theta = fig.add_subplot(131, frameon=False)
    ax_ttheta = fig.add_subplot(132, frameon=False)

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