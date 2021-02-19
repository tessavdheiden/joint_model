import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim


N_STEP = 2


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):

    def __init__(self, input_dim, out_dim, h_dim):
        super(Net, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, h_dim),
                                nn.Sigmoid(), nn.BatchNorm1d(h_dim),
                                nn.Linear(h_dim, h_dim))
        self.mu_head = nn.Sequential(nn.Linear(h_dim, h_dim),
                                nn.Sigmoid(), nn.BatchNorm1d(h_dim),
                                nn.Linear(h_dim, out_dim))
        self.sigma_head = nn.Sequential(nn.Linear(h_dim, h_dim),
                                nn.Sigmoid(), nn.BatchNorm1d(h_dim),
                                nn.Linear(h_dim, out_dim), nn.Softplus())

    def forward(self, x):
        x = self.fc(x)
        mu = self.mu_head(x)
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)


class Empowerment(nn.Module):
    def __init__(self, env, controller):
        super(Empowerment, self).__init__()
        self.h_dim = 128
        self.action_dim = controller.action_space.shape[0]
        self.z_dim = env.observation_space.shape[0]

        if env.name == 'controlled_reacher':
            self.flt_ω = lambda x: x#x[:, -4:] # only pd
            self.flt_q = lambda x: x#torch.cat((x[:, :6], x[:, 8:12],
                                              # x[:, 12:18]), dim=1) # except pd at t+1
            self.ω = Net(self.z_dim, self.action_dim, self.h_dim)
            self.q = Net(self.z_dim*2, self.action_dim, self.h_dim)
            self.opt_q = optim.Adam(self.q.parameters(), lr=1e-4)
            self.forward = self.fwd_step
        else:
            self.ω = Net(self.z_dim, self.action_dim, self.h_dim)
            self.q = Net(self.z_dim*2, self.action_dim, self.h_dim)
            self.auto_regressive = Net(self.z_dim * 2 + self.action_dim, self.action_dim, self.h_dim)
            self.opt_q = optim.Adam(list(self.q.parameters()) + list(self.auto_regressive.parameters()), lr=1e-4)
            self.forward = self.fwd_n_steps

        self.opt_ω = optim.Adam(self.ω.parameters(), lr=1e-5)

        self.t = None
        self.q_steps = 4
        self.use_filter = False
        self.step = env.step_batch
        self.cast = lambda x: x

        self.it = 0

    def set_transition(self, transition_network):
        self.t = transition_network
        self.z_dim = transition_network.z_dim

    def fwd_n_steps(self, z):
        z = self.cast(z)

        all_a_ω = []
        all_log_prob_ω = []
        z_ = z

        for t in range(N_STEP):
            (μ_ω, σ_ω) = self.ω(z_)
            dist_ω = Normal(μ_ω, σ_ω)
            a_ω = dist_ω.rsample()
            all_a_ω.append(a_ω.unsqueeze(1))
            all_log_prob_ω.append(dist_ω.log_prob(a_ω).unsqueeze(1))

            z_ = self.step(z_, a_ω)

        all_a_ω = torch.cat(all_a_ω, dim=1)
        all_log_prob_ω = torch.cat(all_log_prob_ω, dim=1)

        all_log_prob_q = []
        (μ_1_q, σ_1_q) = self.q(torch.cat((z, z_), dim=1))
        dist_1_q = Normal(μ_1_q, σ_1_q)
        all_log_prob_q.append(dist_1_q.log_prob(all_a_ω[:, 0]).unsqueeze(1))

        a_q = dist_1_q.rsample()
        for t in range(1, N_STEP):
            (μ_q, σ_q) = self.auto_regressive(torch.cat((z, a_q, z_), dim=1))
            dist_q = Normal(μ_q, σ_q)
            all_log_prob_q.append(dist_q.log_prob(all_a_ω[:, t]).unsqueeze(1))
            a_q = dist_q.rsample()

        all_log_prob_q = torch.cat(all_log_prob_q, dim=1)

        self.it += 1
        return (all_log_prob_q - all_log_prob_ω).sum(-1).mean(-1)     # sum over action_dim, average over time

    def fwd_step(self, z):
        z = self.cast(z)

        z_ω = self.flt_ω(z)                             # ω only observes PD params
        (μ_ω, σ_ω) = self.ω(z_ω)
        dist_ω = Normal(μ_ω, σ_ω)
        a_ω = dist_ω.rsample()
        z_ = self.step(z, a_ω.detach())                 # ω-step

        for t in range(1, N_STEP):
            z_ = self.step(z_, torch.zeros_like(a_ω))   # n state propagations with no PD update

        z_q = self.flt_q(torch.cat((z, z_), dim=1))     # q does not observe PD_t+n
        (μ_q, σ_q) = self.q(z_q)
        dist_q = Normal(μ_q, σ_q)

        self.it += 1
        return (dist_q.log_prob(a_ω) - dist_ω.log_prob(a_ω)).sum(-1)

    def update(self, s):

        for _ in range(self.q_steps):
            self.opt_q.zero_grad()
            E = self(s)
            L = -E.mean()
            L.backward(retain_graph=True)
            self.opt_q.step()

        self.opt_ω.zero_grad()
        E = self(s)
        L = -E.mean()
        L.backward()
        self.opt_ω.step()

        return E.detach().numpy()

    @property
    def networks(self):
        return [self.ω, self.q]

    def prepare_update(self):
        if DEVICE == 'cuda':
            self.cast = lambda x: x.cuda()
        else:
            self.cast = lambda x: x.cpu()

        for network in self.networks:
            network = self.cast(network)
            network.train()

    def prepare_eval(self):
        self.cast = lambda x: x.cpu()
        for network in self.networks:
            network = self.cast(network)
            network.eval()