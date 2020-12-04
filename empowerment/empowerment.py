import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim


N_STEP = 1


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
    def __init__(self, env, controller, transition_network, use_filter):
        super(Empowerment, self).__init__()
        self.h_dim = 128
        self.action_dim = controller.action_space.shape[0]
        self.transition = transition_network
        self.z_dim = transition_network.z_dim

        self.env = env
        self.source = Net(self.z_dim, self.action_dim, self.h_dim)
        self.planning = Net(self.z_dim*2, self.action_dim, self.h_dim)
        self.auto_regressive = Net(self.z_dim * 2 + self.action_dim, self.action_dim, self.h_dim)

        self.optimizer_planning = optim.Adam(list(self.planning.parameters())
                                    + list(self.auto_regressive.parameters()), lr=1e-4)
        self.optimizer_source = optim.Adam(self.source.parameters(), lr=1e-5)
        self.planning_steps = 4
        self.use_filter = use_filter
        self.step = env.step_batch
        self.cast = lambda x: x

        self.it = 0

    def forward(self, z):
        z = self.cast(z)

        all_a_ω = []
        all_log_prob_ω = []
        z_ = z

        for t in range(N_STEP):
            (μ_ω, σ_ω) = self.source(z_)
            dist_ω = Normal(μ_ω, σ_ω)
            a_ω = dist_ω.rsample()
            all_a_ω.append(a_ω.unsqueeze(1))
            all_log_prob_ω.append(dist_ω.log_prob(a_ω).unsqueeze(1))

            z_ = self.transition(z_, a_ω)[0] if self.use_filter else self.step(z_, a_ω)

        all_a_ω = torch.cat(all_a_ω, dim=1)
        all_log_prob_ω = torch.cat(all_log_prob_ω, dim=1)

        all_log_prob_pln = []
        (μ_1_pln, σ_1_pln) = self.planning(torch.cat((z, z_), dim=1))
        dist_1_pln = Normal(μ_1_pln, σ_1_pln)
        all_log_prob_pln.append(dist_1_pln.log_prob(all_a_ω[:, 0]).unsqueeze(1))

        a_pln = dist_1_pln.rsample()
        for t in range(1, N_STEP):
            (μ_pln, σ_pln) = self.auto_regressive(torch.cat((z, a_pln, z_), dim=1))
            dist_pln = Normal(μ_pln, σ_pln)
            all_log_prob_pln.append(dist_pln.log_prob(all_a_ω[:, t]).unsqueeze(1))
            a_pln = dist_pln.rsample()

        all_log_prob_pln = torch.cat(all_log_prob_pln, dim=1)

        self.it += 1
        return (all_log_prob_pln - all_log_prob_ω).sum(-1).mean(-1)     # sum over action_dim, average over time

    def update(self, s):

        for _ in range(self.planning_steps):
            self.optimizer_planning.zero_grad()
            E = self(s)
            L = -E.mean()
            L.backward(retain_graph=True)
            self.optimizer_planning.step()

        self.optimizer_source.zero_grad()
        E = self(s)
        L = -E.mean()
        L.backward()
        self.optimizer_source.step()

        return E.detach().numpy()

    @property
    def networks(self):
        return [self.source, self.planning]

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