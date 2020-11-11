import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim


class Net(nn.Module):

    def __init__(self, input_dim, out_dim, h_dim):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_dim, h_dim)
        self.mu_head = nn.Linear(h_dim, out_dim)
        self.sigma_head = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = self.mu_head(x)
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma)


class Empowerment(nn.Module):
    def __init__(self, env, controller, transition_network):
        super(Empowerment, self).__init__()
        self.h_dim = 128
        self.action_dim = controller.action_space.shape[0]
        self.transition = transition_network
        self.z_dim = transition_network.z_dim

        self.source = Net(self.z_dim, self.action_dim, self.h_dim)
        self.planning = Net(self.z_dim*2, self.action_dim, self.h_dim)
        self.auto_regressive = Net(self.z_dim * 2 + self.action_dim, self.action_dim, self.h_dim)

        self.optimizer = optim.Adam(list(self.source.parameters()) + list(self.planning.parameters())
                                    + list(self.auto_regressive.parameters()), lr=1e-2)

    def forward(self, z, n_steps=32):
        all_a_ω = []
        all_log_prob_ω = []
        z_ = z
        for t in range(n_steps):
            (μ_ω, σ_ω) = self.source(z_)
            dist_ω = Normal(μ_ω, σ_ω)

            a_ω = dist_ω.rsample()
            all_a_ω.append(a_ω.unsqueeze(1))
            all_log_prob_ω.append(dist_ω.log_prob(a_ω).unsqueeze(1))
            z_ = self.transition(z_, a_ω)

        all_a_ω = torch.cat(all_a_ω, dim=1)
        all_log_prob_ω = torch.cat(all_log_prob_ω, dim=1)
        z_N, z_1 = z_, z

        all_log_prob_pln = []
        (μ_1_pln, σ_1_pln) = self.planning(torch.cat((z_1, z_N), dim=1))
        dist_1_pln = Normal(μ_1_pln, σ_1_pln)
        all_log_prob_pln.append(dist_1_pln.log_prob(all_a_ω[:, 0]).unsqueeze(1))

        a_pln = dist_1_pln.rsample()
        for t in range(1, n_steps):
            (μ_pln, σ_pln) = self.auto_regressive(torch.cat((z_1, a_pln, z_N), dim=1))
            dist_pln = Normal(μ_pln, σ_pln)
            all_log_prob_pln.append(dist_pln.log_prob(all_a_ω[:, t]).unsqueeze(1))
            a_pln = dist_pln.rsample()
        all_log_prob_pln = torch.cat(all_log_prob_pln, dim=1)
        return (all_log_prob_pln - all_log_prob_ω).sum(-1).sum(-1)

    def update(self, s):
        E = self(s)

        self.optimizer.zero_grad()
        L = -E.mean()
        L.backward()
        self.optimizer.step()

        return E.detach().numpy()
