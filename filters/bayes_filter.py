import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
from torch.autograd import Variable


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Generator(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim, w_dim, T):
        super(Generator, self).__init__()
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.T = T
        self.fc = nn.Sequential(nn.Linear(x_dim, h_dim), nn.Sigmoid(), nn.BatchNorm1d(h_dim), nn.Linear(h_dim, h_dim))
        self.rnn = nn.GRU(h_dim, h_dim, bidirectional=True)
        self.p_ξ = nn.Sequential(nn.Linear(h_dim * 2, h_dim), nn.Sigmoid(), nn.BatchNorm1d(h_dim), nn.Linear(h_dim, h_dim))
        self.μ = nn.Sequential(nn.Linear(h_dim, h_dim), nn.Sigmoid(), nn.BatchNorm1d(h_dim), nn.Linear(h_dim, w_dim))
        self.σ = nn.Sequential(nn.Linear(h_dim, h_dim), nn.Sigmoid(), nn.BatchNorm1d(h_dim), nn.Linear(h_dim, w_dim), nn.Softplus())
        self.p_λ = nn.Sequential(nn.Linear(w_dim, h_dim), nn.Sigmoid(), nn.BatchNorm1d(h_dim), nn.Linear(h_dim, z_dim))

    def forward(self, x):
        (batch_size, seq_length, x_dim) = x.shape
        h = self.fc(x.reshape(-1, x_dim)).view(batch_size, seq_length, -1)
        bi_out, _ = self.rnn(h)                 # bi_out: tensor of shape (batch_size, seq_length, h_dim*2)
        h = self.p_ξ(bi_out[:, 0])
        (μ, σ) = self.μ(h), self.σ(h)
        dist = Normal(μ, σ)
        w = dist.rsample()
        z = self.p_λ(w)
        return z, (μ, σ)

    @property
    def params(self):
        return list(self.rnn.parameters()) + list(self.p_ξ.parameters()) + list(self.μ.parameters()) \
               + list(self.σ.parameters()) + list(self.p_λ.parameters())

    @property
    def networks(self):
        return [self.rnn, self.p_ξ, self.μ, self.σ, self.p_λ]


class BayesFilter(nn.Module):
    def __init__(self, seq_length, x_dim, u_dim, z_dim, u_low, u_high, sys):

        super(BayesFilter, self).__init__()
        self.sys = sys
        self.T = seq_length
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.u_low = torch.from_numpy(u_low).float()
        self.u_high = torch.from_numpy(u_high).float()
        self.z_dim = z_dim
        self.w_dim = 6
        self.h_dim = 128
        self.M = 16
        self._initial_generator = Generator(z_dim=self.z_dim, h_dim=self.h_dim, x_dim=self.x_dim, w_dim=self.w_dim, T=self.T)
        self._create_transition_matrices()
        self._create_transition_network()
        self._create_recognition_network()
        self._create_decoding_network()
        self._create_optimizer()
        self.cast = lambda x: x
        self.it = 1
        self.Ta = 1e6
        self.c = .01

    # Initialize potential transition matrices
    def _create_transition_matrices(self, std=1e-4):
        self.q_φ_A = Variable(torch.randn(self.M, self.z_dim, self.z_dim)*std, requires_grad=True)
        self.q_φ_B = Variable(torch.randn(self.M, self.z_dim, self.u_dim)*std, requires_grad=True)
        self.q_φ_C = Variable(torch.randn(self.M, self.z_dim, self.w_dim)*std, requires_grad=True)

    def _create_transition_network(self):
        self.f_ψ = nn.Sequential(nn.Linear(self.z_dim + self.u_dim, self.h_dim),
                                   nn.Sigmoid(), nn.BatchNorm1d(self.h_dim),
                                   nn.Linear(self.h_dim, self.M))

    def _create_recognition_network(self):
        self.q_χ_μ = nn.Sequential(nn.Linear(self.z_dim + self.u_dim + self.x_dim, self.h_dim),
                                   nn.Sigmoid(), nn.BatchNorm1d(self.h_dim),
                                   nn.Linear(self.h_dim, self.w_dim))
        self.q_χ_σ = nn.Sequential(nn.Linear(self.z_dim + self.u_dim + self.x_dim, self.h_dim),
                                   nn.Sigmoid(), nn.BatchNorm1d(self.h_dim),
                                   nn.Linear(self.h_dim, self.w_dim),
                                   nn.Softplus())

    def _create_decoding_network(self):
        self.p_θ_μ = nn.Sequential(nn.Linear(self.z_dim, self.h_dim),
                                   nn.Sigmoid(), nn.BatchNorm1d(self.h_dim),
                                   nn.Linear(self.h_dim, self.h_dim),
                                   nn.Sigmoid(), nn.BatchNorm1d(self.h_dim),
                                   nn.Linear(self.h_dim, self.x_dim))
        self.p_θ_σ = nn.Sequential(nn.Linear(self.z_dim, self.h_dim),
                                   nn.Sigmoid(), nn.BatchNorm1d(self.h_dim),
                                   nn.Linear(self.h_dim, self.h_dim),
                                   nn.Sigmoid(), nn.BatchNorm1d(self.h_dim),
                                   nn.Linear(self.h_dim, self.x_dim),
                                   nn.Softplus())

    def _compute_α(self, zu):
        α = torch.softmax(self.f_ψ(zu).view(-1, self.M, 1, 1), dim=1)
        return α.repeat(1, 1, self.z_dim, self.z_dim), \
               α.repeat(1, 1, self.z_dim, self.u_dim), \
               α.repeat(1, 1, self.z_dim, self.w_dim)

    def _sample_v(self):
        return self.q_φ_A, self.q_φ_B, self.q_φ_C

    def _repeat(self, a, b, c, batch_size):
        return a.repeat(batch_size, 1, 1, 1), b.repeat(batch_size, 1, 1, 1), c.repeat(batch_size, 1, 1, 1)

    def _sample_w(self, z, x, u):
        input = torch.cat((z, x, u), dim=1)
        (w_μ, w_σ) = self.q_χ_μ(input), self.q_χ_σ(input)
        N = Normal(w_μ, w_σ)
        w = N.rsample()
        return w, (w_μ, w_σ)

    def decode(self, z_):
        p_μ, p_σ = self.p_θ_μ(z_), self.p_θ_σ(z_)
        dist_p_θ = Normal(p_μ, p_σ)
        x_ = dist_p_θ.rsample()
        return x_, (p_μ, p_σ)

    def propagate_solution(self, x, u):
        batch_size = x.shape[0]

        z, (w1_μ, w1_σ) = self._initial_generator(x)

        x1, (x1_μ, x1_σ) = self.decode(z)
        x_pred, w_dists, z_pred, x_dists = [], [], [], []
        x_pred.append(x1.unsqueeze(1))
        x_dists.append(torch.stack([x1_μ, x1_σ], dim=-1).unsqueeze(1))
        w_dists.append(torch.stack([w1_μ, w1_σ], dim=-1).unsqueeze(1))
        z_pred.append(z.unsqueeze(1))

        for t in range(1, self.T):
            z_, (w_μ, w_σ) = self.forward(z=z, u=u[:, t - 1], x=x[:, t])
            x_, (x_μ, x_σ) = self.decode(z_)
            # Bookkeeping
            w_dists.append(torch.stack([w_μ, w_σ], dim=-1).unsqueeze(1))
            x_pred.append(x_.unsqueeze(1))
            x_dists.append(torch.stack([x_μ, x_σ], dim=-1).unsqueeze(1))
            z_pred.append(z_.unsqueeze(1))
            z = z_
        x_pred = torch.cat(x_pred, dim=1)
        w_dists = torch.cat(w_dists, dim=1)
        z_pred = torch.cat(z_pred, dim=1)
        x_dists = torch.cat(x_dists, dim=1)
        return x_pred, w_dists, z_pred, x_dists

    def forward(self, z, u, x=None):
        batch_size = u.shape[0]
        u = torch.max(torch.min(u, self.u_high), self.u_low)

        (a, b, c) = self._sample_v()
        (a, b, c) = self._repeat(a, b, c, batch_size)
        (α_A, α_B, α_C) = self._compute_α(torch.cat((z, u), dim=1))
        A = torch.sum(α_A * a, axis=1)
        B = torch.sum(α_B * b, axis=1)
        C = torch.sum(α_C * c, axis=1)
        if x is None:
            (w_μ, w_σ) = torch.zeros((batch_size, self.w_dim)), torch.ones((batch_size, self.w_dim))
        else:
            input = torch.cat((z, x, u), dim=1)
            (w_μ, w_σ) = self.q_χ_μ(input), self.q_χ_σ(input)
        w_dist = Normal(w_μ, w_σ)
        w = w_dist.rsample()
        z_ = torch.bmm(A, z.view(-1, self.z_dim, 1)) + torch.bmm(B, u.view(-1, self.u_dim, 1)) + torch.bmm(C, w.view(-1, self.w_dim, 1))
        return z_.squeeze(2), (w_μ, w_σ)

    @property
    def params(self):
        return self._initial_generator.params + list(self.f_ψ.parameters()) \
               + list(self.q_χ_μ.parameters()) + list(self.q_χ_σ.parameters()) \
               + list(self.p_θ_μ.parameters()) + list(self.p_θ_σ.parameters()) \
               + [self.q_φ_A] + [self.q_φ_B] + [self.q_φ_C]
               # + list(self.p_θ.parameters())

    @property
    def networks(self):
        return self._initial_generator.networks + [self.f_ψ, self.q_χ_μ, self.q_χ_σ, self.p_θ_μ, self.p_θ_σ]

    def _create_optimizer(self):
        # self.optimizer = optim.Adadelta(self.params, lr=1e-1)
        self.optimizer = optim.Adam(self.params, lr=1e-3)
        self.loss_rec = nn.MSELoss()

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

    def update(self, x, u, gradient_updates, debug=False):
        if gradient_updates % 2500 == 0:
           self.c = min(1, 0.01 + gradient_updates / self.Ta)

        x, u = self.cast(x[:, 0:self.T]), self.cast(u[:, 0:self.T])
        x_pred, w_dists, z_pred, x_dists = self.propagate_solution(x, u)

        with torch.no_grad():
            L_rec = self.loss_rec(x_pred[:, 0:self.T].reshape(-1, self.x_dim), x[:, 0:self.T].reshape(-1, self.x_dim))
        x_μ, x_σ = x_dists[:, :, :, 0], x_dists[:, :, :, 1]
        dists = Normal(x_μ, x_σ)
        L_nll = -dists.log_prob(x[:, :]).sum(-1).mean()

        if self.it % 10 == 0 and debug:
            print(f"z[:, 0], z[:, T-1] = {z_pred[0, 0].detach()} {z_pred[0, 1].detach()}")
            print(f"x_pred[:, 0], x[:, 0] = {x_pred[0, 0].detach()} {x[0, 0].detach()}")
            print(f"x_pred[:, self.T-1], x[:, self.T-1] = {x_pred[0, self.T-1].detach()} {x[0, self.T-1].detach()}")

        μ, σ = self.cast(w_dists[:, :, :, 0]), self.cast(w_dists[:, :, :, 1])
        p = Normal(μ, σ)
        q = Normal(torch.zeros_like(μ), torch.ones_like(σ))

        L_KLD = torch.distributions.kl.kl_divergence(p, q).sum(-1).mean()
        self.optimizer.zero_grad()
        L = L_nll + self.c * L_KLD
        L.backward()
        self.optimizer.step()

        self.it += 1
        return L_nll.item(), L_KLD.item(), L_rec.item()

    def save_params(self, path='param/dvbf.pkl'):
        save_dict = {'init_dict': self.init_dict,
                    'networks': [network.state_dict() for network in self.networks],
                    'q_φ_A': self.q_φ_A,
                    'q_φ_B': self.q_φ_B,
                    'q_φ_C': self.q_φ_C}
        torch.save(save_dict, path)

    @classmethod
    def init_from_save(cls, path='param/dvbf.pkl'):
        save_dict = torch.load(path)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for network, params in zip(instance.networks, save_dict['networks']):
            network.load_state_dict(params)
        instance.q_φ_A = save_dict['q_φ_A']
        instance.q_φ_B = save_dict['q_φ_B']
        instance.q_φ_C = save_dict['q_φ_C']

        return instance

    @classmethod
    def init_from_replay_memory(cls, replay_memory, z_dim, u_low, u_high):
        init_dict = {'seq_length': replay_memory.seq_length,
                     'x_dim': replay_memory.state_dim,
                     'u_dim': replay_memory.action_dim,
                     'z_dim': z_dim,
                     'u_low': u_low,
                     'u_high': u_high,
                     'sys': replay_memory.env.name}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance