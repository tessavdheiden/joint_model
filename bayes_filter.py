import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, z_dim, h_dim, x_dim, w_dim, T):
        super(Generator, self).__init__()
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.T = T
        self.rnn = nn.GRU(x_dim, h_dim, bidirectional=True)
        self.p_ξ = nn.Sequential(nn.Linear(h_dim * 2, h_dim), nn.ReLU())
        self.μ = nn.Linear(h_dim, w_dim)
        self.σ = nn.Sequential(nn.Linear(h_dim, w_dim), nn.Softplus())
        self.p_λ = nn.Sequential(nn.Linear(w_dim, h_dim), nn.Linear(h_dim, z_dim))

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, x_dim)
        bi_out, _ = self.rnn(x[:, 0:self.T])                 # bi_out: tensor of shape (batch_size, seq_length, h_dim*2)
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
    def __init__(self, seq_length, x_dim, u_dim):

        super(BayesFilter, self).__init__()
        self.T = seq_length
        self.x_dim = x_dim
        self.u_dim = u_dim

        self.z_dim = 3
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

        self.it = 0

    # Initialize potential transition matrices
    def _create_transition_matrices(self, std=1e-4):
        self.q_φ_A = Variable(torch.randn(self.M, self.z_dim, self.z_dim)*std, requires_grad=True)
        self.q_φ_B = Variable(torch.randn(self.M, self.z_dim, self.u_dim)*std, requires_grad=True)
        self.q_φ_C = Variable(torch.randn(self.M, self.z_dim, self.w_dim)*std, requires_grad=True)

    def _create_transition_network(self):
        self.f_ψ = nn.Sequential(nn.Linear(self.z_dim + self.u_dim, self.h_dim),
                                   nn.Sigmoid(),
                                   nn.Linear(self.h_dim, self.M))

    def _create_recognition_network(self):
        self.q_χ_μ = nn.Sequential(nn.Linear(self.z_dim + self.u_dim + self.x_dim, self.h_dim),
                                   nn.Sigmoid(),
                                   nn.Linear(self.h_dim, self.w_dim))
        self.q_χ_σ = nn.Sequential(nn.Linear(self.z_dim + self.u_dim + self.x_dim, self.h_dim),
                                   nn.Sigmoid(),
                                   nn.Linear(self.h_dim, self.w_dim),
                                   nn.Softplus())

    def _create_decoding_network(self):
        # self.p_θ = nn.Sequential(nn.Linear(self.z_dim, self.h_dim),
        #                          nn.Linear(self.h_dim, self.x_dim))
        self.p_θ_μ = nn.Sequential(nn.Linear(self.z_dim, self.h_dim),
                                   nn.Linear(self.h_dim, self.x_dim))
        self.p_θ_σ = nn.Sequential(nn.Linear(self.z_dim, self.h_dim),
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

    def _sample_x_(self, z_):
        # return self.p_θ(z_)
        p_μ, p_σ = self.p_θ_μ(z_), self.p_θ_σ(z_)
        dist_p_θ = Normal(p_μ, p_σ)
        x_ = dist_p_θ.sample()
        return x_, (p_μ, p_σ)

    def _init_output(self, batch_size):
        x_pred = self.cast(torch.zeros(batch_size, self.T, self.x_dim))
        x_dists = self.cast(torch.zeros(batch_size, self.T, self.x_dim, 2))
        w_dists = self.cast(torch.zeros(batch_size, self.T, self.w_dim, 2))
        z_pred = self.cast(torch.zeros(batch_size, self.T, self.z_dim))
        return x_pred, w_dists, z_pred, x_dists

    def propagate_solution(self, x, u):
        batch_size = x.shape[0]

        z, (w1_μ, w1_σ) = self._initial_generator(x)

        x_pred, w_dists, z_pred, x_dists = self._init_output(batch_size)
        x1, (x1_μ, x1_σ) = self._sample_x_(z)
        x_pred[:, 0] = x1
        x_dists[:, 0] = torch.stack([x1_μ, x1_σ], dim=-1)
        w_dists[:, 0] = torch.stack([w1_μ, w1_σ], dim=-1)
        z_pred[:, 0] = z

        for t in range(1, self.T):
            (a, b, c) = self._sample_v()
            (a, b, c) = self._repeat(a, b, c, batch_size)
            (α_A, α_B, α_C) = self._compute_α(torch.cat((z, u[:, t - 1]), dim=1))
            A = torch.sum(α_A * a, axis=1)
            B = torch.sum(α_B * b, axis=1)
            C = torch.sum(α_C * c, axis=1)
            w, (w_μ, w_σ) = self._sample_w(z, x[:, t], u[:, t - 1])
            w_dists[:, t] = torch.stack([w_μ, w_σ], dim=-1)
            z_ = torch.bmm(A, z.view(-1, self.z_dim, 1)) + torch.bmm(B, u[:, t - 1].view(-1, self.u_dim, 1)) + torch.bmm(C, w.view(-1, self.w_dim, 1))
            z_ = z_.squeeze(2)
            x_, (x_μ, x_σ) = self._sample_x_(z_)
            x_pred[:, t] = x_
            x_dists[:, t] = torch.stack([x_μ, x_σ], dim=-1)
            z_pred[:, t] = z_
            z = z_

        return x_pred, w_dists, z_pred, x_dists

    def forward(self, z, u):
        batch_size = u.shape[0]

        (a, b, c) = self._sample_v()
        (a, b, c) = self._repeat(a, b, c, batch_size)
        (α_A, α_B, α_C) = self._compute_α(torch.cat((z, u), dim=1))
        A = torch.sum(α_A * a, axis=1)
        B = torch.sum(α_B * b, axis=1)
        C = torch.sum(α_C * c, axis=1)
        (w_μ, w_σ) = torch.zeros((batch_size, self.w_dim)), torch.ones((batch_size, self.w_dim))
        w_dist = Normal(w_μ, w_σ)
        w = w_dist.sample()
        z_ = torch.bmm(A, z.view(-1, self.z_dim, 1)) + torch.bmm(B, u.view(-1, self.u_dim, 1)) + torch.bmm(C, w.view(-1, self.w_dim, 1))
        z_ = z_.squeeze(2)
        z = z_
        return z

    @property
    def params(self):
        return list(self._initial_generator.params) + list(self.f_ψ.parameters()) \
               + list(self.q_χ_μ.parameters()) + list(self.q_χ_σ.parameters()) \
               + list(self.p_θ_μ.parameters()) + list(self.p_θ_σ.parameters()) \
               + [self.q_φ_A] + [self.q_φ_B] + [self.q_φ_C]
               # + list(self.p_θ.parameters())

    @property
    def networks(self):
        return self._initial_generator.networks + [self.f_ψ, self.q_χ_μ, self.q_χ_σ, self.p_θ]

    def _prepare_update(self):
        for network in self.networks:
            network.train()
            network.to('cuda:0')

        self.q_φ_A = self.q_φ_A.cuda()
        self.q_φ_B = self.q_φ_B.cuda()
        self.q_φ_C = self.q_φ_C.cuda()

        self.loss_rec = self.loss_rec.cuda()
        self.cast = lambda x: x.cuda()

    def _prepare_compute(self):
        for network in self.networks:
            network.eval()
            network.to('cpu')

        self.q_φ_A = self.q_φ_A.cpu()
        self.q_φ_B = self.q_φ_B.cpu()
        self.q_φ_C = self.q_φ_C.cpu()

        self.loss_rec = self.loss_rec.cpu()
        self.cast = lambda x: x.cpu()

    def _create_optimizer(self):
        # self.optimizer = optim.Adadelta(self.params, lr=1e-1)
        self.optimizer = optim.Adam(self.params, lr=1e-2)
        self.loss_rec = nn.MSELoss()

    def update(self, x, u, debug=False):
        #self._prepare_update()
        x, u = self.cast(x), self.cast(u)
        x_pred, w_dists, _, x_dists = self.propagate_solution(x, u)

        # with torch.no_grad():
        #     L_rec = self.loss_rec(x_pred[:, 0:self.T].reshape(-1, self.x_dim), x[:, 0:self.T].reshape(-1, self.x_dim))
        x_μ, x_σ = x_dists[:, :, :, 0], x_dists[:, :, :, 1] + 1e-3
        dists = Normal(x_μ, x_σ)
        L_nll = -dists.log_prob(x[:, :]).sum(-1).mean()

        if self.it % 10 == 0 and debug:
            print(f"x_pred[:, 0], x[:, 0] = {x_pred[0, 0].detach()} {x[0, 0].detach()}")
            print(f"x_pred[:, self.T-1], x[:, self.T-1] = {x_pred[0, self.T-1].detach()} {x[0, self.T-1].detach()}")

        μ, σ = w_dists[:, :, :, 0], w_dists[:, :, :, 1] + 1e-3
        p = Normal(μ, σ)
        q = Normal(torch.zeros_like(μ), torch.ones_like(σ))
        L_KLD = torch.distributions.kl.kl_divergence(p, q).sum(-1).mean()
        self.optimizer.zero_grad()
        L = L_nll + L_KLD
        L.backward()
        self.optimizer.step()

        self.it += 1
        #self._prepare_compute()
        return L_nll.item(), L_KLD.item()

    def save_params(self, path='param/dvbf_generator_params.pkl'):
        torch.save(self.state_dict(), path)

    def load_params(self, path='param/dvbf_generator_params.pkl'):
        self.load_state_dict(torch.load(path))

    @classmethod
    def init_from_replay_memory(cls, replay_memory):
        instance = cls(seq_length=replay_memory.seq_length, x_dim=replay_memory.state_dim, u_dim=replay_memory.action_dim)
        return instance