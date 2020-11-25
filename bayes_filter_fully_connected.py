import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
from torch.autograd import Variable


class BayesFilterFullyConnected(nn.Module):
    def __init__(self, seq_length, x_dim, u_dim, z_dim, u_max):

        super(BayesFilterFullyConnected, self).__init__()
        self.T = seq_length
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.u_max = u_max
        self.z_dim = z_dim
        self.w_dim = 6
        self.h_dim = 128

        self.num_layers = 1
        self.encoder = nn.LSTM(self.h_dim, self.h_dim, self.num_layers)
        self._create_recognition_network()
        self._create_transition_network()
        self._create_decoding_network()
        self._create_optimizer()
        self.it = 0
        self.beta = 50

    def _create_recognition_network(self):
        self.q_χ = nn.Sequential(nn.Linear(self.x_dim, self.h_dim),
                                   nn.Sigmoid(),
                                   nn.Linear(self.h_dim, self.z_dim))

    def _create_transition_network(self):
        self.f_ψ_μ = nn.Sequential(nn.Linear(self.z_dim + self.u_dim, self.h_dim),
                                   nn.Sigmoid(),
                                   nn.Linear(self.h_dim, self.z_dim))
        self.f_ψ_σ = nn.Sequential(nn.Linear(self.z_dim + self.u_dim, self.h_dim),
                                   nn.Sigmoid(),
                                   nn.Linear(self.h_dim, self.z_dim))

    def _create_decoding_network(self):
        self.p_θ_μ = nn.Sequential(nn.Linear(self.z_dim + self.u_dim, self.h_dim),
                                   nn.Sigmoid(),
                                   nn.Linear(self.h_dim, self.x_dim))
        self.p_θ_σ = nn.Sequential(nn.Linear(self.z_dim + self.u_dim, self.h_dim),
                                   nn.Sigmoid(),
                                   nn.Linear(self.h_dim, self.x_dim))

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.h_dim),
            torch.zeros(self.num_layers, batch_size, self.h_dim)
        )

    def _init_output(self, batch_size):
        x_pred = torch.zeros(batch_size, self.T, self.x_dim)
        x_dists = torch.zeros(batch_size, self.T, self.x_dim, 2)
        z_dists = torch.zeros(batch_size, self.T, self.z_dim, 2)
        z_pred = torch.zeros(batch_size, self.T, self.z_dim)
        return x_pred, z_pred, x_dists, z_dists

    def propagate_solution(self, x, u):
        batch_size, seq_length, x_dim = x.shape
        _, _, u_dim = u.shape

        x_pred, z_pred, x_dists, z_dists = self._init_output(batch_size)
        z_ = self.q_χ(x[:, 0])
        z_pred[:, 0] = z_
        for t in range(1, self.T):
            #z_ = self.q_χ(x[:, t])

            z_, (μ_trans, σ_trans) = self.forward(u=u[:, t - 1], z=z_)
            x_μ, x_σ = self.p_θ_μ(torch.cat((z_, u[:, t - 1]), dim=1)), self.p_θ_σ(torch.cat((z_, u[:, t - 1]), dim=1))
            dist_x = Normal(x_μ, x_σ)
            x_ = dist_x.rsample()
            x_dists[:, t - 1] = torch.stack([x_μ, x_σ], dim=-1)
            z_dists[:, t - 1] = torch.stack([μ_trans, σ_trans], dim=-1)
            x_pred[:, t] = x_
            z_pred[:, t] = z_

        return x_pred, z_dists, z_pred, x_dists

    def forward(self, z, u):
        u = torch.clamp(u, min=-self.u_max, max=self.u_max)

        μ_trans, σ_trans = self.f_ψ_μ(torch.cat((z, u), dim=1)), self.f_ψ_σ(torch.cat((z, u), dim=1))
        dist = Normal(μ_trans, σ_trans)
        z_ = dist.rsample()
        return z_, (μ_trans, σ_trans)

    @property
    def params(self):
        return list(self.f_ψ_μ.parameters()) + list(self.f_ψ_σ.parameters()) \
               + list(self.q_χ.parameters()) \
               + list(self.p_θ_μ.parameters()) + list(self.p_θ_σ.parameters())

    @property
    def networks(self):
        return [self.f_ψ_μ, self.f_ψ_σ, self.q_χ, self.p_θ_μ, self.p_θ_σ]

    def _create_optimizer(self):
        #self.optimizer = optim.Adadelta(self.params, lr=1e-3)
        self.optimizer = optim.Adam(self.params, lr=1e-3)
        self.loss_rec = nn.MSELoss()

    def update(self, x, u, debug=False):
        self.it += 1

        x_pred, z_dists, z_pred, x_dists = self.propagate_solution(x, u)
        x_μ, x_σ = x_dists[:, :, :, 0], x_dists[:, :, :, 1]
        dists = Normal(x_μ, x_σ)
        L_nll = -dists.log_prob(x[:, 0:self.T]).sum(-1).mean()

        L_rec = self.loss_rec(x_pred[:, 0:self.T].reshape(-1, self.x_dim), x[:, 0:self.T].reshape(-1, self.x_dim))

        self.optimizer.zero_grad()
        L = L_nll
        L.backward()
        self.optimizer.step()
        return L_rec.item(), L_rec.item(), L_rec.item()

    def save_params(self, path='param/dvbf_connected.pkl'):
        save_dict = {'init_dict': self.init_dict,
                    'networks': [network.state_dict() for network in self.networks]}
        torch.save(save_dict, path)

    @classmethod
    def init_from_save(cls, path='param/dvbf_connected.pkl'):
        save_dict = torch.load(path)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for network, params in zip(instance.networks, save_dict['networks']):
            network.load_state_dict(params)

        return instance

    @classmethod
    def init_from_replay_memory(cls, replay_memory, z_dim, u_max):
        init_dict = {'seq_length': replay_memory.seq_length // 2,
                     'x_dim': replay_memory.state_dim,
                     'u_dim': replay_memory.action_dim,
                     'z_dim': z_dim,
                     'u_max': u_max}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance