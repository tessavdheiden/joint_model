import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from bayes_filter import Generator


class BayesFilterFullyConnected(nn.Module):
    def __init__(self, seq_length, x_dim, u_dim, z_dim, u_max):

        super(BayesFilterFullyConnected, self).__init__()
        self.T = seq_length
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.u_max = u_max
        self.z_dim = z_dim
        self.w_dim = z_dim
        self.h_dim = 128

        self._initial_generator = Generator(z_dim=self.z_dim, h_dim=self.h_dim, x_dim=self.x_dim, w_dim=self.w_dim, T=self.T)
        self._create_observation_network()
        self._create_decoding_network()
        self._create_optimizer()
        self.cast = lambda x: x
        self.it = 0
        self.c = 1

    def _create_observation_network(self):
        self.q_trans = nn.Sequential(nn.Linear(self.z_dim + self.u_dim, self.h_dim),
                                   nn.Sigmoid(), nn.BatchNorm1d(self.h_dim),
                                   nn.Linear(self.h_dim, self.h_dim))
        self.q_trans_μ = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                   nn.Sigmoid(), nn.BatchNorm1d(self.h_dim),
                                   nn.Linear(self.h_dim, self.z_dim))
        self.q_trans_σ = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                   nn.Sigmoid(), nn.BatchNorm1d(self.h_dim),
                                   nn.Linear(self.h_dim, self.z_dim),
                                   nn.Softplus())
        self.q_trans_σ_bar = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                   nn.Sigmoid(), nn.BatchNorm1d(self.h_dim),
                                   nn.Linear(self.h_dim, self.z_dim),
                                   nn.Softplus())
        self.q_meas = nn.Sequential(nn.Linear(self.x_dim, self.h_dim),
                                   nn.Sigmoid(), nn.BatchNorm1d(self.h_dim),
                                   nn.Linear(self.h_dim, self.h_dim))
        self.q_meas_μ = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                   nn.Sigmoid(), nn.BatchNorm1d(self.h_dim),
                                   nn.Linear(self.h_dim, self.z_dim))
        self.q_meas_σ = nn.Sequential(nn.Linear(self.h_dim, self.h_dim),
                                   nn.Sigmoid(), nn.BatchNorm1d(self.h_dim),
                                   nn.Linear(self.h_dim, self.z_dim),
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

    def decode(self, z_):
        p_μ, p_σ = self.p_θ_μ(z_), self.p_θ_σ(z_)
        dist_p_θ = Normal(p_μ, p_σ)
        x_ = dist_p_θ.rsample()
        return x_, (p_μ, p_σ)

    def propagate_solution(self, x, u):
        batch_size = x.shape[0]

        z, (w1_μ, w1_σ) = self._initial_generator(x)

        # x_pred, w_dists, z_pred, x_dists = self._init_output(batch_size)
        x1, (x1_μ, x1_σ) = self.decode(z)
        x_pred, w_dists, z_pred, x_dists = [],[],[],[]
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
        u = torch.clamp(u, min=-self.u_max, max=self.u_max)

        input = torch.cat((z, u), dim=1)
        trans = self.q_trans(input)
        trans_μ, trans_σ, trans_σ_bar = self.q_trans_μ(trans), self.q_trans_σ(trans), self.q_trans_σ_bar(trans)
        if x is None:  # empowerment
            w_μ, w_σ = torch.zeros_like(trans_μ), torch.ones_like(trans_σ)
            z_dist = Normal(trans_μ, trans_σ)
        else:
            meas = self.q_meas(x)
            meas_μ, meas_σ = self.q_meas_μ(meas), self.q_meas_σ(meas)
            q_μ = (trans_μ * meas_σ ** 2 + meas_μ * trans_σ_bar ** 2) / (meas_σ ** 2 + trans_σ_bar ** 2)
            q_σ = torch.sqrt((meas_σ ** 2 * trans_σ_bar ** 2) / (meas_σ ** 2 + trans_σ_bar ** 2))
            w_μ = (q_μ - trans_μ) / trans_σ
            w_σ = torch.sqrt((q_σ ** 2) / (trans_σ ** 2))
            z_dist = Normal(q_μ, q_σ)

        z_ = z_dist.rsample()
        return z_, (w_μ, w_σ)

    @property
    def params(self):
        return self._initial_generator.params + list(self.q_meas.parameters()) \
               + list(self.q_meas_μ.parameters()) + list(self.q_meas_σ.parameters()) \
               + list(self.q_trans.parameters()) + list(self.q_trans_μ.parameters()) \
               + list(self.q_trans_σ.parameters()) + list(self.q_trans_σ_bar.parameters()) \
               + list(self.p_θ_μ.parameters()) + list(self.p_θ_σ.parameters())

    @property
    def networks(self):
        return self._initial_generator.networks + [self.q_meas, self.q_meas_μ, self.q_meas_σ,
               self.q_trans, self.q_trans_μ, self.q_trans_σ, self.q_trans_σ_bar, self.p_θ_μ, self.p_θ_σ]

    def _create_optimizer(self):
        # self.optimizer = optim.Adadelta(self.params, lr=1e-1)
        self.optimizer = optim.Adam(self.params, lr=1e-3)
        self.loss_rec = nn.MSELoss()

    def save_params(self, path='param/dvbf_connected.pkl'):
        save_dict = {'init_dict': self.init_dict,
                    'networks': [network.state_dict() for network in self.networks]}
        torch.save(save_dict, path)

    def update(self, x, u, gradient_updates, debug=False):
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
            print(f"x_pred[:, self.T-1], x[:, self.T-1] = {x_pred[0, self.T - 1].detach()} {x[0, self.T - 1].detach()}")

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
        init_dict = {'seq_length': replay_memory.seq_length,
                     'x_dim': replay_memory.state_dim,
                     'u_dim': replay_memory.action_dim,
                     'z_dim': z_dim,
                     'u_max': u_max}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance