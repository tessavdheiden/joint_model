import torch
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats


def visualize_predictions_angles(empowerment, bayes_filter):
    x_pxl, y_pxl = 100, 100

    s = torch.Tensor([[np.cos(theta), np.sin(theta), thetadot]
                          for thetadot in np.linspace(-8, 8, y_pxl)
                          for theta in np.linspace(-np.pi, np.pi, x_pxl)])

    v = empowerment(s)
    value_map = v.view(y_pxl, x_pxl).detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(value_map, cmap=plt.cm.spring, interpolation='bicubic')
    plt.colorbar(im, shrink=0.5)
    ax.set_title('Empowerment Landscape')
    ax.set_xlabel('$\\theta$')
    ax.set_xticks(np.linspace(0, x_pxl, 5))
    ax.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_yticks(np.linspace(0, y_pxl, 5))
    ax.set_yticklabels(['-8', '-4', '0', '4', '8'])
    plt.savefig('img/empowerment_landscape.png')


def visualize_predictions_positions(empowerment, bayes_filter, replay_memory):
    replay_memory.reset_batchptr_train()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    X, E = [], []
    for b in range(replay_memory.n_batches_train):
        batch_dict = replay_memory.next_batch_train()
        x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])
        _, _, z_pred, _ = bayes_filter.propagate_solution(x, u)

        E.append(empowerment(z_pred.view(-1, z_pred.shape[2])).detach())
        x = x.reshape(-1, x.shape[2])
        X.append(x)

    X = torch.cat(X, dim=0)
    E = torch.cat(E, dim=0)
    c = ax.hexbin(X[:, 0], X[:, 1], gridsize=10, C=E[:])
    fig.colorbar(c, ax=ax)
    ax.set_title('Empowerment Landscape')
    plt.savefig('img/empowerment_landscape.png')


def visualize_distributions(empowerment, bayes_filter, replay_memory):
    def plot_dist(x, bins, μ_ω, μ_σ, ax, bin_num, title, color='b'):
        inds = np.digitize(x, bins)
        μ_ω = μ_ω[inds == bin_num].mean()
        μ_σ = μ_σ[inds == bin_num].mean()
        x = np.linspace(μ_ω - 3 * μ_σ, μ_ω + 3 * μ_σ, 100)
        ax.plot(x, stats.norm.pdf(x, μ_ω, μ_σ), c=color)
        ax.set_title(title)

    replay_memory.reset_batchptr_train()

    X, Z = [], []
    for b in range(replay_memory.n_batches_train):
        batch_dict = replay_memory.next_batch_train()
        x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])
        _, _, z_pred, _ = bayes_filter.propagate_solution(x, u)

        z_pred = z_pred.view(-1, z_pred.shape[2])
        Z.append(z_pred)

        x = x.reshape(-1, x.shape[2])
        X.append(x)

    X = torch.cat(X, dim=0)
    Z = torch.cat(Z, dim=0)

    (μ_ω, σ_ω) = empowerment.source(Z)
    dist_ω = Normal(μ_ω, σ_ω)
    a_ω = dist_ω.sample()
    p_ω = dist_ω.log_prob(a_ω).exp()

    p_pln = []
    for b, z in enumerate(Z):
        z = z.unsqueeze(0).repeat(Z.shape[0], 1)
        (μ_pln, σ_pln) = empowerment.planning(torch.cat((z, Z), dim=1))     # z = [[z_b, z_b1], [z_b, z_b2], ..., [z_b, z_bB]]
        dist_pln = Normal(μ_pln, σ_pln)     # dist_pln dim = [B, z_dim]
        p_pln.append(dist_pln.log_prob(a_ω[b]).exp().sum(0).unsqueeze(0))

    p_pln = torch.cat(p_pln, dim=0)

    μ_ω_x, μ_ω_y = μ_ω[:, 0], μ_ω[:, 1]
    μ_σ_x, μ_σ_y = σ_ω[:, 0], σ_ω[:, 1]

    bins = np.array([-1., -.5, .5, 1.])
    x, y = X[:, 0], X[:, 1]

    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(10, 10))
    plot_dist(x, bins, μ_ω_x, μ_σ_x, ax[0, 0], bin_num=1, title="-1. < x < -.5")
    plot_dist(x, bins, μ_ω_x, μ_σ_x, ax[0, 1], bin_num=2, title="-.5 < x <  .5")
    plot_dist(x, bins, μ_ω_x, μ_σ_x, ax[0, 2], bin_num=3, title=".5 < x <  1.")
    plot_dist(y, bins, μ_ω_y, μ_σ_y, ax[0, 1], bin_num=1, title="-1. < x < -.5, -1. < y < -.5", color='r')
    plot_dist(y, bins, μ_ω_y, μ_σ_y, ax[1, 1], bin_num=2, title="-.5 < y <  .5", color='r')
    plot_dist(y, bins, μ_ω_y, μ_σ_y, ax[2, 1], bin_num=3, title=".5 < y <  1.", color='r')

    plt.savefig('img/dist_source.png')

