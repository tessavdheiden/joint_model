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

    c = ax.hexbin(X[:, 0], X[:, 1], gridsize=10, C=E[:], mincnt=1)
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

    X, Z, U = [], [], []
    for b in range(1):
        batch_dict = replay_memory.next_batch_train()
        x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])
        _, _, z_pred, _ = bayes_filter.propagate_solution(x, u)

        z_pred = z_pred.view(-1, z_pred.shape[2])
        Z.append(z_pred)

        x = x.reshape(-1, x.shape[2])
        X.append(x)
        U.append(u.reshape(-1, u.shape[2]))

    X = torch.cat(X, dim=0)
    Z = torch.cat(Z, dim=0)
    U = torch.cat(U, dim=0)

    batch_size, z_dim = Z.shape
    (μ_ω, σ_ω) = empowerment.source(Z.unsqueeze(0).repeat(batch_size, 1, 1))
    dist_ω = Normal(μ_ω, σ_ω)
    a_ω = dist_ω.sample()

    Z_hat, _ = empowerment.transition(z=Z.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, Z.shape[-1]), u=a_ω.view(-1, a_ω.shape[-1]))

    Z_1 = Z.unsqueeze(0).repeat(batch_size, 1, 1).view(-1, z_dim)
    Z_N = Z.unsqueeze(1).repeat(1, batch_size, 1).view(-1, z_dim)
    Z_pln = torch.cat((Z_1, Z_N), dim=1)
    (μ_pln, σ_pln) = empowerment.planning(Z_pln)     # z = [[z_b, z_b1], [z_b, z_b2], ..., [z_b, z_bB]]
    dist_pln = Normal(μ_pln.view(batch_size, batch_size, -1), σ_pln.view(batch_size, batch_size, -1))     # dist_pln dim = [B, z_dim]
    a_pln = dist_pln.sample()

    fig, ax = plt.subplots(ncols=6, nrows=1, figsize=(16, 3))
    ax[0].hist(a_pln.view(-1, a_pln.shape[2])[:, 0].numpy(), bins=10)
    ax[0].set_xlabel('$a^q$ at dim=0')
    ax[1].hist(a_pln.view(-1, a_pln.shape[2])[:, 1].numpy(), bins=10)
    ax[1].set_xlabel('$a^q$ at dim=1')
    ax[2].hist(a_ω.view(-1, a_ω.shape[2])[:, 0].numpy(), bins=10)
    ax[2].set_xlabel('$a^\omega$ at dim=0')
    ax[3].hist(a_ω.view(-1, a_ω.shape[2])[:, 1].numpy(), bins=10)
    ax[3].set_xlabel('$a^\omega$ at dim=1')
    ax[4].hist(U[:, 0].numpy(), bins=10)
    ax[4].set_xlabel('a data at dim=0')
    ax[5].hist(U[:, 1].numpy(), bins=10)
    ax[5].set_xlabel('a data at dim=1')
    plt.tight_layout()
    plt.savefig('img/dist_source.png')

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(6, 3))
    ax[0].hist(X[:, 0].numpy(), bins=10)
    ax[0].set_xlabel('x at dim=0')
    ax[1].hist(X[:, 1].numpy(), bins=10)
    ax[1].set_xlabel('x at dim=1')
    plt.tight_layout()
    plt.savefig('img/dist_x.png')

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20, 12))
    ax[0, 0].hist(Z[:, 0].numpy(), bins=10)
    ax[0, 0].set_xlabel('z at dim=0')
    ax[0, 1].hist(Z[:, 1].numpy(), bins=10)
    ax[0, 1].set_xlabel('z at dim=1')
    ax[0, 2].hist(Z[:, 2].numpy(), bins=10)
    ax[0, 2].set_xlabel('z at dim=2')

    ax[1, 0].hist(Z_hat[:, 0].numpy(), bins=10)
    ax[1, 0].set_xlabel('$\hat{z}$ at dim=0')
    ax[1, 1].hist(Z_hat[:, 1].numpy(), bins=10)
    ax[1, 1].set_xlabel('$\hat{z}$ at dim=1')
    ax[1, 2].hist(Z_hat[:, 2].numpy(), bins=10)
    ax[1, 2].set_xlabel('$\hat{z}$ at dim=2')

    plt.savefig('img/dist_z.png')
    # μ_ω_x, μ_ω_y = μ_ω[:, 0], μ_ω[:, 1]
    # μ_σ_x, μ_σ_y = σ_ω[:, 0], σ_ω[:, 1]
    #
    # bins = np.array([-1., -.5, .5, 1.])
    # x, y = X[:, 0], X[:, 1]
    #
    # fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(10, 10))
    # plot_dist(x, bins, μ_ω_x, μ_σ_x, ax[0, 0], bin_num=1, title="-1. < x < -.5")
    # plot_dist(x, bins, μ_ω_x, μ_σ_x, ax[0, 1], bin_num=2, title="-.5 < x <  .5")
    # plot_dist(x, bins, μ_ω_x, μ_σ_x, ax[0, 2], bin_num=3, title=".5 < x <  1.")
    # plot_dist(y, bins, μ_ω_y, μ_σ_y, ax[0, 1], bin_num=1, title="-1. < x < -.5, -1. < y < -.5", color='r')
    # plot_dist(y, bins, μ_ω_y, μ_σ_y, ax[1, 1], bin_num=2, title="-.5 < y <  .5", color='r')
    # plot_dist(y, bins, μ_ω_y, μ_σ_y, ax[2, 1], bin_num=3, title=".5 < y <  1.", color='r')



