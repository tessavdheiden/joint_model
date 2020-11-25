import torch
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats


def visualize_empowerment_landschape_1D(empowerment, bayes_filter, replay_memory, ep=-1):
    replay_memory.reset_batchptr_train()
    x, z, e = [], [], []
    for b in range(replay_memory.n_batches_train):
        batch_dict = replay_memory.next_batch_train()
        x_in = torch.from_numpy(batch_dict["states"])[:, :bayes_filter.T]
        u_in = torch.from_numpy(batch_dict['inputs'])[:, :bayes_filter.T - 1]
        x.append(x_in)

        x_, _, z_out, _ = bayes_filter.propagate_solution(x_in, u_in)
        z_out = z_out.view(-1, bayes_filter.z_dim)
        e_out = empowerment(z_out)
        e.append(e_out)

        z.append(z_out)

    x = torch.cat(x, dim=0).numpy().reshape(-1)
    e = torch.cat(e, dim=0).numpy().reshape(-1)

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(6, 3))
    fig.suptitle(f'epoch = {ep}')

    steps = 11
    bins = np.linspace(0, 1, steps)
    digitized = np.digitize(bins=bins, x=x)
    e_means = [e[digitized == i].mean() for i in range(1, len(bins))]
    e_means = np.asarray(e_means).reshape(-1, 1)
    im = ax[0].imshow(e_means)
    ax[0].set_yticks(np.linspace(0, steps, steps))
    ax[0].set_yticklabels(list(np.around(bins, decimals=2)))

    steps = 100
    bins = np.linspace(0, 1, steps)
    digitized = np.digitize(bins=bins, x=x)
    e_means = [e[digitized == i].mean() for i in range(1, len(bins))]
    e_means = np.asarray(e_means).reshape(-1, 1)
    ax[1].scatter(bins[1:], e_means)

    plt.savefig('img/empowerment_landscape.png')


def visualize_distributions_1D(empowerment, bayes_filter, replay_memory, ep=-1):
    replay_memory.reset_batchptr_train()
    x, z, e = [], [], []
    for b in range(replay_memory.n_batches_train):
        batch_dict = replay_memory.next_batch_train()
        x_in = torch.from_numpy(batch_dict["states"])[:, :bayes_filter.T]
        u_in = torch.from_numpy(batch_dict['inputs'])[:, :bayes_filter.T - 1]
        x.append(x_in)
        x_, _, z_out, _ = bayes_filter.propagate_solution(x_in, u_in)
        z_out = z_out.view(-1, bayes_filter.z_dim)
        e_out = empowerment(z_out)
        e.append(e_out)

        z.append(z_out)

    x = torch.cat(x, dim=0).numpy().reshape(-1)
    e = torch.cat(e, dim=0).numpy().reshape(-1)
    z = torch.cat(z, dim=0).view(-1, 1)

    fig, ax = plt.subplots(ncols=6, nrows=1, figsize=(16, 3))
    fig.suptitle(f'epoch = {ep}')
    (μ_ω, σ_ω) = empowerment.source(z)
    dist_ω = Normal(μ_ω, σ_ω)
    a_ω = dist_ω.sample()

    z_, _ = empowerment.transition(z=z, u=a_ω)
    z_pln = torch.cat((z, z_), dim=1)
    (μ_pln, σ_pln) = empowerment.planning(z_pln)
    dist_pln = Normal(μ_pln, σ_pln)
    a_pln = dist_pln.sample()

    ax[0].hist(a_pln.view(-1, 1)[:, 0].numpy(), bins=10)
    ax[0].set_xlabel('$a^q$ at dim=0')

    ax[2].hist(a_ω.view(-1, 1)[:, 0].numpy(), bins=10)
    ax[2].set_xlabel('$a^\omega$ at dim=0')

    plt.tight_layout()
    plt.savefig('img/dist_source.png')


def visualize_empowerment_landschape_2D(empowerment, bayes_filter, replay_memory, ep=-1):
    replay_memory.reset_batchptr_train()
    x, z, e = [], [], []
    for b in range(replay_memory.n_batches_train):
        batch_dict = replay_memory.next_batch_train()
        x_in = torch.from_numpy(batch_dict["states"])[:, :bayes_filter.T]
        u_in = torch.from_numpy(batch_dict['inputs'])[:, :bayes_filter.T - 1]
        x.append(x_in)

        x_, _, z_out, _ = bayes_filter.propagate_solution(x_in, u_in)
        z_out = z_out.reshape(-1, bayes_filter.z_dim)
        e_out = empowerment(z_out)
        e.append(e_out)

        z.append(z_out)

    x = torch.cat(x, dim=0).numpy().reshape(-1, 2)
    e = torch.cat(e, dim=0).numpy().reshape(-1)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 3))
    c = ax.hexbin(x[:, 0], x[:, 1], gridsize=10, C=e[:], mincnt=1)
    fig.colorbar(c, ax=ax)
    ax.set_title(f'Empowerment Landscape, ep = {ep}')
    plt.savefig('img/empowerment_landscape.png')


def visualize_distributions_2D(empowerment, bayes_filter, replay_memory):
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
    # ax[0, 2].hist(Z[:, 2].numpy(), bins=10)
    # ax[0, 2].set_xlabel('z at dim=2')

    ax[1, 0].hist(Z_hat[:, 0].numpy(), bins=10)
    ax[1, 0].set_xlabel('$\hat{z}$ at dim=0')
    ax[1, 1].hist(Z_hat[:, 1].numpy(), bins=10)
    ax[1, 1].set_xlabel('$\hat{z}$ at dim=1')
    # ax[1, 2].hist(Z_hat[:, 2].numpy(), bins=10)
    # ax[1, 2].set_xlabel('$\hat{z}$ at dim=2')

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



