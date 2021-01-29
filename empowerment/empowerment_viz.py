import torch
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats as stats
from envs.env_pendulum import PendulumEnv


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


def visualize_source_planning_distributions_1D(empowerment, bayes_filter, replay_memory, ep=-1):
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


def visualize_empowerment_landschape_2D(args, empowerment, bayes_filter, replay_memory, ep=-1):
    replay_memory.reset_batchptr_train()
    x, e = [], []
    for b in range(replay_memory.n_batches_train):
        batch_dict = replay_memory.next_batch_train()

        if args.use_filter:
            x_in = torch.from_numpy(batch_dict["states"])[:, :bayes_filter.T]
            u_in = torch.from_numpy(batch_dict['inputs'])[:, :bayes_filter.T - 1]
            x.append(x_in)

            x_, _, z_out, _ = bayes_filter.propagate_solution(x_in, u_in)
            e_out = empowerment(z_out.view(-1, bayes_filter.z_dim))
        else:
            x_in = torch.from_numpy(batch_dict["states"])[:, :]
            u_in = torch.from_numpy(batch_dict['inputs'])[:, :-1]
            x.append(x_in)

            e_out = empowerment(x_in.view(-1, x_in.shape[2]))
        e.append(e_out)

    x = torch.cat(x, dim=0).numpy().reshape(-1, replay_memory.state_dim)
    e = torch.cat(e, dim=0).numpy().reshape(-1)
    if args.env == 0:
        x1 = np.arctan2(x[:, 1], x[:, 0])
        x2 = x[:, 2]
    elif args.env == 4 or args.env == 5:    # arm or reacher
        x1 = x[:, -2]
        x2 = x[:, -1]
    else:
        x1 = x[:, 0]
        x2 = x[:, 1]

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 3))
    c = ax.hexbin(x1, x2, gridsize=20, C=e[:], mincnt=1, vmin=e.mean() - .1)
    if args.env == 1 or args.env == 3: # tanh or ball in box
        ax.axis('square')
    fig.colorbar(c, ax=ax)
    ax.set_title(f'Empowerment Landscape, ep = {ep}')
    ax.set_xlabel('x at dim=0')
    ax.set_ylabel('x at dim=1')
    plt.savefig(f'img/empowerment_landscape.png')
    plt.close()


def visualize_distributions_2D(empowerment, bayes_filter, replay_memory):
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
    plt.close()

