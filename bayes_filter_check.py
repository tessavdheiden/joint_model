import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import namedtuple
import torch
import numpy as np
import os
from sklearn.decomposition import PCA


from filters.bayes_filter import BayesFilter
from filters.bayes_filter_fully_connected import BayesFilterFullyConnected
from filters.simple_filter import SimpleFilter
from replay_memory import ReplayMemory
from controller import Controller
from envs.env_pendulum import PendulumEnv
from envs.env_ball_box import BallBoxEnv
from envs.env_sigmoid import SigmoidEnv
from envs.env_sigmoid2d import Sigmoid2DEnv

Record = namedtuple('Record', ['ep', 'l'])


def visualize_latent_space3D(bayes_filter, replay_memory, ep=-1):
    replay_memory.reset_batchptr_val()
    x, x0, z, z0 = [], [], [], []
    for b in range(replay_memory.n_batches_val):
        batch_dict = replay_memory.next_batch_val()
        x_in = torch.from_numpy(batch_dict["states"])[:, :bayes_filter.T]
        u_in = torch.from_numpy(batch_dict['inputs'])[:, :bayes_filter.T - 1]
        x.append(x_in[:, 1:])
        x_, _, z_out, _ = bayes_filter.propagate_solution(x_in, u_in)
        z.append(z_out[:, 1:])
        z0.append(z_out[:, 0])
        x0.append(x_in[:, 0])

    x = torch.cat(x, dim=0)
    x0 = torch.cat(x0, dim=0)
    z = torch.cat(z, dim=0)
    z0 = torch.cat(z0, dim=0)
    x = x.numpy().reshape(-1, bayes_filter.x_dim)
    x0 = x0.numpy().reshape(-1, bayes_filter.x_dim)
    z = z.detach().numpy().reshape(-1, bayes_filter.z_dim)
    z0 = z0.detach().numpy().reshape(-1, bayes_filter.z_dim)

    # assert bayes_filter.x_dim == 2
    # idx = np.argsort(np.sqrt(np.square(x[:, 0]) + np.square(x[:, 1])))
    idx = np.argsort(x[:, -1])
    colors = cm.rainbow(np.linspace(0, 1, len(idx)))

    fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(10, 6), subplot_kw=dict(projection='3d'))

    assert bayes_filter.z_dim == 3

    degrees = np.linspace(0, 80, 3)

    for i in range(3):
        ax[0, i].scatter(z[idx, 0], z[idx, 1], z[idx, 2], marker='.', color=colors)
        ax[0, i].set_xlabel('$z_0$')
        ax[0, i].set_ylabel('$z_1$')
        ax[0, i].set_zlabel('$z_2$')
        ax[0, i].axis('auto')
        ax[0, i].view_init(30, degrees[i])
        ax[0, i].set_title(f'angle {degrees[i]:.0f}')

    idx = np.argsort(x0[:, -1])
    colors = cm.rainbow(np.linspace(0, 1, len(idx)))
    for i in range(3):
        ax[1, i].scatter(z0[idx, 0], z0[idx, 1], z0[idx, 2], marker='.', color=colors)
        ax[1, i].set_xlabel('$z_0$')
        ax[1, i].set_ylabel('$z_1$')
        ax[1, i].set_zlabel('$z_2$')
        ax[1, i].axis('auto')
        ax[1, i].view_init(30, degrees[i])
        ax[1, i].set_title(f'angle {degrees[i]:.0f}')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f'Latent Space, ep = {ep}')
    plt.savefig(f"img/latent_space.png")
    plt.close()


def visualize_latent_space2D(bayes_filter, replay_memory, ep=-1):
    replay_memory.reset_batchptr_val()
    x0, z0 = [], []
    x, z = [], []
    for b in range(replay_memory.n_batches_val):
        batch_dict = replay_memory.next_batch_val()
        x_in = torch.from_numpy(batch_dict["states"])[:, :bayes_filter.T]
        u_in = torch.from_numpy(batch_dict['inputs'])[:, :bayes_filter.T - 1]
        x_, _, z_out, _ = bayes_filter.propagate_solution(x_in, u_in)
        x0.append(x_in[:, 0])
        z0.append(z_out[:, 0])
        x.append(x_in[:, 1:])
        z.append(z_out[:, 1:])

    x0 = torch.cat(x0, dim=0)
    z0 = torch.cat(z0, dim=0)
    x0 = x0.numpy().reshape(-1, bayes_filter.x_dim)
    z0 = z0.detach().numpy().reshape(-1, bayes_filter.z_dim)
    x = torch.cat(x, dim=0)
    z = torch.cat(z, dim=0)
    x = x.numpy().reshape(-1, bayes_filter.x_dim)
    z = z.detach().numpy().reshape(-1, bayes_filter.z_dim)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    assert bayes_filter.z_dim == 2
    idx = np.argsort(np.sqrt(np.square(x0[:, 0]) + np.square(x0[:, 1])))
    colors = cm.rainbow(np.linspace(0, 1, len(idx)))

    ax[0].scatter(z0[idx, 0], z0[idx, 1], marker='.', color=colors)
    ax[0].set_xlabel('latent z0 at dim=0')
    ax[0].set_ylabel('latent z0 at dim=1')
    ax[0].axis('equal')

    idx = np.argsort(np.sqrt(np.square(x[:, 0]) + np.square(x[:, 1])))
    colors = cm.rainbow(np.linspace(0, 1, len(idx)))
    ax[1].scatter(z[idx, 0], z[idx, 1], marker='.', color=colors)
    ax[1].set_xlabel('latent z at dim=0')
    ax[1].set_ylabel('latent z at dim=1')
    ax[1].axis('equal')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f'Latent Space, ep = {ep}')
    plt.savefig(f"img/latent_space.png")
    plt.close()


def visualize_distributions_2D(bayes_filter, replay_memory):
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

    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(9, 3))
    ax[0].hist(X[:, 0].numpy(), bins=10)
    ax[0].set_xlabel('x at dim=0')
    ax[1].hist(X[:, 1].numpy(), bins=10)
    ax[1].set_xlabel('x at dim=1')
    ax[2].hist2d(X[:, 0].numpy(), X[:, 1].numpy())
    plt.tight_layout()
    plt.savefig('img/dist_x.png')

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(6, 3))
    ax[0].hist(Z[:, 0].numpy(), bins=10)
    ax[0].set_xlabel('z at dim=0')
    ax[1].hist(Z[:, 1].numpy(), bins=10)
    ax[1].set_xlabel('z at dim=1')

    plt.savefig('img/dist_z.png')
    plt.close()


def visualize_latent_space1D(bayes_filter, replay_memory):
    replay_memory.reset_batchptr_val()
    x, z = [], []
    for b in range(replay_memory.n_batches_val):
        batch_dict = replay_memory.next_batch_val()
        x_in = torch.from_numpy(batch_dict["states"])[:, :bayes_filter.T]
        u_in = torch.from_numpy(batch_dict['inputs'])[:, :bayes_filter.T - 1]
        x.append(x_in)
        x_, _, z_out, _ = bayes_filter.propagate_solution(x_in, u_in)
        z.append(z_out)

    x = torch.cat(x, dim=0)
    z = torch.cat(z, dim=0)
    x = x.numpy().reshape(-1)
    z = z.detach().numpy().reshape(-1)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))

    assert bayes_filter.z_dim == 1
    idx = np.argsort(x)
    colors = cm.rainbow(np.linspace(0, 1, len(idx)))

    ax.scatter(x[idx], z[idx], marker='.', color=colors)
    ax.set_xlabel('observed x')
    ax.set_ylabel('latent z')

    plt.tight_layout()
    plt.savefig(f"img/latent_space.png")


def visualize_latent_spaceND(bayes_filter, replay_memory):
    replay_memory.reset_batchptr_val()
    x, z = [], []
    for b in range(replay_memory.n_batches_val):
        batch_dict = replay_memory.next_batch_val()
        x_in = torch.from_numpy(batch_dict["states"])[:, :bayes_filter.T]
        u_in = torch.from_numpy(batch_dict['inputs'])[:, :bayes_filter.T - 1]
        x.append(x_in)
        x_, _, z_out, _ = bayes_filter.propagate_solution(x_in, u_in)
        z.append(z_out)

    x = torch.cat(x, dim=0)
    z = torch.cat(z, dim=0)
    x = x.numpy().reshape(replay_memory.n_batches_val * replay_memory.batch_size * bayes_filter.T, -1)
    z = z.detach().numpy().reshape(replay_memory.n_batches_val * replay_memory.batch_size * bayes_filter.T, -1)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))

    assert bayes_filter.z_dim == z.shape[1]

    pca = PCA(n_components=2)
    pca.fit(z)
    z = pca.transform(z)

    ax.scatter(z[:, 0], z[:, 1], marker='.')
    ax.set_xlabel('latent z at dim=0')
    ax.set_ylabel('latent z at dim=1')
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig(f"img/latent_space.png")


def plot_trajectory(bayes_filter, replay_memory, ep=-1):
    fig, ax = plt.subplots(nrows=1, ncols=replay_memory.state_dim, figsize=(4 * replay_memory.state_dim, 4))
    axc = plt.gca()
    replay_memory.reset_batchptr_train()

    for b in range(replay_memory.n_batches_train):
        batch_dict = replay_memory.next_batch_train()
        x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])
        _, _, z, _ = bayes_filter.propagate_solution(x=x, u=u)

        z = z[:, 0]
        x_dvbf, _ = bayes_filter.decode(z)
        x_hat = []
        x_hat.append(x_dvbf.unsqueeze(1))
        for t in range(1, bayes_filter.T):
            z_, _ = bayes_filter(u=u[:, t - 1], z=z)
            x_dvbf, _ = bayes_filter.decode(z_)
            z = z_
            x_hat.append(x_dvbf.unsqueeze(1))
        x_pred, _, _, _ = bayes_filter.propagate_solution(x, u)
        x_hat = torch.cat(x_hat, dim=1)

        t = np.arange(replay_memory.seq_length)
        x_hat = x_hat.detach().numpy()
        x_pred = x_pred.detach().numpy()
        x = x.detach().numpy()
        for i in range(replay_memory.batch_size):
            c = next(axc._get_lines.prop_cycler)['color']

            for dim in range(x_hat.shape[2]):
                ax[dim].plot(t, x_hat[i, :, dim], linestyle='--', color=c, label='dvbf trans')
                #ax[dim].plot(t, x_pred[i, :, dim], linestyle=':', color=c, label='dvbf prop')
                ax[dim].plot(t, x[i, :, dim], color=c, label='true')
                ax[dim].set_xlabel('time')
                ax[dim].set_ylabel(f'x at dim={dim}')
                ax[dim].legend()
            if i > 0: break
        break

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle(f'{type(bayes_filter).__name__} Prediction, ep = {ep}')
    plt.savefig('img/pred_vs_true.png')
    plt.close(fig)


def main():
    from bayes_filter_train import args
    if not os.path.exists('param'):
        print('bayes filter not trained')

    torch.manual_seed(0)
    np.random.seed(0)
    if args.env == 0:
        env = PendulumEnv()
    elif args.env == 1:
        env = BallBoxEnv()
    elif args.env == 2:
        env = SigmoidEnv()
    elif args.env == 3:
        env = Sigmoid2DEnv()
    env.seed(0)

    controller = Controller(env)
    replay_memory = ReplayMemory(args, controller=controller, env=env)

    if args.filter_type == 0:
        bayes_filter = BayesFilter.init_from_save()
    elif args.filter_type == 1:
        bayes_filter = BayesFilterFullyConnected.init_from_save()
    elif args.filter_type == 2:
        bayes_filter = SimpleFilter.init_from_save()

    plot_trajectory(bayes_filter, replay_memory)
    # if bayes_filter.z_dim == 1:
    #     visualize_latent_space1D(bayes_filter, replay_memory)
    # elif bayes_filter.z_dim == 2:
    #     visualize_latent_space2D(bayes_filter, replay_memory)


if __name__ == '__main__':
    main()