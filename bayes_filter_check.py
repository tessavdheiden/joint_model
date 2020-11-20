import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from collections import namedtuple
import torch
import numpy as np
import os


from bayes_filter import BayesFilter
from replay_memory import ReplayMemory
from controller import Controller
from env_pendulum import PendulumEnv
from env_ball_box import BallBoxEnv
from env_sigmoid import SigmoidEnv

Record = namedtuple('Record', ['ep', 'l'])


def visualize_latent_space3D(bayes_filter, replay_memory):
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
    x = x.numpy().reshape(-1, bayes_filter.x_dim)
    z = z.detach().numpy().reshape(-1, bayes_filter.z_dim)

    assert bayes_filter.x_dim == 2
    idx = np.argsort(np.sqrt(np.square(x[:, 0]) + np.square(x[:, 1])))

    colors = cm.rainbow(np.linspace(0, 1, len(idx)))

    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    axes = [ax1, ax2, ax3]

    assert bayes_filter.z_dim == 3
    ax1.scatter(z[idx, 0], z[idx, 1], z[idx, 2], marker='.', color=colors)
    ax2.scatter(z[idx, 0], z[idx, 1], z[idx, 2], marker='.', color=colors)
    ax3.scatter(z[idx, 0], z[idx, 1], z[idx, 2], marker='.', color=colors)

    degrees = np.linspace(0, 80, 3)
    for i, ax in enumerate(axes):
        ax.set_xlabel('$z_0$')
        ax.set_ylabel('$z_1$')
        ax.set_zlabel('$z_2$')
        ax.axis('auto')
        ax.view_init(30, degrees[i])
        ax.set_title(f'angle {degrees[i]:.0f}')

    plt.savefig(f"img/latent_space.png")


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
    else:
        env = SigmoidEnv()
    env.seed(0)

    controller = Controller(env)
    replay_memory = ReplayMemory(args, controller=controller, env=env)
    bayes_filter = BayesFilter(seq_length=replay_memory.seq_length,
                               x_dim=replay_memory.state_dim,
                               u_dim=replay_memory.action_dim,
                               u_max=env.u_max,
                               z_dim=1)
    bayes_filter.load_params()

    if isinstance(env, SigmoidEnv):
        visualize_latent_space1D(bayes_filter, replay_memory)


if __name__ == '__main__':
    main()