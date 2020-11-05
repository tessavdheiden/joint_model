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
from pendulum_v2 import PendulumEnv

Record = namedtuple('Record', ['ep', 'l'])


def visualize_predictions(bayes_filter, replay_memory):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    x_all = []
    u_all = []

    replay_memory.reset_batchptr_val()
    for b in range(replay_memory.n_batches_val):
        batch_dict = replay_memory.next_batch_val()
        x, u = torch.from_numpy(batch_dict["states"])[:, :bayes_filter.T], torch.from_numpy(batch_dict['inputs'])[:, :bayes_filter.T-1]
        x_all.append(x)
        u_all.append(u)

    x = torch.cat(x_all, dim=0)
    u = torch.cat(u_all, dim=0)

    x_, _, z = bayes_filter(x, u)
    z = z.detach().numpy()

    x = x.reshape(-1, bayes_filter.x_dim)
    angles = np.arctan2(x[:, 1], x[:, 0])
    idx = np.argsort(angles)
    z = z.reshape(-1, bayes_filter.z_dim)
    colors = cm.rainbow(np.linspace(0, 1, len(angles)))

    ax1.scatter(z[idx, 0], z[idx, 1], z[idx, 2], marker='.', color=colors)
    ax2.scatter(z[idx, 0], z[idx, 1], z[idx, 2], marker='.', color=colors)
    ax3.scatter(z[idx, 0], z[idx, 1], z[idx, 2], marker='.', color=colors)

    axes = [ax1, ax2, ax3]
    deg = 0
    for ax in axes:
        ax.set_xlabel('$\\mu_0$')
        ax.set_ylabel('$\\mu_1$')
        ax.set_zlabel('$\\mu_2$')
        ax.axis('auto')
        ax.view_init(30, deg)
        deg += 40

    plt.savefig(f"img/latent_space.png")


def main():
    from args import args
    if not os.path.exists('param'):
        print('bayes filter not trained')

    torch.manual_seed(0)
    np.random.seed(0)
    env = PendulumEnv()
    env.seed(0)

    controller = Controller()
    replay_memory = ReplayMemory(args, controller=controller, env=env)
    bayes_filter = BayesFilter.init_from_replay_memory(replay_memory)
    bayes_filter.load_params()

    visualize_predictions(bayes_filter, replay_memory)

if __name__ == '__main__':
    main()