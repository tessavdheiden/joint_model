import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from collections import namedtuple
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


from bayes_filter import BayesFilter
from replay_memory import ReplayMemory
from controller import Controller
from pendulum_v2 import PendulumEnv

Record = namedtuple('Record', ['ep', 'l'])


def visualize_predictions(args, bayes_filter, replay_memory):
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    x_all = []
    u_all = []
    replay_memory.reset_batchptr_val()

    for b in range(1):
        batch_dict = replay_memory.next_batch_val()
        x, u = torch.from_numpy(batch_dict["states"])[:, :bayes_filter.T], torch.from_numpy(batch_dict['inputs'])[:, :bayes_filter.T-1]
        x_all.append(x)
        u_all.append(u)

    x = torch.cat(x_all, dim=0)
    u = torch.cat(u_all, dim=0)

    x_, z = bayes_filter.compute(x, u)
    z = z.detach().numpy()

    x = x.reshape(-1, bayes_filter.x_dim)
    angles = np.arctan2(x[:, 1], x[:, 0])
    idx = np.argsort(angles)
    z = z.reshape(-1, bayes_filter.z_dim)
    colors = cm.rainbow(np.linspace(0, 1, len(angles)))

    ax1.scatter(z[idx, 0], z[idx, 1], z[idx, 2], marker='.', color=colors)
    ax2.scatter(z[idx, 0], z[idx, 1], z[idx, 2], marker='.', color=colors)
    ax3.scatter(z[idx, 0], z[idx, 1], z[idx, 2], marker='.', color=colors)
    replay_memory.reset_batchptr_train()

    axes = [ax1, ax2, ax3]
    deg = 0
    for ax in axes:
        ax.set_xlabel('$\\mu_0$')
        ax.set_ylabel('$\\mu_1$')
        ax.set_zlabel('$\\mu_2$')
        ax.axis('auto')
        ax.view_init(30, deg)
        deg += 40

    plt.savefig(f"latent_space.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_frac', type=float, default=0.1, help='fraction of data to be witheld in validation set')
    parser.add_argument('--seq_length', type=int, default=32, help='sequence length for training')
    parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')

    parser.add_argument('--num_epochs', type=int, default=400, help='number of epochs')

    parser.add_argument('--n_trials', type=int, default=200, help='number of data sequences to collect in each episode')
    parser.add_argument('--trial_len', type=int, default=256, help='number of steps in each trial')
    parser.add_argument('--n_subseq', type=int, default=8, help='number of subsequences to divide each sequence into')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    controller = Controller()
    env = PendulumEnv()

    replay_memory = ReplayMemory(args, controller=controller, env=env)
    bayes_filter = BayesFilter(args)
    bayes_filter.load_params()

    visualize_predictions(args, bayes_filter, replay_memory)

if __name__ == '__main__':
    main()