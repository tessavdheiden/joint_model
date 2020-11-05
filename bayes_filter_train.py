import matplotlib.pyplot as plt
import argparse
from collections import namedtuple
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


from bayes_filter import BayesFilter
from replay_memory import ReplayMemory
from controller import Controller
from pendulum_v2 import PendulumEnv
from bayes_filter_check import visualize_predictions

Record = namedtuple('Record', ['ep', 'l_r', 'l_k'])


# Plot predictions against true time evolution
def train(args):
    torch.manual_seed(0)
    np.random.seed(0)

    controller = Controller()
    env = PendulumEnv()
    env.seed(0)
    replay_memory = ReplayMemory(args, controller=controller, env=env)
    bayes_filter = BayesFilter.init_from_replay_memory(replay_memory)

    records = [None] * args.num_epochs

    for i in range(args.num_epochs):
        replay_memory.reset_batchptr_train()

        for b in range(replay_memory.n_batches_train):
            batch_dict = replay_memory.next_batch_train()
            x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])
            L_rec, L_KLD = bayes_filter.update(x, u)

        if i % 10 == 0:
            with torch.no_grad():
                visualize_predictions(args, bayes_filter, replay_memory)

        records[i] = Record(i, L_rec, L_KLD)
        print(f'ep = {i}, L_rec = {L_rec:.2f} L_KLD = {L_KLD:.4f}')

    fig, ax = plt.subplots()
    ax.scatter([r.ep for r in records], [r.l_r for r in records], c='b')
    ax.set_ylabel('rec', color='b')
    ax2 = ax.twinx()
    ax2.scatter([r.ep for r in records], [r.l_k for r in records], c='r')
    ax2.set_ylabel('KL', color='r')
    ax2.grid('on')
    fig.tight_layout()
    plt.savefig('loss.png')
    bayes_filter.save_params()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_frac', type=float, default=0.1, help='fraction of data to be witheld in validation set')
    parser.add_argument('--seq_length', type=int, default=32, help='sequence length for training')
    parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')

    parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs')

    parser.add_argument('--n_trials', type=int, default=200, help='number of data sequences to collect in each episode')
    parser.add_argument('--trial_len', type=int, default=256, help='number of steps in each trial')
    parser.add_argument('--n_subseq', type=int, default=8, help='number of subsequences to divide each sequence into')
    args = parser.parse_args()

    train(args)