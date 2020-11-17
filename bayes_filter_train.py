import matplotlib.pyplot as plt
from collections import namedtuple
import torch
import numpy as np
import os
import argparse

from bayes_filter import BayesFilter
from replay_memory import ReplayMemory
from controller import Controller
from pendulum_v2 import PendulumEnv
from ball_box import BallBoxEnv
from bayes_filter_check import visualize_predictions_angle, visualize_predictions_position

Record = namedtuple('Record', ['ep', 'l_r', 'l_nll', 'l_k'])


# Plot predictions against true time evolution
def train(replay_memory, bayes_filter):
    records = [None] * args.num_epochs

    for i in range(args.num_epochs):
        replay_memory.reset_batchptr_train()

        for b in range(replay_memory.n_batches_train):
            batch_dict = replay_memory.next_batch_train()
            x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])
            L_NLL, L_KLD, L_rec = bayes_filter.update(x, u)

        if i % 10 == 0:
            with torch.no_grad():
                if isinstance(env, PendulumEnv):
                    visualize_predictions_angle(bayes_filter, replay_memory)
                elif isinstance(env, BallBoxEnv):
                    visualize_predictions_position(bayes_filter, replay_memory)

        records[i] = Record(i, L_rec, L_NLL, L_KLD)
        print(f'ep = {i},  L_NLL = {L_NLL:.2f} L_rec = {L_rec:.2f} L_KLD = {L_KLD:.4f}')

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
    ax[0].scatter([r.ep for r in records], [r.l_r for r in records])
    ax[0].set_ylabel('MSE')
    ax[1].scatter([r.ep for r in records], [r.l_nll for r in records])
    ax[1].set_ylabel('NLL')
    ax[2].scatter([r.ep for r in records], [r.l_k for r in records])
    ax[2].set_ylabel('KL')
    fig.tight_layout()
    plt.savefig('img/loss.png')
    bayes_filter.save_params()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='fraction of data to be witheld in validation set')
    parser.add_argument('--seq_length', type=int, default=16, help='sequence length for training')
    parser.add_argument('--batch_size', type=int, default=64, help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--n_trials', type=int, default=200,
                        help='number of data sequences to collect in each episode')
    parser.add_argument('--trial_len', type=int, default=32, help='number of steps in each trial')
    parser.add_argument('--n_subseq', type=int, default=4,
                        help='number of subsequences to divide each sequence into')
    args = parser.parse_args()

    if not os.path.exists('img'):
        os.makedirs('img')
    if not os.path.exists('param'):
        os.makedirs('param')

    torch.manual_seed(0)
    np.random.seed(0)
    env = BallBoxEnv()
    env.seed(0)

    controller = Controller(env)
    replay_memory = ReplayMemory(args, controller=controller, env=env)
    bayes_filter = BayesFilter.init_from_replay_memory(replay_memory)

    train(replay_memory, bayes_filter)