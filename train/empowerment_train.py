
import os
import numpy as np
import torch
from collections import namedtuple
import matplotlib.pyplot as plt
import time

from envs import *
from empowerment.empowerment import Empowerment
from controller import Controller
from filters.bayes_filter import BayesFilter
from filters.bayes_filter_fully_connected import BayesFilterFullyConnected
from filters.simple_filter import SimpleFilter
from memory.replay_memory import ReplayMemory
from empowerment.empowerment_viz import visualize_empowerment_landschape_1D, visualize_empowerment_landschape_2D

Record = namedtuple('Transition', ['ep', 'E'])


def train_empowerment(env, empowerment, replay_memory, args, bayes_filter=None):
    records = [None] * args.num_epochs

    for i in range(args.num_epochs):
        replay_memory.reset_batchptr_train()
        t0 = time.time()
        E = np.zeros((replay_memory.n_batches_train, args.batch_size * args.seq_length))
        empowerment.prepare_update()
        for b in range(replay_memory.n_batches_train):
            batch_dict = replay_memory.next_batch_train()
            x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])
            if args.use_filter:
                bayes_filter.prepare_eval()
                _, _, z_pred, _ = bayes_filter.propagate_solution(x, u)
                E[b, :] = empowerment.update(z_pred.reshape(-1, z_pred.shape[2]))
            else:
                E[b, :] = empowerment.update(x.reshape(-1, x.shape[2]))

        if i % 10 == 0:
            with torch.no_grad():
                empowerment.prepare_eval()
                if replay_memory.state_dim == 1:
                    visualize_empowerment_landschape_1D(empowerment, bayes_filter, replay_memory, ep=i)
                else:
                    visualize_empowerment_landschape_2D(args, empowerment, bayes_filter, replay_memory, env, ep=i)
                empowerment.prepare_update()

        records[i] = Record(i, E.mean())
        print(f'ep = {i}, empowerment = {records[i].E:.4f} time = {time.time()-t0:.2f}')

    env.close()
    fig, ax = plt.subplots()
    ax.scatter([r.ep for r in records], [r.E for r in records])
    ax.set_xlabel('Episode')
    ax.set_ylabel('$\\mathcal{E}$')
    ax.grid('on')
    fig.tight_layout()
    plt.savefig('img/curve_empowerment.png')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='fraction of data to be witheld in validation set')
    parser.add_argument('--seq_length', type=int, default=100, help='sequence length for training')
    parser.add_argument('--batch_size', type=int, default=128, help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=2001, help='number of epochs')
    parser.add_argument('--n_trials', type=int, default=3000,
                        help='number of data sequences to collect in each episode')
    parser.add_argument('--trial_len', type=int, default=128, help='number of steps in each trial')
    parser.add_argument('--n_subseq', type=int, default=4,
                        help='number of subsequences to divide each sequence into')
    parser.add_argument('--env', type=str, default='controlled_reacher',
                        help='pendulum, ball_in_box, tanh2d, arm, reacher, controlled_reacher')
    parser.add_argument('--filter_type', type=int, default=1,
                        help='0=bayes filter, 1=bayes filter fully connected')
    parser.add_argument('--use_filter', type=int, default=0,
                        help='0=env, 1=filter')
    args = parser.parse_args()

    if not os.path.exists('../param'):
        print('bayes filter not trained')

    torch.manual_seed(0)
    np.random.seed(0)
    if args.env == 'pendulum':
        env = PendulumEnv()
    elif args.env == 'ball_in_box':
        env = BallBoxEnv()
    elif args.env == 'tanh2d':
        env = Tanh2DEnv()
    elif args.env == 'reacher':
        env = ReacherEnv()
    elif args.env == 'arm':
        env = ArmEnv()
    elif args.env == 'controlled_reacher':
        env = ReacherControlledEnv()
    env.seed(0)

    controller = Controller(env)
    replay_memory = ReplayMemory(args, controller=controller, env=env)

    # assert bayes_filter.T == replay_memory.seq_length

    empowerment = Empowerment(env, controller=controller)
    if args.use_filter:
        if args.filter_type == 0:
            bayes_filter = BayesFilter.init_from_save()
        elif args.filter_type == 1:
            bayes_filter = BayesFilterFullyConnected.init_from_save()
        elif args.filter_type == 2:
            bayes_filter = SimpleFilter.init_from_save()
        empowerment.set_transition(bayes_filter)

        train_empowerment(env, empowerment, replay_memory, args, bayes_filter)
    else:
        train_empowerment(env, empowerment, replay_memory, args)


if __name__ == '__main__':
    main()