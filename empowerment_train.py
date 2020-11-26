import os
import numpy as np
import torch
from collections import namedtuple
import matplotlib.pyplot as plt


from env_pendulum import PendulumEnv
from env_ball_box import BallBoxEnv
from env_sigmoid import SigmoidEnv
from env_sigmoid2d import Sigmoid2DEnv
from empowerment import Empowerment
from controller import Controller
from bayes_filter import BayesFilter
from bayes_filter_fully_connected import BayesFilterFullyConnected
from replay_memory import ReplayMemory
from empowerment_check import visualize_distributions_1D, \
    visualize_empowerment_landschape_1D, visualize_empowerment_landschape_2D, visualize_distributions_2D
from bayes_filter_check import visualize_latent_space1D, visualize_latent_space2D


Record = namedtuple('Transition', ['ep', 'E'])


def train_empowerment(env, empowerment, bayes_filter, replay_memory, args):
    records = [None] * args.num_epochs

    for i in range(args.num_epochs):
        replay_memory.reset_batchptr_train()

        E = np.zeros((replay_memory.n_batches_train, args.batch_size * args.seq_length))
        for b in range(replay_memory.n_batches_train):
            batch_dict = replay_memory.next_batch_train()
            x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])
            _, _, z_pred, _ = bayes_filter.propagate_solution(x, u)
            E[b, :] = empowerment.update(z_pred.reshape(-1, z_pred.shape[2]))

        if i % 10 == 0:
            with torch.no_grad():
                if replay_memory.state_dim == 1:
                    visualize_empowerment_landschape_1D(empowerment, bayes_filter, replay_memory, ep=i)
                    #visualize_distributions_1D(empowerment, bayes_filter, replay_memory, ep=i)
                    visualize_latent_space1D(bayes_filter, replay_memory)
                elif replay_memory.state_dim > 1:
                    visualize_empowerment_landschape_2D(empowerment, bayes_filter, replay_memory, ep=i)
                    #visualize_distributions_2D(empowerment, bayes_filter, replay_memory)

        records[i] = Record(i, E.mean())
        print(f'ep = {i}, empowerment = {records[i].E:.4f}')

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
    parser.add_argument('--seq_length', type=int, default=8, help='sequence length for training')
    parser.add_argument('--batch_size', type=int, default=128, help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=101, help='number of epochs')
    parser.add_argument('--n_trials', type=int, default=2000,
                        help='number of data sequences to collect in each episode')
    parser.add_argument('--trial_len', type=int, default=32, help='number of steps in each trial')
    parser.add_argument('--n_subseq', type=int, default=4,
                        help='number of subsequences to divide each sequence into')
    parser.add_argument('--env', type=int, default=1,
                        help='0=pendulum, 1=ball in box, 2=sigmoid, 3=sigmoid2d')
    args = parser.parse_args()

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
    bayes_filter = BayesFilterFullyConnected.init_from_save()

    empowerment = Empowerment(env, controller=controller, transition_network=bayes_filter)

    train_empowerment(env, empowerment, bayes_filter, replay_memory, args)


if __name__ == '__main__':
    main()