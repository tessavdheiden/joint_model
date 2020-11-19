import os
import numpy as np
import torch
from collections import namedtuple
import matplotlib.pyplot as plt


from env_pendulum import PendulumEnv
from env_ball_box import BallBoxEnv
from env_sigmoid import SigmoidEnv
from empowerment import Empowerment
from controller import Controller
from bayes_filter import BayesFilter
from replay_memory import ReplayMemory
from empowerment_check import visualize_predictions_angles, visualize_predictions_positions, visualize_distributions, visualize_predictions_sigmoid, visualize_distributions_sigmoid


Record = namedtuple('Transition', ['ep', 'E'])


def train_empowerment(env, empowerment, bayes_filter, replay_memory, args):
    records = [None] * args.num_epochs

    for i in range(args.num_epochs):
        replay_memory.reset_batchptr_train()

        E = np.zeros((replay_memory.n_batches_train, args.batch_size * args.seq_length))
        for b in range(replay_memory.n_batches_train):
            batch_dict = replay_memory.next_batch_train()
            x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])

            #_, _, z_pred, _ = bayes_filter.propagate_solution(x, u)
            #E[b, :] = empowerment.update(z_pred.view(-1, z_pred.shape[2]))
            E[b, :] = empowerment.update(x.view(-1, 1))

        if i % 10 == 0:
            with torch.no_grad():
                if isinstance(env, PendulumEnv):
                    visualize_predictions_angles(empowerment, bayes_filter, replay_memory)
                elif isinstance(env, BallBoxEnv):
                    visualize_predictions_positions(empowerment, bayes_filter, replay_memory)
                    #visualize_distributions(empowerment, bayes_filter, replay_memory)
                elif isinstance(env, SigmoidEnv):
                    visualize_predictions_sigmoid(empowerment, bayes_filter, replay_memory, ep=i)
                    visualize_distributions_sigmoid(empowerment, bayes_filter, replay_memory)

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
    parser.add_argument('--seq_length', type=int, default=16, help='sequence length for training')
    parser.add_argument('--batch_size', type=int, default=512, help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('--n_trials', type=int, default=400,
                        help='number of data sequences to collect in each episode')
    parser.add_argument('--trial_len', type=int, default=32, help='number of steps in each trial')
    parser.add_argument('--n_subseq', type=int, default=4,
                        help='number of subsequences to divide each sequence into')
    args = parser.parse_args()

    if not os.path.exists('param'):
        print('bayes filter not trained')

    torch.manual_seed(0)
    np.random.seed(0)
    env = SigmoidEnv()
    env.seed(0)

    controller = Controller(env)
    replay_memory = ReplayMemory(args, controller=controller, env=env)
    bayes_filter = BayesFilter.init_from_replay_memory(replay_memory)
    #bayes_filter.load_params()

    empowerment = Empowerment(env, controller=controller, transition_network=bayes_filter)

    train_empowerment(env, empowerment, bayes_filter, replay_memory, args)


if __name__ == '__main__':
    main()