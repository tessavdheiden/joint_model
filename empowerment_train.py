import os
import numpy as np
import torch
from collections import namedtuple
import matplotlib.pyplot as plt


from pendulum_v2 import PendulumEnv
from ball_box import BallBoxEnv
from empowerment import Empowerment
from controller import Controller
from bayes_filter import BayesFilter
from replay_memory import ReplayMemory
from empowerment_check import visualize_predictions_angles, visualize_predictions_positions


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
            E[b, :] = empowerment.update(z_pred.view(-1, z_pred.shape[2]))

        if i % 10 == 0:
            with torch.no_grad():
                if isinstance(env, PendulumEnv):
                    visualize_predictions_angles(empowerment)
                elif isinstance(env, BallBoxEnv):
                    visualize_predictions_positions(empowerment)

        records[i] = Record(i, E.mean())
        print(f'ep = {i}, empowerment = {records[i].E:.4f}')

    env.close()
    fig, ax = plt.subplots()
    # ax.scatter([r.ep for r in records], [r.r for r in records], c='b')
    # ax.set_ylabel('reward', color='b')
    ax2 = ax.twinx()
    ax2.scatter([r.ep for r in records], [r.E for r in records], c='r')
    ax2.set_ylabel('empowerment', color='r')
    ax2.grid('on')
    fig.tight_layout()
    plt.savefig('img/reward_vs_empowerment.png')
    #empowerment.save_params()


def main():
    from args import args

    if not os.path.exists('param'):
        print('bayes filter not trained')

    torch.manual_seed(0)
    np.random.seed(0)
    env = BallBoxEnv()
    env.seed(0)

    controller = Controller(env)
    replay_memory = ReplayMemory(args, controller=controller, env=env)
    bayes_filter = BayesFilter.init_from_replay_memory(replay_memory)
    bayes_filter.load_params()

    empowerment = Empowerment(env, controller=controller, transition_network=bayes_filter)

    train_empowerment(env, empowerment, bayes_filter, replay_memory, args)


if __name__ == '__main__':
    main()