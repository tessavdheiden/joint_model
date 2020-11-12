import matplotlib.pyplot as plt
from collections import namedtuple
import torch
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D


from bayes_filter import BayesFilter
from replay_memory import ReplayMemory
from controller import Controller
from pendulum_v2 import PendulumEnv
from ball_box import BallBoxEnv
from bayes_filter_check import visualize_predictions_angle, visualize_predictions_position

Record = namedtuple('Record', ['ep', 'l_r', 'l_k'])


# Plot predictions against true time evolution
def train(replay_memory, bayes_filter):
    records = [None] * args.num_epochs

    for i in range(args.num_epochs):
        replay_memory.reset_batchptr_train()

        for b in range(replay_memory.n_batches_train):
            batch_dict = replay_memory.next_batch_train()
            x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])
            L_rec, L_KLD = bayes_filter.update(x, u)

        if i % 10 == 0:
            with torch.no_grad():
                if isinstance(env, PendulumEnv):
                    visualize_predictions_angle(bayes_filter, replay_memory)
                elif isinstance(env, BallBoxEnv):
                    visualize_predictions_position(bayes_filter, replay_memory)

        records[i] = Record(i, L_rec, L_KLD)
        print(f'ep = {i}, L_rec = {L_rec:.2f} L_KLD = {L_KLD:.4f}')

    fig, ax = plt.subplots()
    ax.scatter([r.ep for r in records], [r.l_r for r in records], c='b')
    ax.set_ylabel('NLL', color='b')
    ax2 = ax.twinx()
    ax2.scatter([r.ep for r in records], [r.l_k for r in records], c='r')
    ax2.set_ylabel('KL', color='r')
    ax2.grid('on')
    fig.tight_layout()
    plt.savefig('img/loss.png')
    bayes_filter.save_params()


if __name__ == '__main__':
    from args import args

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