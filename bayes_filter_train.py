import matplotlib.pyplot as plt
from collections import namedtuple
import torch
import numpy as np
import os
import argparse
import time

from filters.bayes_filter import BayesFilter
from filters.bayes_filter_fully_connected import BayesFilterFullyConnected
from filters.simple_filter import SimpleFilter
from memory.replay_memory import ReplayMemory
from controller import Controller
from envs.env_pendulum import PendulumEnv
from envs.env_ball_box import BallBoxEnv
from envs.env_sigmoid import SigmoidEnv
from envs.env_sigmoid2d import Sigmoid2DEnv
from envs.env_reacher import ReacherEnv
from envs.env_arm import ArmEnv
from filters.bayes_filter_viz import visualize_latent_space3D, visualize_latent_space2D, visualize_latent_space1D, \
                                visualize_latent_spaceND, plot_trajectory

Record = namedtuple('Record', ['ep', 'l_r', 'l_nll', 'l_k', 'c'])


# Plot predictions against true time evolution
def train(replay_memory, bayes_filter):
    records = [None] * args.num_epochs

    for i in range(args.num_epochs):
        replay_memory.reset_batchptr_train()

        bayes_filter.prepare_update()
        t = time.time()
        for b in range(replay_memory.n_batches_train):
            batch_dict = replay_memory.next_batch_train()
            x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])
            gu = i * replay_memory.n_batches_train * replay_memory.batch_size + b * replay_memory.batch_size

            L_NLL, L_KLD, L_rec = bayes_filter.update(x, u, gu)

        if i % 10 == 0:
            bayes_filter.prepare_eval()
            with torch.no_grad():
                bayes_filter.save_params()
                if bayes_filter.z_dim == 1:
                    visualize_latent_space1D(bayes_filter, replay_memory)
                elif bayes_filter.z_dim == 2:
                    visualize_latent_space2D(bayes_filter, replay_memory, i)
                elif bayes_filter.z_dim == 3:
                    visualize_latent_space3D(bayes_filter, replay_memory, i)
                else:
                    visualize_latent_spaceND(bayes_filter, replay_memory, i)

                plot_trajectory(bayes_filter, replay_memory, i)
            bayes_filter.prepare_update()
        print(f'{(time.time() - t):.2f}')
        records[i] = Record(i, L_rec, L_NLL, L_KLD, bayes_filter.c)
        print(f'ep = {i},  L_NLL = {L_NLL:.2f} L_rec = {L_rec:.2f} L_KLD = {L_KLD:.4f}')

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(14, 3))
    ax[0].scatter([r.ep for r in records], [r.l_r for r in records])
    ax[0].set_ylabel('MSE')
    ax[1].scatter([r.ep for r in records], [r.l_nll for r in records])
    ax[1].set_ylabel('NLL')
    ax[2].scatter([r.ep for r in records], [r.l_k for r in records])
    ax[2].set_ylabel('KL')
    ax[3].scatter([r.ep for r in records], [r.c for r in records])
    ax[3].set_ylabel('$c_i$')
    for a in ax: a.set_xlabel('Epochs')
    fig.tight_layout()
    plt.savefig('img/loss.png')



parser = argparse.ArgumentParser()
parser.add_argument('--val_frac', type=float, default=0.1,
                    help='fraction of data to be witheld in validation set')
parser.add_argument('--seq_length', type=int, default=32, help='sequence length for training')
parser.add_argument('--batch_size', type=int, default=128, help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=401, help='number of epochs')
parser.add_argument('--n_trials', type=int, default=2000,
                    help='number of data sequences to collect in each episode')
parser.add_argument('--trial_len', type=int, default=32, help='number of steps in each trial')
parser.add_argument('--n_subseq', type=int, default=4,
                    help='number of subsequences to divide each sequence into')
parser.add_argument('--env', type=int, default=5,
                    help='0=pendulum, 1=ball in box, 2=sigmoid, 3=sigmoid2d, 4=reacher, 5=arm')
parser.add_argument('--filter_type', type=int, default=1,
                    help='0=bayes filter, 1=bayes filter fully connected, 3=filter simple')
parser.add_argument('--z_dim', type=int, default=11)
args = parser.parse_args()


if __name__ == '__main__':

    if not os.path.exists('img'):
        os.makedirs('img')
    if not os.path.exists('param'):
        os.makedirs('param')

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
    elif args.env == 4:
        env = ReacherEnv()
    elif args.env == 5:
        env = ArmEnv()
    env.seed(0)

    controller = Controller(env)
    replay_memory = ReplayMemory(args, controller=controller, env=env)
    if args.filter_type == 0:
        bayes_filter = BayesFilter.init_from_replay_memory(replay_memory, u_max=env.u_max, z_dim=args.z_dim)
    elif args.filter_type == 1:
        bayes_filter = BayesFilterFullyConnected.init_from_replay_memory(replay_memory, u_max=env.u_max, z_dim=args.z_dim)
    elif args.filter_type == 2:
        bayes_filter = SimpleFilter.init_from_replay_memory(replay_memory, u_max=env.u_max, z_dim=args.z_dim)

    assert bayes_filter.z_dim <= replay_memory.state_dim


    train(replay_memory, bayes_filter)