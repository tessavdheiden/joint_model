import os
import numpy as np
import torch
import time
import pandas as pd

from envs import *
from viz import *
from empowerment.empowerment import Empowerment
from filters.bayes_filter import BayesFilter
from filters.bayes_filter_fully_connected import BayesFilterFullyConnected
from filters.simple_filter import SimpleFilter
from memory.replay_memory import ReplayMemory


def train_empowerment(env, empowerment, replay_memory, args, bayes_filter=None):
    rp = RecordPlot()
    lp = LandscapePlot()
    sp = SelectionPlot()
    cast = lambda x: x.detach().numpy()
    empowerment.prepare_update()

    for i in range(args.num_epochs):
        replay_memory.reset_batchptr_train()
        t0 = time.time()
        E = np.zeros((replay_memory.n_batches_train, args.batch_size))

        for b in range(replay_memory.n_batches_train):
            batch_dict = replay_memory.next_batch_train()
            x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])
            x = x[:, 0] # first element in sequence

            if args.use_filter:
                z = bayes_filter.propagate_solution(x, u)[2]
            else:
                z = x
            # z = z.reshape(-1, z.shape[2])   # merge seq length and batch_size
            E[b, :] = empowerment.update(z)

        if i % 10 == 0:
            with torch.no_grad():
                empowerment.prepare_eval()
                x = replay_memory.x
                # x = torch.from_numpy(x.reshape(-1, x.shape[2]))
                x = torch.from_numpy(x[:, 0])
                e = empowerment(x)

                if args.state_plot:
                    x = env.get_state_from_obs(x)
                    lp.add(xy=pd.DataFrame(x, index=np.arange(len(x)), columns=env.state_names),
                           z=cast(e).reshape(-1, 1))
                else:
                    lp.add(xy=pd.DataFrame(x, index=np.arange(len(x)), columns=env.obs_names),
                           z=cast(e).reshape(-1, 1))
                sp.add(x=cast(e).reshape(-1, 1))
                lp.plot(f'img/landscape_seed={args.seed}_ep={i}')
                #lp.plot(f'img/landscape_seed_clipped={args.seed}_ep={i}', mi=0, ma=3)
                #sp.plot(f'img/hist_seed={args.seed}_ep={i}')
                empowerment.save_params()
            empowerment.prepare_update()

        rp.add(i, E.mean())
        print(f'ep = {i}, empowerment = {E.mean():.4f} time = {time.time()-t0:.2f}')

    env.close()
    rp.plot('img/empowerment_training_curve.png')



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='fraction of data to be witheld in validation set')
    parser.add_argument('--seq_length', type=int, default=2, help='sequence length for training')
    parser.add_argument('--batch_size', type=int, default=32, help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=51, help='number of epochs')
    parser.add_argument('--n_trials', type=int, default=5000,
                        help='number of data sequences to collect in each episode')
    parser.add_argument('--trial_len', type=int, default=2, help='number of steps in each trial')
    parser.add_argument('--n_subseq', type=int, default=1,
                        help='number of subsequences to divide each sequence into')
    parser.add_argument('--env', type=str, default='controlled_reacher',
                        help='pendulum, ball_in_box, ball_in_box_force, tanh2d, arm, reacher, pendulum')
    parser.add_argument('--filter_type', type=int, default=1,
                        help='0=bayes filter, 1=bayes filter fully connected')
    parser.add_argument('--use_filter', type=int, default=0, help='0=env, 1=filter')
    parser.add_argument('--n_step', type=int, default=1, help='empowerment calculated over n actions')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--load_data', type=bool, default=0, help='generate data')
    parser.add_argument('--state_plot', type=bool, default=0, help='pendulum needs plot states')
    args = parser.parse_args()

    if not os.path.exists('../param'):
        print('bayes filter not trained')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.env == 'pendulum':
        env = PendulumEnv()
    elif args.env == 'ball_in_box':
        env = BallBoxEnv()
    elif args.env == 'ball_in_box_force':
        env = BallBoxForceEnv()
    elif args.env == 'tanh2d':
        env = Tanh2DEnv()
    elif args.env == 'reacher':
        env = ReacherEnv()
    elif args.env == 'arm':
        env = ArmEnv()
    elif args.env == 'controlled_reacher':
        env = ReacherControlledEnv()
    elif args.env == 'controlled_arm':
        env = ControlledArmEnv()
    env.seed(args.seed)

    replay_memory = ReplayMemory(args, env)

    # assert bayes_filter.T == replay_memory.seq_length

    empowerment = Empowerment(env, args.n_step)
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