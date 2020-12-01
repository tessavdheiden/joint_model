import argparse
from collections import namedtuple
import torch

from policy import Policy
from envs.env_pendulum import PendulumEnv
from replay_memory import ReplayMemory
from controller import Controller
from filters.bayes_filter_fully_connected import BayesFilterFullyConnected

parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with PPO')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', default=0, help='render the environment')
parser.add_argument('--val_frac', type=float, default=0.1,
                    help='fraction of data to be witheld in validation set')
parser.add_argument('--seq_length', type=int, default=32, help='sequence length for training')
parser.add_argument('--batch_size', type=int, default=128, help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=201, help='number of epochs')
parser.add_argument('--n_trials', type=int, default=100,
                    help='number of data sequences to collect in each episode')
parser.add_argument('--trial_len', type=int, default=32, help='number of steps in each trial')
parser.add_argument('--n_subseq', type=int, default=4,
                    help='number of subsequences to divide each sequence into')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)

TrainingRecord = namedtuple('TrainingRecord', ['ep', 'reward'])
Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])


def main():
    env = PendulumEnv()
    env.seed(args.seed)

    agent = Policy()
    controller = Controller(env)
    replay_memory = ReplayMemory(args, controller=controller, env=env)

    bayes_filter = BayesFilterFullyConnected.init_from_save()
    running_reward = -1000
    for i_ep in range(1000):
        # batch_dict = replay_memory.next_batch_train()
        # x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])
        # _, _, z, _ = bayes_filter.propagate_solution(x=x, u=u)
        # z, x, u = z[:, 0], x[:, 0], u[:, 0]
        x = env.reset()
        score = 0
        for t in range(200):
            # reward = env.cost(x, u)
            # x_, _ = bayes_filter._sample_x_(z)
            # u, u_prob = agent.select_action(z)
            # z_, _ = bayes_filter(u, z)
            action, action_log_prob = agent.select_action(x)
            x_, reward, done, _ = env.step([action])
            if agent.store(Transition(x, action, action_log_prob, reward, x_)):
                agent.update()

            running_reward = running_reward * 0.9 + score * 0.1
            x = x_
            score += reward
            # z = z_
            # x = x_
        if i_ep % args.log_interval == 0:
            print('Ep {}\tMoving average score: {:.2f}\t'.format(i_ep, running_reward))

        if i_ep >= replay_memory.n_batches_train - 1:
            replay_memory.reset_batchptr_train()

        if running_reward > -200:
            agent.save_param()
            break

    state = env.reset()
    for t in range(200):
        action, action_log_prob = agent.select_action(state)
        state_, reward, done, _ = env.step([action])
        env.render()
        state = state_


if __name__ == '__main__':
    main()