import argparse
import pickle
from collections import namedtuple
import matplotlib.pyplot as plt
import gym
import torch

from policy import Policy

parser = argparse.ArgumentParser(description='Solve the Pendulum-v0 with PPO')
parser.add_argument(
    '--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', default=0, help='render the environment')
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
    env = gym.make('Pendulum-v0')
    env.seed(args.seed)

    agent = Policy()

    training_records = []
    running_reward = -1000

    state = env.reset()
    for i_ep in range(1000):
        score = 0
        state = env.reset()

        for t in range(200):
            action, action_log_prob = agent.select_action(state)
            state_, reward, done, _ = env.step([action])

            if args.render:
                env.render()
            if agent.store(Transition(state, action, action_log_prob, (reward + 8) / 8, state_)):
                agent.update()
            score += reward
            state = state_

        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward))

        if i_ep % args.log_interval == 0:
            print('Ep {}\tMoving average score: {:.2f}\t'.format(i_ep, running_reward))
        if running_reward > -200:
            print("Solved! Moving average score is now {}!".format(running_reward))
            env.close()
            agent.save_param()
            break

    state = env.reset()

    for t in range(200):
        action, action_log_prob = agent.select_action(state)
        state_, reward, done, _ = env.step([action])
        env.render()
        state = state_

    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title('PPO')
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.savefig("img/ppo.png")
    plt.show()


if __name__ == '__main__':
    main()