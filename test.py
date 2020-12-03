import argparse
import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

from envs.env_pendulum import PendulumEnv
from controller import Controller
from policy import Policy


Transition = namedtuple('Transition', ['s', 'a', 'a_prob',  's_', 'r'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_frac', type=float, default=0.1, help='fraction of data to be witheld in validation set')
    parser.add_argument('--seq_length',         type=int,   default= 32,        help='sequence length for training')
    parser.add_argument('--batch_size',         type=int,   default= 1,         help='minibatch size')

    parser.add_argument('--num_epochs',         type=int,   default= 2,        help='number of epochs')

    parser.add_argument('--n_trials',           type=int,   default= 100,       help='number of data sequences to collect in each episode')
    parser.add_argument('--trial_len',          type=int,   default= 256,       help='number of steps in each trial')
    parser.add_argument('--n_subseq',           type=int,   default= 8,         help='number of subsequences to divide each sequence into')
    args = parser.parse_args()

    env = PendulumEnv()
    controller = Controller()
    policy = Policy()

    desired_state = np.array([1, 0, 0]) # theta = 0, pendulum upright

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20, 5))

    for i_episode in range(args.num_epochs):
        state = env.reset()
        integral = 0
        prev_error = 0
        sequence = [None] * args.trial_len
        for t in range(args.trial_len):
            env.render()
            error = state - desired_state

            integral += error
            derivative = error - prev_error
            prev_error = error
            control_input = np.array([error, integral, derivative])
            gain_choice, log_prob = policy.sample(state)
            if i_episode % 2 == 1:
                controller.update(gain_choice)

            pid = controller.compute(control_input)

            state_, reward, done, info = env.step((pid, ))

            #reward = np.sum(error)
            sequence[t] = Transition(state, gain_choice, log_prob, state_, reward)
            state = state_

        policy.update(sequence)

        t = [i for i in range(args.trial_len)]
        ctheta = [t.s[0] for t in sequence]
        stheta = [t.s[1] for t in sequence]
        dtheta = [t.s[2] for t in sequence]
        ddtheta = [dtheta[i+1] - dtheta[i] for i in range(args.trial_len-1)]
        dddtheta = [ddtheta[i + 1] - ddtheta[i] for i in range(args.trial_len - 2)]

        ax[0].plot(t[:], ctheta)
        ax[1].plot(t[:], stheta)
        ax[2].plot(t[:], dtheta)
        ax[3].plot(t[:-1], ddtheta)
        ax[4].plot(t[:-2], dddtheta)

    policy.save_params()

    ax[0].set_ylabel('$\\cos(\\theta)$')
    ax[1].set_ylabel('$\\sin(\\theta)$')
    ax[2].set_ylabel('$\\dot{\\theta}$')
    ax[3].set_ylabel('$\\ddot{\\theta}$')
    ax[4].set_ylabel('$\\dddot{\\theta}$')
    for i in range(5):
        ax[i].set_xlabel('$t$')
    plt.savefig(f'pid_control.png')

    env.close()


if __name__ == '__main__':
    theta = np.linspace(-np.pi, np.pi, 16)
    x = np.array([np.cos(theta), np.sin(theta)])
    t = np.arctan2(x[1, :], x[0, :])
    print(x.shape)
    print(t.shape)
    plt.plot(np.arange(16), t)
    plt.plot(np.arange(16), theta)
    plt.savefig('tmp.png')
    #main()