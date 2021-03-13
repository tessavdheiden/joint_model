import numpy as np
from collections import namedtuple
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


Transition = namedtuple('Transition', ['s', 'a', 'r', 's_'])


from envs import *
from viz import *

np.random.seed(1)
torch.manual_seed(1)

MAX_EPISODES = 200
MAX_EP_STEPS = 100
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = .9
REPLACE_ITER_A = 1100
REPLACE_ITER_C = 1000

MEMORY_CAPACITY = 5000
BATCH_SIZE = 16
VAR_MIN = 0.1

env = BallBoxForceEnv()
STATE_DIM = env.observation_space.shape[0]
H_DIM = 200
ACTION_DIM = env.action_dim
ACTION_SCALE = env.u_high
ACTION_BOUND = env.u_high
RENDER = True
LOAD = True
EMPOWERMENT = False


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(STATE_DIM, H_DIM),
                                nn.ReLU6(),
                                nn.Linear(H_DIM, H_DIM),
                                nn.ReLU6(),
                                nn.Linear(H_DIM, H_DIM))
        self.mu_head = nn.Linear(H_DIM, ACTION_DIM)

        self.fc.apply(init_weights)
        self.mu_head.apply(init_weights)

    def forward(self, s):
        x = F.relu(self.fc(s))
        u = ACTION_SCALE[0] * F.tanh(self.mu_head(x))
        return u


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Sequential(nn.Linear(STATE_DIM + ACTION_DIM, H_DIM),
                                nn.ReLU6(),
                                nn.Linear(H_DIM, H_DIM),
                                nn.ReLU6(),
                                nn.Linear(H_DIM, H_DIM))
        self.v_head = nn.Linear(H_DIM, 1)
        self.fc.apply(init_weights)
        self.v_head.apply(init_weights)

    def forward(self, sa):
        x = F.relu(self.fc(sa))
        state_value = self.v_head(x)
        return state_value


class Memory():

    data_pointer = 0
    isfull = False

    def __init__(self, capacity):
        self.memory = np.empty(capacity, dtype=object)
        self.capacity = capacity

    def update(self, transition):
        self.memory[self.data_pointer] = transition
        self.data_pointer += 1
        if self.data_pointer == self.capacity:
            self.data_pointer = 0
            self.isfull = True

    def sample(self):
        return np.random.choice(self.memory, BATCH_SIZE)


class Agent():

    max_grad_norm = 0.5

    def __init__(self):
        self.training_step = 0
        self.var = 1.
        self.eval_cnet, self.target_cnet = CriticNet().float(), CriticNet().float()
        self.eval_anet, self.target_anet = ActorNet().float(), ActorNet().float()
        self.memory = Memory(MEMORY_CAPACITY)
        self.optimizer_c = optim.RMSprop(self.eval_cnet.parameters(), lr=LR_C)
        self.optimizer_a = optim.RMSprop(self.eval_anet.parameters(), lr=LR_A)
        self.cast = lambda x: x

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = self.eval_anet(state)
        return action

    def store_transition(self, transition):
        self.memory.update(transition)

    def update(self):
        self.training_step += 1

        transitions = self.memory.sample()
        s = torch.tensor([t.s for t in transitions], dtype=torch.float)
        a = torch.tensor([t.a for t in transitions], dtype=torch.float)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float)

        with torch.no_grad():
            q_target = r + GAMMA * self.target_cnet(torch.cat((s_, self.target_anet(s_)), dim=1))
        q_eval = self.eval_cnet(torch.cat((s, a), dim=1))

        # update critic net
        self.optimizer_c.zero_grad()
        c_loss = F.smooth_l1_loss(q_eval, q_target)
        c_loss.backward()
        #nn.utils.clip_grad_norm_(self.eval_cnet.parameters(), self.max_grad_norm)
        self.optimizer_c.step()

        # update actor net
        self.optimizer_a.zero_grad()
        a_loss = -self.eval_cnet(torch.cat((s, self.eval_anet(s)), dim=1)).mean()
        a_loss.backward()
        #nn.utils.clip_grad_norm_(self.eval_anet.parameters(), self.max_grad_norm)
        self.optimizer_a.step()

        if self.training_step % REPLACE_ITER_C == 0:
            self.target_cnet.load_state_dict(self.eval_cnet.state_dict())
        if self.training_step % REPLACE_ITER_A == 0:
            self.target_anet.load_state_dict(self.eval_anet.state_dict())

        self.var = max(self.var * 0.999, 0.01)

        return q_eval.mean().item()

    @property
    def networks(self):
        return [self.eval_anet, self.target_anet, self.eval_cnet, self.target_cnet]

    def prepare_update(self):
        if DEVICE == 'cuda':
            self.cast = lambda x: x.cuda()
        else:
            self.cast = lambda x: x.cpu()

        for network in self.networks:
            network = self.cast(network)
            network.train()

    def prepare_eval(self):
        self.cast = lambda x: x.cpu()
        for network in self.networks:
            network = self.cast(network)
            network.eval()

    def save_param(self, path='param/ddpg.pkl'):
        save_dict = {'networks': [network.state_dict() for network in self.networks]}
        torch.save(save_dict, path)

    def load_param(self, path='param/ddpg.pkl'):
        save_dict = torch.load(path)
        for network, params in zip(self.networks, save_dict['networks']):
            network.load_state_dict(params)

agent = Agent()


def train():
    if not os.path.exists('../param'):
        print('param dir doesnt exit')

    var = 2.  # control exploration
    rewards = []
    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0

        agent.prepare_eval()
        for t in range(MAX_EP_STEPS):
            # while True:
            if RENDER:
                env.render()

            # Added exploration noise
            a = agent.select_action(s)
            a = a.detach().numpy().squeeze(0)
            a = np.clip(np.random.normal(a, var), -ACTION_BOUND, ACTION_BOUND)  # add randomness to action selection for exploration
            s_, r, done, _ = env.step(a)

            agent.store_transition(Transition(s, a, r, s_))

            if agent.memory.isfull:
                agent.prepare_update()
                var = max([var * .9999, VAR_MIN])  # decay the action randomness
                q = agent.update()
                agent.prepare_eval()

            s = s_
            ep_reward += r

            if t == MAX_EP_STEPS - 1 or done:
                # if done:
                result = '| done' if done else '| ----'
                print('Ep:', ep,
                      result,
                      '| R: %i' % int(ep_reward),
                      '| Explore: %.2f' % var,
                      )
                break
        rewards.append(ep_reward)

    agent.save_param()
    plt.scatter(np.arange(len(rewards)), rewards)
    plt.savefig('img/reward_policy.png')


def eval():
    agent.load_param()
    s = env.reset()

    b = BenchmarkPlot()
    v = Video()

#    data = env.get_benchmark_data()

    for t in range(MAX_EP_STEPS):
        if RENDER:
            v.add(env.render(mode='rgb_array'))
        a = agent.select_action(s).detach().numpy().squeeze(0)

        s_, r, done, _ = env.step(a)
        # data = env.get_benchmark_data(data)
        s = s_

    # env.do_benchmark(data)
    # b.add(data)
    # b.plot("img/derivatives.png")
    env.close()

    v.save('img/video.gif')

if __name__ == '__main__':
    if LOAD:
        eval()
    else:
        train()

