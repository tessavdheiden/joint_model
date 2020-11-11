import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.v = nn.Sequential(nn.Linear(3, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Linear(3, 100)
        self.mu_head = nn.Linear(100, 3)
        self.sigma_head = nn.Linear(100, 3)

    def forward(self, x):
        v = self.v(x)
        x = F.relu(self.fc(x))
        mu = torch.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        return (mu, sigma), v


class Policy(object):
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    batch_size = 128

    def __init__(self):
        super(Policy, self).__init__()
        self.net = Net()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def sample(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            (mu, sigma) = self.net(state)[0]
        dist = Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)
        action = action.squeeze().cpu().numpy()
        return action, log_prob.numpy()

    def update(self, transitions, gamma=.9):
        s = torch.tensor([t.s for t in transitions], dtype=torch.float)
        a = torch.tensor([t.a for t in transitions], dtype=torch.float).view(-1, 3)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float)

        old_action_log_probs = torch.tensor([t.a_prob for t in transitions], dtype=torch.float).view(-1, 1)

        with torch.no_grad():
            target_v = r + gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(len(transitions))), self.batch_size, False):

                (mu, sigma) = self.net(s[index])[0]
                dist = Normal(mu, sigma)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_action_log_probs[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def save_params(self, path='net_params.pkl'):
        torch.save(self.net.state_dict(), path)








