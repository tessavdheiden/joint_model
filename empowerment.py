import torch


class Empowerment(object):
    def __init__(self, env, controller, transition_network):
        super(Empowerment).__init__()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = controller.action_space.shape[0]
        self.transition = transition_network

    def compute(self, state):
        return torch.rand(len(state))

    def update(self, transitions):
        s = torch.tensor([t.s for t in transitions], dtype=torch.float)
        a = torch.tensor([t.a for t in transitions], dtype=torch.float)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float)

        return torch.rand(len(transitions))
