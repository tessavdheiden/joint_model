import torch
import torch.nn as nn
import torch.nn.functional as F
device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out).to(device))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out).to(device))
        if activation == 'relu':
            layers.append(nn.ReLU().to(device))
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU().to(device))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout).to(device))
    return nn.Sequential(*layers).to(device)


class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.sigmoid,
                 constrain_out=True, norm_in=True, discrete_action=False, recurrent=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        self.emb = lambda x: x

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x

        self.fc1 = nn.Linear(input_dim, hidden_dim)

        if recurrent:
            self.fc2 = RecurrentUnit(hidden_dim, hidden_dim)
        else:
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        X = self.emb(X)
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out

class RecurrentUnit(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(RecurrentUnit, self).__init__()
        self.lstm = nn.LSTM(input_dim, out_dim)
        self.h_0 = nn.Parameter(torch.randn(input_dim))
        self.c_0 = nn.Parameter(torch.randn(input_dim))

    def init_state(self, batch_size):
        h_0 = self.h_0.repeat(1, batch_size, 1)
        c_0 = self.c_0.repeat(1, batch_size, 1)
        return (h_0, c_0)

    def forward(self, x):
        batch_size, feat_size = x.shape
        state = self.init_state(batch_size)
        x, _ = self.lstm(x.unsqueeze(0), state)
        return x.view(batch_size, feat_size)

