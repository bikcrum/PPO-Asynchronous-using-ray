import torch.nn.functional as F
from torch import nn


class ActorNetworkContinuous(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ActorNetworkContinuous, self).__init__()

        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mu_layer = nn.Linear(64, out_dim)
        self.sigma_layer = nn.Linear(64, out_dim)

    def forward(self, obs):
        out = F.relu(self.fc1(obs))
        out = F.relu(self.fc2(out))
        out_mu = F.tanh(self.mu_layer(out))
        out_sigma = F.softmax(self.sigma_layer(out))

        return out_mu, out_sigma


class CriticNetworkContinuous(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CriticNetworkContinuous, self).__init__()

        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        out = F.relu(self.fc1(obs))
        out = F.relu(self.fc2(out))
        out = F.tanh(self.fc3(out))

        return out


class NetworkDiscrete(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NetworkDiscrete, self).__init__()

        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        relu1 = F.relu(self.fc1(obs))
        relu2 = F.relu(self.fc2(relu1))
        output = self.fc3(relu2)

        return output
