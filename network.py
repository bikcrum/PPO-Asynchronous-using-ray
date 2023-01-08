import torch.nn.functional as F
from torch import nn


class MyNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MyNetwork, self).__init__()

        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        relu1 = F.relu(self.fc1(obs))
        relu2 = F.relu(self.fc2(relu1))
        output = self.fc3(relu2)

        return output
