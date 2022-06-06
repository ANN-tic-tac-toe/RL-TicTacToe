import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, device):
        super(DQN, self).__init__()
        self.device = device
        self.hidden_1 = nn.Linear(18, 128)
        self.hidden_2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, 9)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        return self.output(x)
