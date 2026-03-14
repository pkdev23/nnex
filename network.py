import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallNN(nn.Module):
    """
    Bewusst klein gehalten – damit die Agenten
    das Innenleben klar messen können.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        self.act1 = F.relu(self.fc1(x))   # gespeichert für Agenten
        self.act2 = F.relu(self.fc2(self.act1))
        return self.fc3(self.act2)