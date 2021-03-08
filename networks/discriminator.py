import torch.nn as nn
from .networks import NetworkBase


class Discriminator(NetworkBase):
    """Discriminator."""

    def __init__(self, input_dim=7):
        super(Discriminator, self).__init__()
        self.name = 'discriminator'
        self.fc1 = nn.Linear(input_dim, 48)
        self.fc2 = nn.Linear(48, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.layer_norm1 = nn.LayerNorm(48)
        self.layer_norm2 = nn.LayerNorm(32)
        self.layer_norm3 = nn.LayerNorm(16)
        self.model = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.layer_norm1,
            self.fc2,
            nn.ReLU(),
            self.layer_norm2,
            self.fc3,
            nn.ReLU(),
            self.layer_norm3,
            self.fc4,
        )

    def forward(self, x):
        return self.model(x)
