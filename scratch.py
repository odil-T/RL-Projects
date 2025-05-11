import torch
from torch import nn
import numpy as np

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),

            nn.Flatten(),
            nn.Linear(128, 32),

            nn.Linear(32, 9)
        )

    def forward(self, x):
        """
        Input: torch.Tensor with shape (batch, height, width, 3)
        Output: torch.Tensor with shape (batch, 9)
        """
        x = x.permute(0, 3, 1, 2)
        q_values = self.model(x)
        print(q_values.shape)
        return q_values


# model = DQN()
# image = torch.rand(2, 10, 10, 3)
# print(model.state_dict())
x = -1
print(x + 1)