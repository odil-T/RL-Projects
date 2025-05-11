"""
TODO:
1. display every X episodes
"""

from collections import deque

import torch
from torch import nn
from utils import Environment


# episode configurations
EPISODES = 20000

MIN_REPLAY_BUFFER_SIZE = 1000
MAX_REPLAY_BUFFER_SIZE = 50000

# epsilon configurations

# PyTorch device
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

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

            nn.Linear(32, env.ACTION_SPACE_SIZE)
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




class DQNAgent:
    def __init__(self):
        self.train_model = DQN().to(device)

        self.target_model = DQN().to(device)
        self.target_model.load_state_dict(self.train_model.state_dict())

        self.replay_buffer = deque(maxlen=MAX_REPLAY_BUFFER_SIZE)
        self.target_update_counter = 0

    def update_replay_buffer(self):
        pass

    def get_q_values(self):
        pass



env = Environment(BOARD_SIZE=20)

# for i in range(EPISODES):
#
#     # prepare episode materials
#
#     while True:
#         # episode training