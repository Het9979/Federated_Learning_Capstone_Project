import torch.nn as nn
import config


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.STATE_DIM, config.HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_1, config.HIDDEN_2),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_2, config.ACTION_DIM),
        )
        self.to(config.DEVICE)

    def forward(self, x):
        return self.net(x)


class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()

        # Shared Feature Extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(config.STATE_DIM, config.HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),  # <--- This caused your error before
            nn.Linear(config.HIDDEN_1, config.HIDDEN_2),
            nn.ReLU(),
        )

        # Value Stream: Estimates V(s)
        self.value_stream = nn.Sequential(nn.Linear(config.HIDDEN_2, 1))

        # Advantage Stream: Estimates A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.HIDDEN_2, config.ACTION_DIM)
        )

        self.to(config.DEVICE)

    def forward(self, x):
        features = self.feature_layer(x)

        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine V and A to get Q(s, a)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values
