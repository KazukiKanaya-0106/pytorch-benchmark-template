import torch.nn as nn


class MLPLayer7(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)
