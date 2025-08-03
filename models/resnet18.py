import torch.nn as nn
from torchvision.models import resnet18


class Resnet18(nn.Module):
    def __init__(self, in_channels: int, output_dim: int, pretrained: bool):
        super().__init__()
        self.base_model = resnet18(pretrained=pretrained)
        self.base_model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.base_model.maxpool = nn.Identity()
        self.base_model.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        return self.base_model(x)
