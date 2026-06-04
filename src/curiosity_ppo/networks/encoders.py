"""CNN 编码器: Nature DQN (Atari/RND) 与 Crafter (ICM)."""
import torch
import torch.nn as nn


class NatureDQNEncoder(nn.Module):
    """Nature DQN CNN, 用于 Atari + RND.

    输入: (N, in_channels, 84, 84) -> 输出: (N, out_dim), 默认 out_dim=512.
    结构与 DeepMind 2015 Nature 论文中的卷积前端一致.
    """

    def __init__(self, in_channels: int = 4, out_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # 推断展平后的维度, 避免硬编码
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            flat_dim = self.conv(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class CrafterEncoder(nn.Module):
    """4 层 CNN, 用于 Crafter ICM 编码器, 输出 288 维.

    输入: (N, in_channels, 64, 64) -> 输出: (N, out_dim), 默认 out_dim=288.
    每层 Conv3x3 + ReLU + MaxPool2d, 4 次下采样后展平 -> 线性投影.
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 288):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            flat_dim = self.conv(dummy).shape[1]
        self.fc = nn.Linear(flat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))
