"""Random Network Distillation (RND) 内在奖励网络.

- target: 固定的 NatureDQNEncoder, 不参与训练 (requires_grad=False, eval 模式).
- predictor: NatureDQNEncoder + 3 层 FC, 学习逼近 target 的输出.
- 内在奖励 = (predictor(obs) - target(obs)) ** 2 的特征维均值.
"""
import torch
import torch.nn as nn

from .encoders import NatureDQNEncoder, CrafterEncoder


class RNDNet(nn.Module):
    """RND 预测误差内在奖励.

    Args:
        in_channels: 观测通道数 (Atari 通常为 4 帧堆叠).
        output_dim: target / predictor 输出维度 (默认 512).
        encoder_cls: 编码器类 (NatureDQNEncoder 用于 84×84, CrafterEncoder 用于 64×64).
    """

    def __init__(self, in_channels: int = 4, output_dim: int = 512, encoder_cls=None):
        super().__init__()
        self.output_dim = output_dim

        if encoder_cls is None:
            encoder_cls = NatureDQNEncoder

        # target 网络: 固定随机初始化, 不更新
        self.target = encoder_cls(in_channels=in_channels, out_dim=output_dim)
        for p in self.target.parameters():
            p.requires_grad = False
        self.target.eval()

        # predictor 网络: encoder + FC(512) -> ReLU -> FC(512) -> ReLU -> FC(512)
        self.predictor = nn.Sequential(
            encoder_cls(in_channels=in_channels, out_dim=output_dim),
            nn.Linear(output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, obs: torch.Tensor):
        """返回 (pred_out, target_out).

        - pred_out: (N, output_dim), 带梯度.
        - target_out: (N, output_dim), 在 no_grad 下计算, 不带梯度.
        """
        with torch.no_grad():
            target_out = self.target(obs)
        pred_out = self.predictor(obs)
        return pred_out, target_out

    def intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """计算内在奖励 = ((pred - target) ** 2).mean(dim=-1), 返回 (N,)."""
        pred_out, target_out = self.forward(obs)
        return ((pred_out - target_out) ** 2).mean(dim=-1)

    def train(self, mode: bool = True):
        """重写 train, 保持 target 始终处于 eval 模式 (无 dropout/bn 行为差异)."""
        super().train(mode)
        self.target.eval()
        return self
