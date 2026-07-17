"""Intrinsic Curiosity Module (ICM).

- encoder: 可配置 (CrafterEncoder 或 NatureDQNEncoder), 将观测编码为特征 phi.
- inverse_model: 由 (phi_t, phi_next) 预测动作 a, 用 cross_entropy 监督.
- forward_model: 由 (phi_t, a_onehot) 预测 phi_next, 用 MSE 监督.
- 前向预测误差用作内在奖励; phi_t 作为情景记忆的 controllable embedding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import CrafterEncoder, NatureDQNEncoder


class ICMNet(nn.Module):
    """ICM 好奇心模块.

    Args:
        in_channels: 观测通道数.
        action_dim: 离散动作数.
        feature_dim: encoder 输出特征维度 (默认 288).
        hidden_dim: 逆/前向模型的隐藏层维度 (默认 256).
        encoder_cls: 编码器类, CrafterEncoder (64×64) 或 NatureDQNEncoder (84×84).
    """

    def __init__(
        self,
        in_channels: int = 3,
        action_dim: int = 17,
        feature_dim: int = 288,
        hidden_dim: int = 256,
        encoder_cls=None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.feature_dim = feature_dim

        if encoder_cls is None:
            encoder_cls = CrafterEncoder
        self.encoder = encoder_cls(in_channels=in_channels, out_dim=feature_dim)

        # 逆模型: cat(phi_t, phi_next) -> action_logits
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # 前向模型: cat(phi_t, a_onehot) -> phi_next_pred
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, s_t: torch.Tensor, a: torch.Tensor, s_next: torch.Tensor,
                detach_phi_t_for_forward: bool = False):
        """前向计算.

        Args:
            s_t:     (N, C, H, W) 当前观测.
            a:       (N,) long, 动作索引.
            s_next:  (N, C, H, W) 下一观测.
            detach_phi_t_for_forward: 若 True, 前向模型用 phi_t.detach(),
                阻止 forward_loss 梯度回传到共享编码器 (防编码器发散).

        Returns:
            inverse_loss: 标量, 逆模型交叉熵损失 (可反传到 encoder).
            forward_loss: 标量, 前向模型 MSE 损失 (按 detach 策略反传).
            phi_t:        (N, feature_dim), 当前观测特征, 用于情景记忆嵌入.
        """
        phi_t = self.encoder(s_t)
        phi_next = self.encoder(s_next)

        # 逆模型: 由相邻两帧特征预测动作
        a_pred = self.inverse_model(torch.cat([phi_t, phi_next], dim=-1))
        inverse_loss = F.cross_entropy(a_pred, a)

        # 前向模型: 由当前特征 + 动作预测下一特征
        a_onehot = F.one_hot(a, num_classes=self.action_dim).to(phi_t.dtype)
        # 关键修复: detach phi_t 阻止 forward_loss → encoder 反传
        # NGU 论文中 forward_loss 仅用于计算内在奖励信号,
        # 不应驱动编码器学习; 编码器只由 inverse_loss 驱动 (可控性过滤)
        phi_t_input = phi_t.detach() if detach_phi_t_for_forward else phi_t
        phi_next_pred = self.forward_model(torch.cat([phi_t_input, a_onehot], dim=-1))
        # detach phi_next: 前向损失不更新 encoder 去拟合 phi_next
        # 不再乘 feature_dim: mse_loss 已对特征维度取均值
        forward_loss = F.mse_loss(phi_next_pred, phi_next.detach())

        return inverse_loss, forward_loss, phi_t

    def get_embedding(self, obs: torch.Tensor) -> torch.Tensor:
        """返回 encoder(obs), 作为情景记忆的 controllable embedding."""
        return self.encoder(obs)
