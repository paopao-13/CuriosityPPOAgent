"""ICM 内在奖励模块, 包装 ICMNet.

- compute_reward: 在 no_grad 下计算 eta * forward_loss 作为内在奖励 (float).
- get_loss: 返回 (inverse_loss + forward_loss, phi_t), 用于 ICM 训练.
- get_embedding: 返回 encoder(obs), 作为情景记忆的可控性嵌入.
"""
import torch

from curiosity_ppo.networks.icm import ICMNet


class ICMCuriosity:
    """ICM 内在奖励模块, 包装 ICMNet.

    Args:
        icm_net: 已构建的 ICMNet 实例.
        eta: 内在奖励缩放系数 (内在奖励 = eta * forward_loss).
        device: 计算设备 ('cpu' / 'cuda').
    """

    def __init__(self, icm_net: ICMNet, eta: float = 0.2, device: str = 'cpu'):
        self.icm_net = icm_net.to(device)
        self.eta = eta
        self.device = device

    def compute_reward(self, s_t, a, s_next) -> float:
        """计算内在奖励 = eta * forward_loss (不构建计算图).

        Args:
            s_t:     (N, C, H, W) 当前观测.
            a:       (N,) long, 动作索引.
            s_next:  (N, C, H, W) 下一观测.

        Returns:
            Python float 内在奖励.
        """
        with torch.no_grad():
            inverse_loss, forward_loss, phi_t = self.icm_net(s_t, a, s_next)
        return (self.eta * forward_loss).item()

    def get_loss(self, s_t, a, s_next):
        """获取 ICM 训练损失 = inverse_loss + forward_loss (带梯度).

        Args:
            s_t:     (N, C, H, W) 当前观测.
            a:       (N,) long, 动作索引.
            s_next:  (N, C, H, W) 下一观测.

        Returns:
            (loss, phi_t): loss 为标量 tensor (可反传), phi_t 为 (N, feature_dim).
        """
        inverse_loss, forward_loss, phi_t = self.icm_net(s_t, a, s_next)
        return inverse_loss + forward_loss, phi_t

    def get_embedding(self, obs):
        """获取可控性嵌入, 用于情景记忆.

        Args:
            obs: (N, C, H, W) 观测.

        Returns:
            (N, feature_dim) tensor, 在 no_grad 下计算.
        """
        with torch.no_grad():
            return self.icm_net.get_embedding(obs)
