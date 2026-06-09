"""RND 内在奖励模块 + alpha_t 长期调制.

- compute_reward: RND MSE 预测误差 -> 用 RunningMeanStd 归一化 -> 返回 float.
- compute_alpha: RND 误差归一化后经 sigmoid 映射到 [1, L], 作为 NGU 的长期调制系数.
- _normalize_obs: 可选的观测归一化 (用 RunningMeanStd 在线统计).
"""
import torch
import numpy as np

from curiosity_ppo.networks.rnd import RNDNet
from curiosity_ppo.curiosity.reward_norm import RunningMeanStd, RewardNormalizer


class RNDCuriosity:
    """RND 内在奖励 + alpha_t 长期调制.

    Args:
        rnd_net: 已构建的 RNDNet 实例.
        obs_normalizer: 可选的观测 RunningMeanStd; None 表示不做观测归一化.
        reward_normalizer: 可选的 RewardNormalizer (保留接口, 内部用 _error_rms 归一化误差).
        L: alpha_t 上界, alpha = 1 + (L-1) * sigmoid(normalized_error).
        device: 计算设备.
    """

    def __init__(self, rnd_net: RNDNet, obs_normalizer=None, reward_normalizer=None,
                 L: int = 5, device: str = 'cpu'):
        self.rnd_net = rnd_net.to(device)
        self.device = device
        self.L = L
        self.obs_normalizer = obs_normalizer  # RunningMeanStd for observations
        self.reward_normalizer = reward_normalizer or RewardNormalizer()
        self._error_rms = RunningMeanStd(shape=())

    def _normalize_obs(self, obs):
        """用 obs_normalizer 在线归一化观测; 未提供时直接返回原值."""
        if self.obs_normalizer:
            self.obs_normalizer.update(obs.cpu().numpy())
            obs = (obs - torch.tensor(self.obs_normalizer.mean, device=obs.device, dtype=obs.dtype)) / \
                  (torch.tensor(self.obs_normalizer.std, device=obs.device, dtype=obs.dtype) + 1e-8)
        return obs

    def compute_reward(self, obs) -> float:
        """RND MSE 预测误差 -> 归一化后返回 float.

        Args:
            obs: (N, C, H, W) 观测.

        Returns:
            float 归一化内在奖励 (批次均值).
        """
        obs = self._normalize_obs(obs)
        with torch.no_grad():
            reward = self.rnd_net.intrinsic_reward(obs)
        reward_np = reward.cpu().numpy()
        self._error_rms.update(reward_np)
        normalized = reward_np / (self._error_rms.std + 1e-8)
        return float(normalized.mean())

    def compute_alpha(self, obs) -> float:
        """RND error -> 归一化 -> 映射到 [1, L].

        alpha = 1 + (L - 1) * sigmoid(normalized_error_mean), 用于 NGU 长期调制.

        Args:
            obs: (N, C, H, W) 观测.

        Returns:
            float alpha_t, 范围 (1, L).
        """
        obs = self._normalize_obs(obs)
        with torch.no_grad():
            error = self.rnd_net.intrinsic_reward(obs)
        error_np = error.cpu().numpy()
        self._error_rms.update(error_np)
        normalized = error_np / (self._error_rms.std + 1e-8)
        # alpha = 1 + (L - 1) * sigmoid(normalized)
        alpha = 1.0 + (self.L - 1.0) * (1.0 / (1.0 + np.exp(-normalized.mean())))
        return float(alpha)
