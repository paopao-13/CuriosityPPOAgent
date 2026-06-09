"""奖励/观测归一化: Welford 在线统计 + 奖励归一化器.

- RunningMeanStd: Welford 在线算法, 支持标量与向量 shape, 可分批增量更新.
- RewardNormalizer: 内在奖励归一化 reward / (running_std + 1e-8), 并 clip 到 [-clip, clip].
"""
import numpy as np


class RunningMeanStd:
    """Welford 在线算法, 用于奖励/观测归一化.

    维护流式 mean / var / count, 支持分批增量更新且数值稳定.

    Args:
        shape: 统计量形状, 标量用 () , 逐元素用 (D,).
    """

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        """用新批次 x 更新统计量.

        Args:
            x: 标量 / (D,) 向量 / (N, D) 批次. 当 ndim > 1 时按 batch 聚合.
        """
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0) if x.ndim > 1 else x
        batch_var = x.var(axis=0) if x.ndim > 1 else np.zeros_like(x)
        batch_count = x.shape[0] if x.ndim > 1 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        """合并批次矩与流式矩 (Welford 合并公式)."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    @property
    def std(self):
        """当前标准差."""
        return np.sqrt(self.var)


class RewardNormalizer:
    """内在奖励归一化: reward / (running_std + 1e-8), 并 clip 到 [-clip, clip].

    每次调用都会更新内部 RunningMeanStd 统计量.
    """

    def __init__(self):
        self.running_ms = RunningMeanStd(shape=())
        self.clip = 10.0

    def __call__(self, reward):
        """归一化奖励.

        Args:
            reward: 标量或 array-like 奖励值.

        Returns:
            归一化并 clip 后的奖励 (与输入同形状的 numpy 值).
        """
        self.running_ms.update(reward)
        normalized = reward / (self.running_ms.std + 1e-8)
        return np.clip(normalized, -self.clip, self.clip)
