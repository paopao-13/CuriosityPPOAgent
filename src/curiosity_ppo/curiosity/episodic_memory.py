"""kNN 伪计数情景记忆, 每 episode 清空.

基于 NGU/Agent57 的 episodic memory:
- 维护一个 LRU 内存库, 存储可控性嵌入.
- compute_pseudo_count: 用 kNN 距离 + kernel 计算伪计数 N(x).
- compute_reward: 1 / sqrt(N(x) + 1e-8), 空库时返回大值 (1e3) 表示极度新颖.
"""
import numpy as np

from curiosity_ppo.utils.memory_bank import LRUMemoryBank


class EpisodicMemory:
    """kNN 伪计数情景记忆, 每 episode 清空.

    Args:
        capacity: 内存库容量.
        dim: 嵌入向量维度.
        k: kNN 近邻数.
        epsilon: kernel 平滑参数.
        L: (保留接口) NGU 中 episodic 奖励的上界调制, 此处仅存储.
    """

    def __init__(self, capacity: int = 10000, dim: int = 512, k: int = 5,
                 epsilon: float = 1e-3, L: int = 5):
        self.bank = LRUMemoryBank(capacity, dim)
        self.k = k
        self.epsilon = epsilon
        self.L = L

    def compute_pseudo_count(self, query: np.ndarray) -> float:
        """计算查询点的伪计数.

        N(x) = sum_i kernel(d_i^2 / d_m^2 + epsilon), 其中 d_m^2 为 k 近邻距离平方均值.

        Args:
            query: (dim,) 查询嵌入.

        Returns:
            float 伪计数; 空库返回 0.0.
        """
        if self.bank.size == 0:
            return 0.0
        distances = self.bank.knn_distances(query, k=self.k)
        if len(distances) == 0:
            return 0.0
        d_sq = distances ** 2
        d_m_sq = d_sq.mean()
        kernel = self.epsilon / (d_sq / (d_m_sq + 1e-8) + self.epsilon)
        return float(kernel.sum())

    def compute_reward(self, query: np.ndarray) -> float:
        """计算情景内在奖励 = 1 / sqrt(N(x) + 1e-8).

        空库 (N=0) 时返回 1e3 表示极度新颖.

        Args:
            query: (dim,) 查询嵌入.

        Returns:
            float 情景内在奖励.
        """
        n = self.compute_pseudo_count(query)
        return 1.0 / np.sqrt(n + 1e-8) if n > 0 else 1e3

    def add(self, embedding: np.ndarray):
        """向内存库添加一个嵌入."""
        self.bank.add(embedding)

    def reset(self):
        """清空内存库 (每个 episode 结束时调用)."""
        self.bank.clear()
