"""LRU 内存库 (CPU), 用于情景记忆的 kNN 搜索.

使用 OrderedDict 维护插入顺序, 超容量时按 FIFO 淘汰最旧嵌入.
kNN 采用 numpy 暴力 L2 距离计算, 适合中等规模内存库.
"""
import numpy as np
from collections import OrderedDict


class LRUMemoryBank:
    """CPU 上的 LRU 内存库, 用于 kNN 搜索.

    Args:
        capacity: 最大容量, 超出时淘汰最旧 (FIFO).
        dim: 嵌入向量维度.
    """

    def __init__(self, capacity: int = 10000, dim: int = 512):
        self.capacity = capacity
        self.dim = dim
        self._bank: "OrderedDict[int, np.ndarray]" = OrderedDict()  # key -> np.ndarray
        self._next_key = 0

    @property
    def size(self) -> int:
        """当前库中嵌入数量."""
        return len(self._bank)

    def add(self, embedding: np.ndarray):
        """添加嵌入向量, 超容量时淘汰最旧.

        Args:
            embedding: (dim,) 形状的嵌入向量, 内部会拷贝以防外部修改.
        """
        self._bank[self._next_key] = embedding.copy()
        self._next_key += 1
        if len(self._bank) > self.capacity:
            self._bank.popitem(last=False)  # FIFO 淘汰最旧

    def knn_distances(self, query: np.ndarray, k: int = 5) -> np.ndarray:
        """numpy 暴力 L2 距离, 返回 k 个最近邻距离 (升序).

        Args:
            query: (dim,) 查询向量.
            k: 最近邻数量.

        Returns:
            (min(k, size),) 升序排列的 L2 距离; 空库返回空数组.
        """
        if self.size == 0:
            return np.array([])
        all_embeddings = np.stack(list(self._bank.values()))  # (N, dim)
        diff = all_embeddings - query[np.newaxis, :]  # (N, dim)
        distances = np.sqrt((diff ** 2).sum(axis=1))  # (N,)
        k = min(k, len(distances))
        # np.partition 取 k 个最小, 再排序保证升序
        k_smallest = np.partition(distances, k - 1)[:k]
        return np.sort(k_smallest)

    def clear(self):
        """清空内存库并重置内部计数."""
        self._bank.clear()
        self._next_key = 0
