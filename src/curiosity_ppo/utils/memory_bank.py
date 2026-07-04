"""LRU 内存库 (CPU), 用于情景记忆的 kNN 搜索.

预分配 numpy 数组, 避免 knn_distances 每次调用 np.stack 重建.
使用环形缓冲区管理 FIFO 淘汰, write_ptr 循环写入.
"""
import numpy as np


class LRUMemoryBank:
    """CPU 上的 LRU 内存库, 用于 kNN 搜索.

    Args:
        capacity: 最大容量, 超出时淘汰最旧 (FIFO).
        dim: 嵌入向量维度.
    """

    def __init__(self, capacity: int = 10000, dim: int = 512):
        self.capacity = capacity
        self.dim = dim
        # 预分配数组, 避免每次 knn_distances 调用 np.stack
        self._data = np.zeros((capacity, dim), dtype=np.float32)
        self._size = 0
        self._write_ptr = 0  # 环形缓冲区写指针

    @property
    def size(self) -> int:
        """当前库中嵌入数量."""
        return self._size

    def add(self, embedding: np.ndarray):
        """添加嵌入向量, 超容量时淘汰最旧.

        Args:
            embedding: (dim,) 形状的嵌入向量, 内部会拷贝以防外部修改.
        """
        self._data[self._write_ptr] = embedding
        self._write_ptr = (self._write_ptr + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def knn_distances(self, query: np.ndarray, k: int = 5) -> np.ndarray:
        """numpy 暴力 L2 距离, 返回 k 个最近邻距离 (升序).

        直接使用预分配数组的视图, 无需 np.stack 重建.

        Args:
            query: (dim,) 查询向量.
            k: 最近邻数量.

        Returns:
            (min(k, size),) 升序排列的 L2 距离; 空库返回空数组.
        """
        if self._size == 0:
            return np.array([])
        # 使用预分配数组的有效部分 (视图, 无拷贝)
        active = self._data[:self._size]  # (N, dim)
        diff = active - query[np.newaxis, :]  # (N, dim)
        distances = np.sqrt((diff ** 2).sum(axis=1))  # (N,)
        k = min(k, len(distances))
        k_smallest = np.partition(distances, k - 1)[:k]
        return np.sort(k_smallest)

    def clear(self):
        """清空内存库并重置内部计数."""
        self._size = 0
        self._write_ptr = 0
