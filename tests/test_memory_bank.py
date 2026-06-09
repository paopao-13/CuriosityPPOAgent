"""LRUMemoryBank 单元测试: add/size, knn_distances, LRU 淘汰, clear."""
import numpy as np
import pytest

from curiosity_ppo.utils.memory_bank import LRUMemoryBank


def test_add_and_size():
    """add 后 size 应递增, 容量内不淘汰."""
    bank = LRUMemoryBank(capacity=10000, dim=8)
    assert bank.size == 0

    for i in range(5):
        bank.add(np.full(8, float(i), dtype=np.float32))
        assert bank.size == i + 1


def test_knn_distances_correctness():
    """暴力 L2 距离应返回正确的 k 个最近邻."""
    bank = LRUMemoryBank(capacity=10000, dim=2)
    # 添加若干点
    points = [np.array([0.0, 0.0]), np.array([1.0, 0.0]),
              np.array([2.0, 0.0]), np.array([3.0, 0.0])]
    for p in points:
        bank.add(p.astype(np.float32))

    query = np.array([0.5, 0.0], dtype=np.float32)
    dists = bank.knn_distances(query, k=2)
    # 最近两个点为 [0,0] 和 [1,0], 距离均为 0.5
    assert dists.shape == (2,), f"shape {dists.shape}"
    assert np.allclose(np.sort(dists), [0.5, 0.5]), f"dists {dists}"


def test_knn_distances_empty_bank():
    """空库应返回空数组."""
    bank = LRUMemoryBank(capacity=10000, dim=4)
    dists = bank.knn_distances(np.zeros(4, dtype=np.float32), k=5)
    assert isinstance(dists, np.ndarray)
    assert dists.size == 0


def test_knn_distances_k_larger_than_size():
    """k > 库大小时, 返回全部 (不报错)."""
    bank = LRUMemoryBank(capacity=10000, dim=4)
    bank.add(np.zeros(4, dtype=np.float32))
    bank.add(np.ones(4, dtype=np.float32))
    dists = bank.knn_distances(np.zeros(4, dtype=np.float32), k=5)
    assert dists.shape == (2,), f"shape {dists.shape}"


def test_lru_eviction():
    """超容量时淘汰最旧 (FIFO)."""
    bank = LRUMemoryBank(capacity=3, dim=4)
    bank.add(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))  # 将被淘汰
    bank.add(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    bank.add(np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))
    bank.add(np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32))  # 触发淘汰
    assert bank.size == 3

    # query 0 向量, 被淘汰后最近邻应为 [1,1,1,1] 距离 2.0
    dists = bank.knn_distances(np.zeros(4, dtype=np.float32), k=1)
    assert dists.shape == (1,)
    assert np.isclose(dists[0], 2.0), f"expected 2.0, got {dists[0]}"


def test_clear():
    """clear 应清空库并重置计数."""
    bank = LRUMemoryBank(capacity=10000, dim=4)
    bank.add(np.zeros(4, dtype=np.float32))
    bank.add(np.ones(4, dtype=np.float32))
    assert bank.size == 2

    bank.clear()
    assert bank.size == 0
    # clear 后仍可继续添加
    bank.add(np.zeros(4, dtype=np.float32))
    assert bank.size == 1


def test_add_does_not_mutate_input():
    """add 应拷贝输入, 不应被外部修改影响库内数据."""
    bank = LRUMemoryBank(capacity=10000, dim=4)
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    bank.add(arr)
    arr[0] = 999.0  # 外部修改
    dists = bank.knn_distances(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), k=1)
    assert np.isclose(dists[0], 0.0), f"input mutated, dist {dists[0]}"
