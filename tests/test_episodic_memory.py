"""EpisodicMemory (kNN 伪计数) 单元测试."""
import numpy as np
import pytest

from curiosity_ppo.curiosity.episodic_memory import EpisodicMemory


def test_empty_bank_pseudo_count_zero():
    """空库时 compute_pseudo_count 返回 0.0."""
    mem = EpisodicMemory(capacity=10000, dim=8, k=5, epsilon=1e-3, L=5)
    query = np.zeros(8, dtype=np.float32)
    assert mem.compute_pseudo_count(query) == 0.0


def test_empty_bank_reward_large():
    """空库时 compute_reward 返回大值 (1e3, 表示极度新颖)."""
    mem = EpisodicMemory(capacity=10000, dim=8, k=5, epsilon=1e-3, L=5)
    query = np.zeros(8, dtype=np.float32)
    r = mem.compute_reward(query)
    assert r == 1e3, f"expected 1e3 for empty bank, got {r}"


def test_pseudo_count_positive_after_add():
    """添加嵌入后, 查询邻近点应得到 > 0 的伪计数."""
    mem = EpisodicMemory(capacity=10000, dim=4, k=5, epsilon=1e-3, L=5)
    # 添加若干与 query 相同的点
    query = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    for _ in range(3):
        mem.add(query)
    n = mem.compute_pseudo_count(query)
    assert n > 0.0, f"pseudo-count should be > 0, got {n}"


def test_reward_positive_after_add():
    """添加后 reward 应 > 0 且有限."""
    mem = EpisodicMemory(capacity=10000, dim=4, k=5, epsilon=1e-3, L=5)
    query = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    mem.add(query)
    r = mem.compute_reward(query)
    assert np.isfinite(r), f"reward not finite: {r}"
    assert r > 0.0, f"reward should be > 0, got {r}"


def test_reward_decreases_with_more_visits():
    """同一查询点访问越多 (伪计数越大), 奖励应越小 (新颖性下降)."""
    mem = EpisodicMemory(capacity=10000, dim=4, k=5, epsilon=1e-3, L=5)
    query = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)

    mem.add(query)
    mem.add(query)
    r_few = mem.compute_reward(query)

    # 再多加几次
    for _ in range(8):
        mem.add(query)
    r_many = mem.compute_reward(query)

    assert r_many < r_few, f"reward should decrease with more visits: {r_few} -> {r_many}"


def test_reset_clears_bank():
    """reset 应清空内存库."""
    mem = EpisodicMemory(capacity=10000, dim=4, k=5, epsilon=1e-3, L=5)
    mem.add(np.zeros(4, dtype=np.float32))
    mem.add(np.ones(4, dtype=np.float32))
    assert mem.bank.size == 2

    mem.reset()
    assert mem.bank.size == 0
    # reset 后查询应回到空库行为
    r = mem.compute_reward(np.zeros(4, dtype=np.float32))
    assert r == 1e3


def test_add_and_query_different_points():
    """查询与库内点距离较远时, 伪计数仍应 > 0 (kernel 非零)."""
    mem = EpisodicMemory(capacity=10000, dim=4, k=5, epsilon=1e-3, L=5)
    mem.add(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    n = mem.compute_pseudo_count(np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32))
    # 距离虽远, 但 kernel = epsilon / (d^2/d_m^2 + epsilon) 仍 > 0
    assert n > 0.0, f"pseudo-count should be > 0 due to kernel, got {n}"
