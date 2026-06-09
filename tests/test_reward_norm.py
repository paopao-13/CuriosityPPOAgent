"""RunningMeanStd / RewardNormalizer 单元测试."""
import numpy as np
import pytest

from curiosity_ppo.curiosity.reward_norm import RunningMeanStd, RewardNormalizer


def test_running_mean_std_initial():
    """初始 mean=0, var=1, count=1e-4."""
    rms = RunningMeanStd(shape=())
    assert np.isclose(rms.mean, 0.0)
    assert np.isclose(rms.var, 1.0)
    assert rms.count == 1e-4
    assert np.isclose(rms.std, 1.0)


def test_running_mean_std_update_mean_var():
    """2D 批次更新后 mean/var 应匹配样本统计量 (大批量近似).

    注: spec 的 update 对 ndim>1 走批次分支 (mean/var axis=0),
    因此用 2D (N,1) 数据配合 shape=(1,) 测试批次统计.
    """
    rms = RunningMeanStd(shape=(1,))
    rng = np.random.RandomState(0)
    data = rng.randn(1000, 1)
    rms.update(data)
    # 大样本下 mean/var 应接近真实值 (先验 count=1e-4 影响可忽略)
    assert np.allclose(rms.mean, data.mean(axis=0), atol=1e-6), f"mean {rms.mean}"
    assert np.allclose(rms.var, data.var(axis=0), atol=1e-6), f"var {rms.var}"
    assert np.isclose(rms.count, 1000 + 1e-4)


def test_running_mean_std_incremental_update():
    """分批更新应与一次性更新结果一致 (Welford 在线算法)."""
    rms_a = RunningMeanStd(shape=(1,))
    rms_b = RunningMeanStd(shape=(1,))
    rng = np.random.RandomState(42)
    data = rng.randn(500, 1)

    # 一次性
    rms_a.update(data)
    # 分批 (每批 50)
    for i in range(0, 500, 50):
        rms_b.update(data[i:i + 50])

    assert np.allclose(rms_a.mean, rms_b.mean, atol=1e-9), f"mean {rms_a.mean} vs {rms_b.mean}"
    assert np.allclose(rms_a.var, rms_b.var, atol=1e-9), f"var {rms_a.var} vs {rms_b.var}"
    assert np.isclose(rms_a.count, rms_b.count, atol=1e-9)


def test_running_mean_std_shape_vector():
    """向量 shape 的 RunningMeanStd 应逐元素统计 (大批量稀释先验)."""
    rms = RunningMeanStd(shape=(3,))
    base = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    data = np.tile(base, (500, 1))  # (1000, 3) 重复以稀释 1e-4 先验
    rms.update(data)
    expected_mean = base.mean(axis=0)
    expected_var = base.var(axis=0)
    assert np.allclose(rms.mean, expected_mean, atol=1e-4), f"mean {rms.mean}"
    assert np.allclose(rms.var, expected_var, atol=1e-4), f"var {rms.var}"


def test_running_mean_std_scalar_single_samples():
    """标量 shape 接受单标量输入, 逐次更新 mean."""
    rms = RunningMeanStd(shape=())
    rms.update(2.0)
    # 单样本: mean ≈ 2.0 * 1 / (1 + 1e-4)
    assert np.isclose(rms.mean, 2.0 / (1 + 1e-4), atol=1e-4), f"mean {rms.mean}"
    assert np.isclose(rms.count, 1 + 1e-4)


def test_reward_normalizer_returns_float():
    """RewardNormalizer 调用应返回标量 float (输入标量)."""
    norm = RewardNormalizer()
    out = norm(1.0)
    assert isinstance(out, (float, np.floating)), f"type {type(out)}"


def test_reward_normalizer_normalizes():
    """归一化后奖励应除以 running std.

    __call__ 会先 update 再归一化, 用极大 count 使单样本更新影响可忽略.
    """
    norm = RewardNormalizer()
    # 注入已知统计量 (count 极大, 单次 update 影响可忽略)
    norm.running_ms.mean = 0.0
    norm.running_ms.var = 4.0  # std = 2.0
    norm.running_ms.count = 1e10
    out = norm(4.0)
    # 4.0 / 2.0 = 2.0 (单样本更新被 1e10 count 稀释, 未触发 clip)
    assert np.isclose(out, 2.0, atol=1e-3), f"expected ~2.0, got {out}"


def test_reward_normalizer_clips():
    """归一化后值超过 clip 应被裁剪."""
    norm = RewardNormalizer()
    norm.running_ms.mean = 0.0
    norm.running_ms.var = 1e-8  # 极小 std -> 极大归一化值
    norm.running_ms.count = 100.0
    out = norm(1000.0)
    assert out <= norm.clip, f"expected <= {norm.clip}, got {out}"
    assert out == norm.clip


def test_reward_normalizer_clip_negative():
    """负向 clip."""
    norm = RewardNormalizer()
    norm.running_ms.mean = 0.0
    norm.running_ms.var = 1e-8
    norm.running_ms.count = 100.0
    out = norm(-1000.0)
    assert out >= -norm.clip, f"expected >= -{norm.clip}, got {out}"
    assert out == -norm.clip


def test_reward_normalizer_updates_running_stats():
    """每次调用应更新内部统计量."""
    norm = RewardNormalizer()
    initial_count = norm.running_ms.count
    norm(np.array([1.0, 2.0, 3.0]))
    assert norm.running_ms.count > initial_count
