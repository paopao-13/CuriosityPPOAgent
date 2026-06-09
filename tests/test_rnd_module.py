"""RNDCuriosity 模块测试, 使用真实 RNDNet (in_channels=4)."""
import pytest
import torch

from curiosity_ppo.networks.rnd import RNDNet
from curiosity_ppo.curiosity.rnd_module import RNDCuriosity


@pytest.fixture
def rnd_module():
    """构建真实 RNDNet + RNDCuriosity 包装器 (CPU, L=5)."""
    rnd_net = RNDNet(in_channels=4, output_dim=512)
    return RNDCuriosity(rnd_net, obs_normalizer=None, reward_normalizer=None, L=5, device='cpu')


def test_compute_reward_returns_float(rnd_module):
    """compute_reward 应返回 float."""
    obs = torch.randn(4, 4, 84, 84)
    reward = rnd_module.compute_reward(obs)
    assert isinstance(reward, float), f"type {type(reward)}"
    import math
    assert not math.isnan(reward)


def test_compute_reward_no_nan(rnd_module):
    """compute_reward 多次调用不应产生 NaN."""
    for _ in range(3):
        obs = torch.randn(4, 4, 84, 84)
        r = rnd_module.compute_reward(obs)
        import math
        assert not math.isnan(r), f"NaN reward: {r}"


def test_compute_alpha_in_range(rnd_module):
    """compute_alpha 应返回 [1, L] 范围内的 float."""
    obs = torch.randn(4, 4, 84, 84)
    alpha = rnd_module.compute_alpha(obs)
    assert isinstance(alpha, float), f"type {type(alpha)}"
    L = rnd_module.L
    assert 1.0 <= alpha <= float(L), f"alpha {alpha} not in [1, {L}]"


def test_compute_alpha_multiple_calls_in_range(rnd_module):
    """多次 compute_alpha 都应在 [1, L] 内."""
    for _ in range(5):
        obs = torch.randn(4, 4, 84, 84)
        alpha = rnd_module.compute_alpha(obs)
        assert 1.0 <= alpha <= float(rnd_module.L), f"alpha {alpha} out of range"


def test_compute_alpha_uses_L(rnd_module):
    """alpha 上界应受 L 控制."""
    obs = torch.randn(4, 4, 84, 84)
    # 默认 L=5
    a5 = rnd_module.compute_alpha(obs)

    # 新建一个 L=10 的模块, 用同样的 net 权重
    import copy
    rnd_net2 = copy.deepcopy(rnd_module.rnd_net)
    m10 = RNDCuriosity(rnd_net2, obs_normalizer=None, reward_normalizer=None, L=10, device='cpu')
    a10 = m10.compute_alpha(obs)

    assert a5 <= 5.0, f"a5 {a5}"
    assert a10 <= 10.0, f"a10 {a10}"


def test_compute_reward_updates_error_rms(rnd_module):
    """compute_reward 应更新内部 _error_rms 统计量."""
    initial_count = rnd_module._error_rms.count
    obs = torch.randn(4, 4, 84, 84)
    rnd_module.compute_reward(obs)
    assert rnd_module._error_rms.count > initial_count


def test_compute_alpha_no_nan(rnd_module):
    """compute_alpha 不应产生 NaN/Inf."""
    obs = torch.randn(4, 4, 84, 84)
    alpha = rnd_module.compute_alpha(obs)
    import math
    assert math.isfinite(alpha), f"alpha not finite: {alpha}"
