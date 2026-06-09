"""ICMCuriosity 模块测试, 使用真实 ICMNet (in_channels=3, action_dim=17)."""
import pytest
import torch

from curiosity_ppo.networks.icm import ICMNet
from curiosity_ppo.curiosity.icm_module import ICMCuriosity


@pytest.fixture
def icm_module():
    """构建真实 ICMNet + ICMCuriosity 包装器 (CPU)."""
    icm_net = ICMNet(in_channels=3, action_dim=17, feature_dim=288, hidden_dim=256)
    return ICMCuriosity(icm_net, eta=0.2, device='cpu')


def test_compute_reward_returns_float(icm_module):
    """compute_reward 应返回 Python float."""
    s_t = torch.randn(4, 3, 64, 64)
    s_next = torch.randn(4, 3, 64, 64)
    a = torch.randint(0, 17, (4,))

    reward = icm_module.compute_reward(s_t, a, s_next)
    assert isinstance(reward, float), f"type {type(reward)}"
    assert not np_isnan(reward)
    # 内在奖励 = eta * forward_loss >= 0
    assert reward >= 0.0, f"reward should be non-negative, got {reward}"


def np_isnan(x):
    import math
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return False


def np_isclose(a, b, rtol=1e-5, atol=1e-8):
    import math
    try:
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    except (TypeError, ValueError):
        return False


def test_compute_reward_scales_with_eta(icm_module):
    """compute_reward 应为 eta * forward_loss (与 eta 成正比)."""
    s_t = torch.randn(4, 3, 64, 64)
    s_next = torch.randn(4, 3, 64, 64)
    a = torch.randint(0, 17, (4,))

    # 固定 seed 以保证两次 forward_loss 一致
    torch.manual_seed(0)
    icm_net1 = ICMNet(in_channels=3, action_dim=17)
    m1 = ICMCuriosity(icm_net1, eta=0.2, device='cpu')
    torch.manual_seed(0)
    icm_net2 = ICMNet(in_channels=3, action_dim=17)
    m2 = ICMCuriosity(icm_net2, eta=0.4, device='cpu')

    s_t = torch.randn(4, 3, 64, 64)
    s_next = torch.randn(4, 3, 64, 64)
    a = torch.randint(0, 17, (4,))

    r1 = m1.compute_reward(s_t, a, s_next)
    r2 = m2.compute_reward(s_t, a, s_next)
    # eta 翻倍 -> reward 翻倍
    assert np_isclose(r2, 2.0 * r1, rtol=1e-5), f"r1={r1}, r2={r2}"


def test_get_loss_returns_tuple(icm_module):
    """get_loss 应返回 (loss, phi_t)."""
    s_t = torch.randn(4, 3, 64, 64)
    s_next = torch.randn(4, 3, 64, 64)
    a = torch.randint(0, 17, (4,))

    out = icm_module.get_loss(s_t, a, s_next)
    assert isinstance(out, tuple) and len(out) == 2, f"got {out}"

    loss, phi_t = out
    # loss 应为标量 tensor 且可反传
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0, f"loss not scalar: {loss.shape}"
    assert loss.requires_grad, "loss should require grad"
    # phi_t shape (4, 288)
    assert phi_t.shape == (4, 288), f"phi_t {tuple(phi_t.shape)}"


def test_get_loss_differentiable(icm_module):
    """get_loss 返回的 loss 应可 backward, 梯度传到 encoder."""
    s_t = torch.randn(4, 3, 64, 64)
    s_next = torch.randn(4, 3, 64, 64)
    a = torch.randint(0, 17, (4,))

    loss, phi_t = icm_module.get_loss(s_t, a, s_next)
    loss.backward()
    has_grad = any(
        (p.grad is not None and p.grad.abs().sum().item() > 0)
        for p in icm_module.icm_net.encoder.parameters()
    )
    assert has_grad, "encoder received no gradient from get_loss"


def test_get_embedding_shape(icm_module):
    """get_embedding 应返回 (N, feature_dim)."""
    obs = torch.randn(4, 3, 64, 64)
    emb = icm_module.get_embedding(obs)
    assert emb.shape == (4, 288), f"embedding {tuple(emb.shape)}"
    assert not torch.isnan(emb).any()


def test_compute_reward_no_grad(icm_module):
    """compute_reward 不应构建计算图 (no_grad)."""
    s_t = torch.randn(4, 3, 64, 64)
    s_next = torch.randn(4, 3, 64, 64)
    a = torch.randint(0, 17, (4,))

    # 记录 backward 前所有参数 grad
    icm_module.icm_net.zero_grad(set_to_none=True)
    _ = icm_module.compute_reward(s_t, a, s_next)
    # compute_reward 在 no_grad 下, 参数不应有梯度
    has_grad = any(p.grad is not None for p in icm_module.icm_net.parameters())
    assert not has_grad, "compute_reward should not produce gradients"
