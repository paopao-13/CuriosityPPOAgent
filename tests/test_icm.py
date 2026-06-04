import torch

from curiosity_ppo.networks.icm import ICMNet


def test_icm_forward_returns_three_values():
    """forward 返回 (inverse_loss, forward_loss, phi_t) 共 3 个值."""
    icm = ICMNet(in_channels=3, action_dim=17, feature_dim=288, hidden_dim=256)
    s_t = torch.randn(4, 3, 64, 64)
    s_next = torch.randn(4, 3, 64, 64)
    a = torch.randint(0, 17, (4,))

    out = icm(s_t, a, s_next)
    assert isinstance(out, tuple) and len(out) == 3, f"forward returned {out}"

    inverse_loss, forward_loss, phi_t = out

    # inverse_loss / forward_loss 应为标量
    assert inverse_loss.dim() == 0, f"inverse_loss not scalar: {inverse_loss.shape}"
    assert forward_loss.dim() == 0, f"forward_loss not scalar: {forward_loss.shape}"
    # phi_t shape (4, 288)
    assert phi_t.shape == (4, 288), f"phi_t {tuple(phi_t.shape)}"

    assert not torch.isnan(inverse_loss)
    assert not torch.isnan(forward_loss)
    assert not torch.isnan(phi_t).any()


def test_icm_losses_are_differentiable():
    """inverse_loss 与 forward_loss 都应可反向传播 (有 grad_fn)."""
    icm = ICMNet(in_channels=3, action_dim=17, feature_dim=288, hidden_dim=256)
    s_t = torch.randn(4, 3, 64, 64)
    s_next = torch.randn(4, 3, 64, 64)
    a = torch.randint(0, 17, (4,))

    inverse_loss, forward_loss, phi_t = icm(s_t, a, s_next)
    assert inverse_loss.requires_grad, "inverse_loss not differentiable"
    assert forward_loss.requires_grad, "forward_loss not differentiable"

    total = inverse_loss + forward_loss
    total.backward()
    # 检查 encoder 至少一个参数收到梯度
    has_grad = any(
        (p.grad is not None and p.grad.abs().sum().item() > 0)
        for p in icm.encoder.parameters()
    )
    assert has_grad, "encoder received no gradient"


def test_icm_get_embedding_shape():
    """get_embedding 返回 (4, 288)."""
    icm = ICMNet(in_channels=3, action_dim=17, feature_dim=288, hidden_dim=256)
    obs = torch.randn(4, 3, 64, 64)
    emb = icm.get_embedding(obs)
    assert emb.shape == (4, 288), f"embedding {tuple(emb.shape)}"
    assert not torch.isnan(emb).any()


def test_icm_inverse_loss_is_cross_entropy_scale():
    """inverse_loss 用 cross_entropy, 量级应在合理范围 (0 ~ ~ln(action_dim))."""
    icm = ICMNet(in_channels=3, action_dim=17, feature_dim=288, hidden_dim=256)
    s_t = torch.randn(4, 3, 64, 64)
    s_next = torch.randn(4, 3, 64, 64)
    a = torch.randint(0, 17, (4,))
    inverse_loss, _, _ = icm(s_t, a, s_next)
    import math
    assert 0 <= inverse_loss.item() <= math.log(17) * 5 + 1
