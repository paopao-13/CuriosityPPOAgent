"""测试 AMPManager: autocast 可用、scale_loss、CPU 模式 enabled=False."""
import torch
import pytest

from curiosity_ppo.utils.amp import AMPManager


def test_amp_cpu_mode_disabled_by_default():
    """CPU 上即使 enabled=True, 实际 enabled 也应为 False (无 CUDA)."""
    amp = AMPManager(enabled=True, device='cpu')
    if not torch.cuda.is_available():
        assert amp.enabled is False, "AMP should be disabled on CPU"
    # scaler 应存在
    assert amp.scaler is not None


def test_amp_explicit_disabled():
    amp = AMPManager(enabled=False, device='cpu')
    assert amp.enabled is False


def test_amp_autocast_usable_as_context_manager():
    """autocast() 返回的对象可作为 with 上下文使用."""
    amp = AMPManager(enabled=False, device='cpu')
    x = torch.randn(4, 4)
    with amp.autocast():
        y = x * 2
    assert y.shape == (4, 4)
    assert torch.allclose(y, x * 2)


def test_amp_autocast_returns_callable():
    """autocast() 不带参数调用应返回一个上下文管理器 (有 __enter__/__exit__)."""
    amp = AMPManager(enabled=False, device='cpu')
    ctx = amp.autocast()
    assert hasattr(ctx, '__enter__')
    assert hasattr(ctx, '__exit__')


def test_amp_scale_loss_returns_scaled_loss():
    """scale_loss 返回缩放后的 loss; CPU disabled 模式下应等于原 loss (scale=1)."""
    amp = AMPManager(enabled=False, device='cpu')
    loss = torch.tensor(2.0, requires_grad=True)
    scaled = amp.scale_loss(loss)
    # disabled 时 GradScaler.scale 返回原 loss (scale=1)
    assert torch.allclose(scaled, loss), f"scaled {scaled} != loss {loss}"


def test_amp_step_and_update_on_cpu():
    """CPU disabled 模式下 step/update 可正常调用不报错."""
    amp = AMPManager(enabled=False, device='cpu')
    param = torch.nn.Parameter(torch.randn(3))
    optimizer = torch.optim.Adam([param], lr=1e-3)
    loss = (param * 2).sum()
    scaled = amp.scale_loss(loss)
    scaled.backward()
    amp.unscale_(optimizer)
    amp.step(optimizer)
    amp.update()
    # 梯度应被清零前的 optimizer.step 已执行
    assert param.grad is not None


def test_amp_full_workflow_autocast_backward_step():
    """完整流程: autocast 前向 -> scale_loss backward -> step."""
    amp = AMPManager(enabled=False, device='cpu')
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(8, 4)

    with amp.autocast():
        out = model(x)
        loss = out.sum()

    scaled = amp.scale_loss(loss)
    scaled.backward()
    amp.unscale_(optimizer)
    amp.step(optimizer)
    amp.update()
    # 参数应已更新 (梯度非 None)
    assert model.weight.grad is not None


def test_amp_unscale_then_grad_norm():
    """unscale_ 后梯度可用于 clip_grad_norm_."""
    amp = AMPManager(enabled=False, device='cpu')
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 4)
    with amp.autocast():
        loss = model(x).sum()
    amp.scale_loss(loss).backward()
    amp.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    assert torch.isfinite(grad_norm)
    amp.step(optimizer)
    amp.update()
