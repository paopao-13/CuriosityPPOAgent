import copy
import torch

from curiosity_ppo.networks.rnd import RNDNet


def test_rnd_forward_shapes():
    """forward 返回 (pred, target), shape 均为 (4, 512)."""
    rnd = RNDNet(in_channels=4, output_dim=512)
    obs = torch.randn(4, 4, 84, 84)
    pred, target = rnd(obs)
    assert pred.shape == (4, 512), f"pred {tuple(pred.shape)}"
    assert target.shape == (4, 512), f"target {tuple(target.shape)}"


def test_rnd_forward_no_nan():
    rnd = RNDNet(in_channels=4, output_dim=512)
    obs = torch.randn(4, 4, 84, 84)
    pred, target = rnd(obs)
    assert not torch.isnan(pred).any()
    assert not torch.isnan(target).any()


def test_rnd_intrinsic_reward_shape():
    """intrinsic_reward 返回 (4,)."""
    rnd = RNDNet(in_channels=4, output_dim=512)
    obs = torch.randn(4, 4, 84, 84)
    r = rnd.intrinsic_reward(obs)
    assert r.shape == (4,), f"intrinsic reward {tuple(r.shape)}"
    assert not torch.isnan(r).any()
    # 预测误差应为非负
    assert (r >= 0).all(), "intrinsic reward should be non-negative"


def test_rnd_target_params_not_updated():
    """target 网络参数在反向传播后不应更新."""
    rnd = RNDNet(in_channels=4, output_dim=512)
    # target 参数 requires_grad 应为 False
    for p in rnd.target.parameters():
        assert not p.requires_grad, "target param should have requires_grad=False"

    # 保存 target 参数的初始值
    target_before = [p.clone().detach() for p in rnd.target.parameters()]
    predictor_before = [p.clone().detach() for p in rnd.predictor.parameters()]

    obs = torch.randn(4, 4, 84, 84)
    pred, target = rnd(obs)
    loss = ((pred - target.detach()) ** 2).mean()
    loss.backward()

    # 只更新 predictor
    opt = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-3)
    opt.step()

    # target 参数应不变
    for before, after in zip(target_before, rnd.target.parameters()):
        assert torch.equal(before, after), "target params changed after optimizer step"

    # predictor 参数应有变化 (至少一个参数不同)
    predictor_changed = any(
        not torch.equal(before, after)
        for before, after in zip(predictor_before, rnd.predictor.parameters())
    )
    assert predictor_changed, "predictor params did not update"


def test_rnd_target_has_no_grad_in_forward():
    """target 输出不应携带梯度."""
    rnd = RNDNet(in_channels=4, output_dim=512)
    obs = torch.randn(4, 4, 84, 84)
    pred, target = rnd(obs)
    assert target.requires_grad is False, "target output should not require grad"
    assert pred.requires_grad is True, "predictor output should require grad"
