import torch
import pytest

from curiosity_ppo.networks.encoders import NatureDQNEncoder
from curiosity_ppo.networks.policy import ActorCritic


def _make_actor_critic(action_dim=17, embed_dim=512):
    encoder = NatureDQNEncoder(in_channels=4, out_dim=embed_dim)
    return ActorCritic(encoder, action_dim=action_dim, embed_dim=embed_dim)


def test_actor_critic_forward_shapes():
    """forward 返回 (logits, v_ext, v_int), shape 分别为 (4, A), (4,1), (4,1)."""
    action_dim = 17
    model = _make_actor_critic(action_dim=action_dim, embed_dim=512)
    obs = torch.randn(4, 4, 84, 84)
    logits, v_ext, v_int = model(obs)
    assert logits.shape == (4, action_dim), f"logits {tuple(logits.shape)}"
    assert v_ext.shape == (4, 1), f"v_ext {tuple(v_ext.shape)}"
    assert v_int.shape == (4, 1), f"v_int {tuple(v_int.shape)}"


def test_actor_critic_forward_no_nan():
    model = _make_actor_critic()
    obs = torch.randn(4, 4, 84, 84)
    logits, v_ext, v_int = model(obs)
    assert not torch.isnan(logits).any()
    assert not torch.isnan(v_ext).any()
    assert not torch.isnan(v_int).any()


def test_actor_critic_get_action_returns_four_values():
    """get_action 返回 (action, logprob, v_ext, v_int) 共 4 个值, 且 v 被 squeeze."""
    model = _make_actor_critic(action_dim=17, embed_dim=512)
    obs = torch.randn(4, 4, 84, 84)
    out = model.get_action(obs)
    assert isinstance(out, tuple) and len(out) == 4, f"get_action returned {out}"

    action, logprob, v_ext, v_int = out
    assert action.shape == (4,), f"action {tuple(action.shape)}"
    assert logprob.shape == (4,), f"logprob {tuple(logprob.shape)}"
    # squeeze(-1) 后应为 (4,)
    assert v_ext.shape == (4,), f"v_ext {tuple(v_ext.shape)}"
    assert v_int.shape == (4,), f"v_int {tuple(v_int.shape)}"
    # action 为离散动作索引, dtype 为 long
    assert action.dtype == torch.long, f"action dtype {action.dtype}"
    assert ((action >= 0) & (action < 17)).all()


def test_actor_critic_get_action_no_nan():
    model = _make_actor_critic()
    obs = torch.randn(4, 4, 84, 84)
    action, logprob, v_ext, v_int = model.get_action(obs)
    assert not torch.isnan(logprob).any()
    assert not torch.isnan(v_ext).any()
    assert not torch.isnan(v_int).any()


def test_actor_critic_critic_heads_independent():
    """critic_ext 与 critic_int 是独立的参数, 对同一输入给出不同输出."""
    model = _make_actor_critic()
    obs = torch.randn(4, 4, 84, 84)
    _, v_ext, v_int = model(obs)
    # 极不可能完全相等
    assert not torch.allclose(v_ext, v_int), "ext/int value heads collapsed"
