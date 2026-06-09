"""测试 PPOTrainer: 小 ActorCritic + 小 buffer 的 update 返回正确 dict 与有限 loss."""
import numpy as np
import torch
import pytest

from curiosity_ppo.config import Config, PPOConfig
from curiosity_ppo.networks.encoders import CrafterEncoder
from curiosity_ppo.networks.policy import ActorCritic
from curiosity_ppo.ppo.rollout_buffer import RolloutBuffer
from curiosity_ppo.ppo.gae import compute_gae
from curiosity_ppo.ppo.ppo_trainer import PPOTrainer


def _make_small_config():
    """小 batch 配置: n_steps=16, n_envs=2 -> total=32, batch_size=8 -> 4 batch."""
    config = Config()
    config.use_amp = False  # CPU
    config.ppo = PPOConfig(
        lr=1e-3,
        clip_range=0.2,
        ppo_epochs=2,
        gamma_ext=0.99,
        gamma_int=0.99,
        gae_lambda=0.95,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        batch_size=8,
        accumulation_steps=2,
        n_steps=16,
        n_envs=2,
    )
    return config


def _make_actor_critic(action_dim=7, embed_dim=288):
    encoder = CrafterEncoder(in_channels=3, out_dim=embed_dim)
    return ActorCritic(encoder, action_dim=action_dim, embed_dim=embed_dim)


def _fill_buffer(buffer, model, config, n_steps, n_envs, obs_shape, device='cpu'):
    """用模型采样填充 buffer, 并计算双轨 GAE."""
    model.eval()
    with torch.no_grad():
        for t in range(n_steps):
            obs = torch.randn(n_envs, *obs_shape).to(device)
            action, logprob, v_ext, v_int = model.get_action(obs)
            buffer.add(
                obs=obs.cpu().numpy(),
                action=action.cpu().numpy(),
                logprob=logprob.cpu().numpy(),
                reward_ext=np.random.rand(n_envs).astype(np.float32),
                reward_int=np.random.rand(n_envs).astype(np.float32) * 0.1,
                value_ext=v_ext.cpu().numpy(),
                value_int=v_int.cpu().numpy(),
                done=np.array([1.0 if t == n_steps - 1 else 0.0] * n_envs,
                              dtype=np.float32),
            )
    # 计算 GAE
    last_value_ext = np.zeros(n_envs, dtype=np.float32)
    last_value_int = np.zeros(n_envs, dtype=np.float32)
    # 外在: 情景制, 用真实 dones
    adv_ext, ret_ext = compute_gae(
        buffer.rewards_ext, buffer.values_ext, last_value_ext,
        buffer.dones, config.ppo.gamma_ext, config.ppo.gae_lambda,
    )
    # 内在: 非情景制, dones 全 0
    dones_int = np.zeros_like(buffer.dones)
    adv_int, ret_int = compute_gae(
        buffer.rewards_int, buffer.values_int, last_value_int,
        dones_int, config.ppo.gamma_int, config.ppo.gae_lambda,
    )
    buffer.advantages_ext = adv_ext
    buffer.returns_ext = ret_ext
    buffer.advantages_int = adv_int
    buffer.returns_int = ret_int
    model.train()


def test_ppo_trainer_init():
    config = _make_small_config()
    model = _make_actor_critic()
    trainer = PPOTrainer(model, config, device='cpu')
    assert trainer.device == 'cpu'
    assert trainer.batch_size == 8
    assert trainer.ppo_epochs == 2
    assert trainer.accumulation_steps == 2
    assert trainer.clip_range == 0.2
    assert trainer.optimizer is not None


def test_ppo_trainer_update_returns_dict():
    config = _make_small_config()
    model = _make_actor_critic()
    trainer = PPOTrainer(model, config, device='cpu')

    n_steps, n_envs = 16, 2
    obs_shape = (3, 64, 64)
    buffer = RolloutBuffer(
        n_steps=n_steps, n_envs=n_envs, obs_shape=obs_shape,
        action_dim=7, device='cpu',
    )
    _fill_buffer(buffer, model, config, n_steps, n_envs, obs_shape)

    metrics = trainer.update(buffer)

    assert isinstance(metrics, dict), f"update should return dict, got {type(metrics)}"


def test_ppo_trainer_update_returns_correct_keys():
    config = _make_small_config()
    model = _make_actor_critic()
    trainer = PPOTrainer(model, config, device='cpu')

    n_steps, n_envs = 16, 2
    obs_shape = (3, 64, 64)
    buffer = RolloutBuffer(
        n_steps=n_steps, n_envs=n_envs, obs_shape=obs_shape,
        action_dim=7, device='cpu',
    )
    _fill_buffer(buffer, model, config, n_steps, n_envs, obs_shape)

    metrics = trainer.update(buffer)

    expected_keys = {
        'policy_loss', 'value_ext_loss', 'value_int_loss',
        'entropy', 'clip_fraction', 'n_updates',
    }
    assert set(metrics.keys()) == expected_keys, f"keys: {set(metrics.keys())}"


def test_ppo_trainer_update_metrics_finite():
    """所有 loss/metric 应为有限数值 (非 NaN/Inf)."""
    config = _make_small_config()
    model = _make_actor_critic()
    trainer = PPOTrainer(model, config, device='cpu')

    n_steps, n_envs = 16, 2
    obs_shape = (3, 64, 64)
    buffer = RolloutBuffer(
        n_steps=n_steps, n_envs=n_envs, obs_shape=obs_shape,
        action_dim=7, device='cpu',
    )
    _fill_buffer(buffer, model, config, n_steps, n_envs, obs_shape)

    metrics = trainer.update(buffer)

    import math
    for key in ['policy_loss', 'value_ext_loss', 'value_int_loss', 'entropy', 'clip_fraction']:
        val = metrics[key]
        assert isinstance(val, float), f"{key} should be float, got {type(val)}"
        assert math.isfinite(val), f"{key}={val} is not finite"
    assert metrics['n_updates'] > 0, f"n_updates should be > 0, got {metrics['n_updates']}"


def test_ppo_trainer_update_changes_parameters():
    """一次 update 后参数应发生变化 (确认梯度确实回传)."""
    config = _make_small_config()
    model = _make_actor_critic()
    trainer = PPOTrainer(model, config, device='cpu')

    n_steps, n_envs = 16, 2
    obs_shape = (3, 64, 64)
    buffer = RolloutBuffer(
        n_steps=n_steps, n_envs=n_envs, obs_shape=obs_shape,
        action_dim=7, device='cpu',
    )
    _fill_buffer(buffer, model, config, n_steps, n_envs, obs_shape)

    params_before = [p.detach().clone() for p in model.parameters()]
    trainer.update(buffer)
    params_after = list(model.parameters())

    changed = any(
        not torch.allclose(before, after)
        for before, after in zip(params_before, params_after)
    )
    assert changed, "parameters did not change after update"


def test_ppo_trainer_n_updates_correct():
    """n_steps=16, n_envs=2, batch_size=8, ppo_epochs=2 -> 4 batch/epoch * 2 = 8."""
    config = _make_small_config()
    model = _make_actor_critic()
    trainer = PPOTrainer(model, config, device='cpu')

    n_steps, n_envs = 16, 2
    obs_shape = (3, 64, 64)
    buffer = RolloutBuffer(
        n_steps=n_steps, n_envs=n_envs, obs_shape=obs_shape,
        action_dim=7, device='cpu',
    )
    _fill_buffer(buffer, model, config, n_steps, n_envs, obs_shape)

    metrics = trainer.update(buffer)
    # total=32, batch_size=8 -> 4 batches per epoch, ppo_epochs=2 -> 8
    assert metrics['n_updates'] == 8, f"n_updates={metrics['n_updates']}"


def test_ppo_trainer_clip_fraction_in_range():
    """clip_fraction 应在 [0, 1] 区间."""
    config = _make_small_config()
    model = _make_actor_critic()
    trainer = PPOTrainer(model, config, device='cpu')

    n_steps, n_envs = 16, 2
    obs_shape = (3, 64, 64)
    buffer = RolloutBuffer(
        n_steps=n_steps, n_envs=n_envs, obs_shape=obs_shape,
        action_dim=7, device='cpu',
    )
    _fill_buffer(buffer, model, config, n_steps, n_envs, obs_shape)

    metrics = trainer.update(buffer)
    assert 0.0 <= metrics['clip_fraction'] <= 1.0
