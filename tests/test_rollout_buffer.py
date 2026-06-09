"""测试 RolloutBuffer: add/ptr 递增、reset、get_mini_batches 分片与 shape."""
import numpy as np
import torch

from curiosity_ppo.ppo.rollout_buffer import RolloutBuffer


def _make_buffer(n_steps=8, n_envs=2, obs_shape=(3, 64, 64), action_dim=7):
    return RolloutBuffer(
        n_steps=n_steps,
        n_envs=n_envs,
        obs_shape=obs_shape,
        action_dim=action_dim,
        device='cpu',
    )


def test_buffer_initial_shapes():
    buf = _make_buffer(n_steps=8, n_envs=2, obs_shape=(3, 64, 64))
    assert buf.obs.shape == (8, 2, 3, 64, 64)
    assert buf.actions.shape == (8, 2)
    assert buf.logprobs.shape == (8, 2)
    assert buf.rewards_ext.shape == (8, 2)
    assert buf.rewards_int.shape == (8, 2)
    assert buf.values_ext.shape == (8, 2)
    assert buf.values_int.shape == (8, 2)
    assert buf.dones.shape == (8, 2)
    assert buf.advantages_ext.shape == (8, 2)
    assert buf.advantages_int.shape == (8, 2)
    assert buf.returns_ext.shape == (8, 2)
    assert buf.returns_int.shape == (8, 2)
    assert buf.ptr == 0


def test_buffer_add_increments_ptr():
    buf = _make_buffer(n_steps=4, n_envs=2, obs_shape=(3, 64, 64))
    assert buf.ptr == 0
    for t in range(4):
        buf.add(
            obs=np.zeros((2, 3, 64, 64), dtype=np.float32),
            action=np.array([0, 1], dtype=np.int64),
            logprob=np.array([0.0, 0.0], dtype=np.float32),
            reward_ext=np.array([1.0, 1.0], dtype=np.float32),
            reward_int=np.array([0.1, 0.1], dtype=np.float32),
            value_ext=np.array([0.5, 0.5], dtype=np.float32),
            value_int=np.array([0.2, 0.2], dtype=np.float32),
            done=np.array([0.0, 0.0], dtype=np.float32),
        )
        assert buf.ptr == t + 1, f"ptr should be {t + 1}, got {buf.ptr}"


def test_buffer_add_stores_data_correctly():
    buf = _make_buffer(n_steps=2, n_envs=2, obs_shape=(3, 64, 64))
    obs = np.ones((2, 3, 64, 64), dtype=np.float32) * 0.5
    action = np.array([3, 5], dtype=np.int64)
    logprob = np.array([-1.2, -0.8], dtype=np.float32)
    reward_ext = np.array([1.0, 2.0], dtype=np.float32)
    reward_int = np.array([0.1, 0.2], dtype=np.float32)
    value_ext = np.array([0.5, 0.6], dtype=np.float32)
    value_int = np.array([0.3, 0.4], dtype=np.float32)
    done = np.array([0.0, 1.0], dtype=np.float32)

    buf.add(obs, action, logprob, reward_ext, reward_int, value_ext, value_int, done)

    np.testing.assert_array_equal(buf.obs[0], obs)
    np.testing.assert_array_equal(buf.actions[0], action)
    np.testing.assert_array_equal(buf.logprobs[0], logprob)
    np.testing.assert_array_equal(buf.rewards_ext[0], reward_ext)
    np.testing.assert_array_equal(buf.rewards_int[0], reward_int)
    np.testing.assert_array_equal(buf.values_ext[0], value_ext)
    np.testing.assert_array_equal(buf.values_int[0], value_int)
    np.testing.assert_array_equal(buf.dones[0], done)


def test_buffer_reset():
    buf = _make_buffer(n_steps=4, n_envs=2, obs_shape=(3, 64, 64))
    for _ in range(3):
        buf.add(
            obs=np.zeros((2, 3, 64, 64), dtype=np.float32),
            action=np.array([0, 1], dtype=np.int64),
            logprob=np.array([0.0, 0.0], dtype=np.float32),
            reward_ext=np.array([1.0, 1.0], dtype=np.float32),
            reward_int=np.array([0.1, 0.1], dtype=np.float32),
            value_ext=np.array([0.5, 0.5], dtype=np.float32),
            value_int=np.array([0.2, 0.2], dtype=np.float32),
            done=np.array([0.0, 0.0], dtype=np.float32),
        )
    assert buf.ptr == 3
    buf.reset()
    assert buf.ptr == 0


def test_get_mini_batches_returns_dict_with_correct_keys():
    buf = _make_buffer(n_steps=4, n_envs=2, obs_shape=(3, 64, 64))
    for _ in range(4):
        buf.add(
            obs=np.random.randn(2, 3, 64, 64).astype(np.float32),
            action=np.array([0, 1], dtype=np.int64),
            logprob=np.array([0.0, 0.0], dtype=np.float32),
            reward_ext=np.array([1.0, 1.0], dtype=np.float32),
            reward_int=np.array([0.1, 0.1], dtype=np.float32),
            value_ext=np.array([0.5, 0.5], dtype=np.float32),
            value_int=np.array([0.2, 0.2], dtype=np.float32),
            done=np.array([0.0, 0.0], dtype=np.float32),
        )
    batches = list(buf.get_mini_batches(batch_size=4))
    assert len(batches) > 0
    for batch in batches:
        assert isinstance(batch, dict)
        expected_keys = {
            'obs', 'actions', 'logprobs', 'advantages_ext', 'advantages_int',
            'returns_ext', 'returns_int',
        }
        assert set(batch.keys()) == expected_keys, f"keys mismatch: {set(batch.keys())}"


def test_get_mini_batches_batch_size_splits_correctly():
    """n_steps=8, n_envs=2 -> total=16, batch_size=4 -> 4 个 mini-batch."""
    buf = _make_buffer(n_steps=8, n_envs=2, obs_shape=(3, 64, 64))
    for _ in range(8):
        buf.add(
            obs=np.random.randn(2, 3, 64, 64).astype(np.float32),
            action=np.array([0, 1], dtype=np.int64),
            logprob=np.array([0.0, 0.0], dtype=np.float32),
            reward_ext=np.array([1.0, 1.0], dtype=np.float32),
            reward_int=np.array([0.1, 0.1], dtype=np.float32),
            value_ext=np.array([0.5, 0.5], dtype=np.float32),
            value_int=np.array([0.2, 0.2], dtype=np.float32),
            done=np.array([0.0, 0.0], dtype=np.float32),
        )
    batches = list(buf.get_mini_batches(batch_size=4))
    assert len(batches) == 4, f"expected 4 batches, got {len(batches)}"
    for batch in batches:
        assert batch['obs'].shape[0] == 4
        assert batch['actions'].shape[0] == 4
        assert batch['logprobs'].shape[0] == 4


def test_get_mini_batches_obs_shape_correct():
    """mini-batch obs 应为 (batch_size, *obs_shape)."""
    buf = _make_buffer(n_steps=4, n_envs=2, obs_shape=(3, 64, 64))
    for _ in range(4):
        buf.add(
            obs=np.random.randn(2, 3, 64, 64).astype(np.float32),
            action=np.array([0, 1], dtype=np.int64),
            logprob=np.array([0.0, 0.0], dtype=np.float32),
            reward_ext=np.array([1.0, 1.0], dtype=np.float32),
            reward_int=np.array([0.1, 0.1], dtype=np.float32),
            value_ext=np.array([0.5, 0.5], dtype=np.float32),
            value_int=np.array([0.2, 0.2], dtype=np.float32),
            done=np.array([0.0, 0.0], dtype=np.float32),
        )
    batches = list(buf.get_mini_batches(batch_size=8))
    assert len(batches) == 1
    batch = batches[0]
    assert batch['obs'].shape == (8, 3, 64, 64), f"obs shape {batch['obs'].shape}"
    assert batch['actions'].shape == (8,)
    assert batch['logprobs'].shape == (8,)
    assert batch['advantages_ext'].shape == (8,)
    assert batch['advantages_int'].shape == (8,)
    assert batch['returns_ext'].shape == (8,)
    assert batch['returns_int'].shape == (8,)


def test_get_mini_batches_returns_torch_tensors():
    buf = _make_buffer(n_steps=4, n_envs=2, obs_shape=(3, 64, 64))
    for _ in range(4):
        buf.add(
            obs=np.random.randn(2, 3, 64, 64).astype(np.float32),
            action=np.array([0, 1], dtype=np.int64),
            logprob=np.array([0.0, 0.0], dtype=np.float32),
            reward_ext=np.array([1.0, 1.0], dtype=np.float32),
            reward_int=np.array([0.1, 0.1], dtype=np.float32),
            value_ext=np.array([0.5, 0.5], dtype=np.float32),
            value_int=np.array([0.2, 0.2], dtype=np.float32),
            done=np.array([0.0, 0.0], dtype=np.float32),
        )
    batches = list(buf.get_mini_batches(batch_size=8))
    batch = batches[0]
    assert isinstance(batch['obs'], torch.Tensor)
    assert isinstance(batch['actions'], torch.Tensor)
    assert isinstance(batch['logprobs'], torch.Tensor)
