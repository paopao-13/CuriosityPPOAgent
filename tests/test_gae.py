"""测试 GAE 双轨计算: 情景制 (bootstrap 截断) 与非情景制 (跨 episode bootstrap).

用 gamma=1, gae_lambda=1 简化数学:
- 情景制 returns[t] = sum_{k>=t} r_k (episode 内, done 处截断, last_value=0)
- 非情景制 returns[t] = last_value + sum_{k>=t} r_k (跨 episode, 无 done)
"""
import numpy as np
import pytest

from curiosity_ppo.ppo.gae import compute_gae


def _to_2d(arr):
    """(4,) -> (4, 1) 即 n_envs=1."""
    return np.array(arr, dtype=np.float32).reshape(-1, 1)


# ---------------------------------------------------------------------------
# 情景制 (episodic): dones=[0,0,1,0], last_value=0
# gamma=1, lambda=1, rewards=[1,1,1,1], values=[0,0,0,0]
# 期望 advantages = [3, 2, 1, 1], returns = [3, 2, 1, 1]
# 推导:
#   t=3: next_value=0, nnt=1, delta=1-0=1, gae=1
#   t=2: next_value=0, nnt=0(done), delta=1-0=1, gae=1 (截断, 不累积)
#   t=1: next_value=0, nnt=1, delta=1, gae=1+1*1*1=2
#   t=0: next_value=0, nnt=1, delta=1, gae=1+1*1*1*2=3
# ---------------------------------------------------------------------------
def test_gae_episodic_zero_values():
    rewards = _to_2d([1, 1, 1, 1])
    values = _to_2d([0, 0, 0, 0])
    dones = _to_2d([0, 0, 1, 0])
    last_value = np.array([0], dtype=np.float32)
    gamma, gae_lambda = 1.0, 1.0

    advantages, returns = compute_gae(rewards, values, last_value, dones, gamma, gae_lambda)

    expected_adv = _to_2d([3, 2, 1, 1])
    expected_ret = _to_2d([3, 2, 1, 1])
    np.testing.assert_allclose(advantages, expected_adv, rtol=1e-6,
                               err_msg=f"advantages: {advantages.flatten()}")
    np.testing.assert_allclose(returns, expected_ret, rtol=1e-6,
                               err_msg=f"returns: {returns.flatten()}")


def test_gae_episodic_done_truncates_bootstrap():
    """done=1 处 next_non_terminal=0: 不 bootstrap next_value, 不累积后续 GAE.

    dones=[0,0,1,0]: t=2 处 done, 后续 t=3 的大 value 不应影响 t<=2 的 GAE (截断).
    last_value=0 (情景制), t=3 因 dones[3]=0 仍正常 bootstrap last_value=0.
    """
    rewards = _to_2d([1, 1, 1, 1])
    values = _to_2d([0, 0, 0, 100])  # values[3] 故意设大, 验证不污染 done 步
    dones = _to_2d([0, 0, 1, 0])
    last_value = np.array([0], dtype=np.float32)
    gamma, gae_lambda = 1.0, 1.0

    advantages, returns = compute_gae(rewards, values, last_value, dones, gamma, gae_lambda)

    # t=2 (done): 截断, advantage = r2 - v2 = 1, 不受 values[3]=100 影响
    np.testing.assert_allclose(advantages[2], [1], rtol=1e-6,
                                err_msg=f"adv[2] should ignore values[3]: {advantages.flatten()}")
    np.testing.assert_allclose(returns[2], [1], rtol=1e-6)
    # 截断后 t<=1 的 GAE 仅累积 episode 内 (t<=2) 的 reward
    np.testing.assert_allclose(returns[0], [3], rtol=1e-6)
    np.testing.assert_allclose(returns[1], [2], rtol=1e-6)


def test_gae_episodic_nonzero_values_returns_independent_of_values():
    """gamma=lambda=1 时, returns 仅依赖 rewards (episode 内累计), 与 values 无关."""
    rewards = _to_2d([1, 1, 1, 1])
    values = _to_2d([1, 2, 3, 4])
    dones = _to_2d([0, 0, 1, 0])
    last_value = np.array([0], dtype=np.float32)
    gamma, gae_lambda = 1.0, 1.0

    advantages, returns = compute_gae(rewards, values, last_value, dones, gamma, gae_lambda)

    np.testing.assert_allclose(returns, _to_2d([3, 2, 1, 1]), rtol=1e-6,
                               err_msg=f"returns should be independent of values: {returns.flatten()}")


# ---------------------------------------------------------------------------
# 非情景制 (non-episodic): dones=[0,0,0,0], last_value=[1]
# gamma=1, lambda=1, rewards=[1,1,1,1], values=[0,0,0,0]
# 期望 advantages = [5, 4, 3, 2], returns = [5, 4, 3, 2]
# 推导:
#   t=3: next_value=last_value=1, nnt=1, delta=1+1*1-0=2, gae=2
#   t=2: next_value=0, nnt=1, delta=1, gae=1+1*1*1*2=3
#   t=1: next_value=0, nnt=1, delta=1, gae=1+1*1*1*3=4
#   t=0: next_value=0, nnt=1, delta=1, gae=1+1*1*1*4=5
# ---------------------------------------------------------------------------
def test_gae_non_episodic_uses_last_value_bootstrap():
    rewards = _to_2d([1, 1, 1, 1])
    values = _to_2d([0, 0, 0, 0])
    dones = _to_2d([0, 0, 0, 0])
    last_value = np.array([1], dtype=np.float32)
    gamma, gae_lambda = 1.0, 1.0

    advantages, returns = compute_gae(rewards, values, last_value, dones, gamma, gae_lambda)

    expected_adv = _to_2d([5, 4, 3, 2])
    np.testing.assert_allclose(advantages, expected_adv, rtol=1e-6,
                               err_msg=f"advantages: {advantages.flatten()}")
    np.testing.assert_allclose(returns, expected_adv, rtol=1e-6,
                               err_msg=f"returns: {returns.flatten()}")


def test_gae_non_episodic_last_value_zero_matches_montecarlo():
    """非情景制 + last_value=0 + gamma=lambda=1 => returns = 累计未来 reward."""
    rewards = _to_2d([1, 2, 3, 4])
    values = _to_2d([0, 0, 0, 0])
    dones = _to_2d([0, 0, 0, 0])
    last_value = np.array([0], dtype=np.float32)
    gamma, gae_lambda = 1.0, 1.0

    _, returns = compute_gae(rewards, values, last_value, dones, gamma, gae_lambda)

    # returns[t] = sum_{k>=t} r_k
    expected = _to_2d([1 + 2 + 3 + 4, 2 + 3 + 4, 3 + 4, 4])
    np.testing.assert_allclose(returns, expected, rtol=1e-6)


def test_gae_returns_equals_advantages_plus_values():
    """不变式: returns = advantages + values."""
    rewards = _to_2d([1.0, 0.5, 2.0, 0.3])
    values = _to_2d([0.4, 0.7, 0.1, 0.9])
    dones = _to_2d([0, 0, 1, 0])
    last_value = np.array([0.5], dtype=np.float32)
    gamma, gae_lambda = 0.99, 0.95

    advantages, returns = compute_gae(rewards, values, last_value, dones, gamma, gae_lambda)

    np.testing.assert_allclose(returns, advantages + values, rtol=1e-6)


def test_gae_output_shapes():
    n_steps, n_envs = 16, 4
    rewards = np.random.rand(n_steps, n_envs).astype(np.float32)
    values = np.random.rand(n_steps, n_envs).astype(np.float32)
    dones = np.zeros((n_steps, n_envs), dtype=np.float32)
    last_value = np.zeros(n_envs, dtype=np.float32)

    advantages, returns = compute_gae(rewards, values, last_value, dones, 0.99, 0.95)

    assert advantages.shape == (n_steps, n_envs), f"advantages shape {advantages.shape}"
    assert returns.shape == (n_steps, n_envs), f"returns shape {returns.shape}"


def test_gae_multi_env():
    """多环境并行计算 GAE, 每个环境独立."""
    n_envs = 3
    rewards = np.tile(_to_2d([1, 1, 1, 1]), (1, n_envs))  # (4, 3) 全 1
    values = np.zeros((4, n_envs), dtype=np.float32)
    dones = _to_2d([0, 0, 1, 0]) * np.ones((1, n_envs))  # 每个环境同样 dones
    dones = dones.astype(np.float32)
    last_value = np.zeros(n_envs, dtype=np.float32)
    gamma, gae_lambda = 1.0, 1.0

    advantages, returns = compute_gae(rewards, values, last_value, dones, gamma, gae_lambda)

    expected = _to_2d([3, 2, 1, 1]) * np.ones((1, n_envs))
    np.testing.assert_allclose(advantages, expected, rtol=1e-6)
    np.testing.assert_allclose(returns, expected, rtol=1e-6)
