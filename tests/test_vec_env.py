"""测试向量化环境：DummyVecEnv 与 SubprocVecEnv。"""
import os
import sys
from functools import partial

import numpy as np
import pytest
import gymnasium
from gymnasium import spaces

from curiosity_ppo.envs.vec_env import DummyVecEnv, SubprocVecEnv


# =========================================================================
# 用于 DummyVecEnv 的 mock 环境（进程内，无需 pickle）
# =========================================================================


class _CounterEnv(gymnasium.Env):
    """步数计数环境：每步 reward=1，在 episode_len 步后 done。"""

    def __init__(self, episode_len=5, obs_shape=(3,)):
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(2)
        self.episode_len = episode_len
        self._steps = 0

    def reset(self, **kwargs):
        self._steps = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        self._steps += 1
        obs = np.full(self.observation_space.shape, self._steps, dtype=np.uint8)
        reward = 1.0
        terminated = self._steps >= self.episode_len
        return obs, reward, terminated, False, {"steps": self._steps}


# =========================================================================
# DummyVecEnv
# =========================================================================


def test_dummy_vec_env_reset_shape():
    env = DummyVecEnv([lambda: _CounterEnv() for _ in range(4)])
    obs = env.reset()
    assert obs.shape == (4, 3)
    assert env.num_envs == 4


def test_dummy_vec_env_step_shapes():
    env = DummyVecEnv([lambda: _CounterEnv(episode_len=100) for _ in range(3)])
    env.reset()
    obs, rewards, dones, infos = env.step([0, 1, 0])
    assert obs.shape == (3, 3)
    assert rewards.shape == (3,)
    assert dones.shape == (3,)
    assert len(infos) == 3
    assert rewards.dtype == np.float32


def test_dummy_vec_env_auto_reset_on_done():
    """环境 done 后自动 reset，并将终止观测存入 terminal_observation。"""
    env = DummyVecEnv([lambda: _CounterEnv(episode_len=2)])
    env.reset()
    # 第 1 步：未 done
    obs1, r1, d1, info1 = env.step([0])
    assert not d1[0]
    # 第 2 步：done → 自动 reset，返回的 obs 是新 episode 的 obs (全 0)
    obs2, r2, d2, info2 = env.step([0])
    assert d2[0]
    assert "terminal_observation" in info2[0]
    # terminal_observation 是 done 时的观测 (步数=2)
    np.testing.assert_array_equal(info2[0]["terminal_observation"], np.array([2, 2, 2], dtype=np.uint8))
    # 返回的 obs 是 reset 后的 (全 0)
    np.testing.assert_array_equal(obs2[0], np.array([0, 0, 0], dtype=np.uint8))


def test_dummy_vec_env_spaces_accessible():
    env = DummyVecEnv([lambda: _CounterEnv()])
    assert env.observation_space.shape == (3,)
    assert isinstance(env.action_space, spaces.Discrete)


# =========================================================================
# SubprocVecEnv —— 使用 gymnasium 内置 CartPole（无需额外依赖）
# =========================================================================
#
# SubprocVecEnv 在 Windows 上使用 spawn 启动子进程，子进程需要能 import
# curiosity_ppo (src 目录) 以及本测试模块。这里通过设置 PYTHONPATH 保证。


def _make_cartpole(seed=0):
    """顶层工厂函数，可被 pickle 引用。"""
    env = gymnasium.make("CartPole-v1")
    env.reset(seed=seed)
    return env


@pytest.fixture
def _subproc_paths():
    """为子进程设置 PYTHONPATH，使其能 import curiosity_ppo 与测试模块。"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_dir = os.path.join(project_root, "src")
    tests_dir = os.path.join(project_root, "tests")
    extra = [src_dir, tests_dir]
    old_pp = os.environ.get("PYTHONPATH", "")
    parts = [p for p in old_pp.split(os.pathsep) if p] + extra
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    yield
    os.environ["PYTHONPATH"] = old_pp


def test_subproc_vec_env_reset_and_step(_subproc_paths):
    n = 2
    env = SubprocVecEnv([partial(_make_cartpole, seed=i) for i in range(n)])
    try:
        obs = env.reset()
        assert obs.shape == (n, 4)
        obs, rewards, dones, infos = env.step([0, 1])
        assert obs.shape == (n, 4)
        assert rewards.shape == (n,)
        assert dones.shape == (n,)
        assert len(infos) == n
    finally:
        env.close()


def test_subproc_vec_env_auto_reset(_subproc_paths):
    """CartPole 终止后自动 reset。"""
    env = SubprocVecEnv([partial(_make_cartpole, seed=0)])
    try:
        env.reset()
        done_seen = False
        for _ in range(500):
            obs, rewards, dones, infos = env.step([1])
            if dones[0]:
                done_seen = True
                assert "terminal_observation" in infos[0]
                # reset 后的 obs 形状正确
                assert obs.shape == (1, 4)
                break
        assert done_seen, "CartPole 应在有限步内 done"
    finally:
        env.close()
