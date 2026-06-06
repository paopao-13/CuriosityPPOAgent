"""测试 GymCompatWrapper：旧 gym (4值step / reset返回obs) → gymnasium (5值step / reset返回tuple)。"""
import numpy as np
import pytest
import gymnasium
from gymnasium import spaces

from curiosity_ppo.envs.compat import GymCompatWrapper


class MockOldGymEnv(gymnasium.Env):
    """模拟旧版 gym API：reset 只返回 obs，step 返回 4 元组 (obs, reward, done, info)。"""

    def __init__(self, obs_shape=(4, 4, 3)):
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(2)
        self._step_count = 0

    def reset(self, **kwargs):
        # 旧 gym 风格：仅返回 obs
        return self.observation_space.sample()

    def step(self, action):
        # 旧 gym 风格：4 元组
        self._step_count += 1
        obs = self.observation_space.sample()
        return obs, 1.0, False, {"step": self._step_count}


class MockNewGymEnv(gymnasium.Env):
    """模拟新版 gymnasium API：reset 返回 (obs, info)，step 返回 5 元组。"""

    def __init__(self, obs_shape=(4, 4, 3)):
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(2)

    def reset(self, **kwargs):
        return self.observation_space.sample(), {"new": True}

    def step(self, action):
        return self.observation_space.sample(), 0.5, False, False, {"info": 1}


class MockDoneOldGymEnv(gymnasium.Env):
    """旧 gym，step 立即返回 done=True。"""

    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, **kwargs):
        return np.zeros(2, dtype=np.float32)

    def step(self, action):
        return np.ones(2, dtype=np.float32), 2.0, True, {}


# ---- reset 转换 ----

def test_compat_reset_old_gym_returns_tuple():
    """旧 gym reset 返回单个 obs → 包装后返回 (obs, info) 元组。"""
    env = GymCompatWrapper(MockOldGymEnv())
    result = env.reset()
    assert isinstance(result, tuple)
    assert len(result) == 2
    obs, info = result
    assert obs.shape == (4, 4, 3)
    assert isinstance(info, dict)


def test_compat_reset_new_gym_passthrough():
    """新版 gymnasium reset 已是 tuple → 原样透传。"""
    env = GymCompatWrapper(MockNewGymEnv())
    obs, info = env.reset()
    assert info == {"new": True}
    assert obs.shape == (4, 4, 3)


# ---- step 转换 ----

def test_compat_step_old_gym_returns_5_tuple():
    """旧 gym 4 元组 step → 包装后返回 5 元组 (obs, reward, terminated, truncated, info)。"""
    env = GymCompatWrapper(MockOldGymEnv())
    env.reset()
    result = env.step(0)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (4, 4, 3)
    assert reward == 1.0
    assert terminated is False
    assert truncated is False
    assert info["step"] == 1


def test_compat_step_new_gym_passthrough():
    """新版 gymnasium 5 元组 step → 原样透传。"""
    env = GymCompatWrapper(MockNewGymEnv())
    env.reset()
    result = env.step(0)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert reward == 0.5
    assert info == {"info": 1}


def test_compat_step_done_old_gym():
    """旧 gym done=True → terminated=True, truncated=False。"""
    env = GymCompatWrapper(MockDoneOldGymEnv())
    env.reset()
    result = env.step(0)
    obs, reward, terminated, truncated, info = result
    assert terminated is True
    assert truncated is False
    assert reward == 2.0


def test_compat_preserves_spaces():
    """包装后 observation_space / action_space 仍可访问。"""
    base = MockOldGymEnv()
    env = GymCompatWrapper(base)
    assert env.observation_space.shape == (4, 4, 3)
    assert env.action_space == base.action_space
