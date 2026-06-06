"""测试观测预处理 wrapper：ObsToFloat32 / RunningMeanStd / ObsNormalize / FrameStack / GrayResizeObservation。"""
import collections

import numpy as np
import pytest
import gymnasium
from gymnasium import spaces

from curiosity_ppo.envs.wrappers import (
    ObsToFloat32,
    RunningMeanStd,
    ObsNormalize,
    FrameStack,
    GrayResizeObservation,
)


class _MockEnv(gymnasium.Env):
    """可配置的 mock 环境，支持固定观测或随机采样。"""

    def __init__(self, obs_space, action_space=None, fixed_obs=None):
        self.observation_space = obs_space
        self.action_space = action_space or spaces.Discrete(2)
        self._fixed_obs = fixed_obs

    def _get_obs(self):
        if self._fixed_obs is not None:
            return self._fixed_obs
        return self.observation_space.sample()

    def reset(self, **kwargs):
        return self._get_obs(), {}

    def step(self, action):
        return self._get_obs(), 1.0, False, False, {}


# =========================================================================
# ObsToFloat32
# =========================================================================

def test_obs_to_float32_dtype_and_range():
    obs_space = spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8)
    env = ObsToFloat32(_MockEnv(obs_space))
    obs, info = env.reset()
    assert obs.dtype == np.float32
    assert obs.max() <= 1.0
    assert obs.min() >= 0.0


def test_obs_to_float32_divide_by_255():
    fixed = np.full((8, 8, 3), 255, dtype=np.uint8)
    obs_space = spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8)
    env = ObsToFloat32(_MockEnv(obs_space, fixed_obs=fixed))
    obs, info = env.reset()
    np.testing.assert_allclose(obs, np.ones((8, 8, 3), dtype=np.float32))


def test_obs_to_float32_observation_space_updated():
    obs_space = spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8)
    env = ObsToFloat32(_MockEnv(obs_space))
    assert env.observation_space.dtype == np.float32
    assert env.observation_space.shape == (8, 8, 3)


def test_obs_to_float32_step():
    obs_space = spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8)
    env = ObsToFloat32(_MockEnv(obs_space))
    env.reset()
    obs, reward, term, trunc, info = env.step(0)
    assert obs.dtype == np.float32
    assert obs.shape == (8, 8, 3)


# =========================================================================
# RunningMeanStd (Welford 在线算法)
# =========================================================================

def test_running_mean_std_basic():
    rms = RunningMeanStd(shape=())
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rms.update(data)
    assert rms.count == 5
    np.testing.assert_allclose(rms.mean, 3.0, rtol=1e-6)
    # 总体方差 = 2.0 → std = sqrt(2)
    np.testing.assert_allclose(rms.std, np.sqrt(2.0), rtol=1e-6)


def test_running_mean_std_vector_shape():
    rms = RunningMeanStd(shape=(2,))
    data = np.array([[0.0, 0.0], [2.0, 4.0]])
    rms.update(data)
    np.testing.assert_allclose(rms.mean, [1.0, 2.0], rtol=1e-6)
    # 总体 std: dim0 -> 1.0, dim1 -> 2.0
    np.testing.assert_allclose(rms.std, [1.0, 2.0], rtol=1e-6)
    assert rms.count == 2


def test_running_mean_std_incremental_matches_batch():
    """分批增量更新应与一次性批量更新结果一致 (Chan 并行 Welford)。"""
    rng = np.random.RandomState(0)
    data = rng.rand(100, 3)

    rms_batch = RunningMeanStd(shape=(3,))
    rms_batch.update(data)

    rms_inc = RunningMeanStd(shape=(3,))
    rms_inc.update(data[:30])
    rms_inc.update(data[30:70])
    rms_inc.update(data[70:])

    np.testing.assert_allclose(rms_inc.mean, rms_batch.mean, rtol=1e-6)
    np.testing.assert_allclose(rms_inc.std, rms_batch.std, rtol=1e-6)
    assert rms_inc.count == rms_batch.count == 100


def test_running_mean_std_single_sample():
    """单样本更新（自动补 batch 维）。"""
    rms = RunningMeanStd(shape=(2,))
    rms.update(np.array([3.0, 6.0]))
    np.testing.assert_allclose(rms.mean, [3.0, 6.0], rtol=1e-6)
    assert rms.count == 1


# =========================================================================
# ObsNormalize
# =========================================================================

def test_obs_normalize_output_shape_and_dtype():
    obs_space = spaces.Box(low=0, high=255, shape=(4,), dtype=np.uint8)
    env = ObsNormalize(_MockEnv(obs_space))
    obs, info = env.reset()
    assert obs.shape == (4,)
    assert obs.dtype == np.float32


def test_obs_normalize_updates_statistics():
    """经过多次 step 后 rms 应被更新（mean/std 非零）。"""
    fixed = np.array([100, 200, 50, 250], dtype=np.uint8)
    obs_space = spaces.Box(low=0, high=255, shape=(4,), dtype=np.uint8)
    env = ObsNormalize(_MockEnv(obs_space, fixed_obs=fixed))
    env.reset()
    for _ in range(20):
        env.step(0)
    assert env.rms.count > 0
    np.testing.assert_allclose(env.rms.mean, fixed.astype(np.float64), rtol=1e-5)


def test_obs_normalize_normalizes_constant_obs():
    """常量观测归一化后应接近 0。"""
    fixed = np.array([128, 128, 128], dtype=np.uint8)
    obs_space = spaces.Box(low=0, high=255, shape=(3,), dtype=np.uint8)
    env = ObsNormalize(_MockEnv(obs_space, fixed_obs=fixed))
    env.reset()
    for _ in range(50):
        obs, _, _, _, _ = env.step(0)
    # 常量 → 方差趋近 0 → 归一化后接近 0 (受 eps 限制)
    np.testing.assert_allclose(obs, np.zeros(3), atol=1e-3)


# =========================================================================
# FrameStack
# =========================================================================

def test_frame_stack_shape():
    obs_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)
    env = FrameStack(_MockEnv(obs_space), k=4)
    obs, info = env.reset()
    assert obs.shape == (4, 84, 84)


def test_frame_stack_observation_space():
    obs_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)
    env = FrameStack(_MockEnv(obs_space), k=4)
    assert env.observation_space.shape == (4, 84, 84)


def test_frame_stack_reset_clears_and_fills():
    """reset 后应清空旧帧并用首帧填满 k 帧。"""
    obs_space = spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)
    env = FrameStack(_MockEnv(obs_space), k=4)
    env.reset()
    env.step(0)
    env.step(0)
    obs, info = env.reset()
    assert obs.shape == (4, 84, 84)
    # reset 填充：4 帧应相同
    np.testing.assert_array_equal(obs[0], obs[1])
    np.testing.assert_array_equal(obs[1], obs[2])
    np.testing.assert_array_equal(obs[2], obs[3])


def test_frame_stack_shifts_on_step():
    """连续 step 后，栈中帧应发生移位（最新帧进入栈顶）。"""
    # 使用脚本化的递增观测
    class CounterEnv(gymnasium.Env):
        def __init__(self):
            self.observation_space = spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)
            self.action_space = spaces.Discrete(2)
            self._i = 0

        def reset(self, **kwargs):
            self._i = 0
            return np.array([self._i], dtype=np.uint8), {}

        def step(self, action):
            self._i += 1
            return np.array([self._i], dtype=np.uint8), 0.0, False, False, {}

    env = FrameStack(CounterEnv(), k=3)
    obs, _ = env.reset()  # [0, 0, 0]
    np.testing.assert_array_equal(obs[:, 0], [0, 0, 0])
    obs, *_ = env.step(0)  # [0, 0, 1]
    np.testing.assert_array_equal(obs[:, 0], [0, 0, 1])
    obs, *_ = env.step(0)  # [0, 1, 2]
    np.testing.assert_array_equal(obs[:, 0], [0, 1, 2])
    obs, *_ = env.step(0)  # [1, 2, 3]
    np.testing.assert_array_equal(obs[:, 0], [1, 2, 3])


# =========================================================================
# GrayResizeObservation
# =========================================================================

def test_gray_resize_shape():
    obs_space = spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)
    env = GrayResizeObservation(_MockEnv(obs_space), size=84)
    obs, info = env.reset()
    assert obs.shape == (84, 84)


def test_gray_resize_grayscale():
    obs_space = spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)
    env = GrayResizeObservation(_MockEnv(obs_space), size=84)
    obs, info = env.reset()
    assert obs.ndim == 2  # 灰度图


def test_gray_resize_step():
    obs_space = spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)
    env = GrayResizeObservation(_MockEnv(obs_space), size=84)
    env.reset()
    obs, reward, term, trunc, info = env.step(0)
    assert obs.shape == (84, 84)


def test_gray_resize_observation_space_updated():
    obs_space = spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)
    env = GrayResizeObservation(_MockEnv(obs_space), size=84)
    assert env.observation_space.shape == (84, 84)


def test_gray_resize_preserves_grayscale_input():
    """输入已是灰度 (H, W) 时仍能正确缩放。"""
    obs_space = spaces.Box(low=0, high=255, shape=(100, 100), dtype=np.uint8)
    env = GrayResizeObservation(_MockEnv(obs_space), size=84)
    obs, info = env.reset()
    assert obs.shape == (84, 84)
    assert obs.ndim == 2
