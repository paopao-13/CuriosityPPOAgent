"""Atari 环境工厂及 Atari 专用 wrapper。

链路: gymnasium.make → NoopReset(30) → MaxAndSkip(4) → EpisodicLife
      → GrayResize(84×84) → FrameStack(4) → ClipReward
"""
import numpy as np
import gymnasium
from gymnasium import spaces

from curiosity_ppo.envs.wrappers import FrameStack, GrayResizeObservation
from curiosity_ppo.envs.vec_env import DummyVecEnv, SubprocVecEnv

__all__ = [
    "make_atari_env",
    "NoopReset",
    "MaxAndSkip",
    "EpisodicLife",
    "ClipReward",
]


# =========================================================================
# Atari 专用 wrapper
# =========================================================================


class NoopReset(gymnasium.Wrapper):
    """reset 时执行随机次数 (1..noop_max) 的 NOOP 动作，增加初始状态多样性。"""

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = int(self.unwrapped.np_random.integers(1, self.noop_max + 1))
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkip(gymnasium.Wrapper):
    """重复执行同一动作 skip 帧，并对最后两帧取 max（消除 Atari 闪烁）。"""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=env.observation_space.dtype
        )

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self.skip - 2:
                self._obs_buffer[0] = obs
            if i == self.skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class EpisodicLife(gymnasium.Wrapper):
    """将生命值损失视为 episode 结束（加速训练），仅在真正 game over 时 reset。"""

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # 生命值损失但游戏未结束：执行 NOOP 而非真正 reset
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class ClipReward(gymnasium.Wrapper):
    """将奖励裁剪为 {-1, 0, 1}，使奖励尺度一致。"""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, np.sign(reward), terminated, truncated, info


# =========================================================================
# 工厂函数
# =========================================================================


def _build_single_atari(env_id, seed, rank):
    """构建单个 Atari 子环境 (spawn-safe: 顶层函数 + 显式参数)。"""
    import ale_py  # noqa: F401  注册 ALE 环境

    env = gymnasium.make(env_id)
    env = NoopReset(env, noop_max=30)
    env = MaxAndSkip(env, skip=4)
    env = EpisodicLife(env)
    env = GrayResizeObservation(env, size=84)
    env = FrameStack(env, k=4)
    env = ClipReward(env)
    env.reset(seed=seed + rank)
    return env


def make_atari_env(env_id="ALE/MontezumaRevenge-v5", n_envs=8, seed=42, vec_env_type="dummy"):
    """创建 Atari 向量化环境。

    Args:
        env_id: gymnasium Atari 环境 ID。
        n_envs: 并行环境数量。
        seed: 基础随机种子，各子环境使用 seed + rank。
        vec_env_type: "dummy"(串行) 或 "subproc"(多进程并行, 单卡提速)。

    Returns:
        VecEnv: 观测为 (4, 84, 84) 堆叠灰度图的向量化环境。
    """
    fns = [(_build_single_atari, (env_id, seed, i), {}) for i in range(n_envs)]
    if vec_env_type == "subproc":
        return SubprocVecEnv(fns)
    # Dummy: 同进程, lambda 闭包无需 pickle
    return DummyVecEnv([(lambda i=i: _build_single_atari(env_id, seed, i)) for i in range(n_envs)])
