"""MiniGrid 环境工厂。

链路: gymnasium.make → ImgObsWrapper → ResizeObs(64×64) → ObsToFloat32 → DummyVecEnv
"""
import numpy as np
import gymnasium
from curiosity_ppo.envs.wrappers import ObsToFloat32
from curiosity_ppo.envs.vec_env import DummyVecEnv, SubprocVecEnv

__all__ = ["make_minigrid_env"]


class ResizeObs(gymnasium.ObservationWrapper):
    """将观测缩放到指定尺寸 (默认 64×64×3)，兼容 MiniGrid 7×7 小观测"""

    def __init__(self, env, target_size=(64, 64)):
        super().__init__(env)
        self.target_size = target_size
        old_space = env.observation_space
        if len(old_space.shape) == 3:
            h, w, c = old_space.shape
        else:
            h, w = old_space.shape[:2]
            c = 1
        new_shape = (target_size[0], target_size[1], c)
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, obs):
        obs = np.asarray(obs, dtype=np.uint8)
        if obs.ndim == 2:
            obs = obs[:, :, np.newaxis]
        try:
            import cv2
            resized = cv2.resize(obs, (self.target_size[1], self.target_size[0]),
                                 interpolation=cv2.INTER_NEAREST)
        except ImportError:
            # numpy 最近邻缩放回退
            h, w = obs.shape[:2]
            th, tw = self.target_size
            ry = max(1, th // h)
            rx = max(1, tw // w)
            resized = np.repeat(np.repeat(obs, ry, axis=0), rx, axis=1)
            resized = resized[:th, :tw]
        return resized.astype(np.uint8)


def _build_single_minigrid(env_id, seed, rank):
    """构建单个 MiniGrid 子环境 (spawn-safe: 顶层函数 + 显式参数)。"""
    from minigrid.wrappers import ImgObsWrapper

    env = gymnasium.make(env_id)
    env = ImgObsWrapper(env)
    env = ResizeObs(env, target_size=(64, 64))
    env = ObsToFloat32(env)
    env.reset(seed=seed + rank)
    return env


def make_minigrid_env(env_id="MiniGrid-DoorKey-16x16-v0", n_envs=8, seed=42, vec_env_type="dummy"):
    """创建 MiniGrid 向量化环境。

    Args:
        env_id: MiniGrid 环境 ID。
        n_envs: 并行环境数量。
        seed: 基础随机种子，各子环境使用 seed + rank。
        vec_env_type: "dummy"(串行) 或 "subproc"(多进程并行, 单卡提速)。

    Returns:
        VecEnv: 观测为 float32 (已归一化到 [0, 1]) 的向量化环境。
    """
    fns = [(_build_single_minigrid, (env_id, seed, i), {}) for i in range(n_envs)]
    if vec_env_type == "subproc":
        return SubprocVecEnv(fns)
    return DummyVecEnv([(lambda i=i: _build_single_minigrid(env_id, seed, i)) for i in range(n_envs)])
