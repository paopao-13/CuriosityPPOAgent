"""MiniGrid 环境工厂。

链路: gymnasium.make → ImgObsWrapper → ObsToFloat32 → DummyVecEnv
"""
from curiosity_ppo.envs.wrappers import ObsToFloat32
from curiosity_ppo.envs.vec_env import DummyVecEnv

__all__ = ["make_minigrid_env"]


def make_minigrid_env(env_id="MiniGrid-DoorKey-16x16-v0", n_envs=8, seed=42):
    """创建 MiniGrid 向量化环境。

    Args:
        env_id: MiniGrid 环境 ID。
        n_envs: 并行环境数量。
        seed: 基础随机种子，各子环境使用 seed + rank。

    Returns:
        DummyVecEnv: 观测为 float32 (已归一化到 [0, 1]) 的向量化环境。
    """
    def make_env(rank):
        def _thunk():
            import gymnasium
            from minigrid.wrappers import ImgObsWrapper

            env = gymnasium.make(env_id)
            env = ImgObsWrapper(env)
            env = ObsToFloat32(env)
            env.reset(seed=seed + rank)
            return env

        return _thunk

    return DummyVecEnv([make_env(i) for i in range(n_envs)])
