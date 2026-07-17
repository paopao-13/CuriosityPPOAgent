"""Crafter 环境工厂。

链路: gym.make('CrafterReward-v1') → GymCompatWrapper → ObsToFloat32 → DummyVecEnv
"""
from curiosity_ppo.envs.compat import GymCompatWrapper
from curiosity_ppo.envs.wrappers import ObsToFloat32
from curiosity_ppo.envs.vec_env import DummyVecEnv, SubprocVecEnv

__all__ = ["make_crafter_env"]


def _seed_env(env, seed):
    """兼容地为环境设置种子（gymnasium / 旧 gym）。"""
    try:
        env.reset(seed=seed)
    except TypeError:
        # 旧 gym 的 reset 不接受 seed 关键字
        try:
            env.seed(seed)
            env.reset()
        except Exception:
            pass


def _build_single_crafter(seed, rank):
    """构建单个 Crafter 子环境 (spawn-safe: 顶层函数 + 显式参数)。"""
    import crafter  # noqa: F401

    # 直接创建 Crafter Env, 绕过 gym.make 的 TimeLimit/OrderEnforcing 等
    # wrapper — 这些 wrapper 期望新版 5-value step API, 但 Crafter 返回 4-value
    env = crafter.Env(reward=True)
    env = GymCompatWrapper(env)
    env = ObsToFloat32(env)
    _seed_env(env, seed + rank)
    return env


def make_crafter_env(n_envs=8, seed=42, vec_env_type="dummy"):
    """创建 Crafter 向量化环境。

    Args:
        n_envs: 并行环境数量。
        seed: 基础随机种子，各子环境使用 seed + rank。
        vec_env_type: "dummy"(串行) 或 "subproc"(多进程并行, 单卡提速)。

    Returns:
        VecEnv: 观测为 float32 (已归一化到 [0, 1]) 的向量化环境。
    """
    fns = [(_build_single_crafter, (seed, i), {}) for i in range(n_envs)]
    if vec_env_type == "subproc":
        return SubprocVecEnv(fns)
    return DummyVecEnv([(lambda i=i: _build_single_crafter(seed, i)) for i in range(n_envs)])
