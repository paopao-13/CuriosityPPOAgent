"""MiniGrid 环境工厂。

链路: gymnasium.make → ImgObsWrapper → ResizeObs(64×64) → ObsToFloat32 → DummyVecEnv
"""
import numpy as np
import gymnasium
from curiosity_ppo.envs.wrappers import ObsToFloat32
from curiosity_ppo.envs.vec_env import DummyVecEnv, SubprocVecEnv

__all__ = ["make_minigrid_env"]


class RewardShapingWrapper(gymnasium.RewardWrapper):
    """MiniGrid 轻量奖励塑形: 拿钥匙 / 开门给小奖励，使外部信号能进入策略梯度。

    仅用于稀疏外部奖励任务(DoorKey 等只有终点 +1)。纯好奇心任务(Atari)不启用。
    通过 env.unwrapped 读取内部状态，全部带防御式检查，缺属性则退化为无塑形。
    """

    def __init__(self, env, pickup_reward=0.3, door_reward=0.5, goal_reward=1.0):
        super().__init__(env)
        self.pickup_reward = pickup_reward
        self.door_reward = door_reward
        self.goal_reward = goal_reward
        self._carrying_prev = False
        self._door_open_prev = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._carrying_prev = False
        self._door_open_prev = False
        return obs, info

    def reward(self, reward):
        uw = self.env.unwrapped
        shaped = 0.0
        # 拿钥匙
        try:
            carrying = getattr(uw, "carrying", None) is not None
        except Exception:
            carrying = False
        if carrying and not self._carrying_prev:
            shaped += self.pickup_reward
        self._carrying_prev = carrying
        # 开门
        try:
            door_open = self._any_door_open(uw)
        except Exception:
            door_open = False
        if door_open and not self._door_open_prev:
            shaped += self.door_reward
        self._door_open_prev = door_open
        # 终点(原环境 reward>0 视为到达)
        if reward and reward > 0:
            shaped += self.goal_reward
        return reward + shaped

    @staticmethod
    def _any_door_open(uw):
        grid = getattr(uw, "grid", None)
        if grid is None:
            return False
        w, h = getattr(grid, "width", 0), getattr(grid, "height", 0)
        for i in range(w):
            for j in range(h):
                try:
                    cell = grid.get(i, j)
                except Exception:
                    cell = None
                if cell is not None and type(cell).__name__ == "Door":
                    if getattr(cell, "is_open", False):
                        return True
        return False


class FixedLayoutWrapper(gymnasium.Wrapper):
    """课程学习 phase1：固定单个布局(seed)，使 agent 先学会通用解法再泛化。

    重写 reset 始终使用固定 seed，覆盖训练循环中的随机 reset，
    从而所有 episode 共享同一(或每 rank 固定)布局。
    """

    def __init__(self, env, fixed_seed):
        super().__init__(env)
        self.fixed_seed = fixed_seed

    def reset(self, **kwargs):
        return self.env.reset(seed=self.fixed_seed)


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


def _build_single_minigrid(env_id, seed, rank, reward_shaping=False, fixed_layout_seed=None):
    """构建单个 MiniGrid 子环境 (spawn-safe: 顶层函数 + 显式参数)。"""
    from minigrid.wrappers import ImgObsWrapper

    env = gymnasium.make(env_id)
    env = ImgObsWrapper(env)
    env = ResizeObs(env, target_size=(64, 64))
    env = ObsToFloat32(env)
    if reward_shaping:
        env = RewardShapingWrapper(env)
    if fixed_layout_seed is not None:
        env = FixedLayoutWrapper(env, fixed_layout_seed + rank)
    env.reset(seed=seed + rank)
    return env


def make_minigrid_env(env_id="MiniGrid-DoorKey-16x16-v0", n_envs=8, seed=42,
                      vec_env_type="dummy", reward_shaping=False, fixed_layout_seed=None):
    """创建 MiniGrid 向量化环境。

    Args:
        env_id: MiniGrid 环境 ID。
        n_envs: 并行环境数量。
        seed: 基础随机种子，各子环境使用 seed + rank。
        vec_env_type: "dummy"(串行) 或 "subproc"(多进程并行, 单卡提速)。
        reward_shaping: 是否启用轻量奖励塑形(拿钥匙/开门给小奖励)。
        fixed_layout_seed: 非 None 时固定布局(课程 phase1), 每 rank 用 seed+rank 的固定布局。

    Returns:
        VecEnv: 观测为 float32 (已归一化到 [0, 1]) 的向量化环境。
    """
    fns = [(_build_single_minigrid, (env_id, seed, i, reward_shaping, fixed_layout_seed), {}) for i in range(n_envs)]
    if vec_env_type == "subproc":
        return SubprocVecEnv(fns)
    return DummyVecEnv([(lambda i=i: _build_single_minigrid(env_id, seed, i, reward_shaping, fixed_layout_seed)) for i in range(n_envs)])
