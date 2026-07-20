"""MiniGrid 环境工厂。

链路: gymnasium.make → ImgObsWrapper → ResizeObs(64×64) → ObsToFloat32 → DummyVecEnv
"""
import numpy as np
import gymnasium
from curiosity_ppo.envs.wrappers import ObsToFloat32
from curiosity_ppo.envs.vec_env import DummyVecEnv, SubprocVecEnv

__all__ = ["make_minigrid_env"]


class RewardShapingWrapper(gymnasium.RewardWrapper):
    """MiniGrid 势能塑形(potential-based shaping): 以到当前子目标的曼哈顿距离定义潜能
    Φ(s)，每步奖励 = γ·Φ(s') − Φ(s)，给出连续稠密引导，使外部信号能稳定进入策略梯度。

    子目标随完成度递进: 未拿钥匙 → 朝钥匙; 拿了钥匙未开门 → 朝门; 开门后 → 朝终点。
    终点原始 +1 奖励保留叠加。仅用于稀疏外部奖励任务(DoorKey 等)，纯好奇心(Atari)不启用。
    全部经 env.unwrapped 读取内部状态并带防御式检查，缺属性则退化为无塑形。
    """

    def __init__(self, env, gamma=0.99, scale=0.1, goal_reward=1.0):
        super().__init__(env)
        self.gamma = gamma
        self.scale = scale          # 势能差放大系数
        self.goal_reward = goal_reward
        self._prev_potential = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_potential = self._potential()
        return obs, info

    def reward(self, reward):
        cur = self._potential()
        if self._prev_potential is None:
            self._prev_potential = cur
        shaped = self.gamma * cur - self._prev_potential
        self._prev_potential = cur
        if reward and reward > 0:
            shaped += self.goal_reward
        return reward + shaped

    def _agent_pos(self, uw):
        try:
            return tuple(int(v) for v in uw.agent_pos)
        except Exception:
            return None

    def _find_obj(self, uw, cls_name):
        """在 grid 中找第一个指定类名的对象坐标, 返回 (x, y) 或 None。"""
        grid = getattr(uw, "grid", None)
        if grid is None:
            return None
        w = getattr(grid, "width", 0)
        h = getattr(grid, "height", 0)
        for i in range(w):
            for j in range(h):
                try:
                    cell = grid.get(i, j)
                except Exception:
                    cell = None
                if cell is not None and type(cell).__name__ == cls_name:
                    return (i, j)
        return None

    def _any_door_open(self, uw):
        grid = getattr(uw, "grid", None)
        if grid is None:
            return False
        w = getattr(grid, "width", 0)
        h = getattr(grid, "height", 0)
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

    def _potential(self):
        """返回当前潜能(负值: 越接近当前子目标潜能越高)。防御式返回 0.0。"""
        uw = self.env.unwrapped
        agent = self._agent_pos(uw)
        if agent is None:
            return 0.0
        try:
            carrying = getattr(uw, "carrying", None) is not None
        except Exception:
            carrying = False
        # 子目标递进: 未拿钥匙→钥匙; 拿了未开门→门; 否则→终点
        if not carrying:
            target = self._find_obj(uw, "Key")
        else:
            door_open = False
            try:
                door_open = self._any_door_open(uw)
            except Exception:
                door_open = False
            if not door_open:
                target = self._find_obj(uw, "Door")
            else:
                target = self._find_obj(uw, "Goal")
        if target is None:
            return 0.0
        dist = abs(agent[0] - target[0]) + abs(agent[1] - target[1])
        # 势能 = -距离 (越近越高), 乘以 scale 缩放
        return -float(dist) * self.scale


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
