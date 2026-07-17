"""观测预处理 wrapper 集合。

包含:
    - ObsToFloat32: uint8 → float32 /255.0
    - RunningMeanStd: Welford 在线算法统计量
    - ObsNormalize: 基于 RunningMeanStd 的观测归一化
    - FrameStack: 堆叠最近 N 帧 (Atari 用)
    - GrayResizeObservation: 灰度化 + 缩放到 84×84
"""
import collections

import numpy as np
import gymnasium
from gymnasium import spaces

try:
    import cv2

    _HAS_CV2 = True
except ImportError:  # opencv 未安装时回退到 numpy 实现
    cv2 = None
    _HAS_CV2 = False

__all__ = [
    "ObsToFloat32",
    "RunningMeanStd",
    "ObsNormalize",
    "FrameStack",
    "GrayResizeObservation",
]


# =========================================================================
# ObsToFloat32
# =========================================================================


class ObsToFloat32(gymnasium.ObservationWrapper):
    """将 uint8 观测转换为 float32 并除以 255.0，归一化到 [0, 1]。"""

    def __init__(self, env):
        super().__init__(env)
        old = env.observation_space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=old.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        observation = np.asarray(observation)
        if observation.dtype != np.float32:
            observation = observation.astype(np.float32)
            if observation.max() > 1.0:
                observation = observation / 255.0
        return observation


# =========================================================================
# RunningMeanStd (Welford 在线算法，支持批量更新)
# =========================================================================


class RunningMeanStd:
    """基于 Welford 算法的在线均值/方差估计（支持批量并行更新）。

    使用 Chan 等人的并行归约公式，对批量数据更新统计量，
    当 batch_size=1 时退化为经典 Welford 算法。
    """

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        # 初始方差设为 1，避免未更新时除零；count=0 时方差无影响
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 0

    def update(self, data):
        """用一批样本更新统计量。

        Args:
            data: ndarray，axis 0 为 batch 维。若 shape 与统计量 shape
                  相同（即单个样本），则自动补 batch 维。
        """
        # float32 累积: 观测归一化每步喂入 (8,84,84,4) 批次，float64 在 16GB 主机内存下会 OOM。
        # 改用 float32 将每步内存减半且对归一化统计量精度无损，不影响量化指标。
        data = np.asarray(data, dtype=np.float32)
        # 若传入单个样本（无 batch 维），自动补维
        if data.shape == self.mean.shape:
            data = data[np.newaxis, ...]
        batch_mean = data.mean(axis=0)
        batch_var = data.var(axis=0)  # 总体方差 (ddof=0)
        batch_count = data.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var)


# =========================================================================
# ObsNormalize
# =========================================================================


class ObsNormalize(gymnasium.ObservationWrapper):
    """基于 RunningMeanStd 的在线观测归一化。

    normalization: (obs - mean) / (sqrt(var) + eps)
    """

    def __init__(self, env, eps=1e-8):
        super().__init__(env)
        self.eps = eps
        self.rms = RunningMeanStd(shape=env.observation_space.shape)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        # 在线更新统计量（补 batch 维）
        self.rms.update(observation)
        normalized = (observation - self.rms.mean) / (self.rms.std + self.eps)
        return normalized.astype(np.float32)


# =========================================================================
# FrameStack
# =========================================================================


class FrameStack(gymnasium.ObservationWrapper):
    """堆叠最近 k 帧观测，使用 collections.deque 实现。

    reset 时清空旧帧并用首帧填满 k 帧；step 时新帧入栈、旧帧出栈。
    主要用于 Atari 等需要时序信息的环境。
    """

    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = collections.deque(maxlen=k)

        old = env.observation_space
        # 栈在最后一维 (HWC), 与项目观测约定 (env 出 HWC, 网络入 CHW) 一致
        new_shape = old.shape + (k,)
        low = np.full(new_shape, float(np.min(old.low)), dtype=old.dtype)
        high = np.full(new_shape, float(np.max(old.high)), dtype=old.dtype)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=new_shape, dtype=old.dtype
        )

    def observation(self, observation):
        self.frames.append(observation)
        # 安全填充：若队列未满（理论上 reset 已填满），用当前帧补齐
        while len(self.frames) < self.k:
            self.frames.appendleft(observation)
        # axis=-1 -> (H, W, k) 即 HWC, 配合 _to_tensor 的 permute(0,3,1,2)
        return np.stack(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        self.frames.clear()
        obs, info = self.env.reset(**kwargs)
        # 用首帧填满 k 帧
        for _ in range(self.k):
            self.frames.append(obs)
        return np.stack(list(self.frames), axis=-1), info


# =========================================================================
# GrayResizeObservation
# =========================================================================


def _numpy_resize(img, target):
    """不依赖 cv2 的最近邻缩放（回退实现）。"""
    th, tw = target
    h, w = img.shape[:2]
    row_idx = (np.arange(th) * h // th).clip(0, h - 1)
    col_idx = (np.arange(tw) * w // tw).clip(0, w - 1)
    return img[row_idx][:, col_idx]


class GrayResizeObservation(gymnasium.ObservationWrapper):
    """将 RGB 观测灰度化并缩放到 (size, size)，优先使用 cv2，回退到 numpy。"""

    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size, size), dtype=np.uint8
        )

    def observation(self, observation):
        obs = np.asarray(observation)

        # 灰度化
        if obs.ndim == 3 and obs.shape[-1] == 3:
            if _HAS_CV2:
                obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            else:
                # ITU-R BT.601 加权
                obs = (
                    0.299 * obs[..., 0]
                    + 0.587 * obs[..., 1]
                    + 0.114 * obs[..., 2]
                ).astype(np.uint8)

        # 缩放到 size × size
        if obs.shape[:2] != (self.size, self.size):
            if _HAS_CV2:
                obs = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
            else:
                obs = _numpy_resize(obs, (self.size, self.size))

        return obs.astype(np.uint8)
