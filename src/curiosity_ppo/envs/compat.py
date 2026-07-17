"""GymCompatWrapper：将旧版 gym API 兼容为 gymnasium API。

旧 gym:
    - reset() 返回单个 obs
    - step() 返回 4 元组 (obs, reward, done, info)

gymnasium:
    - reset() 返回 (obs, info) 元组
    - step() 返回 5 元组 (obs, reward, terminated, truncated, info)
"""
import gymnasium

__all__ = ["GymCompatWrapper"]


class GymCompatWrapper(gymnasium.Wrapper):
    """兼容旧版 gym 与 gymnasium 的 step/reset 返回格式。

    注意：绕过 ``gymnasium.Wrapper`` 基类中对 ``isinstance(env, gymnasium.Env)``
    的断言，以便能够包装真正的旧版 ``gym.Env``（例如 Crafter）。
    """

    def __init__(self, env):
        # 直接设置 env，跳过基类的 isinstance(env, gymnasium.Env) 断言，
        # 从而支持旧版 gym 环境。同时初始化基类需要的缓存属性。
        self.env = env
        self._action_space = None
        self._observation_space = None
        self._metadata = None
        self._cached_spec = None

    def reset(self, **kwargs):
        # 拦截 gymnasium 传递的 seed / options 关键字
        # 旧 gym 的 reset() 不接受这些参数，需用 env.seed() 设置种子
        seed = kwargs.pop('seed', None)
        kwargs.pop('options', None)
        if seed is not None:
            try:
                self.env.seed(seed)
            except Exception:
                pass
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            return result
        # 旧 gym 风格：只返回 obs
        return result, {}

    def close(self):
        """安全关闭被包装的旧版环境 (crafter.Env 无 close 属性)。"""
        try:
            self.env.close()
        except (AttributeError, TypeError):
            pass

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            # 旧 gym 的 done → terminated=done, truncated=False
            return obs, reward, done, False, info
        return result
