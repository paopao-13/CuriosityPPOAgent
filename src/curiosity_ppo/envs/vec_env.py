"""向量化环境：DummyVecEnv (顺序) 与 SubprocVecEnv (多进程)。

统一接口:
    reset() -> obs: ndarray (n_envs, *obs_shape)
    step(actions) -> (obs, rewards, dones, infos)
        obs:      ndarray (n_envs, *obs_shape)
        rewards:  ndarray (n_envs,)
        dones:    ndarray (n_envs,)
        infos:    list[dict] 长度 n_envs

当某个子环境 done 时自动 reset，并将终止观测存入 info["terminal_observation"]。
"""
import multiprocessing as mp

import numpy as np

__all__ = ["DummyVecEnv", "SubprocVecEnv"]


# --------------------------------------------------------------------------
# 辅助函数：规范化 reset/step 返回值（兼容旧 gym / gymnasium）
# --------------------------------------------------------------------------


def _normalize_reset(result):
    """reset 结果 → (obs, info)。"""
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, {}


def _normalize_step(result):
    """step 结果 → (obs, reward, terminated, truncated, info)。"""
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, done, False, info
    return result


# ==========================================================================
# DummyVecEnv
# ==========================================================================


class DummyVecEnv:
    """顺序执行多个环境，step/reset 返回 numpy 堆叠数组。"""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self._closed = False

    def reset(self):
        obs_list = []
        for env in self.envs:
            obs, _info = _normalize_reset(env.reset())
            obs_list.append(obs)
        return np.stack(obs_list)

    def step(self, actions):
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, env in enumerate(self.envs):
            result = _normalize_step(env.step(actions[i]))
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)

            if done:
                # 保存终止观测，然后自动 reset
                info = dict(info) if isinstance(info, dict) else {"info": info}
                info["terminal_observation"] = obs
                obs, _ = _normalize_reset(env.reset())

            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        obs = np.stack(obs_list)
        rewards = np.array(reward_list, dtype=np.float32)
        dones = np.array(done_list, dtype=bool)
        return obs, rewards, dones, info_list

    def close(self):
        if getattr(self, "_closed", True):
            return
        for env in self.envs:
            env.close()
        self._closed = True

    def __del__(self):
        if not getattr(self, "_closed", True):
            self.close()


# ==========================================================================
# SubprocVecEnv
# ==========================================================================


def _worker(remote, parent_remote, env_fn):
    """子进程 worker：接收命令，操作环境，返回结果。"""
    parent_remote.close()
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                result = _normalize_step(env.step(data))
                obs, reward, terminated, truncated, info = result
                done = bool(terminated or truncated)
                if done:
                    info = dict(info) if isinstance(info, dict) else {"info": info}
                    info["terminal_observation"] = obs
                    obs, _ = _normalize_reset(env.reset())
                remote.send((obs, reward, done, info))
            elif cmd == "reset":
                obs, _ = _normalize_reset(env.reset())
                remote.send(obs)
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"Unknown command: {cmd}")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


class SubprocVecEnv:
    """基于 multiprocessing.Pipe 的并行向量化环境。

    各子环境在独立进程中运行，接口与 DummyVecEnv 一致。
    """

    def __init__(self, env_fns, start_method=None):
        self.num_envs = len(env_fns)
        ctx = mp.get_context(start_method) if start_method else mp.get_context()

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            p = ctx.Process(target=_worker, args=(work_remote, remote, env_fn))
            p.daemon = True
            p.start()
            self.processes.append(p)
            work_remote.close()

        # 获取 observation/action space
        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        self._closed = False

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.stack([remote.recv() for remote in self.remotes])

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return (
            np.stack(obs),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            list(infos),
        )

    def close(self):
        if getattr(self, "_closed", True):
            return
        for remote in getattr(self, "remotes", []):
            try:
                remote.send(("close", None))
            except (EOFError, BrokenPipeError):
                pass
        for p in getattr(self, "processes", []):
            p.join()
        self._closed = True

    def __del__(self):
        if not getattr(self, "_closed", True):
            self.close()
