"""CPU 上的 rollout 缓冲区, mini-batch 时才传 GPU."""
import numpy as np
import torch


class RolloutBuffer:
    """CPU 上的 rollout 缓冲区，mini-batch 时才传 GPU"""

    def __init__(self, n_steps, n_envs, obs_shape, action_dim, device='cpu'):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device
        self.ptr = 0
        # 全部 numpy 存 CPU
        self.obs = np.zeros((n_steps, n_envs) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.logprobs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards_ext = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.rewards_int = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values_ext = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values_int = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        # GAE 计算后填入
        self.advantages_ext = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages_int = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns_ext = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns_int = np.zeros((n_steps, n_envs), dtype=np.float32)

    def add(self, obs, action, logprob, reward_ext, reward_int, value_ext, value_int, done):
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = action
        self.logprobs[idx] = logprob
        self.rewards_ext[idx] = reward_ext
        self.rewards_int[idx] = reward_int
        self.values_ext[idx] = value_ext
        self.values_int[idx] = value_int
        self.dones[idx] = done
        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def get_mini_batches(self, batch_size=128):
        """生成 mini-batch，torch.from_numpy(arr).to(device) 传 GPU"""
        total = self.n_steps * self.n_envs
        indices = np.random.permutation(total)
        # flatten steps x envs
        obs_flat = self.obs.reshape(total, *self.obs_shape)
        actions_flat = self.actions.reshape(total)
        logprobs_flat = self.logprobs.reshape(total)
        rewards_ext_flat = self.rewards_ext.reshape(total)
        rewards_int_flat = self.rewards_int.reshape(total)
        values_ext_flat = self.values_ext.reshape(total)
        values_int_flat = self.values_int.reshape(total)
        advantages_ext_flat = self.advantages_ext.reshape(total)
        advantages_int_flat = self.advantages_int.reshape(total)
        returns_ext_flat = self.returns_ext.reshape(total)
        returns_int_flat = self.returns_int.reshape(total)

        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]
            yield {
                'obs': torch.from_numpy(obs_flat[idx]).to(self.device),
                'actions': torch.from_numpy(actions_flat[idx]).to(self.device),
                'logprobs': torch.from_numpy(logprobs_flat[idx]).to(self.device),
                'advantages_ext': torch.from_numpy(advantages_ext_flat[idx]).to(self.device),
                'advantages_int': torch.from_numpy(advantages_int_flat[idx]).to(self.device),
                'returns_ext': torch.from_numpy(returns_ext_flat[idx]).to(self.device),
                'returns_int': torch.from_numpy(returns_int_flat[idx]).to(self.device),
            }
