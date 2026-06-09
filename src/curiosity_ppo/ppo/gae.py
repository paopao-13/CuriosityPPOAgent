"""GAE (Generalized Advantage Estimation) 双轨计算.

- ext (情景制): dones 处 last_value=0, bootstrap 截断
- int (非情景制): dones 全 0, 用 V_int(s_{T+1}) bootstrap, 跨 episode
"""
import numpy as np


def compute_gae(rewards, values, last_value, dones, gamma, gae_lambda):
    """通用 GAE 计算.

    - ext (情景制): dones 处 last_value=0，bootstrap 截断
    - int (非情景制): dones 全 0，用 V_int(s_{T+1}) bootstrap，跨 episode
    """
    n_steps, n_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    last_gae = np.zeros(n_envs, dtype=np.float32)
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns
