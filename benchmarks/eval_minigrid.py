"""MiniGrid DoorKey 评测

对比 baseline 2.42M 步收敛, 目标 96.8 万步 (2.5x 样本效率)
"""
import numpy as np

BASELINE_STEPS = 2_420_000
TARGET_STEPS = 968_000


def evaluate_minigrid(agent, n_episodes=100, seed=42, max_steps=1000):
    """评估 MiniGrid (DoorKey-16x16) 智能体

    成功判定: episode 累积奖励 > 0 (MiniGrid 仅在到达目标时给出正奖励),
              或 info['is_success'] 为 True。

    注意:
        返回值中的 ``mean_steps`` 是该检查点对应的 **训练步数**
        (``agent.global_step``), 用于与 baseline 收敛步数对比计算样本效率;
        它并非评测 episode 的平均步长。评测 episode 的平均步长记录在
        ``episode_mean_length`` 字段。

    Args:
        agent: CuriosityPPOAgent, 已加载检查点
        n_episodes: 评测 episode 数
        seed: 随机种子
        max_steps: 单 episode 最大步数 (安全阀)

    Returns:
        dict: {
            'success_rate': float,
            'mean_steps': float,           # 训练步数 (检查点 step)
            'episode_mean_length': float,  # 评测 episode 平均步长
            'episode_successes': list,     # 每个 episode 是否成功 (0/1)
            'episode_lengths': list,
            'baseline_steps': int,
            'target_steps': int,
        }
    """
    from curiosity_ppo.envs.minigrid_env import make_minigrid_env

    env = make_minigrid_env(n_envs=1, seed=seed)

    episode_successes = []
    episode_lengths = []

    obs = env.reset()
    ep_length = 0
    ep_reward = 0.0
    ep_success = False

    for step in range(n_episodes * max_steps):
        action = agent.act(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_reward += float(reward[0])
        ep_length += 1

        info0 = info[0] if isinstance(info, list) else info
        if isinstance(info0, dict) and info0.get("is_success", False):
            ep_success = True

        if bool(done[0]):
            # 成功: 显式 is_success 或累积奖励 > 0 (MiniGrid 仅成功时给正奖励)
            success = ep_success or (ep_reward > 0.0)
            episode_successes.append(int(success))
            episode_lengths.append(ep_length)
            ep_length = 0
            ep_reward = 0.0
            ep_success = False
            # DummyVecEnv 已在 done 时自动 reset
            if len(episode_successes) >= n_episodes:
                break

    try:
        env.close()
    except Exception:
        pass

    n_completed = len(episode_successes)
    success_rate = sum(episode_successes) / max(n_completed, 1)
    episode_mean_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0

    # 训练步数 (用于样本效率对比: efficiency = baseline_steps / mean_steps)
    train_steps = float(getattr(agent, "global_step", 0))

    return {
        "success_rate": success_rate,
        "mean_steps": train_steps,
        "episode_mean_length": episode_mean_length,
        "episode_successes": episode_successes,
        "episode_lengths": episode_lengths,
        "baseline_steps": BASELINE_STEPS,
        "target_steps": TARGET_STEPS,
    }
