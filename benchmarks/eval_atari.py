"""Atari Montezuma's Revenge 评测

对比 PPO baseline ~120 pts, 目标 3500+ pts
"""
import numpy as np

BASELINE_SCORE = 120
TARGET_SCORE = 3500


def _make_atari_eval_env(env_id="ALE/MontezumaRevenge-v5", seed=42):
    """创建用于评测的 Atari 单环境。

    与训练环境保持相同的观测格式 (4×84×84 灰度堆叠), 但:
      - 移除 ClipReward    : 保留真实游戏分数
      - 移除 EpisodicLife  : 按真实 game over 划分 episode
    这样记录到的累积奖励即为 Atari 原始游戏分数。
    """
    import gymnasium

    from curiosity_ppo.envs.atari_env import NoopReset, MaxAndSkip
    from curiosity_ppo.envs.wrappers import FrameStack, GrayResizeObservation
    from curiosity_ppo.envs.vec_env import DummyVecEnv

    def _thunk():
        import ale_py  # noqa: F401  注册 ALE 环境

        env = gymnasium.make(env_id)
        env = NoopReset(env, noop_max=30)
        env = MaxAndSkip(env, skip=4)
        # 评测: 不使用 EpisodicLife (真实 game over 划分 episode)
        env = GrayResizeObservation(env, size=84)
        env = FrameStack(env, k=4)
        # 评测: 不使用 ClipReward (保留真实游戏分数)
        env.reset(seed=seed)
        return env

    return DummyVecEnv([_thunk])


def evaluate_atari(agent, n_episodes=10, seed=42, max_steps=100000):
    """评估 Atari (Montezuma's Revenge) 智能体

    Args:
        agent: CuriosityPPOAgent, 已加载检查点
        n_episodes: 评测 episode 数
        seed: 随机种子
        max_steps: 单 episode 最大步数 (安全阀, 防止卡死)

    Returns:
        dict: {
            'mean_score': float,
            'max_score': float,
            'min_score': float,
            'episode_scores': list,   # 每个 episode 的游戏分数
            'episode_lengths': list,
            'baseline': int,   # PPO 基线分数(默认120)；不使用硬编码绝对 target
        }
    """
    env = _make_atari_eval_env(seed=seed)

    episode_scores = []
    episode_lengths = []

    obs = env.reset()
    ep_score = 0.0
    ep_length = 0

    for step in range(n_episodes * max_steps):
        action = agent.act(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # 未裁剪奖励之和即为原始游戏分数
        ep_score += float(reward[0])
        ep_length += 1

        if bool(done[0]):
            episode_scores.append(ep_score)
            episode_lengths.append(ep_length)
            ep_score = 0.0
            ep_length = 0
            # DummyVecEnv 已在 done 时自动 reset
            if len(episode_scores) >= n_episodes:
                break

    try:
        env.close()
    except Exception:
        pass

    mean_score = float(np.mean(episode_scores)) if episode_scores else 0.0
    max_score = float(np.max(episode_scores)) if episode_scores else 0.0
    min_score = float(np.min(episode_scores)) if episode_scores else 0.0

    return {
        "mean_score": mean_score,
        "max_score": max_score,
        "min_score": min_score,
        "episode_scores": episode_scores,
        "episode_lengths": episode_lengths,
        "baseline": BASELINE_SCORE,
        # 注意：不再注入硬编码绝对 target(原 TARGET_SCORE=3500)。
        # EXPERIMENT.md 口径为"相对基线 120 显著提升"，由 benchmarks/report.py 判定 target_met。
    }
