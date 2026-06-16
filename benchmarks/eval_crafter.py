"""Crafter 评测: 22 成就成功率 → 几何均值 → Score (%)

对比 PPO(ResNet) baseline 15.6%, 目标 19.0%
"""
import numpy as np

# Crafter 全部 22 个成就 (固定列表, 保证几何均值覆盖所有成就)
CRAFTER_ACHIEVEMENTS = [
    "collect_coal", "collect_cow", "collect_diamond", "collect_drink",
    "collect_iron", "collect_sapling", "collect_stone", "collect_wood",
    "defeat_cow", "defeat_skeleton", "defeat_zombie",
    "eat_cow", "eat_plant",
    "make_iron_pickaxe", "make_iron_sword",
    "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword",
    "place_furnace", "place_plant", "place_table",
]

BASELINE_SCORE = 15.6
TARGET_SCORE = 19.0


def evaluate_crafter(agent, n_episodes=100, seed=42, max_steps=1000):
    """评估 Crafter 智能体

    Args:
        agent: CuriosityPPOAgent, 已加载检查点
        n_episodes: 评测 episode 数
        seed: 随机种子
        max_steps: 单个 episode 最大步数

    Returns:
        dict: {
            'score': float,            # 归一化得分 (%) — 22 成就几何均值
            'success_rates': dict,     # 22 个成就各自成功率
            'episode_rewards': list,   # 每个 episode 的累积外在奖励
            'episode_lengths': list,
            'baseline_score': float,
            'target_score': float,
        }
    """
    from curiosity_ppo.envs.crafter_env import make_crafter_env

    env = make_crafter_env(n_envs=1, seed=seed)

    # 成就计数 (初始化全部 22 个为 0, 保证几何均值覆盖所有成就)
    achievements = {name: 0 for name in CRAFTER_ACHIEVEMENTS}
    episode_rewards = []
    episode_lengths = []

    obs = env.reset()
    ep_reward = 0.0
    ep_length = 0
    ep_achievements = set()  # 本 episode 内达成过的成就

    for step in range(n_episodes * max_steps):  # 最多 max_steps 步/episode
        action = agent.act(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_reward += float(reward[0])
        ep_length += 1

        # 记录本 episode 达成的成就
        # (info['achievements'] 为本步解锁标志, 累积到 set 中去重)
        info0 = info[0] if isinstance(info, list) else info
        if isinstance(info0, dict) and "achievements" in info0:
            for ach, achieved in info0["achievements"].items():
                if achieved:
                    ep_achievements.add(ach)

        if bool(done[0]):
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            for ach in ep_achievements:
                achievements[ach] = achievements.get(ach, 0) + 1
            ep_reward = 0.0
            ep_length = 0
            ep_achievements = set()
            # DummyVecEnv 已在 done 时自动 reset, obs 已是新 episode 首帧
            if len(episode_rewards) >= n_episodes:
                break

    # 计算成就成功率
    n_completed = len(episode_rewards)
    success_rates = {k: v / max(n_completed, 1) for k, v in achievements.items()}

    # 几何均值得分 (Crafter 标准: 对 0 成功率用极小值避免 log(0))
    if success_rates:
        scores = [max(s, 1e-6) for s in success_rates.values()]
        geo_mean = float(np.exp(np.mean(np.log(scores))) * 100)
    else:
        geo_mean = 0.0

    try:
        env.close()
    except Exception:
        pass

    return {
        "score": geo_mean,
        "success_rates": success_rates,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "baseline_score": BASELINE_SCORE,
        "target_score": TARGET_SCORE,
    }
