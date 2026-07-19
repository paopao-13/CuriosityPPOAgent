"""评测入口

用法:
    python scripts/evaluate.py --checkpoint results/checkpoints/last.pt --env crafter
    python scripts/evaluate.py --checkpoint last.pt --env atari --n-episodes 10
    python scripts/evaluate.py --checkpoint last.pt --env minigrid --config experiments/config.yaml

流程: 加载模型 → 根据环境调用对应评测函数 → 生成 JSON + Markdown 报告
"""
import argparse
import os
import sys

# ---- 路径引导: 让 curiosity_ppo (src/) 与 benchmarks (项目根) 可被导入 ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402

from curiosity_ppo.config import Config, load_config  # noqa: E402
from curiosity_ppo.ppo.agent import CuriosityPPOAgent  # noqa: E402
from curiosity_ppo.utils.checkpoint import load_checkpoint  # noqa: E402

from benchmarks.eval_crafter import evaluate_crafter  # noqa: E402
from benchmarks.eval_atari import evaluate_atari  # noqa: E402
from benchmarks.eval_minigrid import evaluate_minigrid  # noqa: E402
from benchmarks.report import generate_report  # noqa: E402


# 各环境默认评测 episode 数
DEFAULT_EPISODES = {
    "crafter": 100,
    "atari": 10,
    "minigrid": 100,
}


def build_eval_env(env_name, seed):
    """根据环境名创建单环境评测用向量化环境"""
    if env_name == "crafter":
        from curiosity_ppo.envs.crafter_env import make_crafter_env
        return make_crafter_env(n_envs=1, seed=seed)
    elif env_name == "atari":
        from curiosity_ppo.envs.atari_env import make_atari_env
        return make_atari_env(n_envs=1, seed=seed)
    elif env_name == "minigrid":
        from curiosity_ppo.envs.minigrid_env import make_minigrid_env
        return make_minigrid_env(n_envs=1, seed=seed)
    else:
        raise ValueError(f"Unknown env: {env_name}")


def load_config_for_env(checkpoint_path, config_path, env_name):
    """加载配置: 优先 checkpoint 内置 config (保证架构与训练一致), 其次 --config, 最后默认 Config。

    强制将 env.name 与 env.n_envs 与评测环境对齐 (env.name 影响编码器选择)。
    """
    # 优先用 checkpoint 内置 config: 训练时 agent.save 存了完整 Config,
    # 用它重建网络可 100% 匹配编码器/通道数/动作数, 避免选错 CrafterEncoder/NatureDQN。
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ckpt = load_checkpoint(checkpoint_path, device="cpu")
            saved = ckpt.get("agent_state", {}).get("config")
            if saved is not None:
                saved.env.n_envs = 1
                return saved
        except Exception:
            pass
    if config_path and os.path.exists(config_path):
        cfg = load_config(config_path)
    else:
        cfg = Config()
    # 与评测环境对齐
    cfg.env.name = env_name
    cfg.env.n_envs = 1
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Curiosity-PPO 评测入口")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点路径 (.pt)")
    parser.add_argument("--env", type=str, default="crafter",
                        choices=["crafter", "atari", "minigrid"],
                        help="评测环境")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径 (可选, 优先级最高)")
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="评测 episode 数 (默认: crafter=100, atari=10, minigrid=100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help="cpu / cuda / auto")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="报告输出目录")
    args = parser.parse_args()

    # 设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # episode 数
    n_episodes = args.n_episodes if args.n_episodes is not None else DEFAULT_EPISODES[args.env]

    # 加载配置
    config = load_config_for_env(args.checkpoint, args.config, args.env)
    config.seed = args.seed

    # 创建评测环境
    env = build_eval_env(args.env, args.seed)

    # 构建 Agent 并加载权重
    agent = CuriosityPPOAgent(env, config, device=device)
    agent.load(args.checkpoint)
    agent.actor_critic.eval()
    print(f"[evaluate] 已加载检查点: {args.checkpoint} (global_step={agent.global_step})")

    # 运行对应评测
    print(f"[evaluate] 环境={args.env}, episodes={n_episodes}, device={device}")
    if args.env == "crafter":
        results = evaluate_crafter(agent, n_episodes=n_episodes, seed=args.seed)
    elif args.env == "atari":
        results = evaluate_atari(agent, n_episodes=n_episodes, seed=args.seed)
    else:
        results = evaluate_minigrid(agent, n_episodes=n_episodes, seed=args.seed)

    # 生成报告
    report = generate_report({args.env: results}, output_dir=args.output_dir)
    json_path = os.path.join(args.output_dir, "benchmark_report.json")
    md_path = os.path.join(args.output_dir, "benchmark_report.md")
    print(f"[evaluate] 报告已保存: {json_path} / {md_path}")

    # 打印摘要
    summary = report["summary"].get(args.env, {})
    print(f"[evaluate] 摘要: {summary}")


if __name__ == "__main__":
    main()
