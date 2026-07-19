"""通用训练入口

用法:
    python scripts/train.py --config experiments/crafter_full.yaml --total-steps 1000000
    python scripts/train.py --config experiments/atari_montezuma_full.yaml --use-wandb
    python scripts/train.py --config experiments/minigrid_doorkey_full.yaml \
        --resume results/checkpoints/step_100000.pt
"""
import argparse
import os
import sys

# 将 src 加入模块搜索路径, 使脚本可独立运行
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from curiosity_ppo.config import load_config
from curiosity_ppo.utils.seed import set_seed
from curiosity_ppo.utils.logger import TrainLogger


def make_env(config):
    """根据 config.env.name 创建向量化环境"""
    name = config.env.name.lower()
    vec_type = getattr(config.env, "vec_env_type", "dummy")
    if "crafter" in name:
        from curiosity_ppo.envs.crafter_env import make_crafter_env

        return make_crafter_env(n_envs=config.env.n_envs, seed=config.seed, vec_env_type=vec_type)
    elif "atari" in name or "montezuma" in name:
        from curiosity_ppo.envs.atari_env import make_atari_env

        return make_atari_env(n_envs=config.env.n_envs, seed=config.seed, vec_env_type=vec_type)
    elif "minigrid" in name or "doorkey" in name:
        from curiosity_ppo.envs.minigrid_env import make_minigrid_env

        return make_minigrid_env(
            n_envs=config.env.n_envs, seed=config.seed, vec_env_type=vec_type,
            reward_shaping=getattr(config.env, "reward_shaping", False),
            env_id=getattr(config.env, "env_id", None),
            fixed_layout_seed=getattr(config.env, "fixed_layout_seed", None),
        )
    else:
        raise ValueError(f"Unknown env: {name}")


def main():
    parser = argparse.ArgumentParser(description="Curiosity-PPO 通用训练入口")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--total-steps", type=int, default=None, help="总训练步数 (覆盖配置)")
    parser.add_argument("--resume", type=str, default=None, help="检查点路径")
    parser.add_argument("--use-wandb", action="store_true", help="启用 Wandb")
    parser.add_argument("--run-name", type=str, default=None, help="Wandb 运行名称")
    parser.add_argument("--checkpoint-dir", type=str, default="results/checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None, help="随机种子 (覆盖配置, 便于多种子统计)")
    parser.add_argument("--use-compile", action="store_true", help="启用 torch.compile 网络 (数学等价提速)")
    parser.add_argument("--vec-env-type", type=str, default=None,
                        help="向量化环境类型: dummy(串行) 或 subproc(多进程并行提速)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.total_steps:
        config.env.total_steps = args.total_steps
    if args.seed is not None:
        config.seed = args.seed
    if args.use_compile:
        config.use_compile = True
    if args.vec_env_type is not None:
        config.env.vec_env_type = args.vec_env_type
    set_seed(config.seed)

    env = make_env(config)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    from curiosity_ppo.ppo.agent import CuriosityPPOAgent

    logger = TrainLogger(
        project=config.wandb_project,
        config=config,
        use_wandb=args.use_wandb,
        run_name=args.run_name or f"{config.env.name}_{config.ablation}",
    )
    agent = CuriosityPPOAgent(env, config, device=device, logger=logger)

    if args.resume:
        agent.load(args.resume)
        print(f"Resumed from {args.resume}, step={agent.global_step}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(
        f"Training {config.env.name} ({config.ablation}) for "
        f"{config.env.total_steps} steps on {device}"
    )
    agent.train(
        total_steps=config.env.total_steps,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
