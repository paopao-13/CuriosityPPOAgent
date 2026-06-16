"""消融实验调度器

用法:
    python scripts/run_ablation.py --env crafter --steps 1000000
    python scripts/run_ablation.py --env atari --steps 1000000 --use-wandb

依次运行 full/no_icm/no_episodic/no_rnd 四组实验。
每组实验对应一个 experiments/<env>_<ablation>.yaml 配置, 通过 scripts/train.py 启动训练。

注意: 请在项目根目录下运行本脚本 (依赖相对路径 scripts/train.py 与 experiments/)。
"""
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="消融实验调度器")
    parser.add_argument("--env", type=str, default="crafter",
                        choices=["crafter", "atari", "minigrid"])
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--use-wandb", action="store_true")
    args = parser.parse_args()

    ablations = ["full", "no_icm", "no_episodic", "no_rnd"]
    for ablation in ablations:
        config_path = f"experiments/{args.env}_{ablation}.yaml"
        if not os.path.exists(config_path):
            print(f"Config not found: {config_path}, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"Running ablation: {ablation}")
        print(f"{'='*60}\n")
        cmd = [sys.executable, "scripts/train.py", "--config", config_path,
               "--total-steps", str(args.steps)]
        if args.use_wandb:
            cmd.append("--use-wandb")
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
