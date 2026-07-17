"""消融实验 + 多种子调度器

用法:
    # 单种子 (默认 seed=42), 跑 full/no_icm/no_episodic/no_rnd 四组
    python scripts/run_ablation.py --env crafter --steps 1000000

    # 多种子统计 (seed 42/43/44), 每组写入独立 seed 子目录, 跑完自动聚合
    python scripts/run_ablation.py --env crafter --steps 1000000 --seeds 42,43,44

    # 只跑 full 配置 (用于 headline 指标的多种子), 跳过其他消融
    python scripts/run_ablation.py --env crafter --steps 1000000 --ablations full --seeds 42,43,44

    # 先批量起训练, 不评测, 之后用 aggregate_multiseed.py 单独聚合
    python scripts/run_ablation.py --env crafter --seeds 42,43,44 --no-eval

每组对应 experiments/<prefix>_<ablation>.yaml, 训练产物写入独立的
results/ablation/<prefix>_<ablation>/seed_<seed>/ 目录 (每种子隔离, 不会互相覆盖)。
训练完成后默认自动加载该种子最新检查点进行评测, 生成
results/ablation/<prefix>_<ablation>/seed_<seed>/benchmark_report.json(.md)。
全部 (ablation, seed) 跑完后, 逐组聚合多种子结果到
results/ablation/<prefix>_<ablation>/aggregate/aggregate_report.json(.md) (mean ± std)。

环境名 -> 配置文件前缀映射 (与 experiments/ 中 *_full.yaml 命名对齐):
    crafter   -> crafter
    atari     -> atari_montezuma
    minigrid  -> minigrid_doorkey

注意: 请在项目根目录下运行本脚本 (依赖相对路径 scripts/train.py 与 experiments/)。
"""
import argparse
import os
import subprocess
import sys

# 让 curiosity_ppo (src/) 可导入
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

# 环境 CLI 名 -> 配置文件前缀
ENV_PREFIX = {
    "crafter": "crafter",
    "atari": "atari_montezuma",
    "minigrid": "minigrid_doorkey",
}

ABLATIONS = ["full", "no_icm", "no_episodic", "no_rnd"]


def _parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _run(cmd):
    print(">> " + " ".join(cmd))
    return subprocess.run(cmd).returncode


def main():
    parser = argparse.ArgumentParser(description="消融实验 + 多种子调度器")
    parser.add_argument("--env", type=str, default="crafter",
                        choices=list(ENV_PREFIX.keys()))
    parser.add_argument("--ablations", type=str, default=",".join(ABLATIONS),
                        help="要运行的消融组, 逗号分隔 (默认全4组: full,no_icm,no_episodic,no_rnd)")
    parser.add_argument("--steps", type=int, default=None,
                        help="总训练步数 (覆盖配置; 不传则用 YAML 中的 total_steps)")
    parser.add_argument("--seeds", type=str, default="42",
                        help="随机种子列表, 逗号分隔 (默认 42; 多种子如 42,43,44)")
    parser.add_argument("--use-wandb", action="store_true", help="启用 Wandb")
    parser.add_argument("--no-eval", action="store_true",
                        help="训练后不自动评测 (也不聚合)")
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="评测 episode 数 (默认随环境: crafter=100, atari=10, minigrid=100)")
    args = parser.parse_args()

    prefix = ENV_PREFIX[args.env]
    seeds = _parse_int_list(args.seeds)
    ablations = [a.strip() for a in args.ablations.split(",") if a.strip()]
    # 过滤掉不存在的配置, 避免误报
    ablations = [a for a in ablations
                 if os.path.exists(f"experiments/{prefix}_{a}.yaml")]

    if not ablations:
        print("[ERROR] 没有任何可用的消融配置, 请检查 --ablations 与环境前缀。")
        return

    print(f"\n{'='*60}")
    print(f"  调度: env={args.env} prefix={prefix}")
    print(f"  ablations={ablations}")
    print(f"  seeds={seeds}")
    print(f"  steps={args.steps if args.steps else 'YAML默认'}")
    print(f"{'='*60}\n")

    for ablation in ablations:
        for seed in seeds:
            config_path = f"experiments/{prefix}_{ablation}.yaml"
            ckpt_dir = f"results/ablation/{prefix}_{ablation}/seed_{seed}"

            print(f"\n{'-'*60}")
            print(f"  [{ablation} | seed={seed}]")
            print(f"  config : {config_path}")
            print(f"  ckpt   : {ckpt_dir}")
            print(f"{'-'*60}\n")

            cmd = [sys.executable, "scripts/train.py",
                   "--config", config_path,
                   "--seed", str(seed),
                   "--checkpoint-dir", ckpt_dir]
            if args.steps:
                cmd += ["--total-steps", str(args.steps)]
            if args.use_wandb:
                cmd += ["--use-wandb",
                        "--run-name", f"{prefix}_{ablation}_s{seed}"]

            rc = _run(cmd)
            if rc != 0:
                print(f"[ERROR] {ablation} seed={seed} 训练失败 (rc={rc}), 跳过评测")
                continue

            if args.no_eval:
                continue

            from curiosity_ppo.utils.checkpoint import get_latest_checkpoint
            latest = get_latest_checkpoint(ckpt_dir)
            if not latest:
                print(f"[WARN] 未找到检查点: {ckpt_dir}, 跳过评测")
                continue

            eval_cmd = [sys.executable, "scripts/evaluate.py",
                         "--checkpoint", latest,
                         "--env", args.env,
                         "--output-dir", ckpt_dir]
            if args.n_episodes:
                eval_cmd += ["--n-episodes", str(args.n_episodes)]
            print(f"\n[EVAL] {ablation} seed={seed} -> {latest}")
            _run(eval_cmd)

    # ---- 多种子聚合 (仅当做了评测) ----
    if not args.no_eval and len(seeds) >= 1:
        from multiseed_agg import aggregate_results, save_aggregate
        print(f"\n{'='*60}")
        print(f"  多种子聚合 (env={args.env}, seeds={seeds})")
        print(f"{'='*60}\n")
        for ablation in ablations:
            try:
                agg = aggregate_results("results/ablation", prefix,
                                       ablation, args.env, seeds)
            except RuntimeError as e:
                print(f"[AGG][SKIP] {ablation}: {e}")
                continue
            out_dir = f"results/ablation/{prefix}_{ablation}/aggregate"
            json_path, md_path = save_aggregate(agg, out_dir)
            print(f"[AGG] {prefix}_{ablation}: "
                  f"n={agg['n_seeds']} 已写 {json_path}")

    print(f"\n=== 调度完成 ===")
    print(f"各组检查点与评测报告位于: results/ablation/<prefix>_<ablation>/seed_<seed>/")
    if not args.no_eval:
        print(f"多种子聚合报告位于: results/ablation/<prefix>_<ablation>/aggregate/")


if __name__ == "__main__":
    main()
