"""多种子聚合 CLI (独立运行)

适用于: 在不同机器/分批跑完多种子后, 汇总各 seed 的评测报告。
自动发现 <group-dir>/seed_* 子目录, 或按 --seeds 指定。

用法:
    # 自动发现 crafter_full 下所有 seed_* 子目录
    python scripts/aggregate_multiseed.py \
        --group-dir results/ablation/crafter_full \
        --env crafter

    # 显式指定种子与组
    python scripts/aggregate_multiseed.py \
        --base-dir results/ablation --prefix crafter --ablation full \
        --env crafter --seeds 42,43,44

    # Atari / MiniGrid
    python scripts/aggregate_multiseed.py \
        --group-dir results/ablation/atari_montezuma_full --env atari

输出:
    <group-dir>/aggregate/aggregate_report.json(.md)   (--group-dir 模式)
    <base-dir>/<prefix>_<ablation>/aggregate/...       (--base-dir 模式)
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from multiseed_agg import aggregate_results, save_aggregate


def _discover_seeds(group_dir):
    if not os.path.isdir(group_dir):
        return []
    seeds = []
    for name in os.listdir(group_dir):
        full = os.path.join(group_dir, name)
        if os.path.isdir(full) and name.startswith("seed_"):
            try:
                seeds.append(int(name[len("seed_"):]))
            except ValueError:
                continue
    return sorted(seeds)


def main():
    parser = argparse.ArgumentParser(description="多种子评测结果聚合")
    # 两种定位方式: 直接给 group-dir, 或给 base-dir+prefix+ablation
    parser.add_argument("--group-dir", type=str, default=None,
                        help="组目录, 如 results/ablation/crafter_full (自动发现 seed_*)")
    parser.add_argument("--base-dir", type=str, default="results/ablation",
                        help="基础目录 (与 --prefix/--ablation 配合)")
    parser.add_argument("--prefix", type=str, default=None,
                        help="环境前缀, 如 crafter / atari_montezuma / minigrid_doorkey")
    parser.add_argument("--ablation", type=str, default=None,
                        help="消融组, 如 full / no_icm / no_episodic / no_rnd")
    parser.add_argument("--env", type=str, required=True,
                        choices=["crafter", "atari", "minigrid"],
                        help="环境名 (决定主指标)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="显式种子列表, 逗号分隔; 不传则自动发现 seed_* 目录")
    args = parser.parse_args()

    # 解析 group-dir
    if args.group_dir:
        group_dir = args.group_dir
        prefix = None
        ablation = None
        # 从路径末两段推断 prefix_ablation
        parent, last = os.path.split(os.path.normpath(group_dir))
        if last.startswith("seed_"):
            # 落到了 seed 子目录, 上移一层
            parent, last = os.path.split(parent)
        # last 形如 crafter_full
        if "_" in last:
            prefix, ablation = last.split("_", 1)
        base_dir = parent
    else:
        if not args.prefix or not args.ablation:
            print("[ERROR] 需提供 --group-dir, 或同时提供 --prefix 与 --ablation")
            return
        prefix, ablation = args.prefix, args.ablation
        base_dir = args.base_dir
        group_dir = os.path.join(base_dir, f"{prefix}_{ablation}")

    # 解析种子
    if args.seeds:
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    else:
        seeds = _discover_seeds(group_dir)
        if not seeds:
            print(f"[ERROR] 未在 {group_dir} 发现任何 seed_* 子目录, "
                  f"请用 --seeds 显式指定")
            return

    print(f"[AGG] group={prefix}_{ablation} env={args.env} seeds={seeds}")
    try:
        agg = aggregate_results(base_dir, prefix, ablation, args.env, seeds)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    if args.group_dir:
        out_dir = os.path.join(args.group_dir, "aggregate")
    else:
        out_dir = os.path.join(base_dir, f"{prefix}_{ablation}", "aggregate")
    json_path, md_path = save_aggregate(agg, out_dir)
    print(f"[AGG] 已写: {json_path}")
    print(f"[AGG] 已写: {md_path}")
    print(f"[AGG] n={agg['n_seeds']} 均值: {agg['summary_mean']}")


if __name__ == "__main__":
    main()
