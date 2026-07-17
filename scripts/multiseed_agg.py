"""多种子评测结果聚合 (mean ± std)

读取某组实验下各 seed 子目录的 benchmark_report.json,
计算各指标在多种子上的均值与标准差, 输出聚合报告 (JSON + Markdown)。

用法 (作为库被 run_ablation.py / aggregate_multiseed.py 调用):
    from multiseed_agg import aggregate_results, save_aggregate
    agg = aggregate_results("results/ablation", "crafter", "full", "crafter", [42, 43, 44])
    save_aggregate(agg, "results/ablation/crafter_full/aggregate")
"""
import json
import os
from datetime import datetime

import numpy as np


# 各环境的"主指标"字段 (用于 Markdown 摘要表)
#   field   : benchmark_report.summary[env] 中的键
#   label   : 展示名称
#   scale   : 展示倍数 (success_rate 存的是 0~1 分数, 展示成 % 需 ×100)
PRIMARY = {
    "crafter": [("score", "得分 (%)", 1.0)],
    "atari": [("mean_score", "平均分数", 1.0)],
    "minigrid": [
        ("success_rate", "成功率 (%)", 100.0),
        ("mean_steps", "平均步数", 1.0),
        ("efficiency", "样本效率 (x)", 1.0),
    ],
}


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def aggregate_results(base_dir, prefix, ablation, env, seeds):
    """聚合某组实验跨多种子的评测结果。

    Args:
        base_dir : 如 "results/ablation"
        prefix   : 环境前缀, 如 crafter / atari_montezuma / minigrid_doorkey
        ablation : full / no_icm / no_episodic / no_rnd
        env      : crafter / atari / minigrid
        seeds    : int 列表, 如 [42, 43, 44]

    Returns:
        dict: 聚合报告 (含 per_seed / summary_mean / summary_std)
    """
    group_dir = os.path.join(base_dir, f"{prefix}_{ablation}")
    per_seed = {}
    found_seeds = []
    for seed in seeds:
        report_path = os.path.join(group_dir, f"seed_{seed}", "benchmark_report.json")
        if not os.path.exists(report_path):
            print(f"[AGG][SKIP] 缺报告: {report_path}")
            continue
        rep = _load_json(report_path)
        summary = rep.get("summary", {}).get(env, {})
        if not summary:
            print(f"[AGG][WARN] 报告无 {env} 摘要: {report_path}")
            continue
        per_seed[str(seed)] = summary
        found_seeds.append(seed)

    if not per_seed:
        raise RuntimeError(f"未找到任何种子的报告于 {group_dir}")

    # 收集所有出现过的数值字段 (并集)
    all_fields = set()
    for s in per_seed.values():
        all_fields.update(k for k, v in s.items() if _is_num(v))

    summary_mean = {}
    summary_std = {}
    for field in sorted(all_fields):
        vals = [float(s[field]) for s in per_seed.values()
                 if field in s and _is_num(s[field])]
        arr = np.array(vals, dtype=float)
        summary_mean[field] = float(arr.mean())
        # 总体标准差 (种子数少时比样本标准差更稳健)
        summary_std[field] = float(arr.std(ddof=0))

    agg = {
        "timestamp": datetime.now().isoformat(),
        "env": env,
        "group": f"{prefix}_{ablation}",
        "n_seeds": len(found_seeds),
        "seeds": found_seeds,
        "per_seed": per_seed,
        "summary_mean": summary_mean,
        "summary_std": summary_std,
    }
    return agg


def render_markdown(agg):
    env = agg["env"]
    group = agg["group"]
    lines = [
        f"# 多种子聚合报告: {group}",
        "",
        f"- 环境: `{env}`",
        f"- 种子数: {agg['n_seeds']} ({', '.join(str(s) for s in agg['seeds'])})",
        f"- 时间: {agg['timestamp']}",
        "",
        "## 主指标 (mean ± std)",
        "",
    ]
    primaries = PRIMARY.get(env, [])
    if primaries:
        lines.append("| 指标 | mean | std | 各种子值 |")
        lines.append("|------|------|-----|-----------|")
        for field, label, scale in primaries:
            m = agg["summary_mean"].get(field)
            if m is None:
                continue
            s = agg["summary_std"].get(field, 0.0)
            per = ", ".join(
                f"{sd}={agg['per_seed'][sd].get(field)}"
                for sd in agg["per_seed"]
            )
            lines.append(
                f"| {label} | {m * scale:.2f} | {s * scale:.2f} | {per} |"
            )
    else:
        for field in sorted(agg["summary_mean"]):
            m = agg["summary_mean"][field]
            s = agg["summary_std"].get(field, 0.0)
            lines.append(f"- **{field}**: {m:.2f} ± {s:.2f}")
    lines.append("")
    return "\n".join(lines)


def save_aggregate(agg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "aggregate_report.json")
    md_path = os.path.join(out_dir, "aggregate_report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(agg))
    return json_path, md_path
