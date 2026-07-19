"""聚合三基准评测结果, 生成 JSON + Markdown 报告"""
import json
import os
from datetime import datetime

import numpy as np


def _json_default(obj):
    """JSON 序列化兜底: 处理 numpy 标量/数组。"""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


def generate_report(results, output_dir="results"):
    """生成评测报告

    Args:
        results: dict, 包含 crafter/atari/minigrid 的评测结果
        output_dir: 输出目录

    Returns:
        dict: 报告内容
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "summary": {},
    }

    # 汇总
    if "crafter" in results:
        r = results["crafter"]
        baseline = r.get("baseline_score", 15.6)
        target = r.get("target_score", 19.0)
        improvement = (r["score"] - baseline) / baseline * 100 if baseline else 0.0
        report["summary"]["crafter"] = {
            "score": r["score"],
            "baseline": baseline,
            "improvement_pct": improvement,
            "target_met": r["score"] >= target,
        }

    if "atari" in results:
        r = results["atari"]
        baseline = r.get("baseline", 120)
        # EXPERIMENT.md: Atari 不再以 3500 为硬指标，只需相对 PPO 基线(120)显著提升。
        # 兼容旧调用：若显式传入 target 则仍按绝对线判，否则按"相对基线提升"判。
        target = r.get("target", None)
        improvement = (r["mean_score"] - baseline) / baseline * 100 if baseline else 0.0
        if target is None:
            target_met = r["mean_score"] > baseline
            target_note = f"相对基线 {baseline} 提升 {improvement:.1f}% (EXPERIMENT.md 口径)"
        else:
            target_met = r["mean_score"] >= target
            target_note = f"绝对目标 {target} (兼容旧口径)"
        report["summary"]["atari"] = {
            "mean_score": r["mean_score"],
            "max_score": r.get("max_score", r["mean_score"]),
            "baseline": baseline,
            "improvement_pct": improvement,
            "target_met": target_met,
            "target_note": target_note,
        }

    if "minigrid" in results:
        r = results["minigrid"]
        baseline_steps = r.get("baseline_steps", 2420000)
        target_steps = r.get("target_steps", 968000)
        efficiency = baseline_steps / max(r["mean_steps"], 1)
        report["summary"]["minigrid"] = {
            "success_rate": r["success_rate"],
            "mean_steps": r["mean_steps"],
            "baseline_steps": baseline_steps,
            "efficiency": efficiency,
            "target_met": efficiency >= 2.5,
        }

    # 保存 JSON
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "benchmark_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=_json_default)

    # 生成 Markdown
    md_path = os.path.join(output_dir, "benchmark_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_report(report))

    return report


def markdown_report(report):
    """生成 Markdown 格式报告"""
    lines = ["# 基准评测报告", "", f"**时间**: {report['timestamp']}", ""]

    for env_name, data in report["summary"].items():
        lines.append(f"## {env_name.upper()}")
        lines.append("")
        if env_name == "crafter":
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| 得分 | {data['score']:.1f}% |")
            lines.append(f"| PPO基线 | {data['baseline']:.1f}% |")
            lines.append(f"| 提升 | {data['improvement_pct']:.1f}% |")
            lines.append(f"| 达标 | {'✓' if data['target_met'] else '✗'} |")
        elif env_name == "atari":
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| 平均分数 | {data['mean_score']:.0f} |")
            lines.append(f"| 最高分数 | {data['max_score']:.0f} |")
            lines.append(f"| PPO基线 | {data['baseline']} |")
            lines.append(f"| 达标 | {'✓' if data['target_met'] else '✗'} |")
        elif env_name == "minigrid":
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| 成功率 | {data['success_rate']*100:.1f}% |")
            lines.append(f"| 平均步数 | {data['mean_steps']:.0f} |")
            lines.append(f"| 基线步数 | {data['baseline_steps']:.0f} |")
            lines.append(f"| 样本效率 | {data['efficiency']:.1f}x |")
            lines.append(f"| 达标 | {'✓' if data['target_met'] else '✗'} |")
        lines.append("")

    return "\n".join(lines)
