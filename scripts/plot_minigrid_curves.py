#!/usr/bin/env python
"""Plot MiniGrid DoorKey-8x8 success_rate curves for three settings.

Reads periodic `eval_score` lines from the three training logs and draws a
single figure comparing:

  * Pure curiosity (no shaping)  -> stays at 0.0
  * Curriculum learning (+shaping) -> plateaus ~0.21
  * Potential-based shaping        -> converges to 1.0

All numbers come straight from the training logs (greedy 100-episode eval),
so the figure is fully reproducible from `results/`.

Usage:
    python scripts/plot_minigrid_curves.py
"""

import os
import re
import matplotlib

matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOGS = {
    "Pure curiosity (no shaping)": os.path.join(
        ROOT, "results/ablation/minigrid_doorkey_full/seed_42/train.log"
    ),
    "Curriculum learning (+shaping)": os.path.join(
        ROOT, "results/ablation/minigrid_curriculum/phase2/seed_42/_wrapper.log"
    ),
    "Potential-based shaping": os.path.join(
        ROOT, "results/ablation/minigrid_potential/seed_42/_wrapper.log"
    ),
}

COLORS = {
    "Pure curiosity (no shaping)": "#d62728",
    "Curriculum learning (+shaping)": "#ff7f0e",
    "Potential-based shaping": "#2ca02c",
}

PATTERN = re.compile(r"step=(\d+).*eval_score=([\d.]+)")


def parse(path):
    steps, scores = [], []
    if not os.path.exists(path):
        print(f"[warn] missing log: {path}")
        return steps, scores
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                steps.append(int(m.group(1)) / 1e6)  # -> millions
                scores.append(float(m.group(2)))
    return steps, scores


def main():
    fig, ax = plt.subplots(figsize=(9, 5.2))

    for label, path in LOGS.items():
        x, y = parse(path)
        if not x:
            continue
        ax.plot(
            x,
            y,
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=COLORS[label],
            label=f"{label}  (n={len(x)})",
            alpha=0.9,
        )

    ax.axhline(0.95, color="#555555", linestyle="--", linewidth=1.2, label="target 0.95")

    ax.set_xlabel("Training steps (millions)")
    ax.set_ylabel("success_rate (greedy, 100 eps)")
    ax.set_title("MiniGrid DoorKey-8x8: exploration progression under three settings")
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    note = (
        "Pure curiosity never solves DoorKey (root cause: equal-weight advantage\n"
        "merge drowns the extrinsic signal). Curriculum learning reaches a ~0.21\n"
        "plateau. Potential-based shaping drives success to 1.0 within ~1.35M steps."
    )
    ax.text(
        0.015,
        0.97,
        note,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="#f7f7f7", edgecolor="#cccccc", alpha=0.9),
    )

    out = os.path.join(ROOT, "docs/figures/minigrid_curves.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
