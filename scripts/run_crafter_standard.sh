#!/usr/bin/env bash
# Crafter 无好奇心标准 PPO 对照训练 (本地 Windows + RTX 3060)
# 用法 (Git Bash, 后台):
#   bash scripts/run_crafter_standard.sh &
# 或前台看日志:
#   bash scripts/run_crafter_standard.sh
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="C:/Users/jx/AppData/Local/Programs/Python/Python311/python.exe"
LOGDIR="results/ablation/crafter_standard_ppo/seed_42"
mkdir -p "$LOGDIR"
echo "[run_crafter_standard] starting... log -> $LOGDIR/train.log"
"$PYTHON" -u scripts/train.py \
  --config experiments/crafter_standard_ppo.yaml \
  --checkpoint-dir "$LOGDIR" \
  --run-name crafter_standard_ppo_seed42 \
  > "$LOGDIR/train.log" 2>&1
echo "[run_crafter_standard] finished. See $LOGDIR/train.log"
