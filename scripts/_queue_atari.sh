#!/usr/bin/env bash
# Atari full 排队器：等 MiniGrid(>=1.47M) + Crafter(>=0.98M) 跑完后，
# 用 SubprocVecEnv + cudnn.benchmark + torch.compile 加速启动 Atari 10M 步。
#
# 用法 (后台):
#   bash scripts/_queue_atari.sh > results/ablation/atari_montezuma_full/seed_42/queue.log 2>&1 &
set -e
ROOT="D:/简历/curiosity-ppo"
cd "$ROOT"
PY="C:/Users/jx/AppData/Local/Programs/Python/Python311/python.exe"
LOG="results/ablation/atari_montezuma_full/seed_42/train.log"
mkdir -p "$(dirname "$LOG")"

# 取某目录下最大 step 编号
maxstep(){
  ls "$1" 2>/dev/null | grep -o "step_[0-9]*" | grep -o "[0-9]*" | sort -n | tail -1
}

MG="results/ablation/minigrid_doorkey_full/seed_42"
CR="results/ablation/crafter_full/seed_42"
MG_TARGET=1470000   # 1.5M × 0.98
CR_TARGET=980000    # 1M × 0.98

echo "[queue $(date +%H:%M:%S)] waiting for minigrid(>=${MG_TARGET}) + crafter(>=${CR_TARGET})..."

MAX_WAIT=600   # 最多等 600 × 30s = 5 小时
i=0
while true; do
  mg=$(maxstep "$MG"); mg=${mg:-0}
  cr=$(maxstep "$CR"); cr=${cr:-0}
  if [ "$mg" -ge "$MG_TARGET" ] && [ "$cr" -ge "$CR_TARGET" ]; then
    break
  fi
  i=$((i+1))
  if [ $i -ge $MAX_WAIT ]; then
    echo "[queue $(date +%H:%M:%S)] TIMEOUT after 5h (mg=$mg cr=$cr), launching anyway"
    break
  fi
  sleep 30
done

echo "[queue $(date +%H:%M:%S)] both done (mg=$mg cr=$cr)"
echo "[queue $(date +%H:%M:%S)] launching Atari full: 10M steps, subproc+compile, seed 42"

# 用 SubprocVecEnv + torch.compile + cudnn.benchmark 加速
# 日志直接写文件 (不加 | tail)，可实时 cat 追进度
$PY scripts/train.py \
  --config experiments/atari_montezuma_full.yaml \
  --total-steps 10000000 \
  --seed 42 \
  --vec-env-type subproc \
  --use-compile \
  --checkpoint-dir results/ablation/atari_montezuma_full/seed_42 \
  > "$LOG" 2>&1

EXIT=$?
echo "[queue $(date +%H:%M:%S)] Atari full done. exit=$EXIT"
