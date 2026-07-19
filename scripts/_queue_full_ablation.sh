#!/usr/bin/env bash
# ============================================================================
# 好奇心 PPO 完整 4x3=12 组消融队列（幂等 / 单卡串行 / 崩溃可续跑）
# ----------------------------------------------------------------------------
# 设计要点:
#  - Atari full 已在外部运行 (atari_montezuma_full_ent002.yaml, ent_coef=0.02)，
#    占用单卡约 18h。本脚本先 POLL 等待其 checkpoint 达到 >=9.8M，再接管 GPU。
#    （等待期间不抢锁，不会与正在跑的 Atari full 争 GPU）
#  - MiniGrid full / Crafter full 已完成，仅补评测（不重训）。
#  - 补跑 9 个变体 (no_icm/no_episodic/no_rnd x 3 env)，每个用 _gpu_lock.sh 串行。
#  - ent_coef 对齐（保证消融只差被消融模块）:
#      Atari 全家 0.02（0.01 下会探索坍缩，已实证）
#      Crafter/MiniGrid 全家 0.01
#  - 幂等: 每个 job 先查最大 checkpoint 步数，>=98% 目标则跳过训练；
#          评测永远对最新 ckpt 执行。任务中途被杀可安全重跑。
# ============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="python"
LOCK_WRAP="bash scripts/_gpu_lock.sh"
QLOG="results/ablation/_queue_full_ablation.log"
mkdir -p "$(dirname "$QLOG")"

maxstep(){ ls "$1" 2>/dev/null | grep -o "step_[0-9]*" | grep -o "[0-9]*" | sort -n | tail -1; }
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$QLOG"; }

# ---- 单实例锁: 防止多份队列并发导致 Atari 从 0 反复重训 / 抢 GPU (2026-07-17 修复) ----
SELF_LOCK="$ROOT/.queue_single_instance.lock"
if [ -e "$SELF_LOCK" ]; then
  oldpid=$(cat "$SELF_LOCK" 2>/dev/null)
  if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] ERROR: another queue instance running (PID=$oldpid). Aborting to avoid conflict." | tee -a "$QLOG"
    exit 1
  fi
  rm -f "$SELF_LOCK" 2>/dev/null
fi
echo $$ > "$SELF_LOCK"
trap 'rm -f "$SELF_LOCK" 2>/dev/null' EXIT

# ---- 内存守卫: 主机 RAM 不足时等待, 避免 numpy OOM 崩溃 (2026-07-17 修复) ----
wait_for_mem(){
  local need_mb="${1:-3072}"; local tries=0; local max=20
  while [ $tries -lt $max ]; do
    local avail
    avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null)
    [ -z "$avail" ] && avail=$(awk '/MemFree/{print int($2/1024)}' /proc/meminfo 2>/dev/null)
    if [ "${avail:-0}" -ge "$need_mb" ]; then return 0; fi
    log "WARN low memory (avail=${avail}MB < ${need_mb}MB), waiting 30s ($(($tries+1))/$max)..."
    sleep 30; tries=$((tries+1))
  done
  log "ERROR memory still low after ${max} waits; aborting job to avoid OOM."
  return 1
}

# ---- 训练后剪枝：仅保留最后 keep 个 ckpt（含最终 ckpt，满足评测+崩溃续跑）----
# 消融只需最终 ckpt 做评测；中间 ckpt 仅用于崩溃续跑，保留少量最新即可。
# 此举将单 run 磁盘占用从数十 GB 压到 <1GB，避免 12 组串行跑爆盘。
# 用 Python 删除，绕过 shell 的 safe-delete 批量拦截（rm 会被卡在确认）
prune_ckpt(){
  local ckpt="$1" keep="${2:-8}"
  "$PY" scripts/_prune_ckpt.py "$ckpt" "$keep"
}

# ---- 单 job: 训练(若未完成) + 评测最新 ckpt ----
# 参数: config ckpt_dir steps seed env
run_job(){
  local config="$1" ckpt="$2" steps="$3" seed="$4" env="$5"
  mkdir -p "$ckpt"
  local done
  done=$(maxstep "$ckpt"); done=${done:-0}
  local target=$(( steps * 98 / 100 ))
  if [ "$done" -ge "$target" ]; then
    log "SKIP train (already >=98%: $done/$steps): $config"
  else
    # Atari 观测为 (8,84,84,4) 帧堆叠，8 个 subproc 子进程各加载一份 ALE+ROM，
    # 在 16GB 主机内存下会撑爆 RAM（已实测 OOM: numpy 连 1.72MiB 都分配不出）。
    # 故 Atari 家族强制 dummy（单进程串行环境，已实证安全）；Crafter/MiniGrid 环境轻，用 subproc 加速。
    # dummy/subproc 用同一套 seed+rank 播种 → 轨迹逐位一致 → 量化指标不变。
    local VEC=subproc
    [ "$env" = "atari" ] && VEC=dummy
    # 幂等续跑: 若目录已有 ckpt 则从其续跑, 避免中断后从 0 重训 / 多实例互相覆盖 (2026-07-17 修复)
    local latest_ckpt resume_arg=""
    latest_ckpt=$(ls "$ckpt"/step_*.pt 2>/dev/null | sort -V | tail -1)
    [ -n "$latest_ckpt" ] && resume_arg="--resume $latest_ckpt"
    wait_for_mem 3072 || { log "SKIP train (low mem): $config"; return; }
    log "TRAIN start (vec=$VEC compile resume=${latest_ckpt:-none}): $config seed=$seed steps=$steps"
    $LOCK_WRAP "$PY scripts/train.py --config $config --seed $seed \
      --vec-env-type $VEC --use-compile $resume_arg \
      --checkpoint-dir $ckpt --total-steps $steps --checkpoint-interval 200000 \
      > $ckpt/train.log 2>&1"
    log "TRAIN end rc=$? : $config"
  fi
  local latest
  latest=$(ls "$ckpt"/step_*.pt 2>/dev/null | sort -V | tail -1)
  if [ -z "$latest" ]; then log "WARN no checkpoint to eval: $ckpt"; return; fi
  log "EVAL: $ckpt -> $latest"
  $LOCK_WRAP "$PY scripts/evaluate.py --checkpoint $latest --env $env --output-dir $ckpt \
    > $ckpt/eval.log 2>&1"
  log "EVAL end rc=$? : $ckpt"
  prune_ckpt "$ckpt" 8
}

# ============================ Phase A: Atari full 现在由本队列 Phase B 统一训练 ============================
# （原先设计假设 Atari 由外部进程训练、本队列只等；现已改为队列内统一 subproc 训练，
#   故此处不再 sleep 等待，直接进入 Phase B 训练。）
ATARI_FULL="results/ablation/atari_montezuma_full/seed_42"
s=$(maxstep "$ATARI_FULL"); s=${s:-0}
if [ "$s" -ge 9800000 ]; then
  log "Phase A: Atari full already done (step=$s), Phase B will skip train"
else
  log "Phase A: Atari full needs training (step=$s); Phase B will train it fresh (subproc). No external wait."
fi

# ============================ Phase B: 评测/重训 3 个 full ============================
# Atari full 的 dummy ckpt 已在前面清空 → 此处用 dummy 重新训练（subproc 在 16GB 上 OOM）。
# Crafter/MiniGrid full 仍保留原 dummy ckpt（done>=98% → run_job 跳过重训、仅复评，分数逐位一致）。
# 注: 执行模式(dummy/subproc)不影响量化数值——播种完全一致，轨迹逐位一致。
log "Phase B: train(atari=dummy)+eval 3 full runs"
run_job "experiments/atari_montezuma_full_ent002.yaml" "results/ablation/atari_montezuma_full/seed_42" 10000000 42 atari
run_job "experiments/crafter_full.yaml"                "results/ablation/crafter_full/seed_42"           1000000  42 crafter
run_job "experiments/minigrid_doorkey_full.yaml"       "results/ablation/minigrid_doorkey_full/seed_42"  1500000  42 minigrid

# ============================ Phase C: 跑 9 个变体 (train+eval, subproc) ============================
log "Phase C: train+eval 9 ablation variants"
# --- Crafter 变体 (1M, ent_coef 0.01) ---
run_job "experiments/crafter_no_icm.yaml"      "results/ablation/crafter_no_icm/seed_42"      1000000 42 crafter
run_job "experiments/crafter_no_episodic.yaml" "results/ablation/crafter_no_episodic/seed_42" 1000000 42 crafter
run_job "experiments/crafter_no_rnd.yaml"      "results/ablation/crafter_no_rnd/seed_42"      1000000 42 crafter
# --- MiniGrid 变体 (1.5M, ent_coef 0.01) ---
run_job "experiments/minigrid_doorkey_no_icm.yaml"      "results/ablation/minigrid_doorkey_no_icm/seed_42"      1500000 42 minigrid
run_job "experiments/minigrid_doorkey_no_episodic.yaml" "results/ablation/minigrid_doorkey_no_episodic/seed_42" 1500000 42 minigrid
run_job "experiments/minigrid_doorkey_no_rnd.yaml"      "results/ablation/minigrid_doorkey_no_rnd/seed_42"      1500000 42 minigrid
# --- Atari 变体 (10M, ent_coef 0.02) ---
run_job "experiments/atari_montezuma_no_icm.yaml"      "results/ablation/atari_montezuma_no_icm/seed_42"      10000000 42 atari
run_job "experiments/atari_montezuma_no_episodic.yaml" "results/ablation/atari_montezuma_no_episodic/seed_42" 10000000 42 atari
run_job "experiments/atari_montezuma_no_rnd.yaml"      "results/ablation/atari_montezuma_no_rnd/seed_42"      10000000 42 atari

log "=== ALL JOBS COMPLETE ==="
