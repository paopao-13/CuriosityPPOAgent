#!/usr/bin/env bash
# ============================================================
# cloud_train_atari.sh  (自愈版 watch-dog)
# 在阿里云 PAI (DSW 交互式 / DLC 训练任务) 上重训
# Atari Montezuma's Revenge (Curiosity-PPO, 全模块) 10M 步
#
# 设计目标：
#   1. 自愈续训   —— 崩溃后自动从 CKPT_DIR 中最新的 step_*.pt 续跑，
#                   最坏只丢一个 checkpoint 间隔(~10k 步)，不再丢 3.87M。
#   2. 崩溃安全   —— checkpoint 间隔 10240 步。
#   3. 持久化     —— CKPT_DIR 指向 OSS 挂载目录，实例释放/被抢占不丢进度。
#   4. 抗坍缩     —— 默认使用 ent_coef=0.02 配置（已验证可避免熵坍缩）。
#   5. 致命即停   —— 日志出现 nan/inf/Traceback/MemoryError 立即放弃，
#                   避免反复空转烧钱。
#   6. 日志可见   —— 通过 tee 同时写文件与 stdout（DLC 日志服务捕获 stdout）。
#
# 用法：
#   # 首次从 0 跑（CKPT_DIR 建议挂 OSS，例如 /mnt/oss/atari_seed42）
#   CKPT_DIR=/mnt/oss/atari_seed42 bash scripts/cloud_train_atari.sh
#
#   # 内存偏紧的小实例（RAM<32G）：改用 dummy 向量环境
#   VEC_ENV=dummy CKPT_DIR=/mnt/oss/atari_seed42 bash scripts/cloud_train_atari.sh
#
#   # 关闭 torch.compile（若 A100 上 compile 报错）
#   USE_COMPILE=0 CKPT_DIR=/mnt/oss/atari_seed42 bash scripts/cloud_train_atari.sh
# ============================================================
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---------- 可调参数（环境变量覆盖，均有默认值）----------
CKPT_DIR="${CKPT_DIR:-results/ablation/atari_montezuma_full/seed_42}"
CONFIG="${CONFIG:-experiments/atari_montezuma_full.yaml}"   # 已含 ent_coef=0.02
TOTAL_STEPS="${TOTAL_STEPS:-10000000}"
SEED="${SEED:-42}"
VEC_ENV="${VEC_ENV:-subproc}"        # 云端 RAM 充足用 subproc；RAM<32G 改 dummy
USE_COMPILE="${USE_COMPILE:-1}"      # 1=开 torch.compile（A100/A10 下安全提速）
CKPT_INTERVAL="${CKPT_INTERVAL:-10240}"
MAX_RETRIES="${MAX_RETRIES:-20}"      # 最多自愈重启次数（防无限循环）
USE_WANDB="${USE_WANDB:-0}"          # 1=启用 wandb（需先 export WANDB_API_KEY）

mkdir -p "$CKPT_DIR"
LOG="$CKPT_DIR/train.log"

latest_ckpt() {
  # 返回 CKPT_DIR 中修改时间最新、且全局步数最大的 step_*.pt
  ls -1t "$CKPT_DIR"/step_*.pt 2>/dev/null | head -1
}

attempt=0
while (( attempt <= MAX_RETRIES )); do
  attempt=$((attempt + 1))
  RESUME="$(latest_ckpt)"
  if [ -n "$RESUME" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] attempt $attempt: resume from $RESUME"
  else
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] attempt $attempt: fresh start (no checkpoint)"
  fi

  CMD=(python scripts/train.py
    --config "$CONFIG"
    --total-steps "$TOTAL_STEPS"
    --seed "$SEED"
    --vec-env-type "$VEC_ENV"
    --checkpoint-dir "$CKPT_DIR"
    --checkpoint-interval "$CKPT_INTERVAL"
  )
  [ "$USE_COMPILE" = "1" ] && CMD+=(--use-compile)
  [ -n "$RESUME" ]         && CMD+=(--resume "$RESUME")
  [ "$USE_WANDB" = "1" ]   && CMD+=(--use-wandb --run-name "atari_full_seed42")

  echo "[$(date +'%Y-%m-%d %H:%M:%S')] launching: ${CMD[*]}"
  # tee 同时写文件(train.log)与 stdout(DLC 日志服务捕获)
  "${CMD[@]}" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}

  if [ "$rc" -eq 0 ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] training completed cleanly (exit 0). DONE."
    exit 0
  fi

  # 致命标记：出现即放弃自愈，避免反复空转烧钱
  if grep -qiE "nan|inf|nan_detected=True|Traceback|MemoryError|CUDA out of memory" "$LOG"; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] FATAL markers in log — aborting watchdog (needs manual fix). See $LOG"
    exit 2
  fi

  echo "[$(date +'%Y-%m-%d %H:%M:%S')] attempt $attempt crashed (rc=$rc); auto-resume from latest ckpt after 10s..."
  sleep 10
done

echo "[$(date +'%Y-%m-%d %H:%M:%S')] exceeded MAX_RETRIES=$MAX_RETRIES — giving up. Inspect $LOG"
exit 3
