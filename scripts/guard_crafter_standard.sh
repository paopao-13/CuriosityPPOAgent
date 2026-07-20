#!/usr/bin/env bash
# 守护任务：周期性调用 finalize_crafter_standard.py，训练完成后自动收尾。
# 崩溃检测：finalize 返回 STALLED 时写 ALERT.txt 并退出。
set -u
# 项目根：Git Bash 后台运行时 BASH_SOURCE 相对解析会失真(算出 d:\d\... 错误路径)，
# 故优先用环境变量 CURIOSITY_PPO_ROOT，否则默认用户本机绝对路径。
ROOT="${CURIOSITY_PPO_ROOT:-D:/简历/curiosity-ppo}"
PYTHON="C:/Users/jx/AppData/Local/Programs/Python/Python311/python.exe"
[ -x "$PYTHON" ] || PYTHON="python"
CKPT_DIR="$ROOT/results/ablation/crafter_standard_ppo/seed_42"
GUARD_LOG="$CKPT_DIR/guard.log"
ALERT="$CKPT_DIR/ALERT.txt"

echo "$(date) guard started" >> "$GUARD_LOG"
while true; do
  OUT=$("$PYTHON" "$ROOT/scripts/finalize_crafter_standard.py" 2>&1 | tail -1)
  echo "$(date) finalize -> $OUT" >> "$GUARD_LOG"
  case "$OUT" in
    STATUS=DONE|STATUS=ALREADY_FINALIZED|STATUS=PUSH_FAILED)
      echo "$(date) GUARD DONE ($OUT)" >> "$GUARD_LOG"
      break ;;
    STATUS=STALLED*)
      echo "$(date) TRAINING STALLED: $OUT" > "$ALERT"
      echo "$(date) GUARD EXIT ON STALL" >> "$GUARD_LOG"
      break ;;
    STATUS=NOT_DONE*|STATUS=EVAL_FAILED*)
      sleep 60 ;;
    *)
      echo "$(date) UNEXPECTED: $OUT" >> "$GUARD_LOG"
      sleep 60 ;;
  esac
done
