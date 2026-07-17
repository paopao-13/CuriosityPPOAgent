#!/bin/bash
# GPU 互斥锁包装器 (单卡 RTX 3060 多任务排队)
# 不依赖 flock (Git Bash 环境无 flock), 改用原子 mkdir 抢锁
# 用法: _gpu_lock.sh "要执行的命令"
LOCK_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCK="$LOCK_DIR/_gpu_lock.lockdir"
# 原子抢锁: 抢不到则每 5s 轮询, 直到拿到
while ! mkdir "$LOCK" 2>/dev/null; do sleep 5; done
# 退出时自动释放锁
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
# 执行被包装的命令
eval "$1"
