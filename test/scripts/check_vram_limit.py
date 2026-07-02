#!/usr/bin/env python3
"""实时显存占用监控脚本

功能:
  - 训练过程持续采集 GPU 显存峰值
  - 一旦超过 2.2GB 红线, 自动终止训练进程并输出完整显存日志
  - 适配 RTX 3060 Laptop 6GB 显卡

用法:
  方式一 (独立监控, 手动启动训练后运行):
    python test/scripts/check_vram_limit.py --threshold 2200 --interval 2

  方式二 (包裹训练命令, 自动启动并监控):
    python test/scripts/check_vram_limit.py --wrap "python scripts/train.py --config experiments/crafter_full.yaml --total-steps 1000000"

  方式三 (仅检测当前环境显存基线):
    python test/scripts/check_vram_limit.py --baseline-only
"""
import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

# 红线常量 (MB), 不可修改
VRAM_LIMIT_MB = 2200

LOG_DIR = "test/logs"
LOG_FILE = os.path.join(LOG_DIR, "vram_log.csv")
ALERT_FILE = os.path.join(LOG_DIR, "vram_alert.txt")


def check_cuda_available():
    """检查 CUDA 是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_vram_mb():
    """获取当前 GPU 显存占用 (MB)"""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0, 0.0
        allocated = torch.cuda.memory_allocated() / 1e6
        peak = torch.cuda.max_memory_allocated() / 1e6
        return allocated, peak
    except Exception:
        return 0.0, 0.0


def write_log_row(writer, timestamp, allocated, peak, status):
    """写入一行 CSV 日志"""
    writer.writerow({
        "timestamp": timestamp,
        "allocated_mb": f"{allocated:.2f}",
        "peak_mb": f"{peak:.2f}",
        "status": status,
    })


def monitor_loop(threshold_mb, interval_sec, target_pid=None):
    """持续监控显存, 超阈值时告警并终止目标进程

    Args:
        threshold_mb: 显存红线 (MB)
        interval_sec: 采样间隔 (秒)
        target_pid: 被监控的训练进程 PID, None 则仅告警不终止
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    alert_triggered = False
    peak_ever = 0.0

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["timestamp", "allocated_mb", "peak_mb", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print(f"[VRAM Monitor] 开始监控 | 红线={threshold_mb}MB | 间隔={interval_sec}s")
        print(f"[VRAM Monitor] 日志文件: {LOG_FILE}")

        while True:
            allocated, peak = get_vram_mb()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if peak > peak_ever:
                peak_ever = peak

            if peak > threshold_mb:
                status = "ALERT"
                write_log_row(writer, ts, allocated, peak, status)
                f.flush()

                if not alert_triggered:
                    alert_triggered = True
                    alert_msg = (
                        f"[{ts}] 显存超限! peak={peak:.2f}MB > 红线={threshold_mb}MB\n"
                        f"  allocated={allocated:.2f}MB\n"
                        f"  训练进程 PID={target_pid}\n"
                    )
                    print(alert_msg)

                    with open(ALERT_FILE, "w", encoding="utf-8") as af:
                        af.write(alert_msg)
                        af.write(f"历史峰值: {peak_ever:.2f}MB\n")

                    # 终止训练进程
                    if target_pid is not None:
                        try:
                            os.kill(target_pid, signal.SIGTERM)
                            print(f"[VRAM Monitor] 已发送 SIGTERM 终止进程 {target_pid}")
                        except (ProcessLookupError, PermissionError):
                            try:
                                if sys.platform == "win32":
                                    subprocess.run(
                                        ["taskkill", "/PID", str(target_pid), "/F"],
                                        capture_output=True,
                                    )
                                else:
                                    os.kill(target_pid, signal.SIGKILL)
                                print(f"[VRAM Monitor] 已强制终止进程 {target_pid}")
                            except Exception:
                                pass

                        # 终止后再记录几秒确认显存释放
                        for _ in range(5):
                            time.sleep(interval_sec)
                            a, p = get_vram_mb()
                            write_log_row(writer, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                          a, p, "POST_KILL")
                            f.flush()
                        break
                    else:
                        # 仅告警模式, 继续监控
                        pass
            else:
                status = "OK"
                write_log_row(writer, ts, allocated, peak, status)
                f.flush()

                # 每 30 秒打印一次状态
                if int(time.time()) % 30 < interval_sec:
                    print(f"[{ts}] allocated={allocated:.0f}MB peak={peak:.0f}MB | OK")

            time.sleep(interval_sec)

    print(f"\n[VRAM Monitor] 监控结束 | 历史峰值={peak_ever:.2f}MB")
    if alert_triggered:
        print(f"[VRAM Monitor] 触发过显存超限告警, 详见: {ALERT_FILE}")
        sys.exit(1)
    else:
        print(f"[VRAM Monitor] 全程未超限, 测试合格")
        sys.exit(0)


def run_with_wrap(cmd_str, threshold_mb, interval_sec):
    """启动训练命令并监控其显存"""
    print(f"[VRAM Monitor] 启动训练命令: {cmd_str}")
    proc = subprocess.Popen(
        cmd_str,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(f"[VRAM Monitor] 训练进程 PID={proc.pid}")

    # 在子线程中监控显存
    import threading
    monitor_thread = threading.Thread(
        target=monitor_loop,
        args=(threshold_mb, interval_sec, proc.pid),
        daemon=True,
    )
    monitor_thread.start()

    # 主线程等待训练完成
    ret = proc.wait()
    monitor_thread.join(timeout=5)

    print(f"[VRAM Monitor] 训练进程退出, returncode={ret}")
    return ret


def main():
    parser = argparse.ArgumentParser(description="GPU 显存实时监控 (红线 2.2GB)")
    parser.add_argument("--threshold", type=int, default=VRAM_LIMIT_MB,
                        help=f"显存红线 (MB), 默认 {VRAM_LIMIT_MB}")
    parser.add_argument("--interval", type=int, default=2,
                        help="采样间隔 (秒), 默认 2")
    parser.add_argument("--wrap", type=str, default=None,
                        help="包裹训练命令, 自动启动并监控")
    parser.add_argument("--baseline-only", action="store_true",
                        help="仅检测当前环境显存基线")
    args = parser.parse_args()

    if not check_cuda_available():
        print("[VRAM Monitor] CUDA 不可用, 无法监控显存")
        print("[VRAM Monitor] CPU 模式下此脚本无意义, 请在 GPU 环境运行")
        sys.exit(0)

    if args.baseline_only:
        allocated, peak = get_vram_mb()
        print(f"[VRAM Monitor] 当前显存: allocated={allocated:.2f}MB, peak={peak:.2f}MB")
        print(f"[VRAM Monitor] 红线: {args.threshold}MB")
        if peak > args.threshold:
            print("[VRAM Monitor] 当前峰值已超限!")
            sys.exit(1)
        else:
            print("[VRAM Monitor] 当前峰值正常")
            sys.exit(0)

    if args.wrap:
        ret = run_with_wrap(args.wrap, args.threshold, args.interval)
        sys.exit(ret)
    else:
        monitor_loop(args.threshold, args.interval, target_pid=None)


if __name__ == "__main__":
    main()
