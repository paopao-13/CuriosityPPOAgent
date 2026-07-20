"""Crafter 标准 PPO 训练收尾（幂等，可重复执行）。

流程：
  1. 检查是否已完成（FINALIZED.txt 存在则跳过）
  2. 检查训练是否完成（最大 ckpt step >= 990000）
  3. 等训练进程释放 GPU（train.log 静默 120s）
  4. 运行 evaluate.py 拿 Crafter 几何均值分数
  5. 回填 README 性能表（插入对照行）
  6. git commit + push
  7. 写 FINALIZED.txt 记录分数与 commit

stdout 只输出状态码（STATUS=XXX），详细日志写 finalize.log。
状态码：DONE / ALREADY_FINALIZED / NOT_DONE / STALLED / EVAL_FAILED / PUSH_FAILED
"""
import os
import re
import sys
import time
import json
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "results/ablation/crafter_standard_ppo/seed_42")
TRAIN_LOG = os.path.join(CKPT_DIR, "train.log")
README = os.path.join(ROOT, "README.md")
REPORT_JSON = os.path.join(ROOT, "results", "benchmark_report.json")
FINALIZED = os.path.join(CKPT_DIR, "FINALIZED.txt")
SCORE_CACHE = os.path.join(CKPT_DIR, "SCORE.txt")
FIN_LOG = os.path.join(CKPT_DIR, "finalize.log")
DONE_MARK = "无好奇心标准 PPO"

# 训练用的解释器（系统 Python 3.11，带 torch 2.7.1+cu118）
PYTHON = r"C:/Users/jx/AppData/Local/Programs/Python/Python311/python.exe"
if not os.path.exists(PYTHON):
    PYTHON = sys.executable

TARGET_STEP = 990000
STALL_SEC = 900  # 日志静默超过 15 分钟且未完成 -> 判定卡死


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(FIN_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def find_max_ckpt():
    if not os.path.isdir(CKPT_DIR):
        return None, 0
    pts = [f for f in os.listdir(CKPT_DIR) if re.match(r"step_(\d+)\.pt$", f)]
    if not pts:
        return None, 0
    best = max(pts, key=lambda f: int(re.search(r"step_(\d+)\.pt$", f).group(1)))
    step = int(re.search(r"step_(\d+)\.pt$", best).group(1))
    return os.path.join(CKPT_DIR, best), step


def log_idle_sec():
    if not os.path.exists(TRAIN_LOG):
        return 1e9
    return time.time() - os.path.getmtime(TRAIN_LOG)


def wait_for_log_idle(idle_sec=120, timeout=600):
    """等 train.log 静默 idle_sec 秒（训练进程已退出），超时则继续。"""
    waited = 0
    while waited < timeout:
        if log_idle_sec() >= idle_sec:
            return True
        time.sleep(15)
        waited += 15
    log(f"warn: 等日志静默超时({timeout}s)，仍继续")
    return True


def run_eval(ckpt):
    log(f"运行 evaluate.py: {ckpt}")
    r = subprocess.run(
        [PYTHON, "scripts/evaluate.py", "--checkpoint", ckpt,
         "--env", "crafter", "--n-episodes", "100", "--output-dir", "results"],
        cwd=ROOT, capture_output=True, text=True, timeout=1200,
    )
    log(f"evaluate returncode={r.returncode}")
    if r.stdout:
        for ln in r.stdout.strip().splitlines()[-8:]:
            log(f"  stdout: {ln}")
    if r.stderr:
        for ln in r.stderr.strip().splitlines()[-8:]:
            log(f"  stderr: {ln}")
    return r.returncode == 0


def read_score():
    if not os.path.exists(REPORT_JSON):
        return None
    try:
        with open(REPORT_JSON, encoding="utf-8") as f:
            rep = json.load(f)
        return float(rep["summary"]["crafter"]["score"])
    except Exception as e:
        log(f"read_score error: {e}")
        return None


def insert_readme(score, step):
    with open(README, encoding="utf-8") as f:
        content = f.read()
    if DONE_MARK in content:
        log("README 已含对照行，跳过插入")
        return
    new_line = (f"| Crafter 无好奇心标准 PPO（1M 步） | 15.6% | **{score:.1f}%** | "
                f"19.0% | 关掉 rnd/icm/episodic 三模块，纯标准 PPO 对照基线（step_{step}） |\n")
    lines = content.split("\n")
    out, inserted = [], False
    for line in lines:
        out.append(line)
        if not inserted and line.startswith("| Crafter") and "0.2%" in line:
            out.append(new_line)
            inserted = True
    if not inserted:
        raise RuntimeError("找不到 Crafter 纯好奇心行，无法插入对照行")
    with open(README, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    log(f"README 已插入对照行: {score:.1f}%")


def commit_and_push(score):
    r = subprocess.run(["git", "status", "--porcelain", "README.md"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.stdout.strip():
        subprocess.run(["git", "add", "README.md"], cwd=ROOT, check=True)
        msg = (f"feat(crafter): 新增无好奇心标准 PPO 对照基线，1M步实测 {score:.1f}%\n\n"
               f"- 关掉 rnd/icm/episodic 三模块，纯标准 PPO 对照纯好奇心 0.2%\n"
               f"- 验证架构在常规条件下可运行，补全探索权衡叙事")
        subprocess.run(["git", "commit", "-m", msg], cwd=ROOT, check=True)
        log("git commit 完成")
    else:
        log("README 无新改动，跳过 commit")
    # push（失败重试一次）
    for attempt in range(2):
        p = subprocess.run(["git", "push", "origin", "main"],
                           cwd=ROOT, capture_output=True, text=True)
        if p.returncode == 0:
            log("git push 成功")
            return True
        log(f"push 失败(attempt {attempt+1}): {p.stderr.strip()[:200]}")
        time.sleep(5)
    return False


def main():
    if os.path.exists(FINALIZED):
        print("STATUS=ALREADY_FINALIZED")
        return 0

    ckpt, step = find_max_ckpt()
    if ckpt is None or step < TARGET_STEP:
        # 崩溃检测
        if step < TARGET_STEP and log_idle_sec() > STALL_SEC:
            log(f"STALLED: 训练静默超 {STALL_SEC}s，step={step}")
            print("STATUS=STALLED")
            return 2
        print(f"STATUS=NOT_DONE step={step}")
        return 0

    log(f"训练完成，最终 ckpt step={step}")

    # README 是否已更新？
    with open(README, encoding="utf-8") as f:
        content = f.read()
    if DONE_MARK not in content:
        wait_for_log_idle(120, 600)
        # 取分数（优先缓存）
        score = None
        if os.path.exists(SCORE_CACHE):
            try:
                score = float(open(SCORE_CACHE).read().strip())
            except Exception:
                score = None
        if score is None:
            if not run_eval(ckpt):
                print("STATUS=EVAL_FAILED")
                return 1
            score = read_score()
            if score is None:
                print("STATUS=EVAL_FAILED")
                return 1
            with open(SCORE_CACHE, "w", encoding="utf-8") as f:
                f.write(str(score))
        log(f"评测分数={score:.1f}%")
        insert_readme(score, step)
    else:
        log("README 已含对照行，直接走 commit/push")

    ok = commit_and_push(score if 'score' in dir() and score is not None else 0.0)
    # 写 FINALIZED
    try:
        with open(FINALIZED, "w", encoding="utf-8") as f:
            f.write(f"score={score:.1f}%\nstep={step}\npush_ok={ok}\n")
    except Exception:
        pass
    print("STATUS=DONE" if ok else "STATUS=PUSH_FAILED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log(f"FATAL: {e}")
        print("STATUS=EVAL_FAILED")
        sys.exit(1)
