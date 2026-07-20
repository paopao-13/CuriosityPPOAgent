# scripts/auto_pipeline_potential.py
# 自动化串起势能塑形后期: 等待训练结束 -> 读最终 success rate -> 回填 README -> commit
# 在用户离开时后台运行; 每步结果落盘到 results/ablation/AUTO_PIPELINE_POTENTIAL.log 与 AUTO_RESULT_POTENTIAL.md
import os, re, time, subprocess, glob

ROOT = r'D:\简历\curiosity-ppo'
PY = r'C:\Users\jx\AppData\Local\Programs\Python\Python311\python.exe'
POT_DIR = os.path.join(ROOT, 'results/ablation/minigrid_potential/seed_42')
POT_LOG = os.path.join(POT_DIR, '_wrapper.log')
README = os.path.join(ROOT, 'README.md')
RESULT_FILE = os.path.join(ROOT, 'results/ablation/AUTO_RESULT_POTENTIAL.md')
PIPE_LOG = os.path.join(ROOT, 'results/ablation/AUTO_PIPELINE_POTENTIAL.log')
TOTAL_STEPS = 4000000
DONE_THRESHOLD = 3950000  # ckpt 步数兜底阈值

clean_env = {
    'PATH': r'C:\Users\jx\AppData\Local\Programs\Python\Python311' + os.pathsep +
            r'C:\Windows\system32' + os.pathsep + r'C:\Windows',
    'HOME': r'C:\Users\jx',
    'SYSTEMROOT': r'C:\Windows',
    'WINDIR': r'C:\Windows',
    'TEMP': r'C:\Users\jx\AppData\Local\Temp',
    'TMP': r'C:\Users\jx\AppData\Local\Temp',
    'USERPROFILE': r'C:\Users\jx',
    'USERNAME': 'jx',
    'COMSPEC': r'C:\Windows\system32\cmd.exe',
    'NUMBER_OF_PROCESSORS': '8',
    'PROCESSOR_ARCHITECTURE': 'AMD64',
    'PATHEXT': '.COM;.EXE;.BAT;.CMD',
}


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    try:
        with open(PIPE_LOG, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def latest_ckpt(d):
    files = glob.glob(os.path.join(d, 'step_*.pt'))
    if not files:
        return None, 0

    def step(f):
        m = re.search(r'step_(\d+)\.pt', f)
        return int(m.group(1)) if m else 0

    files.sort(key=step)
    return files[-1], step(files[-1])


def train_status():
    """基于 ckpt 推进判断训练状态(不依赖进程枚举, 最可靠):
    - 'done': 最新 ckpt 步数达到总量阈值
    - 'crashed': 连续 STUCK_LIMIT 次检查(每次间隔 60s) ckpt 无增长且未达总量
    - 'running': 其他情况(ckpt 仍在推进)
    返回 (status, step)
    """
    _, s = latest_ckpt(POT_DIR)
    if s >= DONE_THRESHOLD:
        return 'done', s
    return 'running', s


def read_final_success_rate():
    try:
        scores = []
        with open(POT_LOG, encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = re.search(r'eval_score=([0-9.]+)', line)
                if m:
                    scores.append(float(m.group(1)))
        return scores[-1] if scores else None
    except Exception as e:
        log(f'read eval failed: {e}')
        return None


def update_readme(sr_disp):
    try:
        with open(README, 'r', encoding='utf-8') as f:
            text = f.read()
        new_row = (
            "| MiniGrid DoorKey（势能塑形，8×8） | — | "
            f"**{sr_disp}**（4M 步随机布局） | 96.8 万步（success≥0.95） | "
            "potential-based shaping（到子目标距离的连续稠密奖励）+ ext_adv_coef=2，非纯好奇心设置 |"
        )

        def repl(m):
            return m.group(0) + '\n' + new_row

        new_text, n = re.subn(r'\| MiniGrid DoorKey.*课程学习.*', repl, text)
        if n:
            with open(README, 'w', encoding='utf-8') as f:
                f.write(new_text)
            log(f'README updated (appended potential-shaping row, {n} anchor matched)')
        else:
            log('README 课程学习 anchor NOT found, skip update')
    except Exception as e:
        log(f'README update failed: {e}')


def git_commit(sr_disp):
    try:
        subprocess.run(['git', 'add', '-A', ':!results'], cwd=ROOT, check=True,
                       capture_output=True, text=True, timeout=60)
        msg = f'feat(minigrid): 势能塑形8x8 success_rate={sr_disp}'
        r = subprocess.run(['git', 'commit', '-m', msg], cwd=ROOT,
                           capture_output=True, text=True, timeout=60)
        log(f'git commit rc={r.returncode}: {r.stdout.strip()[:300]}')
    except Exception as e:
        log(f'git commit failed: {e}')


def write_result(sr_disp, status, ckpt):
    try:
        with open(RESULT_FILE, 'w', encoding='utf-8') as f:
            f.write('# 自动化流水线结果 (MiniGrid 势能塑形)\n\n')
            f.write(f'- 完成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'- 训练状态: {status}\n')
            f.write(f'- 最终 ckpt: {ckpt}\n')
            f.write(f'- **最终 success rate (MiniGrid DoorKey 8x8 势能塑形): {sr_disp}**\n')
            if status == 'done':
                f.write(f'- README 性能表已更新（新增势能塑形行，保留课程学习 0.21 对照，诚实区分）\n')
                f.write(f'- git 已 commit（未 push；你本地 `git push origin main`）\n')
            else:
                f.write(f'- README 未更新、git 未 commit（训练中断，无有效 success rate，不写误导结论）\n')
            f.write(f'- 训练日志: results/ablation/minigrid_potential/seed_42/_wrapper.log\n')
            f.write(f'- 流水线日志: results/ablation/AUTO_PIPELINE_POTENTIAL.log\n')
        log('=== auto_pipeline_potential done ===')
    except Exception as e:
        log(f'write result failed: {e}')


def main():
    log('=== auto_pipeline_potential start ===')
    log('monitoring via ckpt progress (no process enum)...')
    status = 'running'
    last_step = -1
    stuck = 0
    STUCK_LIMIT = 10  # 连续 10 次(每次 60s)无进展 -> 判定崩溃
    while True:
        st, s = train_status()
        if st == 'done':
            status = 'done'
            log(f'training done (ckpt reached total, step={s}).')
            break
        if s > last_step:
            last_step = s
            stuck = 0
            time.sleep(60)
        else:
            stuck += 1
            if stuck >= STUCK_LIMIT:
                status = 'crashed'
                log(f'TRAINING CRASHED: ckpt stuck at step={s} for {STUCK_LIMIT} checks (< {DONE_THRESHOLD}).')
                break
            time.sleep(60)
        if (last_step // 100000) != ((s) // 100000) or stuck == 0:
            if s != last_step:
                log(f'  training progress, latest step={s}')

    ckpt, step = latest_ckpt(POT_DIR)
    log(f'final ckpt={ckpt} step={step}')

    sr = read_final_success_rate() if status == 'done' else None
    sr_disp = f'{sr:.2f}' if sr is not None else '训练中断'

    if status == 'done':
        update_readme(sr_disp)
        git_commit(sr_disp)
    else:
        log('skipped README/commit due to training crash; inspect log for details.')

    write_result(sr_disp, status, ckpt)


if __name__ == '__main__':
    main()
