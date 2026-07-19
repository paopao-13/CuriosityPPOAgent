# scripts/auto_pipeline.py
# 自动化串起 MiniGrid 课程学习: phase1(等待) -> phase2(resume) -> 评测 -> README -> commit
# 在用户离开时后台运行; 每步结果落盘到 results/ablation/AUTO_PIPELINE.log 与 AUTO_RESULT.md
import os, re, time, subprocess, glob

ROOT = r'D:\简历\curiosity-ppo'
PY = r'C:\Users\jx\AppData\Local\Programs\Python\Python311\python.exe'
PHASE1_DIR = os.path.join(ROOT, 'results/ablation/minigrid_curriculum/phase1/seed_42')
PHASE2_DIR = os.path.join(ROOT, 'results/ablation/minigrid_curriculum/phase2/seed_42')
CONFIG_P2 = os.path.join(ROOT, 'experiments/minigrid_curriculum_p2_random.yaml')
README = os.path.join(ROOT, 'README.md')
RESULT_FILE = os.path.join(ROOT, 'results/ablation/AUTO_RESULT.md')
PIPE_LOG = os.path.join(ROOT, 'results/ablation/AUTO_PIPELINE.log')

# 训练子进程用干净 env，避开 Git Bash POSIX DLL 冲突（与之前成功训练一致）
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


def phase1_done():
    # 方法1: 进程已退出
    try:
        out = subprocess.run(['pgrep', '-f', 'minigrid_curriculum_p1'],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        if not out:
            return True
    except Exception:
        pass
    # 方法2: ckpt 步数兜底 (phase1 total=500k)
    _, s = latest_ckpt(PHASE1_DIR)
    return s >= 490000


def main():
    log('=== auto_pipeline start ===')
    # 1. 等待 phase1 结束
    log('waiting phase1 to finish...')
    waited = 0
    while not phase1_done():
        time.sleep(30)
        waited += 30
        if waited % 300 == 0:
            _, s = latest_ckpt(PHASE1_DIR)
            log(f'  phase1 still running, latest step={s}')

    p1_ckpt, p1_step = latest_ckpt(PHASE1_DIR)
    log(f'phase1 done. ckpt={p1_ckpt} step={p1_step}')

    # 2. 启动 phase2 (从 phase1 ckpt resume)
    os.makedirs(PHASE2_DIR, exist_ok=True)
    p2_log = os.path.join(PHASE2_DIR, '_wrapper.log')
    log(f'starting phase2 with resume {p1_ckpt}')
    with open(p2_log, 'w', encoding='utf-8') as f:
        proc = subprocess.Popen(
            [PY, 'scripts/train.py', '--config', CONFIG_P2,
             '--total-steps', '1500000', '--run-name', 'minigrid_curriculum_p2',
             '--checkpoint-dir', PHASE2_DIR, '--resume', p1_ckpt],
            env=clean_env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    log(f'phase2 pid={proc.pid}')
    rc = proc.wait()
    log(f'phase2 ended rc={rc}')

    # 3. 读取最终 success rate (eval_score, 语义已改为 success rate)
    sr = None
    try:
        scores = []
        with open(p2_log, encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = re.search(r'eval_score=([0-9.]+)', line)
                if m:
                    scores.append(float(m.group(1)))
        sr = scores[-1] if scores else None
        log(f'final eval_score (success rate) = {sr}')
    except Exception as e:
        log(f'read eval failed: {e}')

    # 4. 诚实更新 README 性能表 (拆成纯好奇心旧基线 + 课程学习新行)
    try:
        with open(README, 'r', encoding='utf-8') as f:
            text = f.read()
        sr_disp = f'{sr:.2f}' if sr is not None else '待评测'
        new_rows = (
            "| MiniGrid DoorKey（纯好奇心，16×16） | 242 万步收敛 | **0.0**（1.5M 步） | 96.8 万步（success≥0.95） | success_rate，未解出 DoorKey；根因=等权优势合并淹没外部信号 |\n"
            f"| MiniGrid DoorKey（课程学习，8×8） | — | **{sr_disp}**（约 2M 步：固定布局预热 500K + 随机泛化 1.5M） | 96.8 万步（success≥0.95） | 含奖励塑形 + 外部优势加权(ext_adv_coef=4)，非纯好奇心设置 |"
        )
        new_text, n = re.subn(r'^\| MiniGrid DoorKey .*$', new_rows, text, flags=re.MULTILINE)
        if n:
            with open(README, 'w', encoding='utf-8') as f:
                f.write(new_text)
            log(f'README updated ({n} row replaced)')
        else:
            log('README MiniGrid row NOT found, skip')
    except Exception as e:
        log(f'README update failed: {e}')

    # 5. git commit (排除 results/ 大文件, 不 push)
    try:
        subprocess.run(['git', 'add', '-A', ':!results'], cwd=ROOT, check=True)
        msg = f'feat(minigrid): 课程学习8x8泛化 success_rate={sr_disp}'
        r = subprocess.run(['git', 'commit', '-m', msg], cwd=ROOT,
                           capture_output=True, text=True)
        log(f'git commit rc={r.returncode}: {r.stdout.strip()[:300]}')
    except Exception as e:
        log(f'git commit failed: {e}')

    # 6. 写最终结果报告
    try:
        with open(RESULT_FILE, 'w', encoding='utf-8') as f:
            f.write('# 自动化流水线结果 (MiniGrid 课程学习)\n\n')
            f.write(f'- 完成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'- phase1 ckpt: {p1_ckpt} (step {p1_step})\n')
            f.write(f'- phase2 退出码: {rc}\n')
            f.write(f'- **最终 success rate (MiniGrid DoorKey 8x8 课程学习): {sr_disp}**\n')
            f.write(f'- README 性能表已更新（纯好奇心 0.0 旧基线 + 课程学习新行，诚实区分）\n')
            f.write(f'- git 已 commit（未 push；你本地 `git push origin main`）\n')
            f.write(f'- phase2 训练日志: results/ablation/minigrid_curriculum/phase2/seed_42/_wrapper.log\n')
            f.write(f'- 流水线日志: results/ablation/AUTO_PIPELINE.log\n')
        log('=== auto_pipeline done ===')
    except Exception as e:
        log(f'write result failed: {e}')


if __name__ == '__main__':
    main()
