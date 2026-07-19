# 一键训练 + 评测方案 (RUN_PLAN)

> 配套本仓库已修复的四处缺陷 (见 `METRIC_FEASIBILITY.md` 第三节):
> 1. `scripts/run_ablation.py` 已为每组写入独立 `--checkpoint-dir`, 不再互相覆盖；
> 2. `agent.load()` 已恢复优化器状态, 续训不丢动量；
> 3. 已补齐 Atari / MiniGrid 的 `no_icm/no_episodic/no_rnd` 三组消融 YAML；
> 4. `docs/ABLATION_REPORT.md` §2.5 评测命令已对齐 `results/ablation/<prefix>_<ablation>/step_*.pt` 实际产出。
>
> ✅ **步数口径（重要，已核实）**: `agent.py` 的 `global_step += n_envs` 位于
> `collect_rollout` 的 `for step in range(n_steps)` 循环**内部**, 每 rollout 累加 `n_steps×n_envs`
> = **环境步数**。原始代码**本就正确**, 配置里的 `total_steps` **就是环境步数**
> (Crafter `1000000` = 严格 1M 环境步; Atari `10000000` = 10M 环境步; MiniGrid `1500000` = 1.5M 环境步)。
> 检查点文件名 `step_{global_step}.pt` 中的数字即环境步数。下文命令无需再乘系数。
> （注：曾误以为需要改成 `+= n_envs*n_steps`, 实测暴露 128× 放大后已**回退还原**, 切勿再改。）

---

## 0. 后台运行约定（必读，避免踩坑）

⚠️ **日志必须写文件，不要管道 `| tail`**：后台跑训练时用
`> results/ablation/<x>/train.log 2>&1`（不加 `| tail -N`）。
`| tail` 会缓冲到进程退出才 flush，导致后台任务全程"无输出"、无法实时追进度，
且 `ps`/`tasklist` 均无法稳定列出 Windows 的 `python.exe` 进程——判断"是否在跑"以 checkpoint 文件 mtime 是否持续更新为准。**写文件可随时 `cat` 追进度。**

⚠️ **单卡 3060 显存互斥**：多个训练不能同时占 GPU。用仓库自带互斥锁脚本
`scripts/_gpu_lock.sh`（原子 `mkdir` 抢锁，5s 轮询，退出自动放锁）串行化：

```bash
# 互斥锁包裹单次训练（先抢锁再跑 python，跑完自动放锁）
scripts/_gpu_lock.sh "C:/Users/jx/AppData/Local/Programs/Python/Python311/python.exe \
  scripts/train.py --config experiments/minigrid_doorkey_full.yaml \
  --total-steps 1500000 --seed 42 \
  --checkpoint-dir results/ablation/minigrid_doorkey_full/seed_42 \
  > results/ablation/minigrid_doorkey_full/seed_42/train.log 2>&1"

# 后台并行起多个（同一把锁 → 自动排队，交替占 GPU，互不冲突）
scripts/_gpu_lock.sh "..."  # MiniGrid
scripts/_gpu_lock.sh "..."  # Crafter（排队）
```

> 注：本机 Git Bash **无 `flock`**，故 `_gpu_lock.sh` 改用原子 `mkdir` 抢锁，切勿手写 `flock`。

> 💡 **排队等待的可靠写法**（不用 `ps`，它查不到 Windows 进程）：轮询 checkpoint 目录的最大步数，达到阈值（如 MiniGrid≥1.47M、Crafter≥0.98M）再接力起下一个训练。

> 也可用 `run_ablation.py`（内部已串行、单卡安全），多种子/消融直接交给它调度。

📌 **进度查看**：`cat results/ablation/<x>/seed_<s>/train.log`（实时追加）。
📌 **完成判定**：`ls` 该目录最大 `step_*.pt` 的步数 ≥ 0.98×total_steps 且 60s 内不再增长即跑完（末步因 checkpoint 间隔 10240 不整除 total_steps，不一定是精确 total_steps）。
📌 **崩溃排查**：日志末尾有 `Traceback` 即异常；`step_*.pt` 没到目标步数就是中途挂了。

> 解释器：本机 `python` 默认指向 managed 3.13（空环境）。**训练/验证必须用系统 3.11**：
> `C:\Users\jx\AppData\Local\Programs\Python\Python311\python.exe`

---

## 0. 环境与冒烟测试

```bash
# 安装 (Windows)
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 跑通 144 单元测试
python -m pytest tests/ -q

# 冒烟: 小步快跑, 确认端到端链路 (Crafter 5 万 global_step)
python scripts/train.py --config experiments/crafter_full.yaml --total-steps 50000 --checkpoint-dir results/smoke/crafter
python scripts/evaluate.py --checkpoint results/smoke/crafter/step_50000.pt --env crafter --n-episodes 5
```

> 冒烟通过后再跑全量, 避免几小时后才发现链路问题。

---

## 1. 三基准「完整模型 (full)」全量训练

```bash
# 日志写文件 (勿用 | tail); 单卡用互斥锁串行 (见 §0)
# Crafter  (严格 1M 环境步 = total_steps 即环境步数)
flock -x scripts/_gpu_lock.lock -c \
  "C:/Users/jx/AppData/Local/Programs/Python/Python311/python.exe \
   scripts/train.py --config experiments/crafter_full.yaml \
   --total-steps 1000000 --seed 42 \
   --checkpoint-dir results/ablation/crafter_full/seed_42 \
   > results/ablation/crafter_full/seed_42/train.log 2>&1"

# Atari Montezuma  (严格 10M 环境步 = total_steps 即环境步数)
# 交付: 报 10M 步实测分 + 相对 PPO 基线 120 分的提升 (方案 a, 不再写 3500+)
flock -x scripts/_gpu_lock.lock -c \
  "C:/Users/jx/AppData/Local/Programs/Python/Python311/python.exe \
   scripts/train.py --config experiments/atari_montezuma_full.yaml \
   --total-steps 10000000 --seed 42 \
   --checkpoint-dir results/ablation/atari_montezuma_full/seed_42 \
   > results/ablation/atari_montezuma_full/seed_42/train.log 2>&1"

# MiniGrid DoorKey  (严格 1.5M 环境步 = total_steps 即环境步数)
flock -x scripts/_gpu_lock.lock -c \
  "C:/Users/jx/AppData/Local/Programs/Python/Python311/python.exe \
   scripts/train.py --config experiments/minigrid_doorkey_full.yaml \
   --total-steps 1500000 --seed 42 \
   --checkpoint-dir results/ablation/minigrid_doorkey_full/seed_42 \
   > results/ablation/minigrid_doorkey_full/seed_42/train.log 2>&1"
```

训练结束后评测 (取该组最新检查点; 单种子默认写入 seed_42 子目录):

```bash
python scripts/evaluate.py --checkpoint results/ablation/crafter_full/seed_42/step_1000000.pt --env crafter
python scripts/evaluate.py --checkpoint results/ablation/atari_montezuma_full/seed_42/step_10000000.pt --env atari
python scripts/evaluate.py --checkpoint results/ablation/minigrid_doorkey_full/seed_42/step_1500000.pt --env minigrid
```

---

## 2. 四组消融 (一键: 训练 + 自动评测)

`run_ablation.py` 已内置「训练 → 取最新检查点 → 评测」全流程, 且每组写入独立目录。

```bash
# Crafter 四组消融 (full / no_icm / no_episodic / no_rnd)
python scripts/run_ablation.py --env crafter --steps 1000000 --use-wandb

# Atari 四组消融 (配置已补齐)
python scripts/run_ablation.py --env atari --steps 10000000 --use-wandb

# MiniGrid 四组消融 (配置已补齐)
python scripts/run_ablation.py --env minigrid --steps 1500000 --use-wandb
```

每组产物 (按种子隔离, 默认 seed_42):
- 检查点: `results/ablation/<prefix>_<ablation>/seed_<seed>/step_*.pt`
- 评测报告: `results/ablation/<prefix>_<ablation>/seed_<seed>/benchmark_report.json` 与 `.md`
- 多种子聚合 (用 `--seeds` 时自动生成): `results/ablation/<prefix>_<ablation>/aggregate/aggregate_report.json` 与 `.md` (mean ± std)

> 若只训练不评测 (例如先批量起训练), 加 `--no-eval`, 之后用
> `scripts/aggregate_multiseed.py` 单独聚合 (见 §3)。

---

## 3. 多种子统计 (降低随机性)

`run_ablation.py` 现已内置 `--seeds` (逗号分隔) 与 `--ablations` (限定组),
每个 (ablation, seed) 写入独立 `seed_<seed>` 目录, 跑完自动聚合 mean±std。
`train.py` 的 `--seed` 覆盖已就位, 无需再改 YAML。

```bash
# Crafter full 的多种子 (headline 指标), 仅跑 full 组
python scripts/run_ablation.py --env crafter --steps 1000000 \
    --ablations full --seeds 42,43,44 --use-wandb
# -> results/ablation/crafter_full/seed_{42,43,44}/... + aggregate/aggregate_report.*

# Crafter 全四组 × 三种子 (完整消融统计)
python scripts/run_ablation.py --env crafter --steps 1000000 \
    --seeds 42,43,44 --use-wandb
# -> 每组下多 seed 目录 + 各组 aggregate/

# Atari / MiniGrid 同理
python scripts/run_ablation.py --env atari    --steps 10000000 --ablations full --seeds 42,43,44
python scripts/run_ablation.py --env minigrid --steps 1500000  --ablations full --seeds 42,43,44
```

若分批/跨机器跑, 可先 `--no-eval` 起训练, 再单独聚合:

```bash
# 自动发现某组下所有 seed_* 子目录并聚合
python scripts/aggregate_multiseed.py \
    --group-dir results/ablation/crafter_full --env crafter

# 或显式指定
python scripts/aggregate_multiseed.py \
    --base-dir results/ablation --prefix crafter --ablation full \
    --env crafter --seeds 42,43,44
```

> 建议保留 seed=42 为主实验; 补 43/44 得到均值±标准差, 提升结果可信度。

---

## 4. 导出 ONNX + Web Demo + 运行视频

```bash
# 导出 ONNX (MiniGrid DoorKey 用于 Web Demo)
python scripts/export_onnx.py \
    --checkpoint results/ablation/minigrid_doorkey_full/step_1500000.pt \
    --output web/public/models/model.onnx --env minigrid

# 启动网页 Demo (浏览器打开 http://localhost:5173)
cd web && npm install && npm run dev

# 录制智能体运行演示视频 (需先有训练好的检查点)
python scripts/record_video.py \
    --checkpoint results/ablation/crafter_full/step_1000000.pt --env crafter \
    --output results/videos/crafter_demo.mp4
```

---

## 5. 结果汇总到消融报告

把第 2 步各组 `benchmark_report.json` 的实测数值填入
`docs/ABLATION_REPORT.md` 第三节表格 (替换 `_____` 占位符),
并删除 §5.4 的"预期数值"或明确标注"实测 vs 预期"。

---

## 6. 推荐执行顺序 (单卡 RTX3060 时间预算参考)

| 阶段 | 命令 | 预估耗时 (3060 6GB) |
|------|------|------------------------|
| 0 冒烟 | §0 | ~5 min |
| 1 Crafter full | §1 | 数小时–1 天 |
| 2 Crafter 消融 ×4 | §2 crafter | ~1 天 |
| 1 MiniGrid full | §1 | 数小时 |
| 2 MiniGrid 消融 ×4 | §2 minigrid | ~1 天 |
| 1 Atari full | §1 | 数小时–1 天 (严格 10M 环境步) |
| 2 Atari 消融 ×4 | §2 atari | 1–数天 |
| 4 ONNX/Demo/视频 | §4 | ~30 min |

> Atari 现严格 10M 环境步 (≈40M 帧), 远轻于旧口径; 交付报实测分 + 相对 PPO 基线 120 分提升 (方案 a),
> 不再追求 3500+。若时间有限, 可先交付 Crafter + MiniGrid 的完整「full+四组消融」,
> Atari 先跑 full 与 no_rnd/no_episodic 两组。
