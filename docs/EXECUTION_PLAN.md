# 好奇心 PPO 项目 — 执行计划 (EXECUTION PLAN)

> 版本：v1（优化版，务实目标 + 可靠性优先）
> 适用硬件：RTX 3060 Laptop 6GB / 主机 16GB RAM
> 编写依据：基于当前仓库真实状态核查（非文档声明）

---

## 0. 当前真实状态（实测，非文档声明）

| 基准 | 目标 | 实际产物 | 状态 |
|---|---|---|---|
| MiniGrid DoorKey | 1.5M 步 | 146 ckpt，末行 `eval_score=0.0`，30 次评测全程 0% | ⚠️ 训练跑完但**未解出**（评测协议缺陷导致假 0%，算法未必失败） |
| Crafter | 1M 步 | 97 ckpt，末行 `eval_score=0.2`，训练内分数 0.7→0.2 衰减 | ⚠️ 不稳定，几何均值未实测 |
| Atari Montezuma | 10M 步 | **0 个 ckpt，进度全失** | ❌ 崩塌丢失（RAM 饥饿 OOM + 误删 ckpt） |

待交付缺口：
- 四组消融实验：**未跑**（仅有 `no_icm/no_episodic/no_rnd` 配置，无 `baseline` 全关配置，无结果）
- `benchmark_report.json`：README 引用但**不存在**
- 演示视频 / 真实 ONNX / 可用 Web Demo / Wandb run：**均无**

关键代码事实：
- `src/curiosity_ppo/utils/checkpoint.py:15` → `torch.save(checkpoint, path)` **非原子写**（被中途 kill 会损坏/消失）
- `src/curiosity_ppo/ppo/agent.py:603` → 评测用 `logits.argmax` **贪婪策略**，且复用训练 `vec_env`（seed=42 固定）→ MiniGrid 假 0% 根因
- `scripts/evaluate.py` → **已正确创建独立新环境**（`build_eval_env` n_envs=1，独立 seed），评测入口本身没问题，需确认基准函数是否随机采样 + 是否算 success_rate / 22 成就几何均值

---

## 1. 务实目标（降低后，可辩护 + 可复现 + 经得起答辩追问）

核心思想：**不堆绝对高分，改为"同预算下好奇心版严格胜过 PPO-only 基线 + 多种子可复现"**。

| 基准 | 必达线（务实） | 主指标 | 辅指标 |
|---|---|---|---|
| **MiniGrid DoorKey** | 解出 DoorKey | **多种子平均成功率 ≥ 0.95** | 首次到达 goal 的步数中位数（收敛步数） |
| **Crafter** | 22 成就几何均值 **≥ 15%**，且 **≥ 同预算 PPO-only 基线** | 几何均值% | 相对基线提升（百分点）、新解锁成就数 |
| **Atari Montezuma** | 10M 步实测均分 + **相对 PPO 120 分的提升** | 均分 | **首房通关率**（过第一道门锁即算，目标 ≥ 1/10 episode） |

> 注：原目标 "Atari 3500+ 分" 在 10M 步单卡下物理不可达（项目文档 `BENCHMARKS.md` / `ABLATION_REPORT.md` 已自认），故改为"相对 120 分有提升"，更严谨。
> 全部目标均不强求精确绝对数值，而以"相对基线增益 + 多种子可复现"为证据链。

---

## 2. 可靠性底座（Phase 0，必须先做，否则重跑也白跑）

### 2.1 原子化 checkpoint 落盘（修复 `checkpoint.py`）

把 `save_checkpoint` 改为先写 `.tmp` 再 `os.replace`（原子重命名，中途 kill 不丢旧 ckpt）：

```python
# src/curiosity_ppo/utils/checkpoint.py
import os, tempfile

def save_checkpoint(path, agent_state, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {'agent_state': agent_state}
    if extra:
        checkpoint.update(extra)
    tmp = path + ".tmp"
    torch.save(checkpoint, tmp)        # 先写临时文件
    os.replace(tmp, path)              # 原子替换（POSIX 保证）
```

同时改 `get_latest_checkpoint`：忽略 `.tmp` 文件（`if f.endswith('.pt')` 已天然排除）。

### 2.2 评测协议修正（修复 MiniGrid 假 0%）

**A. `agent.evaluate()`（训练内评测）改为随机采样 + 成功率：**

```python
# src/curiosity_ppo/ppo/agent.py  evaluate() 内
with torch.no_grad():
    logits, _, _ = self.actor_critic(obs_norm)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample()             # 改 argmax → 随机采样
eval_obs, reward, done, info = self.vec_env.step(action.cpu().numpy())
# 记录 success（MiniGrid DoorKey 在 info 中返回 is_success）
for i in range(self.n_envs):
    if done[i]:
        succ = bool(info[i].get('is_success', False)) if isinstance(info, (list, tuple)) else bool(info.get('is_success', False))
        self._eval_success.append(succ)
        ...
# 返回 (mean_reward, mean_success)
```

> 若 `info` 无 `is_success`，改由基准函数（`benchmarks/eval_minigrid.py`）用环境自身的 goal 到达判定。

**B. `scripts/evaluate.py` 的基准函数必须：**
- 随机采样动作（非 argmax），seed 固定可复现；
- MiniGrid 算 `success_rate`（到达 goal 的 episode 占比）；
- Crafter 算 **22 成就成功率 → 几何均值 → 归一化 Score%**（已有 `benchmarks/eval_crafter.py`，需确认其算的是该口径）；
- Atari 算首房通关率 + 均分。

**C. 评测时关闭 intrinsic reward**：eval 不应再给好奇心奖励，否则评的是"带作弊器"而非真策略（确认基准函数只用 `ext_reward`，不混入 `int_reward`）。

### 2.3 资源守卫固化（已修复的 `scripts/_queue_full_ablation.sh`，直接复用）

该脚本已实现：`--resume`（从目录内最新 ckpt 续跑）、`wait_for_mem 3072`（RAM<3GB 等待，不足则跳过）、单实例锁（防多份并发抢 RAM）、`--checkpoint-interval 10240`（小间隔落盘，崩了最多丢 10k 步）。

**规则**：所有训练/评测/视频录制一律经该脚本或等价守卫启动；**严禁 `kill -9` 父进程树**（上次 Atari ckpt 即因此消失），清理只杀明确 PID。

### 2.4 RUN_MANIFEST.json + 磁盘守卫

- 训练启动即写 `RUN_MANIFEST.json`：`{config_hash, seed, start_time, total_steps, checkpoint_dir}` → 崩溃后一眼可知"该跑到哪、跑到哪了"。
- 磁盘守卫：除 RAM 外加 `df` 检查，可用 <10GB 时暂停并报警（历史磁盘曾到 94%）。

---

## 3. 实验矩阵（Phase 1：对照组 + 好奇心组）

### 3.1 新建 PPO-only 基线配置（全关好奇心）

仓库已有 `crafter_full.yaml`（三模块全开）、`crafter_no_icm/no_episodic/no_rnd.yaml`，但**缺"全关"基线**。新建三份 `*_baseline.yaml`，内容 = 对应 `full.yaml` 但三个模块 `enabled: false` + `ablation: baseline`：

```yaml
# experiments/crafter_baseline.yaml  （minigrid/atari 同构）
ppo: { lr: 1.0e-4, clip_range: 0.1, ppo_epochs: 4, gamma_ext: 0.999,
       gamma_int: 0.99, gae_lambda: 0.95, ent_coef: 0.01, vf_coef: 0.5,
       max_grad_norm: 0.5, batch_size: 128, accumulation_steps: 4,
       n_steps: 128, n_envs: 8 }
rnd:    { enabled: false }
icm:    { enabled: false }
episodic: { enabled: false }
env: { name: crafter, n_envs: 8, total_steps: 1000000 }
ablation: baseline
use_amp: true
seed: 42
wandb_project: curiosity-ppo
```

### 3.2 命令（全部经守卫启动；Atari 最久，最先开跑）

```bash
PY="C:/Users/jx/AppData/Local/Programs/Python/Python311/python.exe"
CK="results/ablation"

# —— Crafter（1M 步，快，优先验证可行性）——
bash scripts/_queue_full_ablation.sh   # 已封装 full 三基准；或显式：
"$PY" scripts/train.py --config experiments/crafter_baseline.yaml \
  --checkpoint-dir $CK/crafter_baseline/seed_42 --total-steps 1000000 \
  --checkpoint-interval 10240 --use-compile
"$PY" scripts/train.py --config experiments/crafter_full.yaml \
  --checkpoint-dir $CK/crafter_full/seed_42 --total-steps 1000000 \
  --checkpoint-interval 10240 --use-compile

# —— MiniGrid（1.5M 步）——
"$PY" scripts/train.py --config experiments/minigrid_doorkey_full.yaml \
  --checkpoint-dir $CK/minigrid_doorkey_full/seed_42 --total-steps 1500000 \
  --checkpoint-interval 10240 --use-compile
# (baseline 同理 minigrid_doorkey_baseline.yaml)

# —— Atari（10M 步，后台长跑，最先启动）——
"$PY" scripts/train.py --config experiments/atari_montezuma_full_ent002.yaml \
  --checkpoint-dir $CK/atari_montezuma_full/seed_42 --total-steps 10000000 \
  --checkpoint-interval 10240
```

> **优化点（省 GPU）**：PPO-only 对照组**只需在 Crafter 上跑一份**（最具判别力、最便宜）；MiniGrid/Atari 的"纯 PPO"对照用**文献公认值**（PPO 在 Montezuma ≈120 分、DoorKey 纯 PPO 收敛约 2.4M 步）即可，不必自跑两份 10M/1.5M，省下数十小时。

---

## 4. 统一评测 → `benchmark_report.json`（Phase 2，消除文档硬伤）

写 `scripts/run_benchmarks.py`：对三基准分别调用 `evaluate_*` + `evaluate_crafter`，**5 个 seed（42/43/44/45/46）**，输出：

```json
{
  "minigrid_doorkey": {
    "curiosity":  {"success": 0.97, "std": 0.03, "conv_step_median": 920000},
    "ppo_baseline": {"success": 0.55, "std": 0.08, "conv_step_median": 2400000}
  },
  "crafter": {
    "curiosity":  {"geom_mean": 0.162, "std": 0.012, "new_achievements": 11},
    "ppo_baseline": {"geom_mean": 0.156, "std": 0.009, "new_achievements": 9}
  },
  "atari_montezuma": {
    "curiosity":  {"mean_score": 540, "std": 120, "first_room_rate": 0.3},
    "ppo_baseline": {"mean_score": 120}
  }
}
```

README 已引用此文件但缺失 —— 补上即堵住最大评审漏洞。

---

## 5. 四组消融（Phase 3，你明确列的交付项）

仓库已有 `no_icm/no_episodic/no_rnd` 配置，补跑 + baseline：

```bash
for ab in baseline no_icm no_episodic no_rnd; do
  for s in 42 43 44; do
    "$PY" scripts/train.py --config experiments/crafter_$ab.yaml \
      --seed $s --checkpoint-dir $CK/crafter_$ab/seed_$s \
      --total-steps 1000000 --checkpoint-interval 10240
  done
done
```

结果填入 `docs/ABLATION_REPORT.md` 待填表格，证明 ICM 特征过滤 / 短期记忆 / 长期 RND 三模块**各自独立增益**。

---

## 6. 交付物补齐（Phase 4）

```bash
# 演示视频（Crafter 最出片）
"$PY" scripts/record_video.py --checkpoint $CK/crafter_full/seed_42/step_*.pt \
  --env crafter --output results/videos/demo_crafter.mp4

# ONNX 导出真实 ckpt → web/public/models/ → Vite Demo 真能推理
"$PY" scripts/export_onnx.py --checkpoint $CK/crafter_full/seed_42/step_XXXXXX.pt \
  --output web/public/models/crafter.onnx

# Wandb：续跑加 --use-wandb，补 8 panel 仪表盘截图进 docs/WANDB_DASHBOARD.md
```

---

## 7. 文档对齐（Phase 5）

将 README / `release_note.md` / `BENCHMARKS.md` 里的「19.0% / 3500+ / 96.8 万步」统一改为**实测值**或明确标注「目标/理论值」。Atari 不写 3500+，如实报相对 120 分提升。

---

## 8. 统计显著性检验（答辩级可复现）

多种子结果用 **Welch t-test / bootstrap CI** 报 p 值：

```python
import numpy as np
from scipy import stats
def significance(cur, base):
    t, p = stats.ttest_ind(cur, base, equal_var=False)   # Welch
    diff = np.mean(cur) - np.mean(base)
    ci = np.percentile(
        [np.mean(cur)-np.mean(base) for _ in range(2000)
         for cur,base in bootstrap_resample(cur,base)], [2.5,97.5])
    return diff, ci, p   # 报告 "好奇心版显著优于基线 (p<0.05)"
```

---

## 9. 可靠性检查清单（每次启动前核对）

- [ ] checkpoint.py 已改原子写（`.tmp` + `os.replace`）
- [ ] 评测用随机采样 + 关 intrinsic + 算 success_rate / 几何均值
- [ ] 经资源守卫启动（`wait_for_mem` + 单实例锁 + `--resume` + `checkpoint-interval 10240`）
- [ ] 未用 `kill -9` 父进程树；清理只杀明确 PID
- [ ] 已写 `RUN_MANIFEST.json`；磁盘可用 >10GB
- [ ] 每 2h 巡检自动化在跑（进程静默死亡会报警）

---

## 10. 里程碑与时间（粗估）

| 阶段 | 内容 | 耗时 | 风险 |
|---|---|---|---|
| Phase 0 | 原子写 + 评测协议修正 + 守卫固化 | 0.5 天 | 低 |
| Phase 1 | 对照组 + 好奇心组（Atari 10M 最久，后台） | Atari 数十 h / 其余 1–2 h | 中（靠守卫缓解） |
| Phase 2–3 | 统一评测 + 四组消融 | 1–2 天 | 低 |
| Phase 4–5 | 视频/ONNX/Web/Wandb + 文档对齐 | 1 天 | 低 |

---

## 11. 仍存在的真实风险（诚实告知）

1. **Crafter 衰减未根治**：之前 eval_score 0.7→0.2，若因 `value_int_loss` 不稳/熵过高，重训可能再衰减 → 先跑 **200k 步小试验**验证稳定性，再投 1M 全量。
2. **Atari 10M 步能否"显著过首房"未知**：RND 原论文 ~1e8 帧才稳过首房，单卡 10M 步（~4e7 帧）可能只偶尔过 → 首房率目标设为"≥1/10"，否则可能达不成。
3. **RAM 仍是天花板**：任何并行/分段受 16GB 约束，必须守住 `wait_for_mem` 守卫。

---

## 12. 建议执行顺序

1. **先做 Phase 0 代码修复**（原子写 + 评测协议）—— 可逆、安全、信息量大。
2. **立即后台启动 Atari 10M 重训**（耗时最长，先开跑）。
3. 用修复后的评测协议**多种子复评现有 MiniGrid/Crafter ckpt**，立刻判断是"算法失败"还是"评测假象"。
4. 再跑 Crafter baseline + 消融，出 `benchmark_report.json` + `ABLATION_REPORT.md`。
5. 最后补视频/ONNX/Web/Wandb/文档。

> 本计划为"规划文档"，**未启动任何训练/评测进程**（遵守"先别跑了"指令）。确认后从 Phase 0 开始执行。
