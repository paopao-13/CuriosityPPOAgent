# Atari 训练完成后续 + 消融实验完整操作手册

> 适用范围：DSW 实例（A10）上已续训完成 Montezuma 10M 主线后，做评测 + 消融实验。
> 计费提醒：实例按运行时长烧资源包（约 7 CU·H/小时），**任何不跑训练/评测的空档都要停实例**。
> 资源包剩余（截至 07-18 中午）：约 65,778,000 / 100,000,000 ≈ 66 CU·H。

---

## 总流程一览

```
主线训练完成
   └─ ① 确认完成（tail log）
        └─ ② 评测主线（evaluate.py，10 episodes）
             └─ ③ 核心消融 3 组（no_icm / no_episodic / no_rnd，各 3M）
                  └─ ④ 可选：主线+消融多种子（seed 42/43/44）拉统计量
                       └─ ⑤ 备份（tar） + 停实例
```

⚠️ **关键省钱原则**：步骤 ① ② ⑤ 可以连续做（都在同一实例会话内）；
步骤 ③ ④ 耗时长，建议**单独开一次实例**跑完立刻停，不要和主线评测混着空转。

---

## ① 确认主线训练完成

```bash
tail -20 /mnt/workspace/atari_seed42/train.log
```
确认最后一行 `global_step` ≈ 10000000，且无 `Traceback` / `NaN` / `OOM`。

---

## ② 评测主线（实例还开着、GPU 可用，几分钟）

```bash
# 自动找最新 ckpt 并评测
LATEST=$(python -c "import glob,os; fs=sorted(glob.glob('/mnt/workspace/atari_seed42/step_*.pt'), key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0])); print(fs[-1])")
echo "Evaluating: $LATEST"
python scripts/evaluate.py \
  --config experiments/atari_montezuma_full.yaml \
  --checkpoint "$LATEST" \
  --eval-episodes 10 \
  2>&1 | tee /mnt/workspace/atari_seed42/eval.log

# 看结果
cat /mnt/workspace/atari_seed42/eval.log
```

产物：`benchmark_report.json / .md`（蒙特祖玛真实得分）。
注意：训练时 `ext_reward_mean=0` 是稀疏奖励假象，**评测才出真分**。

---

## ③ 核心消融实验（单独开实例跑，跑完立刻停）

> 主线条已用 10M 跑完，消融只跑**对比组**（跳过 full，避免重复 10M）。
> 仓库已含配置：`atari_montezuma_{no_icm, no_episodic, no_rnd}.yaml`

**推荐命令（单 seed 42，每组 3M 步）：**
```bash
# 在 /mnt/workspace/curiosity-ppo 目录下
python scripts/run_ablation.py \
  --env atari \
  --steps 3000000 \
  --seeds 42 \
  --ablations no_icm,no_episodic,no_rnd
```

脚本会自动：
1. 对每组跑 `scripts/train.py --config experiments/atari_montezuma_<ablation>.yaml --total-steps 3000000 --seed 42 --checkpoint-dir results/ablation/atari_montezuma_<ablation>/seed_42`
2. 训练完自动加载最新 ckpt 评测（atari 默认 10 episodes）
3. 结果写入 `results/ablation/atari_montezuma_<ablation>/seed_42/benchmark_report.json(.md)`

**耗时估算**：3 组 × 3M ≈ 9M 步 ≈ 7 小时（按 ~1.23M 步/小时）。
**额度**：约 7 CU·H × 7h ≈ 49 CU·H —— 但当前剩余仅 ~66 CU·H，**跑完消融就剩 ~17 CU·H，不够做多种子了**。

### 省额度选项（三选一，按需）

| 方案 | 命令改动 | 耗时 | 额度消耗 | 适用 |
|------|---------|------|---------|------|
| **A. 标准（推荐）** | 原命令 | ~7h | ~49 CU·H | 额度够、要完整对比 |
| **B. 减步数** | `--steps 2000000` | ~4.7h | ~33 CU·H | 想留点余量 |
| **C. 只跑核心组** | `--ablations no_icm` | ~2.4h | ~17 CU·H | 只想证「好奇心有增益」这一条结论 |

> 消融实验的意义：`no_icm`（PPO+RND+Episodic 基线）得分若明显低于 full，就证明 **ICM 内在好奇心模块带来实质增益**——这是作品集的核心故事点。

---

## ④ 可选：多种子统计（拉 mean ± std，增强说服力）

> ⚠️ **多种子很费额度**：3  ablation × 3 seed × 3M = 27M 步 ≈ 22h ≈ 154 CU·H —— **远超当前剩余 66 CU·H，做不了。**
> 若以后有额度，命令如下（仅供参考，本次不执行）：

```bash
# 核心消融多种子（需充足额度）
python scripts/run_ablation.py --env atari --steps 3000000 --seeds 42,43,44 \
  --ablations no_icm,no_episodic,no_rnd

# 主线 full 多种子（headline 指标统计）
python scripts/run_ablation.py --env atari --steps 10000000 --seeds 42,43,44 \
  --ablations full
```
跑完自动聚合到 `results/ablation/atari_montez_<ablation>/aggregate/aggregate_report.json(.md)`（mean ± std）。

**本次建议**：主线 full 只跑了 seed 42（10M），消融也只 seed 42（3M）。报告里如实写「单种子」，不编造统计量。

---

## ⑤ 备份 + 停实例（保钱包 + 防 07-25 释放）

消融跑完后，在**同一实例会话内**立刻打包：

```bash
cd /mnt/workspace
# 主线结果
tar czf atari_seed42_backup.tar.gz atari_seed42/
# 消融结果
tar czf atari_ablation_backup.tar.gz results/ablation/
ls -lh atari_seed42_backup.tar.gz atari_ablation_backup.tar.gz
```

然后在 **DSW 控制台 → 点「停止」实例**。

> 数据下载：实例停了磁盘还在（释放截止 07-25）。下次启动实例后用 OSS / 文件浏览器下载这两个 tar.gz 到本地。

---

## ⑥ 结果怎么用（写到作品集）

最终你要的对比表（单种子，10M full vs 3M 消融）：

| 配置 | 模块状态 | Montezuma 得分（10 episodes 均值） |
|------|---------|----------------------------------|
| **full** | RND + ICM + Episodic | （主线评测结果） |
| no_icm | RND + Episodic（无 ICM 好奇心） | （消融结果） |
| no_episodic | RND + ICM（无情景记忆） | （消融结果） |
| no_rnd | ICM + Episodic（无 RND 长期调制） | （消融结果） |

**核心结论模板**：
> "去掉 ICM 后得分从 X 降到 Y，证明内在好奇心模块贡献了 Z% 的相对提升；
> 去掉 Episodic/RND 分别带来 ... 说明三者协同工作。"

（具体数字等评测跑出来再填，不要提前编。）

---

## 命令速查卡

```bash
# ② 评测主线
LATEST=$(python -c "import glob,os; fs=sorted(glob.glob('/mnt/workspace/atari_seed42/step_*.pt'), key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0])); print(fs[-1])")
python scripts/evaluate.py --config experiments/atari_montezuma_full.yaml --checkpoint "$LATEST" --eval-episodes 10 2>&1 | tee /mnt/workspace/atari_seed42/eval.log

# ③ 核心消融（推荐方案 A）
python scripts/run_ablation.py --env atari --steps 3000000 --seeds 42 --ablations no_icm,no_episodic,no_rnd

# ⑤ 备份
cd /mnt/workspace && tar czf atari_seed42_backup.tar.gz atari_seed42/ && tar czf atari_ablation_backup.tar.gz results/ablation/ && ls -lh *.tar.gz
# 然后去控制台停实例
```
