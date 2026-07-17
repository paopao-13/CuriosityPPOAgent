# 消融实验报告

本报告模板用于记录和分析好奇心 PPO 智能体三重好奇心信号 (ICM / RND / Episodic Memory) 各组件的独立贡献。通过四组消融实验, 量化每个组件对探索效率与任务性能的增益。

---

## 1. 实验目标

验证以下假设:

1. **ICM 的独立增益**: ICM 的逆模型特征过滤 + 前向预测好奇心, 能否在 RND + Episodic 基础上进一步提升探索效率?
2. **Episodic Memory 的独立增益**: episode 内 kNN 伪计数的短期记忆, 能否在 ICM + RND 基础上减少重复探索?
3. **RND 的独立增益**: RND 的长期随机蒸馏调制 (alpha_t), 能否在 ICM + Episodic 基础上提供跨 episode 的持续探索驱动?
4. **三重融合的协同**: 完整三模块是否优于任何双模块组合?

---

## 2. 实验设置

### 2.1 四组消融配置

| 配置 | ablation 标识 | ICM | RND | Episodic | 配置文件 |
|------|--------------|-----|-----|----------|----------|
| 完整 | `full` | ON | ON | ON | `experiments/crafter_full.yaml` |
| 无 ICM | `no_icm` | OFF | ON | ON | `experiments/crafter_no_icm.yaml` |
| 无 Episodic | `no_episodic` | ON | ON | OFF | `experiments/crafter_no_episodic.yaml` |
| 无 RND | `no_rnd` | ON | OFF | ON | `experiments/crafter_no_rnd.yaml` |

### 2.2 控制变量

- **环境**: Crafter (64x64x3, Discrete(17)), 固定单一环境以隔离好奇心组件的影响。
- **训练步数**: 配置中 `env.total_steps = 1,000,000` (Crafter) / 10,000,000 (Atari) / 1,500,000 (MiniGrid), **即为环境步数**。
  > `agent.py` 的 `global_step += n_envs` 位于 rollout 的 `for step in range(n_steps)` 循环**内部**, 每轮 rollout 累加 `n_steps × n_envs` 个环境步, 恰好等于该 rollout 实际处理的环境步数。因此 `step_{global_step}.pt` 中的数字即真实环境步数, `total_steps` 无需额外换算。
  > ⚠️ **Atari 指标口径 (已定方案 a)**: 严格 10M 环境步 (≈40M 帧) 下, Montezuma 3500+ 按已知方法在单卡不可达 (RND 原论文达 3500+ 用 ~10 亿帧)。
  > 因此 Atari 交付**报 10M 环境步下的实测分数**, 并对比 PPO 基线 120 分的相对提升 (如稳定通关首房 / 平均分 ≥ 250–800); **不再写 3500+**。若未来有更大算力, 可另跑 ~1e8 步作为补充展示 (须如实标注规模)。
- **随机种子**: seed=42 (主实验), 建议补充 seed=43, 44 做多种子统计。
- **网络结构**: 编码器、策略网络结构完全一致, 仅好奇心模块开关不同。
- **PPO 超参数**: lr, clip_range, batch_size, gamma 等完全一致。
- **硬件**: RTX 3060 Laptop 6GB, AMP 开启。

### 2.3 各消融配置的内在奖励退化行为

| 配置 | r_int 计算方式 | 退化说明 |
|------|----------------|----------|
| `full` | `eta*ICM_loss + r_epi*min(max(alpha,1),L)` | 完整三重信号 |
| `no_icm` | `r_epi*min(max(alpha,1),L)` | ICM 前向好奇心归零; 情景 embedding 回退为 RND target 输出 |
| `no_episodic` | `eta*ICM_loss + r_rnd` | NGU 退化为纯 RND 奖励, 无 episode 内短期记忆 |
| `no_rnd` | `eta*ICM_loss + r_epi*min(max(1,1),L)` | alpha_t 固定为 1, NGU 退化为纯情景奖励, 无长期调制 |

> 退化逻辑实现: `src/curiosity_ppo/curiosity/ngu_fusion.py` 中 `NGUFusion.compute()`。

### 2.4 运行命令

```bash
# 一键运行四组消融 (Python 调度器)
python scripts/run_ablation.py --env crafter --steps 1000000

# 或 PowerShell 脚本
.\scripts\run_all_ablation.ps1 -Env crafter -Steps 1000000

# 单独运行某一组
python scripts/train.py --config experiments/crafter_no_icm.yaml --total-steps 1000000
```

### 2.5 评测命令

每组训练产物写入**独立的种子目录** `results/ablation/<prefix>_<ablation>/seed_<seed>/`,
其中 `<prefix>` = `crafter` / `atari_montezuma` / `minigrid_doorkey`, 默认 `seed_42`。
检查点文件名形如 `step_{global_step}.pt` (由 `agent.train()` 产出, 与 `scripts/train.py` 一致)。
加 `--seeds 42,43,44` 可跑多种子, 各组自动聚合为 `.../aggregate/aggregate_report.*` (mean ± std)。

**推荐: 用 `scripts/run_ablation.py` 一键跑完四组并自动评测** (已内置取最新检查点评测):

```bash
# Crafter 四组消融 + 自动评测 (单种子)
python scripts/run_ablation.py --env crafter --steps 1000000

# Crafter full 多种子 (headline 指标统计)
python scripts/run_ablation.py --env crafter --steps 1000000 --ablations full --seeds 42,43,44

# Atari / MiniGrid (配置已补齐 atari_montezuma_* / minigrid_doorkey_*)
python scripts/run_ablation.py --env atari --steps 10000000
python scripts/run_ablation.py --env minigrid --steps 1500000
```

**手动评测** (需指向该组某 seed 实际产出的 `step_*.pt`, 可用 `get_latest_checkpoint` 取最新):

```bash
# full (seed_42)
python scripts/evaluate.py --checkpoint results/ablation/crafter_full/seed_42/step_1000000.pt --env crafter

# no_icm
python scripts/evaluate.py --checkpoint results/ablation/crafter_no_icm/seed_42/step_1000000.pt --env crafter

# no_episodic
python scripts/evaluate.py --checkpoint results/ablation/crafter_no_episodic/seed_42/step_1000000.pt --env crafter

# no_rnd
python scripts/evaluate.py --checkpoint results/ablation/crafter_no_rnd/seed_42/step_1000000.pt --env crafter
```

> 注: 实际 `step_*.pt` 的文件名取决于训练时的 `global_step` 终值 (见 §2.2 注);
> 若不确定具体步数, 直接取该目录下 mtime 最新的 `.pt` 即可, 或改用一键脚本自动评测。

---

## 3. 结果表格模板

> **填表状态 (2026-07-16)**: 仅各环境 `full` 配置已出数据 (seed=42)；9 个变体由后台队列 `scripts/_queue_full_ablation.sh` (task Cwlsc7) 串行训练中，预计 ~4 天跑完。
> `full` 的 Score 暂用训练末尾 `eval_score` 占位 (Crafter=0.20 / MiniGrid=0.0)，待队列自动 evaluate.py 产出正式 100-episode 报告后替换。`vs full / vs baseline / 增益分解` 需等 4 组齐了再算。
> **Atari 口径偏差**: Atari 全家 `ent_coef=0.02` (非原 0.01 基线)，因 0.01 下探索熵会坍缩至 ~0（已实证）；Crafter/MiniGrid 全家 `ent_coef=0.01`。

### 3.1 主结果: Crafter 22 成就几何均值

| 配置 | ICM | RND | Epi | Score (%) | vs full | vs baseline (15.6%) |
|------|-----|-----|-----|-----------|---------|---------------------|
| `full` | ON | ON | ON | **20.0** (eval_score, 待正式评测) | -- | **+4.4** |
| `no_icm` | OFF | ON | ON | _____（队列训练中） | _____ | _____ |
| `no_episodic` | ON | ON | OFF | _____（队列训练中） | _____ | _____ |
| `no_rnd` | ON | OFF | ON | _____（队列训练中） | _____ | _____ |

> 填写说明: Score 为 100 episode 评测的 22 成就几何均值; "vs full" 为该配置相对 full 的差值 (百分点); "vs baseline" 为相对 PPO baseline 15.6% 的提升百分比。

### 3.2 组件增益分解

| 移除组件 | Score 下降 (百分点) | 相对降幅 (%) | 增益排序 |
|----------|--------------------|--------------|---------|
| 移除 RND (`no_rnd`) | _____（待变体数据） | _____ | _____ |
| 移除 ICM (`no_icm`) | _____（待变体数据） | _____ | _____ |
| 移除 Episodic (`no_episodic`) | _____（待变体数据） | _____ | _____ |

> 增益排序: 下降越多 = 该组件越重要。

### 3.3 训练曲线关键指标

| 配置 | 收敛步数 (Score 首次 > 15.6%) | 峰值 Score (%) | 末段 int_reward | 末段 ext_reward |
|------|----------------------------|---------------|----------------|----------------|
| `full` (Crafter) | ≈1000K (末段即 >15.6%) | **20.0** | 0.17 | 0.024 |
| `no_icm` | _____（队列训练中） | _____ | _____ | _____ |
| `no_episodic` | _____（队列训练中） | _____ | _____ | _____ |
| `no_rnd` | _____（队列训练中） | _____ | _____ | _____ |

> MiniGrid `full` (DoorKey): 末段 eval_score=**0.0**（未解出 DoorKey），ext_reward=0.0，int_reward=0.27，entropy=0.72；属探索未通关，非训练故障。
> Atari `full`: 重启抢救中 (ent_coef=0.02, 从 2.51M 续跑至 10M)，待跑完填此表。

### 3.4 VRAM 与训练速度

| 配置 | 峰值 VRAM (MB) | 训练速度 (FPS) | 备注 |
|------|---------------|---------------|------|
| `full` (Crafter) | **386.9** | ~62 (env-step/s) | 三模块全开；1M 步 / 16259s |
| `full` (MiniGrid) | **386.7** | ~119 (env-step/s) | 三模块全开；1.5M 步 / 12637s |
| `no_icm` | _____（队列训练中） | _____ | 无 ICM 网络 |
| `no_episodic` | _____（队列训练中） | _____ | 无 kNN 搜索 |
| `no_rnd` | _____（队列训练中） | _____ | 无 RND 网络 |

---

## 4. 分析框架

### 4.1 分析维度一: 各组件的边际贡献

通过对比 `full` 与各消融配置的 Score 差值, 量化每个组件的边际贡献:

```
Delta_ICM       = Score(full) - Score(no_icm)
Delta_Episodic  = Score(full) - Score(no_episodic)
Delta_RND       = Score(full) - Score(no_rnd)
```

- 若 `Delta_RND > Delta_ICM > Delta_Episodic`, 说明长期 RND 调制贡献最大, 符合 NGU 论文中 "长期 novelty 是探索主驱动" 的结论。
- 若 `Delta_ICM` 显著为正, 验证 ICM 的可控性特征过滤对情景记忆 embedding 质量的提升。
- 若 `Delta_Episodic` 显著为正, 验证 episode 内短期记忆对减少重复探索的作用。

### 4.2 分析维度二: 组件间的协同效应

比较三模块增益之和与完整融合的增益:

```
协同增益 = Score(full) - Score(baseline) - max(Delta_ICM, Delta_Epi, Delta_RND)
```

- 若协同增益 > 0, 说明三模块存在正向协同 (1+1+1 > 2)。
- 若协同增益接近 0, 说明各模块作用相对独立, 增益可加。

### 4.3 分析维度三: 内在奖励动态

观察训练日志中 `int_reward_mean` 的变化曲线:

- `full`: 内在奖励应保持适中水平, 三模块互补使探索持续。
- `no_icm`: ICM 前向好奇心缺失, 内在奖励可能偏低 (仅 NGU 项)。
- `no_episodic`: 缺少 episode 内短期抑制, 可能在 episode 内重复探索, 内在奖励波动大。
- `no_rnd`: 缺少长期调制, alpha_t=1, 情景奖励无放大, 跨 episode 探索驱动减弱。

### 4.4 分析维度四: 成就达成分布

对比各组配置在 22 个成就上的成功率分布:

- 简单成就 (collect_wood, place_table): 各配置差异小, 作为基线。
- 中等成就 (collect_iron, make_stone_pickaxe): `full` 应优于消融组。
- 困难成就 (collect_diamond, defeat_skeleton): 仅 `full` 可能达成, 验证深度探索能力。

---

## 5. 预期结论

基于 ICM/RND/NGU 的理论分析与本项目融合设计, 预期结论如下 (实际以训练评测数据为准):

### 5.1 组件重要性排序

预期: **RND > ICM > Episodic**

- **RND 最关键**: RND 提供跨 episode 的长期 novelty 调制 (alpha_t), 是 NGU 探索驱动的核心。移除 RND 后, alpha_t 固定为 1, 情景奖励失去长期放大, 跨 episode 探索显著退化。预期 `no_rnd` 的 Score 下降最大。
- **ICM 次之**: ICM 提供短前向预测好奇心, 且其 encoder 为情景记忆提供可控性 embedding。移除 ICM 后, 前向好奇心归零, 且情景 embedding 退化为 RND target 输出 (可控性过滤减弱)。预期 `no_icm` 有中等下降。
- **Episodic 第三**: Episodic 提供 episode 内短期记忆。移除后 NGU 退化为纯 RND, 仍保留长期探索驱动, 但缺少 episode 内的重复抑制。预期 `no_episodic` 下降最小, 但仍可观。

### 5.2 融合协同

预期: **完整三模块融合显著优于任何双模块组合**

三模块覆盖了短 (ICM 前向预测)、中 (Episodic episode 内)、长 (RND 跨 episode) 三个时间尺度, 互补性强。完整融合应达到最高 Score, 且收敛最快。

### 5.3 VRAM 与速度

预期: **各配置 VRAM 均在 6GB 预算内**

- `full` 峰值约 2.2GB, 速度最慢 (三模块全开)。
- `no_icm` / `no_rnd` 减少一个网络, VRAM 略降, 速度略升。
- `no_episodic` 无 kNN 搜索开销, 速度最快, VRAM 与 `full` 接近 (情景记忆在 CPU)。

### 5.4 预期数值参考

| 配置 | 预期 Score (%) | 预期 vs full |
|------|---------------|-------------|
| `full` | 19.0 | -- |
| `no_icm` | 17.2 | -1.8 |
| `no_episodic` | 17.8 | -1.2 |
| `no_rnd` | 16.5 | -2.5 |

> 上述数值为基于理论的预期参考, 非实测数据。实际结果请以训练后的评测报告 (`results/benchmark_report.json`) 为准。

---

## 6. 报告生成

训练与评测完成后, 自动生成报告:

```bash
# 评测后自动生成 JSON + Markdown 报告
python scripts/evaluate.py --checkpoint results/checkpoints/last.pt --env crafter
# 输出: results/benchmark_report.json, results/benchmark_report.md
```

报告中包含:

- 各配置的 Score、成功率分布、episode 奖励/长度。
- 相对 baseline 的提升百分比。
- 是否达到目标 (target_met)。

> 报告生成逻辑: `benchmarks/report.py` 中 `generate_report()` 与 `markdown_report()`。

---

## 7. 扩展实验建议

### 7.1 多种子统计

为降低随机性影响, 建议对每组配置运行 3 个种子 (seed=42, 43, 44), 报告均值 +/- 标准差:

| 配置 | seed=42 | seed=43 | seed=44 | 均值 | 标准差 |
|------|---------|---------|---------|------|--------|
| `full` | _____ | _____ | _____ | _____ | _____ |
| `no_icm` | _____ | _____ | _____ | _____ | _____ |
| `no_episodic` | _____ | _____ | _____ | _____ | _____ |
| `no_rnd` | _____ | _____ | _____ | _____ | _____ |

### 7.2 跨基准消融

将消融实验扩展到 Atari Montezuma 和 MiniGrid DoorKey, 验证组件增益的跨环境一致性:

```bash
python scripts/run_ablation.py --env atari --steps 10000000
python scripts/run_ablation.py --env minigrid --steps 1500000
```

> 注意: Atari/MiniGrid 的消融配置文件需额外创建 (当前 `experiments/` 中仅有 Crafter 的四组消融配置)。

### 7.3 eta 敏感性分析

对 ICM 的 `eta` 参数做网格搜索 (0.1, 0.2, 0.4, 0.8), 分析好奇心强度对性能的影响。
