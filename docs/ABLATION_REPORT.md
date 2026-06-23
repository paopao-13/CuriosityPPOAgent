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
- **训练步数**: 1,000,000 步 (所有配置相同)。
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

每组训练完成后, 加载检查点评测:

```bash
# full
python scripts/evaluate.py --checkpoint results/checkpoints/crafter_full.pt --env crafter

# no_icm
python scripts/evaluate.py --checkpoint results/checkpoints/crafter_no_icm.pt --env crafter

# no_episodic
python scripts/evaluate.py --checkpoint results/checkpoints/crafter_no_episodic.pt --env crafter

# no_rnd
python scripts/evaluate.py --checkpoint results/checkpoints/crafter_no_rnd.pt --env crafter
```

---

## 3. 结果表格模板

### 3.1 主结果: Crafter 22 成就几何均值

| 配置 | ICM | RND | Epi | Score (%) | vs full | vs baseline (15.6%) |
|------|-----|-----|-----|-----------|---------|---------------------|
| `full` | ON | ON | ON | _____ | -- | _____ |
| `no_icm` | OFF | ON | ON | _____ | _____ | _____ |
| `no_episodic` | ON | ON | OFF | _____ | _____ | _____ |
| `no_rnd` | ON | OFF | ON | _____ | _____ | _____ |

> 填写说明: Score 为 100 episode 评测的 22 成就几何均值; "vs full" 为该配置相对 full 的差值 (百分点); "vs baseline" 为相对 PPO baseline 15.6% 的提升百分比。

### 3.2 组件增益分解

| 移除组件 | Score 下降 (百分点) | 相对降幅 (%) | 增益排序 |
|----------|--------------------|--------------|---------|
| 移除 RND (`no_rnd`) | _____ | _____ | _____ |
| 移除 ICM (`no_icm`) | _____ | _____ | _____ |
| 移除 Episodic (`no_episodic`) | _____ | _____ | _____ |

> 增益排序: 下降越多 = 该组件越重要。

### 3.3 训练曲线关键指标

| 配置 | 收敛步数 (Score 首次 > 15.6%) | 峰值 Score (%) | 末段 int_reward | 末段 ext_reward |
|------|----------------------------|---------------|----------------|----------------|
| `full` | _____ | _____ | _____ | _____ |
| `no_icm` | _____ | _____ | _____ | _____ |
| `no_episodic` | _____ | _____ | _____ | _____ |
| `no_rnd` | _____ | _____ | _____ | _____ |

### 3.4 VRAM 与训练速度

| 配置 | 峰值 VRAM (MB) | 训练速度 (FPS) | 备注 |
|------|---------------|---------------|------|
| `full` | _____ | _____ | 三模块全开 |
| `no_icm` | _____ | _____ | 无 ICM 网络 |
| `no_episodic` | _____ | _____ | 无 kNN 搜索 |
| `no_rnd` | _____ | _____ | 无 RND 网络 |

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
