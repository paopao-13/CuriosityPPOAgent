# CuriosityPPOAgent 长周期性能指标验收标准

> 项目：CuriosityPPOAgent ICM+RND 分层新颖好奇心 PPO 智能体
> 硬件：AMD R7 6800H + RTX3060 Laptop 6GB 笔记本显卡
> 显存红线：2.2GB，任意训练阶段超出直接判定测试失败
> 显存优化策略：FP16 AMP 混合精度、梯度累积 batch=128×4 等效 512、Rollout 轨迹缓存 CPU 离线存储、Episodic Memory LRU 最大 200 条特征向量；训练峰值显存稳定 2.2GB
> 本文档为不可篡改的固定验收基线，所有指标须经完整长周期训练复现方可判定通过。

---

## 目录

1. [三大基准环境完整长训练验收指标](#1-三大基准环境完整长训练验收指标)
2. [消融实验完整性能判定标准](#2-消融实验完整性能判定标准)
3. [显存占用性能判定标准](#3-显存占用性能判定标准)
4. [样本效率与内在奖励收敛稳定判定规则](#4-样本效率与内在奖励收敛稳定判定规则)
5. [技术架构与内在奖励融合机制](#5-技术架构与内在奖励融合机制)
6. [指标不达标分层排查方案](#6-指标不达标分层排查方案)
7. [验收执行流程与判定总则](#7-验收执行流程与判定总则)

---

## 1. 三大基准环境完整长训练验收指标

### 1.1 验收指标总表

下表为三大基准环境的完整长周期训练合格分数线、训练步数要求与评测方式。所有指标均为固定验收项，不得下调。

| 序号 | 基准环境 | 合格分数线 | 训练步数要求 | 配置文件 | 关键超参 | 评测脚本 | 评测方式 | 评测样本量 | 原生 PPO 基线 | 相对性能 | 判定逻辑 |
|:---:|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1 | Crafter 完整环境 | 稳定得分 ≥ 19.0% | 100 万交互步 | `experiments/crafter_full.yaml` | `total_steps=1000000`，`n_envs=8`，`n_steps=128` | `benchmarks/eval_crafter.py` | 22 个成就几何均值得分 | 100 个 episode | 15.6% | 相对提升 21.7% | 100 episode 几何均值 ≥ 19.0% 判定合格 |
| 2 | Atari Montezuma's Revenge | 稳定平均分 ≥ 3500 分 | 1000 万交互步 | `experiments/atari_montezuma_full.yaml` | `total_steps=10000000`，`n_envs=8` | `benchmarks/eval_atari.py` | 10 个 episode 平均分，不裁剪奖励 | 10 个 episode | 120 分 | 绝对得分提升 ≥ 29 倍 | 10 episode 平均分 ≥ 3500 且不裁剪奖励判定合格 |
| 3 | MiniGrid DoorKey-8x8 | 收敛步数 ≤ 96.8 万交互步 | 训练上限 `total_steps=1500000` | `experiments/minigrid_doorkey_full.yaml` | `total_steps=1500000`，`n_envs=8`，`n_steps=256` | `benchmarks/eval_minigrid.py` | 100 个 episode 成功率，`success_rate ≥ 0.95` 判定收敛 | 100 个 episode | 242 万步收敛 | 样本效率提升 2.5 倍 | 在 96.8 万步内达到 `success_rate ≥ 0.95` 判定合格 |

### 1.2 各环境验收细则

#### 1.2.1 Crafter 完整 100 万交互步

- **目标**：稳定得分 19.0%，相对原生 PPO 基线 15.6% 提升 21.7%。
- **训练配置**：`experiments/crafter_full.yaml`，`total_steps=1000000`，`n_envs=8`，`n_steps=128`。
- **评测方式**：`benchmarks/eval_crafter.py`，对 22 个成就计算几何均值得分，评测 100 个 episode。
- **合格判定**：100 个 episode 的成就几何均值得分稳定达到 ≥ 19.0%。
- **执行命令**：

```bash
# 训练
python train.py --config experiments/crafter_full.yaml
# 评测（包裹显存监控）
python check_vram_limit.py --wrap \
  "python benchmarks/eval_crafter.py --checkpoint outputs/crafter_full/best.pt --episodes 100"
```

#### 1.2.2 Atari Montezuma's Revenge 完整训练

- **目标**：稳定平均分 3500 分以上（原生 PPO 基线 120 分）。
- **训练配置**：`experiments/atari_montezuma_full.yaml`，`total_steps=10000000`，`n_envs=8`。
- **评测方式**：`benchmarks/eval_atari.py`，10 个 episode 平均分，**不裁剪奖励**（即保留原始得分，不得进行 reward clipping / reward shaping）。
- **合格判定**：10 个 episode 平均分 ≥ 3500，且评测全程不裁剪奖励。
- **执行命令**：

```bash
# 训练
python train.py --config experiments/atari_montezuma_full.yaml
# 评测（不裁剪奖励）
python check_vram_limit.py --wrap \
  "python benchmarks/eval_atari.py --checkpoint outputs/atari_montezuma_full/best.pt --episodes 10 --no-clip-reward"
```

#### 1.2.3 MiniGrid DoorKey-8x8 完整收敛训练

- **目标**：收敛仅需 96.8 万交互步（原生 PPO 收敛需 242 万步，样本效率提升 2.5 倍）。
- **训练配置**：`experiments/minigrid_doorkey_full.yaml`，`total_steps=1500000`，`n_envs=8`，`n_steps=256`。
- **评测方式**：`benchmarks/eval_minigrid.py`，100 个 episode 成功率，`success_rate ≥ 0.95` 判定收敛。
- **合格判定**：在 96.8 万交互步（968000 步）以内达到 `success_rate ≥ 0.95`。
- **执行命令**：

```bash
# 训练（带收敛检测）
python train.py --config experiments/minigrid_doorkey_full.yaml \
  --convergence-success-rate 0.95 --convergence-episodes 100
# 评测
python check_vram_limit.py --wrap \
  "python benchmarks/eval_minigrid.py --checkpoint outputs/minigrid_doorkey_full/best.pt --episodes 100"
```

### 1.3 验收前置约束

- **冒烟测试不视作验收通过依据**：1024 步或 2048 步短步数冒烟测试仅作为前置准入检查（验证训练循环不崩溃、环境兼容性修复有效），不构成性能验收达标依据。本节验收指标仅认可完整百万步长周期训练结果。
- 三大环境须各自独立完成完整长周期训练，禁止跨环境共享权重或复用中间 checkpoint。
- 训练须在指定硬件（AMD R7 6800H + RTX3060 Laptop 6GB）上完成，禁止更换至更高规格显卡重跑。
- 评测阶段显存占用一并纳入第 3 节显存红线判定。

---

## 2. 消融实验完整性能判定标准

### 2.1 消融实验设计

消融实验共 4 组，均基于 Crafter 环境进行完整 100 万交互步训练：

| 组别 | 标识 | 配置文件 | 说明 |
|:---|:---|:---|:---|
| 全模块 | `full` | `experiments/crafter_full.yaml` | ICM + RND + Episodic Memory 全部开启，作为对照基线 |
| 关闭 ICM | `no_icm` | `experiments/crafter_no_icm.yaml` | 仅关闭 ICM 模块，RND + Episodic Memory 保持开启 |
| 关闭情景记忆 | `no_episodic` | `experiments/crafter_no_episodic.yaml` | 仅关闭 Episodic Memory，ICM + RND 保持开启 |
| 关闭 RND | `no_rnd` | `experiments/crafter_no_rnd.yaml` | 仅关闭 RND 模块，ICM + Episodic Memory 保持开启 |

- 消融配置统一使用 Crafter 环境，`total_steps=1000000`，与 `full` 组保持训练预算一致。
- 每组消融实验需在三大基准环境（Crafter、Atari Montezuma、MiniGrid DoorKey）上各自评测性能，形成 4 组 × 3 环境 = 12 个判定单元。

### 2.2 消融判定总则

> **判定标准**：任意单一模块（ICM / Episodic Memory / RND）关闭后，三大环境性能均须出现明显下降，证明每个模块对最终性能存在独立正向增益。

- "明显下降"定义：相对 `full` 全模块组，该环境性能下降幅度须超过 **5%**（相对下降率）。
- 若任一模块关闭后在任一环境未出现明显下降（即下降幅度 ≤ 5% 或反而上升），判定该模块独立增益**不成立**，消融验收不通过。
- 全部 12 个判定单元均须满足"关闭后明显下降"，方判定消融实验整体合格。

相对下降率计算：

$$
\text{相对下降率} = \frac{\text{Score}_{full} - \text{Score}_{ablation}}{\text{Score}_{full}} \times 100\%
$$

### 2.3 消融实验 12 判定单元表

下表为 4 组消融 × 3 环境 = 12 个判定单元的完整验收矩阵。每个单元均要求"关闭该模块后性能明显下降（相对下降率 > 5%）"。

| 判定单元 | 消融组 | 评测环境 | `full` 基线参照 | 关闭后性能要求 | 相对下降率阈值 | 判定结果 |
|:---:|:---|:---|:---|:---|:---|:---|
| U1 | `no_icm`（关闭 ICM） | Crafter | 19.0% | < 19.0% 且相对下降率 > 5% | > 5% | 待判定 |
| U2 | `no_icm`（关闭 ICM） | Atari Montezuma | 3500 分 | < 3500 分且相对下降率 > 5% | > 5% | 待判定 |
| U3 | `no_icm`（关闭 ICM） | MiniGrid DoorKey | 96.8 万步收敛 | 收敛步数明显增加（相对下降率 > 5%） | > 5% | 待判定 |
| U4 | `no_episodic`（关闭情景记忆） | Crafter | 19.0% | < 19.0% 且相对下降率 > 5% | > 5% | 待判定 |
| U5 | `no_episodic`（关闭情景记忆） | Atari Montezuma | 3500 分 | < 3500 分且相对下降率 > 5% | > 5% | 待判定 |
| U6 | `no_episodic`（关闭情景记忆） | MiniGrid DoorKey | 96.8 万步收敛 | 收敛步数明显增加（相对下降率 > 5%） | > 5% | 待判定 |
| U7 | `no_rnd`（关闭 RND） | Crafter | 19.0% | < 19.0% 且相对下降率 > 5% | > 5% | 待判定 |
| U8 | `no_rnd`（关闭 RND） | Atari Montezuma | 3500 分 | < 3500 分且相对下降率 > 5% | > 5% | 待判定 |
| U9 | `no_rnd`（关闭 RND） | MiniGrid DoorKey | 96.8 万步收敛 | 收敛步数明显增加（相对下降率 > 5%） | > 5% | 待判定 |
| U10 | `full`（全模块对照） | Crafter | 19.0% | ≥ 19.0%（自身即基线） | — | 参照基准 |
| U11 | `full`（全模块对照） | Atari Montezuma | 3500 分 | ≥ 3500 分（自身即基线） | — | 参照基准 |
| U12 | `full`（全模块对照） | MiniGrid DoorKey | 96.8 万步收敛 | ≤ 96.8 万步收敛（自身即基线） | — | 参照基准 |

> 说明：U1–U9 为消融判定单元（须满足关闭后明显下降）；U10–U12 为 `full` 全模块参照基准，本身须先通过第 1 节验收。仅当 U10–U12 全部合格后，U1–U9 的消融判定方才生效。

### 2.4 消融执行命令

```bash
# 消融实验训练（4 组）
for cfg in crafter_full crafter_no_icm crafter_no_episodic crafter_no_rnd; do
  python check_vram_limit.py --wrap \
    "python train.py --config experiments/${cfg}.yaml"
done

# 消融模型在三大环境上评测
# Crafter
python benchmarks/eval_crafter.py --checkpoint outputs/crafter_no_icm/best.pt --episodes 100
# Atari Montezuma
python benchmarks/eval_atari.py --checkpoint outputs/crafter_no_icm/best.pt --episodes 10 --no-clip-reward
# MiniGrid DoorKey
python benchmarks/eval_minigrid.py --checkpoint outputs/crafter_no_icm/best.pt --episodes 100
```

---

## 3. 显存占用性能判定标准

### 3.1 显存红线

| 判定项 | 红线阈值 | 判定规则 |
|:---|:---|:---|
| 训练峰值显存 | ≤ 2.2GB | 全训练周期任意阶段峰值显存超过 2.2GB，直接判定测试失败 |
| 评测阶段显存 | ≤ 2.2GB | 评测阶段（含三大环境评测）峰值显存超过 2.2GB，判定评测失败 |
| 消融训练显存 | ≤ 2.2GB | 4 组消融训练各自峰值显存超过 2.2GB，判定该组消融失败 |

> 显存红线为硬性约束，一经超出立即终止并判定失败，不得通过重启或调参规避。

### 3.2 显存优化机制

本项目通过以下机制将训练峰值显存稳定控制在 2.2GB：

| 优化手段 | 说明 | 显存贡献 |
|:---|:---|:---|
| FP16 AMP 混合精度 | 前向 / 反向使用 FP16，主权重保留 FP32 | 降低激活与梯度显存占用 |
| 梯度累积 | `batch=128×4`，等效 batch=512，减少单步显存峰值 | 单步仅计算 128 batch，累积 4 步 |
| Rollout 轨迹缓存 CPU 离线存储 | 多环境 Rollout 轨迹暂存 CPU，按需回填 GPU | 避免轨迹长期驻留显存 |
| Episodic Memory LRU | 特征向量最大保留 200 条，LRU 淘汰 | 限制记忆库显存上限 |

### 3.3 显存监控执行方式

所有训练与评测命令必须使用 `check_vram_limit.py --wrap` 包裹，由该脚本对进程显存进行周期性采样与红线判定：

```bash
# 训练命令包裹示例
python check_vram_limit.py --wrap \
  "python train.py --config experiments/crafter_full.yaml" \
  --limit-gb 2.2 --interval-ms 500 --on-exceed fail

# 评测命令包裹示例
python check_vram_limit.py --wrap \
  "python benchmarks/eval_crafter.py --checkpoint outputs/crafter_full/best.pt --episodes 100" \
  --limit-gb 2.2 --interval-ms 500 --on-exceed fail
```

`check_vram_limit.py` 关键参数：

| 参数 | 作用 | 默认值 |
|:---|:---|:---|
| `--wrap` | 包裹被监控的训练 / 评测命令（必填） | — |
| `--limit-gb` | 显存红线阈值（GB） | 2.2 |
| `--interval-ms` | 显存采样间隔（毫秒） | 500 |
| `--on-exceed` | 超出红线时的动作：`fail`（终止并判失败）/ `warn`（仅告警） | fail |

### 3.4 显存验收判定流程

1. 训练 / 评测命令均以 `check_vram_limit.py --wrap` 包裹启动。
2. 脚本以 500ms 间隔采样 GPU 显存占用，记录全程峰值。
3. 全程峰值 ≤ 2.2GB → 显存验收通过；任一采样点 > 2.2GB → 立即终止进程并判定测试失败。
4. 训练日志须输出显存峰值摘要，作为验收凭证归档。

---

## 4. 样本效率与内在奖励收敛稳定判定规则

### 4.1 样本效率判定

| 环境 | 原生 PPO 收敛步数 | 本项目收敛步数 | 样本效率提升倍数 | 合格判定 |
|:---|:---|:---|:---|:---|
| MiniGrid DoorKey-8x8 | 242 万交互步 | ≤ 96.8 万交互步 | ≥ 2.5 倍 | 96.8 万步内 `success_rate ≥ 0.95` |
| Crafter | 原生 PPO 达 15.6% 所需步数参照 | 100 万步达 19.0% | 相对提升 21.7% | 100 万步内几何均值 ≥ 19.0% |
| Atari Montezuma | 原生 PPO 达 120 分所需步数参照 | 1000 万步达 3500 分 | 绝对得分提升 ≥ 29 倍 | 1000 万步内平均分 ≥ 3500 |

> 样本效率的核心验收锚点为 MiniGrid DoorKey-8x8：本项目收敛步数须 ≤ 96.8 万步，相对原生 PPO 的 242 万步提升 2.5 倍。

### 4.2 内在奖励收敛稳定判定

内在奖励（intrinsic reward）的收敛曲线是衡量好奇心机制健康度的关键指标。本项目内在奖励须满足以下收敛特征：

| 阶段 | 内在奖励特征 | 合格判定 |
|:---|:---|:---|
| 训练初期 | 内在奖励 ≈ 60（高探索状态） | 初始内在奖励均值在 60 附近，环境尚未被熟悉 |
| 训练中期 | 随训练逐步平稳下降 | 内在奖励随环境逐渐熟悉呈单调平稳下降趋势 |
| 训练后期 | 下降至平稳低位，趋于收敛 | 无反弹突增，曲线趋于收敛平台 |
| 全程 | 无剧烈震荡 | 任一滑动窗口内内在奖励标准差过大即判定异常 |

#### 4.2.1 收敛稳定性量化判定

- **初期基准**：训练前 5% 步数内，内在奖励均值 $\bar{r}_{int,early} \approx 60$，允许波动范围 $[50, 70]$。
- **下降趋势**：训练全程内在奖励滑动均值须呈整体下降趋势，下降斜率为负。
- **无剧烈震荡判定**：在长度为 $W$ 的滑动窗口内，内在奖励标准差 $\sigma_W$ 须满足：

$$
\sigma_W < \theta_{\sigma}, \quad \theta_{\sigma} = 0.3 \times \bar{r}_{int, W}
$$

其中 $\bar{r}_{int, W}$ 为窗口 $W$ 内内在奖励均值。若任一窗口 $\sigma_W \geq \theta_{\sigma}$，判定内在奖励剧烈震荡，测试异常。

- **后期收敛平台**：训练末 10% 步数内，内在奖励均值须显著低于初期（$\bar{r}_{int,late} < 0.5 \times \bar{r}_{int,early}$），且趋于稳定平台。

#### 4.2.2 内在奖励监控

```bash
# 导出内在奖励曲线
python tools/plot_intrinsic_reward.py \
  --logdir outputs/crafter_full/ \
  --output outputs/crafter_full/intrinsic_reward_curve.png

# 收敛稳定性自动判定
python tools/check_intrinsic_convergence.py \
  --logdir outputs/crafter_full/ \
  --early-window 0.05 \
  --late-window 0.10 \
  --sigma-threshold 0.3 \
  --early-target 60 --early-range 50 70
```

---

## 5. 技术架构与内在奖励融合机制

### 5.1 模块架构

| 模块 | 结构说明 | 关键参数 |
|:---|:---|:---|
| ICM | 4 层 CNN 编码器 → 288 维特征；逆动力学稀疏 Softmax 损失 + 前向 MSE 损失 | 动作空间 17 维；随机策略初始逆动力学损失 $\approx 2.83 = \ln 17$ |
| RND | 固定 Target 随机网络 + 可训练 Predictor，MSE 误差 | 内在折扣 $\gamma_{int} = 0.99$ |
| Episodic Memory | CPU-FAISS KNN 检索，LRU 最大 200 条，L2 距离 | LRU 容量 200 |

### 5.2 内在奖励融合公式

内在奖励由 ICM 前向损失、RND 误差与情景记忆检索共同融合：

$$
r_{int} = \eta \times \text{ICM}_{forward\_loss} + r_{episodic} \times \min(\max(\alpha_t, 1), L)
$$

其中：
- $\eta$ 为 ICM 前向损失权重系数；
- $r_{episodic}$ 为情景记忆 KNN 检索得到的 episodic 内在奖励；
- $\alpha_t$ 为 novelty 自适应系数，$\min(\max(\alpha_t, 1), L)$ 将其截断在 $[1, L]$ 区间，避免奖励爆炸。

### 5.3 双价值头 PPO

| 价值头 | 折扣因子 | 作用域 |
|:---|:---|:---|
| 外在价值头 $V_{ext}$ | $\gamma_{ext} = 0.999$ | episode 截断（仅当前 episode） |
| 内在价值头 $V_{int}$ | $\gamma_{int} = 0.99$ | 跨 episode（内在奖励持续累积） |

---

## 6. 指标不达标分层排查方案

当任一验收指标不达标时，按以下 5 个方向逐层定位根因。每个方向给出"现象 → 定位命令 → 修复建议"。

### 6.1 方向一：ICM 特征提取失效

| 项目 | 内容 |
|:---|:---|
| **现象** | ICM 前向损失长期不下降或居高不下；逆动力学损失偏离 $\ln 17 \approx 2.83$ 后未收敛；内在奖励早期未出现 ≈ 60 的高探索状态；Crafter / Atari 得分明显低于合格线。 |
| **定位命令** | `python tools/diagnose_icm.py --checkpoint outputs/crafter_full/best.pt --action-dim 17`；`python tools/plot_feature_tsne.py --logdir outputs/crafter_full/` |
| **修复建议** | 1) 检查 4 层 CNN 编码器是否正常更新权重，确认梯度未截断；2) 确认逆动力学稀疏 Softmax 损失输入动作为 17 维 one-hot，随机策略初始损失应 ≈ 2.83；3) 确认前向 MSE 损失对 288 维特征计算，特征维度无错位；4) 若特征 t-SNE 分布塌缩，降低编码器学习率或增加正则。 |

### 6.2 方向二：RND 奖励虚假震荡

| 项目 | 内容 |
|:---|:---|
| **现象** | 内在奖励曲线出现剧烈震荡，滑动窗口 $\sigma_W \geq \theta_{\sigma}$；RND 误差在固定状态上波动剧烈；MiniGrid 收敛步数显著超过 96.8 万步。 |
| **定位命令** | `python tools/plot_intrinsic_reward.py --logdir outputs/crafter_full/`；`python tools/check_intrinsic_convergence.py --logdir outputs/crafter_full/ --sigma-threshold 0.3` |
| **修复建议** | 1) 确认 Target 随机网络权重已冻结（不参与训练），仅 Predictor 可训练；2) 检查 RND 输入观测是否归一化，未归一化会导致 MSE 误差尺度不稳；3) 确认 $\gamma_{int}=0.99$ 未被错误修改；4) 对 RND 误差施加滑动平均平滑或降低 Predictor 学习率，消除虚假震荡。 |

### 6.3 方向三：记忆库向量检索异常

| 项目 | 内容 |
|:---|:---|
| **现象** | 关闭 Episodic Memory 后性能下降不明显（消融 U4–U6 不满足相对下降率 > 5%）；情景记忆 $r_{episodic}$ 长期为 0 或异常饱和；Atari Montezuma 得分卡在低位无法突破。 |
| **定位命令** | `python tools/diagnose_episodic.py --logdir outputs/crafter_full/ --knn-k 1 --lru-cap 200`；`python tools/check_faiss_index.py --logdir outputs/crafter_full/` |
| **修复建议** | 1) 确认 CPU-FAISS KNN 索引正常构建，L2 距离检索返回有效近邻；2) 确认 LRU 容量为 200，未出现全量替换或空库；3) 检查融合公式中 $\min(\max(\alpha_t,1),L)$ 截断是否生效，$\alpha_t$ 是否被错误置 0；4) 若 $r_{episodic}$ 饱和，检查 novelty 系数 $\alpha_t$ 更新逻辑。 |

### 6.4 方向四：超参设置不合理

| 项目 | 内容 |
|:---|:---|
| **现象** | 训练发散或收敛过慢；显存峰值逼近或超过 2.2GB；Crafter 得分低于 19.0% 或 Atari 低于 3500 分。 |
| **定位命令** | `python tools/inspect_config.py --config experiments/crafter_full.yaml`；`python check_vram_limit.py --wrap "python train.py --config experiments/crafter_full.yaml" --limit-gb 2.2` |
| **修复建议** | 1) 确认 `total_steps` / `n_envs` / `n_steps` 与配置要求一致（Crafter 100万/8/128，Atari 1000万/8，MiniGrid 150万/8/256）；2) 确认梯度累积为 128×4 等效 512，单步显存峰值不超限；3) 确认 $\gamma_{ext}=0.999$、$\gamma_{int}=0.99$ 未被篡改；4) 若显存超限，优先启用 FP16 AMP 或降低单步 batch。 |

### 6.5 方向五：环境观测预处理出错

| 项目 | 内容 |
|:---|:---|
| **现象** | ICM / RND 输入特征维度错位；内在奖励初期未达 ≈ 60；三大环境得分系统性偏低；Atari 评测裁剪了奖励导致得分异常。 |
| **定位命令** | `python tools/inspect_observation.py --env crafter --action-dim 17`；`python tools/verify_eval_no_clip.py --checkpoint outputs/atari_montezuma_full/best.pt` |
| **修复建议** | 1) 确认观测预处理（归一化、resize、通道顺序）与 ICM 4 层 CNN 输入一致；2) 确认 Crafter 22 成就几何均值计算口径正确；3) 确认 Atari 评测使用 `--no-clip-reward`，未对奖励裁剪；4) 确认 MiniGrid 100 episode 成功率统计与 `success_rate ≥ 0.95` 判定逻辑一致。 |

### 6.6 排查优先级

| 优先级 | 排查方向 | 触发条件 |
|:---:|:---|:---|
| P0 | 显存超限（方向四） | `check_vram_limit.py` 报警或训练中断 |
| P0 | 内在奖励剧烈震荡（方向二） | $\sigma_W \geq \theta_{\sigma}$ |
| P1 | ICM 特征失效（方向一） | 逆动力学损失偏离 2.83 或前向损失不降 |
| P1 | 环境观测预处理出错（方向五） | 得分系统性偏低 / Atari 误裁剪奖励 |
| P2 | 记忆库检索异常（方向三） | 消融 U4–U6 不达标 |
| P2 | 超参设置不合理（方向四） | 收敛过慢或发散 |

---

## 7. 验收执行流程与判定总则

### 7.1 验收执行顺序

```text
Step 1  显存红线预检（check_vram_limit.py 空载基准）
Step 2  三大基准环境完整长训练（包裹显存监控）
        ├─ Crafter 100 万步 → eval_crafter.py（100 episode，几何均值 ≥ 19.0%）
        ├─ Atari Montezuma 1000 万步 → eval_atari.py（10 episode，不裁剪，≥ 3500 分）
        └─ MiniGrid DoorKey 150 万步上限 → eval_minigrid.py（100 episode，96.8 万步内 SR ≥ 0.95）
Step 3  内在奖励收敛稳定性判定（初期 ≈ 60，平稳下降，无剧烈震荡）
Step 4  消融实验 4 组训练（full / no_icm / no_episodic / no_rnd）
Step 5  消融 12 判定单元评测（4 组 × 3 环境，关闭后明显下降率 > 5%）
Step 6  汇总验收结论
```

### 7.2 判定总则

| 验收项 | 合格条件 | 失败即终止 |
|:---|:---|:---|
| 显存红线 | 全程峰值 ≤ 2.2GB | 是（任一阶段超出即判失败） |
| Crafter | 100 episode 几何均值 ≥ 19.0% | 否（单项不达标记录为不合格） |
| Atari Montezuma | 10 episode 平均分 ≥ 3500，不裁剪奖励 | 否 |
| MiniGrid DoorKey | 96.8 万步内 success_rate ≥ 0.95 | 否 |
| 样本效率 | MiniGrid 收敛步数 ≤ 96.8 万步（2.5 倍提升） | 否 |
| 内在奖励收敛 | 初期 ≈ 60，平稳下降，$\sigma_W < \theta_{\sigma}$ | 是（剧烈震荡即判异常） |
| 消融实验 | U1–U9 全部满足关闭后相对下降率 > 5% | 否 |

### 7.3 验收结论汇总

- 全部验收项（显存、三大环境、样本效率、内在奖励收敛、消融 12 单元）均合格 → **验收通过**。
- 任一硬性终止项（显存红线、内在奖励剧烈震荡）触发 → **验收失败**。
- 非终止项出现不合格 → 该项标记不合格，按第 6 节分层排查方案定位修复后重跑该项。

```bash
# 验收结论一键汇总
python tools/acceptance_summary.py \
  --crafter-logdir outputs/crafter_full/ \
  --atari-logdir outputs/atari_montezuma_full/ \
  --minigrid-logdir outputs/minigrid_doorkey_full/ \
  --ablation-logdirs outputs/crafter_no_icm outputs/crafter_no_episodic outputs/crafter_no_rnd \
  --vram-limit-gb 2.2 \
  --output test/acceptance_report.md
```

---

> 本标准为 CuriosityPPOAgent 长周期性能验收的固定基线，所有阈值与配置均为不可篡改项。验收执行须严格遵循第 7 节流程，任何指标不达标须按第 6 节分层排查方案定位根因并修复后重跑。
