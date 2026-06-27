# Wandb 训练可视化 — 8 面板仪表盘配置

本项目的 Wandb 仪表盘包含 8 个核心面板，全面监控好奇心 PPO 训练过程。

## 面板设计

### 1. 奖励曲线 (Rewards)
- **图表类型**: 折线图 (双线)
- **指标**:
  - `ext_reward_mean` — 外在奖励均值 (环境奖励)
  - `int_reward_mean` — 内在奖励均值 (好奇心驱动)
- **X 轴**: `global_step`
- **解读**: ext_reward 上升说明任务性能提升; int_reward 下降说明探索逐渐充分

### 2. 好奇心分解 (Curiosity Breakdown)
- **图表类型**: 折线图 (三线)
- **指标**:
  - `icm_forward_loss` — ICM 前向预测损失 (短前向预测好奇心)
  - `rnd_loss` — RND 预测误差 (全局长期好奇心)
  - `episode_count` — Episode 计数 (训练进度)
- **解读**: 三条线各自反映不同时间尺度的新颖性信号

### 3. 价值损失 (Value Losses)
- **图表类型**: 折线图 (双线)
- **指标**:
  - `value_ext_loss` — 外在价值函数损失
  - `value_int_loss` — 内在价值函数损失
- **解读**: 双轨价值头各自收敛情况

### 4. 策略指标 (Policy Metrics)
- **图表类型**: 折线图 (三线)
- **指标**:
  - `policy_loss` — PPO 策略损失
  - `entropy` — 策略熵 (探索-利用平衡)
  - `clip_fraction` — PPO clip 比例
- **解读**: entropy 下降说明策略逐渐确定; clip_fraction 应在 0.1-0.3 范围

### 5. 评测分数 (Evaluation Score)
- **图表类型**: 折线图 (单线 + 基线水平线)
- **指标**:
  - `eval_score` — 评测得分 (Crafter: 归一化%, Atari: 游戏分数, MiniGrid: 成功率)
- **基线**:
  - Crafter: 15.6 (PPO baseline)
  - Atari: 120 (PPO baseline)
  - MiniGrid: 95% 成功率
- **解读**: 曲线应持续上升并超过基线

### 6. VRAM 使用 (Memory)
- **图表类型**: 面积图 (双面积)
- **指标**:
  - `vram_allocated_mb` — 当前分配显存 (MB)
  - `vram_peak_mb` — 峰值显存 (MB)
- **阈值**: 6000 (RTX 3060 6GB 上限)
- **解读**: 峰值应 < 3000MB, 确保 6GB 约束内运行

### 7. 训练效率 (Efficiency)
- **图表类型**: 折线图 + 直方图
- **指标**:
  - `fps` — 每秒交互步数 (折线)
  - `episode_length` — Episode 长度分布 (直方图)
- **解读**: FPS 反映训练速度; episode_length 变化反映探索行为

### 8. alpha_t 分布 (NGU Modulation)
- **图表类型**: 折线图 + 直方图
- **指标**:
  - `alpha_t_mean` — alpha_t 均值 (折线, 范围 [1, L=5])
  - `alpha_t_hist` — alpha_t 分布 (直方图)
- **解读**: alpha_t 反映 RND 长期新颖性调制; 新区域 alpha_t 接近 L, 已探索区域接近 1

## 消融实验对比

在 Wandb 中使用 **Group** 功能对比四组消融:

| 分组 | 配置 | 预期 ext_reward | 预期 int_reward |
|------|------|-----------------|-----------------|
| `full` | ICM + RND + Episodic | 最高 | 最稳定下降 |
| `no_icm` | RND + Episodic | 下降 | 缺少特征过滤 |
| `no_episodic` | ICM + RND | 下降 | 缺少短期新颖性 |
| `no_rnd` | ICM + Episodic | 下降 | 缺少长期调制 |

### 对比面板
- **奖励对比**: 4 条 `ext_reward_mean` 曲线, 按 `ablation` 分组
- **好奇心对比**: 4 条 `int_reward_mean` 曲线
- **效率对比**: 4 条 `eval_score` 曲线

## 使用方法

```bash
# 启用 Wandb 训练
python scripts/train.py --config experiments/crafter_full.yaml --use-wandb

# 运行消融对比 (自动分组)
python scripts/run_ablation.py --env crafter --steps 1000000 --use-wandb

# 或使用 PowerShell 一键脚本
.\scripts\run_all_ablation.ps1 -UseWandb
```

在 Wandb Web UI 中:
1. 进入项目页面
2. 点击 "Add Panel" 创建上述 8 个面板
3. 使用 "Group By" 功能按 `ablation` 分组对比消融实验
