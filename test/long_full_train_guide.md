# CuriosityPPOAgent 完整长周期性能验收训练指南

> **适用硬件**：AMD R7 6800H + RTX3060 Laptop 6GB
> **显存红线**：2.2GB（任意训练阶段超出直接判定失败）
> **验收性质**：Github 上传前最终硬性性能达标，冒烟测试不作为验收依据

---

## 一、RTX3060 显卡训练前置环境准备

### 1.1 独显直连开启

RTX3060 Laptop 需确保训练进程使用独显而非核显，操作步骤：

1. 桌面右键 → **NVIDIA 控制面板** → **管理 3D 设置** → **全局设置**
2. 首选图形处理器选择 **高性能 NVIDIA 处理器**
3. 电源管理模式选择 **最高性能**
4. 点击 **应用**

验证独显已激活：

```powershell
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')"
```

合格输出：

```text
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
VRAM: 6.0 GB
```

### 1.2 系统性能模式

1. 设置 → 系统 → 电源 → 电源模式选择 **最佳性能**
2. 确保笔记本电脑接通电源（禁止电池模式训练）
3. 关闭不必要的后台程序（浏览器、IDE 等），释放显存占用

### 1.3 显存监控脚本启动

在训练开始前，先验证显存基线：

```powershell
python test/scripts/check_vram_limit.py --baseline-only
```

合格输出：

```text
[VRAM Monitor] 当前显存: allocated=0.00MB, peak=0.00MB
[VRAM Monitor] 红线: 2200MB
[VRAM Monitor] 当前峰值正常
```

训练过程中使用包裹模式自动监控显存，超 2.2GB 自动终止：

```powershell
python test/scripts/check_vram_limit.py --wrap "python scripts/train.py --config experiments/crafter_full.yaml --use-wandb" --threshold 2200 --interval 2
```

---

## 二、三大环境完整长训练 PowerShell 执行命令

训练优先级：**MiniGrid（轻量）→ Crafter（中等）→ Atari Montezuma（大容量）**

### 2.1 MiniGrid DoorKey 完整收敛训练（目标 96.8 万步收敛）

```powershell
# 全新训练
python test/scripts/check_vram_limit.py --wrap "python scripts/train.py --config experiments/minigrid_doorkey_full.yaml --total-steps 1500000 --use-wandb --run-name minigrid_full --checkpoint-dir results/checkpoints/minigrid --checkpoint-interval 50000" --threshold 2200 --interval 2

# 断点续训（如训练中断，从最近检查点恢复）
python test/scripts/check_vram_limit.py --wrap "python scripts/train.py --config experiments/minigrid_doorkey_full.yaml --total-steps 1500000 --use-wandb --run-name minigrid_full --resume results/checkpoints/minigrid/step_XXXXXX.pt --checkpoint-dir results/checkpoints/minigrid --checkpoint-interval 50000" --threshold 2200 --interval 2
```

| 参数 | 值 | 说明 |
|------|----|------|
| `--config` | `experiments/minigrid_doorkey_full.yaml` | MiniGrid 配置 |
| `--total-steps` | `1500000` | 训练上限 150 万步 |
| `--use-wandb` | — | 启用 Wandb 日志 |
| `--checkpoint-interval` | `50000` | 每 5 万步保存检查点 |
| `--checkpoint-dir` | `results/checkpoints/minigrid` | 检查点目录 |
| 显存红线 | `2200` MB | 超 2.2GB 自动终止 |

### 2.2 Crafter 完整 100 万步训练（目标得分 19.0%）

```powershell
# 全新训练
python test/scripts/check_vram_limit.py --wrap "python scripts/train.py --config experiments/crafter_full.yaml --total-steps 1000000 --use-wandb --run-name crafter_full --checkpoint-dir results/checkpoints/crafter --checkpoint-interval 50000" --threshold 2200 --interval 2

# 断点续训
python test/scripts/check_vram_limit.py --wrap "python scripts/train.py --config experiments/crafter_full.yaml --total-steps 1000000 --use-wandb --run-name crafter_full --resume results/checkpoints/crafter/step_XXXXXX.pt --checkpoint-dir results/checkpoints/crafter --checkpoint-interval 50000" --threshold 2200 --interval 2
```

| 参数 | 值 | 说明 |
|------|----|------|
| `--config` | `experiments/crafter_full.yaml` | Crafter 配置 |
| `--total-steps` | `1000000` | 完整 100 万交互步 |
| `--checkpoint-interval` | `50000` | 每 5 万步保存检查点 |

### 2.3 Atari Montezuma's Revenge 完整训练（目标 3500+ 分）

```powershell
# 全新训练
python test/scripts/check_vram_limit.py --wrap "python scripts/train.py --config experiments/atari_montezuma_full.yaml --total-steps 10000000 --use-wandb --run-name atari_full --checkpoint-dir results/checkpoints/atari --checkpoint-interval 100000" --threshold 2200 --interval 2

# 断点续训
python test/scripts/check_vram_limit.py --wrap "python scripts/train.py --config experiments/atari_montezuma_full.yaml --total-steps 10000000 --use-wandb --run-name atari_full --resume results/checkpoints/atari/step_XXXXXX.pt --checkpoint-dir results/checkpoints/atari --checkpoint-interval 100000" --threshold 2200 --interval 2
```

| 参数 | 值 | 说明 |
|------|----|------|
| `--config` | `experiments/atari_montezuma_full.yaml` | Atari 配置 |
| `--total-steps` | `10000000` | 完整 1000 万交互步 |
| `--checkpoint-interval` | `100000` | 每 10 万步保存检查点 |

> **注意**：Atari 训练时间最长（1000 万步），显存峰值最高（4×84×84 帧堆叠 + NatureDQN 编码器），需密切监控显存。若峰值逼近 2.2GB，优先降低 `n_envs`（从 8 降到 4）。

---

## 三、4 组消融完整长训练批量执行脚本

消融实验均基于 Crafter 环境，完整 100 万步训练，4 组依次执行：

```powershell
# =====================================================================
# CuriosityPPOAgent 4 组消融完整长训练 (Windows PowerShell)
# 硬件: RTX3060 Laptop 6GB | 显存红线: 2.2GB
# 用法: 保存为 .ps1 文件后执行, 或直接在 PowerShell 中粘贴运行
# =====================================================================

$ablations = @("full", "no_icm", "no_episodic", "no_rnd")
$envName = "crafter"
$totalSteps = 1000000
$checkpointInterval = 50000

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  4 组消融完整长训练 ($totalSteps 步/组)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

foreach ($ablation in $ablations) {
    $configPath = "experiments/${envName}_${ablation}.yaml"

    if (-not (Test-Path $configPath)) {
        Write-Host "[SKIP] 配置不存在: $configPath" -ForegroundColor Yellow
        continue
    }

    $checkpointDir = "results/checkpoints/ablation_${ablation}"

    Write-Host "`n------ 消融组: $ablation ------`n" -ForegroundColor Green

    python test/scripts/check_vram_limit.py `
        --wrap "python scripts/train.py --config $configPath --total-steps $totalSteps --use-wandb --run-name crafter_${ablation} --checkpoint-dir $checkpointDir --checkpoint-interval $checkpointInterval" `
        --threshold 2200 --interval 2

    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] $ablation 训练完成" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] $ablation 训练失败 (exit=$LASTEXITCODE)" -ForegroundColor Red
        Write-Host "  可使用 --resume 断点续训:" -ForegroundColor Yellow
        $latestCkpt = Get-ChildItem "$checkpointDir/step_*.pt" -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        if ($latestCkpt) {
            Write-Host "  python scripts/train.py --config $configPath --total-steps $totalSteps --use-wandb --run-name crafter_${ablation} --resume $($latestCkpt.FullName) --checkpoint-dir $checkpointDir --checkpoint-interval $checkpointInterval" -ForegroundColor Yellow
        }
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  4 组消融训练全部完成" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan
```

消融配置对照表：

| 组名 | ICM | RND | Episodic | 配置文件 | 预期结果 |
|------|-----|-----|----------|----------|----------|
| `full` | ON | ON | ON | `experiments/crafter_full.yaml` | 得分最高（≥19.0%） |
| `no_icm` | OFF | ON | ON | `experiments/crafter_no_icm.yaml` | 得分下降（<19.0%） |
| `no_episodic` | ON | ON | OFF | `experiments/crafter_no_episodic.yaml` | 得分下降（<19.0%） |
| `no_rnd` | ON | OFF | ON | `experiments/crafter_no_rnd.yaml` | 得分下降最多 |

---

## 四、长训练过程监控指标观测标准

### 4.1 训练日志关键字段

训练日志每 `n_steps × n_envs` 步输出一行，格式如下：

```text
step=XXXXX | policy_loss=X.XXXX | value_ext_loss=X.XXXX | value_int_loss=X.XXXX | entropy=X.XXXX | clip_fraction=X.XXXX | n_updates=XX | icm_inverse_loss=X.XXXX | icm_forward_loss=X.XXXX | rnd_loss=X.XXXX | global_step=XXXXX | episode_count=XX | vram_allocated_mb=XX.XX | vram_peak_mb=XX.XX | ext_reward_mean=X.XXXX | int_reward_mean=X.XXXX | elapsed=XXs
```

### 4.2 核心监控指标合格标准

| 指标 | 早期阶段（前 5%步数） | 中期阶段（5%-50%） | 后期阶段（50%-100%） | 异常判定 |
|------|----------------------|-------------------|---------------------|----------|
| `icm_inverse_loss` | ≈2.83（=ln17，随机策略） | 逐步下降至 <2.5 | 稳定在 <2.0 | 全程 ≈2.83 不降 → ICM 编码器未学习 |
| `icm_forward_loss` | ≈0.003 | 缓慢下降 | 稳定下降趋势 | 爆炸增长 → 特征维度错位 |
| `rnd_loss` | ≈0.002 | 逐步下降 | 趋于稳定 | 剧烈震荡 → Target 网络未冻结 |
| `int_reward_mean` | ≈60（Crafter）/≈24（MiniGrid） | 逐步平稳下降 | 趋于较低值 | 剧烈震荡或 NaN → 融合公式异常 |
| `ext_reward_mean` | ≈0 | 缓慢上升 | 明显上升 | 全程为 0 且不涨 → 策略未学习 |
| `entropy` | ≈2.83（Crafter 17动作） | 缓慢下降 | 稳定在 1.0-2.0 | 迅速降至 0 → 策略过早收敛 |
| `vram_peak_mb` | <2200 | <2200 | <2200 | ≥2200 → 显存超限，自动终止 |
| `clip_fraction` | 0.0-0.1 | 0.0-0.2 | 0.0-0.3 | 持续 >0.5 → 学习率过大 |

### 4.3 Wandb 仪表盘 8 面板观测

启用 `--use-wandb` 后，在 `https://wandb.ai` 查看以下 8 类监控面板：

| 面板编号 | 指标 | 观测重点 |
|----------|------|----------|
| 1 | `policy_loss` | 应为负值（PPO 目标取负），逐步趋近 0 |
| 2 | `value_ext_loss` + `value_int_loss` | 双轨价值损失，各自逐步下降 |
| 3 | `entropy` | 策略熵，缓慢下降表示探索逐步收敛 |
| 4 | `ext_reward_mean` + `int_reward_mean` | 外在奖励上升、内在奖励下降 |
| 5 | `icm_inverse_loss` + `icm_forward_loss` | ICM 双损失下降趋势 |
| 6 | `rnd_loss` | RND 预测误差下降趋势 |
| 7 | `vram_peak_mb` | 全程 ≤2200MB |
| 8 | `clip_fraction` | PPO 裁剪比例，正常 0.0-0.3 |

### 4.4 内在奖励收敛趋势判定

内在奖励反映智能体对环境的"陌生程度"，合格趋势：

$$r_{int}(t) \text{ 应从初期高值（}\approx 60\text{）单调平稳下降}$$

判定规则：
- 用最近 1000 步滑动窗口计算标准差 $\sigma_W$ 和均值 $\bar{r}_{int,W}$
- 合格：$\sigma_W < 0.3 \times \bar{r}_{int,W}$（平稳下降，无剧烈震荡）
- 不合格：$\sigma_W \geq 0.3 \times \bar{r}_{int,W}$（震荡剧烈，需排查 RND 或融合公式）

---

## 五、训练完成后得分评测执行命令

### 5.1 MiniGrid DoorKey 评测

```powershell
# 使用最终检查点评测（XXXXXX 替换为实际步数）
python scripts/evaluate.py --checkpoint results/checkpoints/minigrid/step_XXXXXX.pt --env minigrid --n-episodes 100 --seed 42 --output-dir results/eval/minigrid
```

合格标准：

| 指标 | 合格线 | 判定 |
|------|--------|------|
| `success_rate` | ≥0.95 | 收敛判定 |
| `mean_steps`（训练步数） | ≤968,000 | 在 96.8 万步内达到 success_rate≥0.95 |
| 样本效率 | ≥2.5× | 968,000 / 2,420,000 = 2.5× |

### 5.2 Crafter 评测

```powershell
python scripts/evaluate.py --checkpoint results/checkpoints/crafter/step_1000000.pt --env crafter --n-episodes 100 --seed 42 --output-dir results/eval/crafter
```

合格标准：

| 指标 | 合格线 | 判定 |
|------|--------|------|
| `score`（22 成就几何均值） | ≥19.0% | 100 episode 几何均值 |
| 相对提升 | ≥21.7% | (19.0-15.6)/15.6 = 21.7% |

### 5.3 Atari Montezuma 评测

```powershell
python scripts/evaluate.py --checkpoint results/checkpoints/atari/step_XXXXXX.pt --env atari --n-episodes 10 --seed 42 --output-dir results/eval/atari
```

合格标准：

| 指标 | 合格线 | 判定 |
|------|--------|------|
| `mean_score`（10 episode 平均分） | ≥3500 | 不裁剪奖励 |
| 相对提升 | ≥29× | 3500/120 ≈ 29× |

### 5.4 消融实验评测

对 4 组消融检查点分别评测：

```powershell
# full 组
python scripts/evaluate.py --checkpoint results/checkpoints/ablation_full/step_1000000.pt --env crafter --n-episodes 100 --seed 42 --output-dir results/eval/ablation_full

# no_icm 组
python scripts/evaluate.py --checkpoint results/checkpoints/ablation_no_icm/step_1000000.pt --env crafter --n-episodes 100 --seed 42 --output-dir results/eval/ablation_no_icm

# no_episodic 组
python scripts/evaluate.py --checkpoint results/checkpoints/ablation_no_episodic/step_1000000.pt --env crafter --n-episodes 100 --seed 42 --output-dir results/eval/ablation_no_episodic

# no_rnd 组
python scripts/evaluate.py --checkpoint results/checkpoints/ablation_no_rnd/step_1000000.pt --env crafter --n-episodes 100 --seed 42 --output-dir results/eval/ablation_no_rnd
```

消融判定矩阵（4 组 × 3 环境 = 12 单元，此处以 Crafter 为例）：

| 组名 | 目标得分 | 合格判定 | 相对下降率要求 |
|------|----------|----------|---------------|
| `full` | ≥19.0% | 基准上限 | — |
| `no_icm` | <19.0% | 下降 >5% | >5% |
| `no_episodic` | <19.0% | 下降 >5% | >5% |
| `no_rnd` | <19.0% | 下降 >5% | >5% |

### 5.5 评测结果记录规范

评测完成后在 `results/eval/` 目录生成 `benchmark_report.json` 和 `benchmark_report.md`，记录格式：

```text
results/eval/
├── minigrid/
│   ├── benchmark_report.json    # 机器可读
│   └── benchmark_report.md      # 人类可读
├── crafter/
│   ├── benchmark_report.json
│   └── benchmark_report.md
├── atari/
│   ├── benchmark_report.json
│   └── benchmark_report.md
├── ablation_full/
│   ├── benchmark_report.json
│   └── benchmark_report.md
├── ablation_no_icm/
│   └── ...
├── ablation_no_episodic/
│   └── ...
└── ablation_no_rnd/
    └── ...
```

---

## 六、指标不达标分层排查流程

参照 `test/performance_acceptance_standard.md` 第 6 节排查方案，按以下优先级分层定位。

### 6.1 排查优先级总表

| 优先级 | 排查方向 | 触发现象 | 定位命令 |
|--------|----------|----------|----------|
| P0 | 显存超限 | `vram_peak_mb ≥ 2200` | `python test/scripts/check_vram_limit.py --baseline-only` |
| P0 | 训练崩溃/OOM | 进程异常退出 | 查看 `test/logs/vram_alert.txt` |
| P1 | ICM 特征失效 | `icm_inverse_loss` 全程 ≈2.83 不降 | `python -m pytest tests/test_icm.py tests/test_icm_module.py -v` |
| P1 | RND 奖励虚假震荡 | `rnd_loss` 或 `int_reward_mean` 剧烈震荡 | `python -m pytest tests/test_rnd.py tests/test_rnd_module.py -v` |
| P1 | 内在奖励异常 | `int_reward_mean` 出现 NaN 或不下降 | `python -m pytest tests/test_ngu_fusion.py -v` |
| P2 | 记忆库检索异常 | `episodic_memory` 报错或 kNN 超时 | `python -m pytest tests/test_episodic_memory.py tests/test_memory_bank.py -v` |
| P2 | 超参被误改 | 指标整体偏离预期 | `diff experiments/crafter_full.yaml experiments/crafter_no_icm.yaml` |
| P2 | 环境观测预处理出错 | 观测形状/范围异常 | `python -m pytest tests/test_wrappers.py tests/test_compat.py -v` |

### 6.2 P0：显存超限排查

```text
现象：vram_peak_mb ≥ 2200 或训练进程被 check_vram_limit.py 终止
定位：
  1. 查看 test/logs/vram_log.csv 确认超限时间点和峰值
  2. 确认 FP16 AMP 已启用 (config.use_amp=true)
  3. 确认梯度累积 accumulation_steps=4 (等效 batch=512)
  4. 确认 Episodic Memory LRU capacity 未超过 200
修复：
  - 降低 n_envs 从 8 → 4（显存约降 40%）
  - 降低 n_steps 从 128 → 64（Rollout 缓存减半）
  - 确认 RND Target 网络 requires_grad=False
  - 确认 Episodic Memory 在 CPU 上运行（非 GPU）
```

### 6.3 P1：ICM 特征提取失效

```text
现象：icm_inverse_loss 全程 ≈2.83 不下降，或 icm_forward_loss 爆炸增长
定位：
  1. 运行 python -m pytest tests/test_icm.py -v 确认单测通过
  2. 检查 ICM 编码器是否为 4 层 CNN → 288 维
  3. 确认逆动力学损失输入为 17 维 one-hot 动作
  4. 确认前向 MSE 损失对 288 维特征计算
修复：
  - 若编码器权重未更新：检查 lr=1e-4 是否被覆盖
  - 若特征维度错位：确认 encoder_cls 与环境匹配（Crafter→CrafterEncoder, Atari→NatureDQNEncoder）
  - 若 forward_loss 过大：确认未误乘 feature_dim
```

### 6.4 P1：RND 奖励虚假震荡

```text
现象：rnd_loss 或 int_reward_mean 出现周期性剧烈震荡
定位：
  1. 确认 Target 随机网络权重已冻结（不参与训练）
  2. 检查 RND 输入观测是否归一化（obs_normalize=true）
  3. 确认 γ_int=0.99 未被错误修改
  4. 检查 α_t 计算是否被错误置 0
修复：
  - 对 RND 误差施加滑动平均平滑
  - 降低 Predictor 学习率
  - 确认 obs_normalizer RunningMeanStd 正常更新
```

### 6.5 P1：内在奖励不收敛

```text
现象：int_reward_mean 全程不下降，或出现 NaN
定位：
  1. 检查融合公式 r_int = η×ICM前向损失 + r_episodic×min(max(α_t,1),L)
  2. 确认 η=0.2、L=5 未被修改
  3. 确认 min(max(α_t,1),L) 截断生效
  4. 检查 Episodic Memory 是否正确 add 和 reset
修复：
  - 若 r_episodic 饱和：检查 kNN 检索是否返回有效近邻
  - 若 α_t 异常：检查 RND compute_alpha 逻辑
  - 若 NaN：检查 epsilon=1e-3 是否被移除
```

### 6.6 P2：超参设置不合理

```text
现象：训练发散或收敛过慢
定位：
  1. 确认 lr=1e-4、clip_range=0.1、ent_coef=0.001
  2. 确认 gamma_ext=0.999、gamma_int=0.99
  3. 确认 batch_size=128、accumulation_steps=4、ppo_epochs=4
  4. 确认 n_envs=8（MiniGrid n_steps=256，Crafter/Atari n_steps=128）
修复：
  - 对比 experiments/ 目录下 YAML 配置是否与预期一致
  - 若发散：降低 lr 或增加 ent_coef
  - 若过慢：确认学习率衰减 update_lr 正常调用
```

---

## 七、长训练完成后验收核对清单

以下全部项目打勾合格后，方可视为全量测试收尾，才可生成 Github 仓库物料。

### 7.1 训练完成核对

| 编号 | 检查项 | 合格判定 | 勾选 |
|------|--------|----------|------|
| T-1 | MiniGrid 训练完成 | `global_step` 达到 968,000 以上 | ☐ |
| T-2 | Crafter 训练完成 | `global_step` 达到 1,000,000 | ☐ |
| T-3 | Atari 训练完成 | `global_step` 达到 1,000,000 以上 | ☐ |
| T-4 | MiniGrid 全程显存 ≤2.2GB | `test/logs/vram_log.csv` 无 ALERT 记录 | ☐ |
| T-5 | Crafter 全程显存 ≤2.2GB | `test/logs/vram_log.csv` 无 ALERT 记录 | ☐ |
| T-6 | Atari 全程显存 ≤2.2GB | `test/logs/vram_log.csv` 无 ALERT 记录 | ☐ |
| T-7 | ICM 逆损失正常下降 | 从 ≈2.83 下降至 <2.0 | ☐ |
| T-8 | 内在奖励平稳收敛 | 从 ≈60 平稳下降，无剧烈震荡 | ☐ |
| T-9 | Wandb 日志完整 | 8 类面板数据无缺失 | ☐ |
| T-10 | 检查点完整保存 | 每个环境最终检查点文件存在且可加载 | ☐ |

### 7.2 评测达标核对

| 编号 | 检查项 | 合格判定 | 勾选 |
|------|--------|----------|------|
| E-1 | MiniGrid success_rate ≥0.95 | 100 episode 成功率 ≥95% | ☐ |
| E-2 | MiniGrid 收敛步数 ≤968,000 | 在 96.8 万步内达到 E-1 | ☐ |
| E-3 | MiniGrid 样本效率 ≥2.5× | 2,420,000 / 实际收敛步数 ≥2.5 | ☐ |
| E-4 | Crafter 得分 ≥19.0% | 100 episode 22 成就几何均值 ≥19.0% | ☐ |
| E-5 | Crafter 相对提升 ≥21.7% | (实际得分-15.6)/15.6 ≥21.7% | ☐ |
| E-6 | Atari 平均分 ≥3500 | 10 episode 平均分 ≥3500（不裁剪奖励） | ☐ |
| E-7 | 评测报告已生成 | `results/eval/` 下 JSON+MD 报告完整 | ☐ |

### 7.3 消融实验达标核对

| 编号 | 检查项 | 合格判定 | 勾选 |
|------|--------|----------|------|
| A-1 | `full` 组 Crafter 得分 ≥19.0% | 基准组达标 | ☐ |
| A-2 | `no_icm` 组得分 < `full` 组 | 相对下降率 >5% | ☐ |
| A-3 | `no_episodic` 组得分 < `full` 组 | 相对下降率 >5% | ☐ |
| A-4 | `no_rnd` 组得分 < `full` 组 | 相对下降率 >5% | ☐ |
| A-5 | 单调性成立 | `full` 得分最高 | ☐ |
| A-6 | 消融评测报告已生成 | 4 组 JSON+MD 报告完整 | ☐ |

### 7.4 显存红线最终核对

| 编号 | 检查项 | 合格判定 | 勾选 |
|------|--------|----------|------|
| V-1 | 三大环境训练全程峰值 ≤2.2GB | `vram_log.csv` 全行 status=OK | ☐ |
| V-2 | 4 组消融训练全程峰值 ≤2.2GB | `vram_log.csv` 全行 status=OK | ☐ |
| V-3 | 评测阶段峰值 ≤2.2GB | 评测过程无 OOM | ☐ |

### 7.5 最终验收结论

| 编号 | 检查项 | 合格判定 | 勾选 |
|------|--------|----------|------|
| F-1 | T-1 ~ T-10 全部打勾 | 训练阶段无遗留 | ☐ |
| F-2 | E-1 ~ E-7 全部打勾 | 评测指标全部达标 | ☐ |
| F-3 | A-1 ~ A-6 全部打勾 | 消融实验全部达标 | ☐ |
| F-4 | V-1 ~ V-3 全部打勾 | 显存红线全程未超 | ☐ |

> **最终判定**：F-1 ~ F-4 全部打勾 → 全量测试收尾通过 → 可生成 Github 仓库物料并推送。
>
> **任何一项未打勾**：禁止 `git push`，按第六节排查流程定位修复后重跑对应环境训练。
