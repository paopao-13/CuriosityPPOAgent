# 超参数表

本文档列出好奇心 PPO 智能体的全部超参数, 按模块分组, 包含默认值、说明与配置位置。所有超参数均可通过 YAML 配置文件覆盖 (参见 `experiments/` 目录)。

> 配置数据类定义: `src/curiosity_ppo/config.py`
> 默认配置文件: `experiments/config.yaml`

---

## 1. PPO 超参数

PPO (Proximal Policy Optimization) 核心训练参数, 定义于 `PPOConfig`。

| 超参数 | 默认值 | 说明 | 配置键 |
|--------|--------|------|--------|
| `lr` | 1.0e-4 | 学习率 (Adam), 策略网络与好奇心网络共用 | `ppo.lr` |
| `clip_range` | 0.1 | PPO 裁剪范围 epsilon, 限制策略更新幅度 | `ppo.clip_range` |
| `ppo_epochs` | 4 | 每轮 rollout 数据的 PPO 更新轮数 | `ppo.ppo_epochs` |
| `gamma_ext` | 0.999 | 外在回报折扣因子 (episodic, episode 结束截断) | `ppo.gamma_ext` |
| `gamma_int` | 0.99 | 内在回报折扣因子 (non-episodic, 跨 episode 累积) | `ppo.gamma_int` |
| `gae_lambda` | 0.95 | GAE (Generalized Advantage Estimation) 的 lambda 参数 | `ppo.gae_lambda` |
| `ent_coef` | 0.001 | 熵正则化系数, 鼓励策略探索 | `ppo.ent_coef` |
| `vf_coef` | 0.5 | 价值函数损失权重 | `ppo.vf_coef` |
| `max_grad_norm` | 0.5 | 梯度裁剪的最大范数 | `ppo.max_grad_norm` |
| `batch_size` | 128 | PPO mini-batch 大小 (每次前向的样本数) | `ppo.batch_size` |
| `accumulation_steps` | 4 | 梯度累积步数, 有效 batch = 128 x 4 = 512 | `ppo.accumulation_steps` |
| `n_steps` | 128 | 每个 rollout 的步数 (每环境) | `ppo.n_steps` |
| `n_envs` | 8 | 并行环境数 (在 env 配置中同步设置) | `ppo.n_envs` |

### 关键说明

- **gamma_ext vs gamma_int**: 外在回报 `gamma_ext=0.999` (接近 1, 长期规划), 在 episode 边界截断; 内在回报 `gamma_int=0.99` (略低), 跨 episode 不截断 (non-episodic), 这是 RND/NGU 的关键设计。
- **有效 batch**: `batch_size * accumulation_steps = 512`, 通过梯度累积实现大 batch 而不增加 VRAM 峰值。
- **clip_range=0.1**: 比标准 PPO 的 0.2 更保守, 适合好奇心奖励带来的非平稳目标。

---

## 2. RND 超参数

RND (Random Network Distillation) 内在奖励参数, 定义于 `RNDConfig`。

| 超参数 | 默认值 | 说明 | 配置键 |
|--------|--------|------|--------|
| `enabled` | true | 是否启用 RND 模块 | `rnd.enabled` |
| `output_dim` | 512 | target / predictor 网络输出维度 | `rnd.output_dim` |
| `predictor_hidden` | 512 | predictor FC 层隐藏维度 | `rnd.predictor_hidden` |
| `obs_normalize` | true | 是否对观测做 RunningMeanStd 归一化 | `rnd.obs_normalize` |
| `reward_normalize` | true | 是否对 RND 内在奖励做归一化 | `rnd.reward_normalize` |

### 关键说明

- **obs_normalize**: RND 对观测尺度极其敏感, 必须开启。使用 Welford 在线算法 (`RunningMeanStd`) 流式统计 mean/std。
- **reward_normalize**: RND 误差在训练初期极大、后期趋近 0, 必须除以 running std 归一化, 否则破坏 PPO 优势估计。
- **output_dim=512**: target 和 predictor 输出维度一致, 512 维提供足够的随机投影容量。
- **predictor 结构**: encoder + 3 层 FC (512 -> 512 -> 512 -> 512), 见 `networks/rnd.py`。

---

## 3. ICM 超参数

ICM (Intrinsic Curiosity Module) 内在奖励参数, 定义于 `ICMConfig`。

| 超参数 | 默认值 | 说明 | 配置键 |
|--------|--------|------|--------|
| `enabled` | true | 是否启用 ICM 模块 | `icm.enabled` |
| `feature_dim` | 288 | encoder 输出特征维度 (phi 空间维度) | `icm.feature_dim` |
| `eta` | 0.2 | 内在奖励缩放系数: r_icm = eta * forward_loss | `icm.eta` |
| `hidden_dim` | 256 | 逆模型 / 前向模型的 MLP 隐藏层维度 | `icm.hidden_dim` |

### 关键说明

- **feature_dim=288**: CrafterEncoder 的输出维度, 同时作为情景记忆 controllable embedding 的维度。
- **eta=0.2**: ICM 前向损失缩放系数, 控制好奇心信号强度。过大可能导致策略过度追逐新颖性而忽视外在奖励。
- **hidden_dim=256**: 逆模型 `Linear(288*2, 256) -> Linear(256, action_dim)`, 前向模型 `Linear(288+action_dim, 256) -> Linear(256, 288)`。
- **encoder 复用**: ICM 的 encoder 输出 `phi_t` 既用于前向预测, 又作为情景记忆 embedding, 避免独立编码器。

---

## 4. Episodic Memory 超参数

NGU 情景记忆参数, 定义于 `EpisodicConfig`。

| 超参数 | 默认值 | 说明 | 配置键 |
|--------|--------|------|--------|
| `enabled` | true | 是否启用情景记忆模块 | `episodic.enabled` |
| `k` | 5 | kNN 搜索的近邻数 | `episodic.k` |
| `epsilon` | 1.0e-3 | kernel 平滑参数: kernel(x) = epsilon / (x + epsilon) | `episodic.epsilon` |
| `L` | 5 | NGU 调制上界: min(max(alpha_t, 1), L) | `episodic.L` |
| `capacity` | 10000 | LRU 内存库最大容量 (超出 FIFO 淘汰) | `episodic.capacity` |

### 关键说明

- **k=5**: kNN 近邻数, 用于计算伪计数 N(x)。k 越大, 伪计数越平滑, 但计算开销增加。
- **epsilon=1e-3**: kernel 函数的平滑参数, 控制 "近距离" 的灵敏度。epsilon 越小, 只有非常近的邻居才贡献高伪计数。
- **L=5**: NGU 长期调制系数 alpha_t 的上界。alpha_t = 1 + (L-1) * sigmoid(error), 映射到 [1, 5]。L 越大, 长期新颖区域的探索放大越强。
- **capacity=10000**: 内存库容量, 限制 kNN 搜索开销。10000 x 512 x 4B = 20MB CPU RAM。
- **reset**: 每 episode 结束时调用 `reset()` 清空记忆库, 实现 "短期" 探索。

---

## 5. AMP 超参数

FP16 自动混合精度参数, 定义于顶层 `Config`。

| 超参数 | 默认值 | 说明 | 配置键 |
|--------|--------|------|--------|
| `use_amp` | true | 是否启用 FP16 自动混合精度 | `use_amp` |

### 关键说明

- **use_amp=true**: 启用后通过 `torch.autocast(dtype=torch.float16)` + `GradScaler` 实现, 节省约 40% 激活显存。
- **无 CUDA 时自动降级**: `AMPManager` 检测 `torch.cuda.is_available()`, 无 GPU 时 AMP 自动禁用, 训练回退到 CPU FP32。
- **GradScaler**: 自动管理 loss scale, 防止 FP16 小梯度下溢。

---

## 6. 环境与全局超参数

环境与全局训练参数, 定义于 `EnvConfig` 与顶层 `Config`。

| 超参数 | 默认值 | 说明 | 配置键 |
|--------|--------|------|--------|
| `env.name` | crafter | 环境名称 (决定编码器选择) | `env.name` |
| `env.n_envs` | 8 | 并行环境数 | `env.n_envs` |
| `env.total_steps` | 1000000 | 总训练步数 (Crafter 默认 1M) | `env.total_steps` |
| `ablation` | full | 消融配置标识 (full / no_icm / no_episodic / no_rnd) | `ablation` |
| `seed` | 42 | 全局随机种子 | `seed` |
| `wandb_project` | curiosity-ppo | Wandb 项目名 | `wandb_project` |

### 各基准的 total_steps 差异

| 基准 | total_steps | n_steps | 说明 |
|------|-------------|---------|------|
| Crafter | 1,000,000 | 128 | 22 成就, 中等探索难度 |
| Atari Montezuma | 10,000,000 | 128 | 极稀疏分数, 需要长训练 |
| MiniGrid DoorKey | 1,500,000 | 256 | 收敛目标 96.8 万步 |

---

## 7. 网络结构参数 (隐式)

以下参数由代码结构决定, 不在 YAML 中配置, 但影响模型容量与显存:

| 参数 | Crafter / MiniGrid | Atari | 说明 |
|------|--------------------|-------|------|
| Encoder 类型 | CrafterEncoder | NatureDQNEncoder | 4 层 CNN / Nature DQN CNN |
| Encoder 输出 | 288 (icm.feature_dim) | 512 | 嵌入维度 |
| Actor 隐藏层 | 512 | 512 | `Linear(embed, 512) -> ReLU -> Linear(512, action_dim)` |
| Critic_ext 隐藏层 | 512 | 512 | `Linear(embed, 512) -> ReLU -> Linear(512, 1)` |
| Critic_int 隐藏层 | 512 | 512 | `Linear(embed, 512) -> ReLU -> Linear(512, 1)` |
| RND Predictor FC | 512 x 3 层 | 512 x 3 层 | encoder + 3 层 FC (512->512->512->512) |
| ICM 逆模型 | 256 | 256 | `Linear(288*2, 256) -> Linear(256, action_dim)` |
| ICM 前向模型 | 256 | 256 | `Linear(288+action_dim, 256) -> Linear(256, 288)` |

---

## 8. 完整默认配置 (YAML)

以下为 `experiments/config.yaml` 的完整默认配置, 各基准配置在此基础上覆盖:

```yaml
ppo:
  lr: 1.0e-4
  clip_range: 0.1
  ppo_epochs: 4
  gamma_ext: 0.999
  gamma_int: 0.99
  gae_lambda: 0.95
  ent_coef: 0.001
  vf_coef: 0.5
  max_grad_norm: 0.5
  batch_size: 128
  accumulation_steps: 4
  n_steps: 128
  n_envs: 8

rnd:
  enabled: true
  output_dim: 512
  predictor_hidden: 512
  obs_normalize: true
  reward_normalize: true

icm:
  enabled: true
  feature_dim: 288
  eta: 0.2
  hidden_dim: 256

episodic:
  enabled: true
  k: 5
  epsilon: 1.0e-3
  L: 5
  capacity: 10000

env:
  name: crafter
  n_envs: 8
  total_steps: 1000000

ablation: full
use_amp: true
seed: 42
wandb_project: curiosity-ppo
```

---

## 9. 消融配置差异

四组消融实验仅在 `enabled` 开关与 `ablation` 标识上不同, 其余参数完全一致:

| 配置文件 | ablation | icm.enabled | rnd.enabled | episodic.enabled |
|----------|----------|-------------|-------------|------------------|
| `crafter_full.yaml` | full | true | true | true |
| `crafter_no_icm.yaml` | no_icm | **false** | true | true |
| `crafter_no_episodic.yaml` | no_episodic | true | true | **false** |
| `crafter_no_rnd.yaml` | no_rnd | true | **false** | true |

> 注: 关闭 ICM 时, 情景记忆的 controllable embedding 回退为 RND target 输出; 关闭 Episodic 时, NGU 融合退化为 `r_ngu = r_rnd`; 关闭 RND 时, alpha_t 固定为 1.0, NGU 退化为纯情景奖励。详见 `src/curiosity_ppo/curiosity/ngu_fusion.py`。
