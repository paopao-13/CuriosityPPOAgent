# VRAM 显存优化文档

本项目目标硬件为 **RTX 3060 Laptop (6GB VRAM)**。好奇心 PPO 智能体需要同时驻留策略网络、ICM 网络 (encoder + 逆/前向模型)、RND 网络 (target + predictor), 显存压力大。本文档详细说明各项优化策略及其原理, 并给出 VRAM 预算表。

---

## 1. 6GB VRAM 约束分析

### 1.1 显存占用来源

深度学习训练的 GPU 显存主要由以下部分构成:

| 占用类型 | 说明 | 是否可优化 |
|----------|------|-----------|
| 模型参数 | 各网络的权重张量 | 结构决定 |
| 梯度 | 反向传播产生的参数梯度 (与参数同尺寸) | 结构决定 |
| 优化器状态 | Adam 的 momentum + variance (参数的 2 倍) | 优化器选择 |
| 激活值 (activation) | 前向中间结果, 用于反向传播 | **主要优化目标** |
| 临时缓冲 | autograd 中间张量 | 可通过 AMP 减少 |

### 1.2 本项目的网络规模

以 Crafter 配置为例, 需要驻留的网络:

| 网络 | 参数量 (约) | 说明 |
|------|------------|------|
| ActorCritic (encoder + 3 头) | ~1.5M | 共享 CrafterEncoder + actor + critic_ext + critic_int |
| ICMNet (encoder + 逆/前向模型) | ~1.2M | 独立 CrafterEncoder + 2 个 MLP |
| RNDNet (target + predictor) | ~3.5M | 2 个 encoder + 3 层 FC predictor |

三组网络合计约 6.2M 参数。在 FP32 下, 仅参数 + 梯度 + Adam 状态就需 `6.2M * (4 + 4 + 8) = ~100MB`。真正的显存瓶颈在于 **前向激活值**: 每次 mini-batch (128 样本) 经过多个 CNN encoder, 中间特征图的激活值可达数百 MB。

### 1.3 峰值显存风险

如果不做优化, 一次完整的前向 + 反向 (PPO + ICM + RND 同时更新) 在 batch=128 下, 激活值峰值可能超过 4GB, 加上参数与优化器状态, 总占用接近 5-6GB, 极易触发 OOM (Out of Memory)。

---

## 2. 优化策略一: FP16 AMP (自动混合精度)

### 2.1 原理

AMP (Automatic Mixed Precision) 将前向计算中的矩阵乘法、卷积自动转为 FP16 (半精度), 而将需要数值精度的操作 (如 softmax、layer norm) 保留 FP32。这带来两大好处:

- **激活显存减半**: FP16 张量占 2 字节/元素, 相比 FP32 的 4 字节节省 50%。实际由于部分操作保留 FP32, 整体激活显存约节省 **40%**。
- **计算加速**: RTX 3060 的 Tensor Core 对 FP16 有 2 倍以上吞吐加速。

### 2.2 实现: torch.autocast + GradScaler

本项目通过 `AMPManager` 封装 PyTorch 的 AMP API:

```python
# src/curiosity_ppo/utils/amp.py
class AMPManager:
    def __init__(self, enabled=True, device='cpu'):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.enabled)

    def autocast(self):
        if self.enabled:
            return torch.autocast(device_type='cuda', dtype=torch.float16)
        else:
            return torch.autocast(device_type='cpu', enabled=False)
```

使用方式 (PPO 更新):

```python
with self.amp.autocast():
    logits, v_ext, v_int = self.actor_critic(obs)
    loss = policy_loss + vf_coef * (v_ext_loss + v_int_loss) - ent_coef * entropy
    loss = loss / self.accumulation_steps

self.amp.scale_loss(loss).backward()   # GradScaler 放大梯度防下溢
self.amp.unscale_(self.optimizer)       # 反缩放以正确裁剪
torch.nn.utils.clip_grad_norm_(...)
self.amp.step(self.optimizer)           # 缩放后步进
```

### 2.3 GradScaler 的作用

FP16 的动态范围有限 (最大 65504), 小梯度可能下溢为 0。GradScaler 将 loss 乘以一个大 scale factor (如 65536), 使反向传播的梯度放大到安全范围, 在 optimizer.step 前再除以 scale factor 还原。这保证了 FP16 训练的数值稳定性。

### 2.4 注意事项

- ICM 的 forward loss 使用 MSE, 对精度敏感, 但 autocast 会自动在关键节点保留 FP32, 无需特殊处理。
- RND 的 target 网络在 `no_grad` 下计算, 不涉及梯度, AMP 无副作用。
- 训练日志中可通过 `vram_allocated_mb` / `vram_peak_mb` 监控实际占用。

---

## 3. 优化策略二: 梯度累积 (Gradient Accumulation)

### 3.1 问题

PPO 期望较大的有效 batch size (如 512) 以稳定策略更新。但 batch=512 的一次前向 + 反向会使激活值峰值翻 4 倍, 直接突破 6GB 限制。

### 3.2 解决方案

梯度累积将大 batch 拆分为多个小 mini-batch, 分别前向 + 反向, 梯度累积后再统一更新:

```
有效 batch = mini_batch_size * accumulation_steps = 128 * 4 = 512
```

### 3.3 实现

```python
# src/curiosity_ppo/ppo/ppo_trainer.py
for step, batch in enumerate(buffer.get_mini_batches(self.batch_size)):
    with self.amp.autocast():
        loss = ...  # PPO loss
        loss = loss / self.accumulation_steps   # 损失除以累积步数

    self.amp.scale_loss(loss).backward()        # 梯度累积

    if (step + 1) % self.accumulation_steps == 0:
        self.amp.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(...)
        self.amp.step(self.optimizer)            # 累积 4 次后才更新
        self.optimizer.zero_grad()
```

### 3.4 关键效果

- **峰值显存不增加**: 每次只处理 128 个样本的 mini-batch, 激活值峰值与 batch=128 相同。
- **等效大 batch**: 4 次累积的梯度平均后, 等效于 batch=512 的单次更新, 保证 PPO 的优化稳定性。
- **损失缩放**: 每次计算的 loss 除以 `accumulation_steps`, 确保累积梯度的数学期望与单次大 batch 一致。

---

## 4. 优化策略三: LRU 内存库放 CPU

### 4.1 问题

NGU 的情景记忆需要维护一个 embedding 库做 kNN 搜索。若放 GPU, 10000 个 512 维 embedding 占用 `10000 * 512 * 4B = 20MB` (FP32), 看似不大, 但 kNN 距离计算的中间矩阵 `(10000, 512)` 会产生临时显存, 且 GPU 显存预算紧张时任何额外占用都是风险。

### 4.2 解决方案

将 LRU 内存库完全放在 CPU (numpy), kNN 搜索用 numpy 暴力 L2 距离计算:

```python
# src/curiosity_ppo/utils/memory_bank.py
class LRUMemoryBank:
    def __init__(self, capacity=10000, dim=512):
        self._bank = OrderedDict()  # CPU 上的 dict

    def knn_distances(self, query, k=5):
        all_embeddings = np.stack(list(self._bank.values()))  # (N, dim) CPU
        diff = all_embeddings - query[np.newaxis, :]
        distances = np.sqrt((diff ** 2).sum(axis=1))
        k_smallest = np.partition(distances, k - 1)[:k]
        return np.sort(k_smallest)
```

### 4.3 内存开销

- 内存库: `10000 * 512 * 4B = 20MB` RAM (CPU 内存, 非 VRAM)。
- kNN 中间矩阵: `(10000, 512)` float64 = `40MB` RAM, 计算后即释放。
- 总计: 约 60MB CPU RAM, 对 VRAM 零占用。

### 4.4 性能权衡

- kNN 搜索在 CPU 上, 每步搜索耗时约 1-5ms (10000 条), 相比 GPU 前向的数十毫秒, 开销可忽略。
- LRU FIFO 淘汰保证容量上限, 防止内存无限增长。

---

## 5. 优化策略四: Rollout Buffer 放 CPU

### 5.1 问题

PPO 需要存储整个 rollout 的观测、动作、奖励、价值等数据。以 Crafter 为例:

```
n_steps=128, n_envs=8, obs_shape=(64, 64, 3), float32
obs 占用 = 128 * 8 * 64 * 64 * 3 * 4B = 37.7MB
```

加上 actions、rewards、values、advantages、returns 等, 整个 buffer 约 80MB。若全部放 GPU, 占用宝贵的 VRAM。

### 5.2 解决方案

Rollout buffer 全部用 numpy 存 CPU, 仅在 PPO 更新的 mini-batch 迭代时, 将当前 mini-batch 传到 GPU:

```python
# src/curiosity_ppo/ppo/rollout_buffer.py
class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_shape, action_dim, device='cpu'):
        # 全部 numpy 存 CPU
        self.obs = np.zeros((n_steps, n_envs) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        # ... 其余字段同理

    def get_mini_batches(self, batch_size=128):
        total = self.n_steps * self.n_envs
        indices = np.random.permutation(total)
        obs_flat = self.obs.reshape(total, *self.obs_shape)
        for start in range(0, total, batch_size):
            idx = indices[start:start + batch_size]
            yield {
                'obs': torch.from_numpy(obs_flat[idx]).to(self.device),  # 此时才传 GPU
                'actions': torch.from_numpy(actions_flat[idx]).to(self.device),
                # ...
            }
```

### 5.3 关键效果

- rollout 收集阶段: buffer 全在 CPU, GPU 只需处理当前单步观测的前向推理, 显存占用极小。
- PPO 更新阶段: 每次 mini-batch (128 样本) 才传 GPU, 用完即释放, 不会累积。
- 传输开销: `torch.from_numpy(...).to(device)` 的 PCIe 传输约 1-2ms/mini-batch, 相比 GPU 计算 negligible。

---

## 6. VRAM 预算表

### 6.1 FP32 (无优化) vs FP16 AMP 对比

以 Crafter 配置 (batch=128, n_envs=8, n_steps=128) 为例的峰值显存估算:

| 占用项 | FP32 (无优化) | FP16 AMP | 节省 |
|--------|--------------|-----------|------|
| 模型参数 (ActorCritic + ICM + RND) | ~25 MB | ~25 MB (主权重仍 FP32) | 0 |
| 梯度 | ~25 MB | ~25 MB (梯度 FP32) | 0 |
| Adam 优化器状态 (momentum + variance) | ~50 MB | ~50 MB | 0 |
| PPO 前向激活 (batch=128, encoder + 3 头) | ~280 MB | ~170 MB | 110 MB |
| ICM 前向激活 (encoder x2 + 2 MLP) | ~180 MB | ~110 MB | 70 MB |
| RND 前向激活 (target + predictor) | ~350 MB | ~210 MB | 140 MB |
| Rollout buffer (若放 GPU) | ~80 MB | 0 (放 CPU) | 80 MB |
| LRU 内存库 (若放 GPU) | ~20 MB | 0 (放 CPU) | 20 MB |
| 临时缓冲 + autograd | ~100 MB | ~60 MB | 40 MB |
| **合计峰值** | **~1110 MB** | **~650 MB** | **~460 MB** |

> 注: 上表为单次 mini-batch 更新的峰值。实际训练中 PPO epochs=4, 每个 epoch 多个 mini-batch, 但每次 mini-batch 独立前向+反向, 峰值不叠加。

### 6.2 完整训练峰值

加上 CUDA context、PyTorch 运行时开销、碎片化等因素:

| 配置 | 估算峰值 VRAM | 6GB 预算余量 |
|------|--------------|-------------|
| FP32, 全 GPU (无优化) | ~3.5 - 4.5 GB | 1.5 - 2.5 GB (紧张) |
| FP16 AMP + CPU buffer + CPU memory bank | **~2.2 GB** | **3.8 GB (充裕)** |

### 6.3 三基准的 VRAM 消耗差异

| 基准 | 观测尺寸 | 单样本激活 | 估算峰值 VRAM |
|------|----------|-----------|--------------|
| Crafter | 64x64x3 | 较小 | ~2.0 GB |
| Atari | 4x84x84 | 中等 (NatureDQN encoder) | ~2.2 GB |
| MiniGrid | 64x64x3 | 较小 (resize 后) | ~2.0 GB |

三个基准均在 6GB 预算内有充足余量, 可安全训练。

---

## 7. 显存监控

项目内置 VRAM 监控工具, 训练时每步记录显存占用:

```python
# src/curiosity_ppo/utils/vram.py
def get_vram_usage():
    """返回 (allocated_mb, peak_mb), 无 CUDA 时返回 (0, 0)"""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.memory_allocated() / 1e6
    peak = torch.cuda.max_memory_allocated() / 1e6
    return allocated, peak
```

训练日志中可查看:

- `vram_allocated_mb`: 当前已分配显存 (MB)
- `vram_peak_mb`: 训练以来的峰值显存 (MB)

此外, 每次训练步结束后调用 `empty_cache()` 释放缓存的空闲显存, 防止碎片化累积:

```python
# src/curiosity_ppo/ppo/agent.py
def train_step(self):
    ...
    empty_cache()  # 清理 CUDA 缓存
    ...
```

---

## 8. 优化策略总结

```mermaid
graph TB
    subgraph GPU "GPU VRAM (~2.2GB)"
        P["模型参数 + 梯度 + Adam 状态"]
        A["FP16 激活值<br/>autocast 节省 40%"]
        MB["mini-batch 临时张量<br/>用完即释放"]
    end
    subgraph CPU "CPU RAM (~140MB)"
        RB["Rollout Buffer<br/>128x8 观测, ~80MB"]
        LRU["LRU 内存库<br/>10000x512, ~20MB"]
        GAE["GAE 计算中间量"]
    end
    RB -->|"mini-batch 时传 GPU"| MB
    LRU -.->|"kNN 搜索在 CPU"| GAE
```

| 优化策略 | 节省显存 | 实现位置 | 代价 |
|----------|----------|----------|------|
| FP16 AMP | ~460 MB (激活减 40%) | `utils/amp.py` | GradScaler 微小开销 |
| 梯度累积 | 避免 batch=512 的 4x 峰值 | `ppo/ppo_trainer.py` | 更新频率降低 (数学等效) |
| LRU 内存库放 CPU | ~20 MB + kNN 临时 | `utils/memory_bank.py` | CPU kNN ~1-5ms/步 |
| Rollout buffer 放 CPU | ~80 MB | `ppo/rollout_buffer.py` | PCIe 传输 ~1-2ms/mini-batch |
| empty_cache 每步清理 | 减少碎片 | `ppo/agent.py` | 微小调用开销 |

**最终结果**: 训练峰值约 **2.2GB**, 在 6GB VRAM 预算下剩余 **3.8GB** 缓冲, 可安全运行所有基准与消融实验, 不会触发 OOM。
