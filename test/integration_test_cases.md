# CuriosityPPOAgent 集成功能完整测试用例集

> **项目**：CuriosityPPOAgent — ICM+RND 分层新颖好奇心 PPO 智能体
> **硬件**：AMD R7 6800H + RTX3060 Laptop 6GB
> **显存红线**：2.2GB
> **技术架构**：ICM(4层CNN→288维, 17维动作空间, 初始损失≈2.83) + RND(Target+Predictor MSE, γ=0.99) + Episodic Memory(CPU-FAISS KNN, LRU 200条) + 融合 r_int=η×ICM+r_epi×min(max(α,1),L) + 双价值头PPO(γ_ext=0.999, γ_int=0.99)
> **脚本入口**：`scripts/train.py` / `scripts/evaluate.py` / `scripts/export_onnx.py` / `scripts/run_ablation.py` / `scripts/run_all_ablation.ps1`
> **配置文件**：`experiments/` 目录下 7 个 YAML 配置
> **早期内在奖励**：≈60（训练初期），随训练逐步平稳下降

---

## 目录

| 编号 | 测试场景 | 优先级 |
|------|----------|--------|
| TC-01 | 消融脚本一键批量启动 4 组完整消融对照实验 | P0 |
| TC-02 | 训练中途中断、加载 checkpoint 断点续训 | P0 |
| TC-03 | 模型权重保存、离线评测脚本加载权重输出得分 | P0 |
| TC-04 | PyTorch 模型导出 ONNX 轻量化推理全流程 | P0 |
| TC-05 | Episodic Memory LRU 自动淘汰机制有效性 | P1 |
| TC-06 | Wandb 仪表盘 8 类监控指标完整记录 | P1 |
| TC-07 | 全训练周期显存峰值监控校验 | P0 |
| TC-08 | 三大环境种子固定复现一致性测试 | P1 |
| TC-09 | MiniGrid 轻量冒烟集成训练 | P2 |
| TC-10 | Crafter 中等规模冒烟训练 | P2 |

---

## 测试前置条件

### 环境准备

```bash
# 1. 确认工作目录为项目根目录
cd curiosity-ppo

# 2. 激活 Python 虚拟环境
.\.venv\Scripts\Activate.ps1

# 3. 确认关键依赖已安装
pip list | Select-String "torch|onnx|onnxruntime|gym|crafter|minigrid|faiss|wandb|yaml"

# 4. 确认配置文件存在
Get-ChildItem experiments\*.yaml | Select-Object Name
# 预期输出：crafter_full, crafter_no_icm, crafter_no_rnd, crafter_no_episodic, atari_montezuma_full, minigrid_doorkey_full, config

# 5. 确认脚本存在
Get-ChildItem scripts\*.py, scripts\*.ps1 | Select-Object Name
```

### 通用清理操作

```bash
# 清理旧检查点与输出（每次测试前执行）
Remove-Item -Recurse -Force checkpoints\test_* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force outputs\test_* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force onnx_exports\test_* -ErrorAction SilentlyContinue
```

---

## TC-01: 消融脚本一键批量启动 4 组完整消融对照实验

### 测试目的

验证 `scripts/run_ablation.py` 和 `scripts/run_all_ablation.ps1` 能依次自动运行 4 组消融实验（full / no_icm / no_episodic / no_rnd），每组实验使用对应配置文件正确加载或移除对应模块，最终生成可对比的结果汇总。

### 分步执行操作

```bash
# 步骤1: 确认 4 组消融配置文件存在且字段正确
# experiments/crafter_full.yaml        → icm: true,  rnd: true,  episodic: true
# experiments/crafter_no_icm.yaml      → icm: false, rnd: true,  episodic: true
# experiments/crafter_no_episodic.yaml → icm: true,  rnd: true,  episodic: false
# experiments/crafter_no_rnd.yaml      → icm: true,  rnd: false, episodic: true
Get-ChildItem experiments\crafter_*.yaml | ForEach-Object { Write-Host "=== $_ ==="; Get-Content $_ | Select-String "icm|rnd|episodic" }
```

```bash
# 步骤2: 使用 Python 消融脚本启动（短步数冒烟）
python scripts/run_ablation.py --env crafter --steps 512 --use-wandb
```

```bash
# 步骤3: 使用 PowerShell 脚本启动（等效功能）
.\scripts\run_all_ablation.ps1 -Env crafter -Steps 512 -UseWandb
```

```bash
# 步骤4: 检查消融输出目录结构
Get-ChildItem -Recurse outputs\ablation_* | Select-Object FullName
```

```bash
# 步骤5: 检查每组消融实验是否生成了 metrics 汇总
Get-ChildItem outputs\ablation_*\*.json | ForEach-Object { Write-Host "=== $_ ==="; Get-Content $_ }
```

### 预期合格输出

1. `run_ablation.py` 依次运行 4 组实验，控制台输出包含 4 段日志：
   ```
   [ABLATION] Running: crafter_full (icm=True, rnd=True, episodic=True)
   [ABLATION] Running: crafter_no_icm (icm=False, rnd=True, episodic=True)
   [ABLATION] Running: crafter_no_episodic (icm=True, rnd=True, episodic=False)
   [ABLATION] Running: crafter_no_rnd (icm=True, rnd=False, episodic=True)
   [ABLATION] All 4 experiments completed. Summary saved to outputs/ablation_summary_*.json
   ```

2. `outputs/` 目录下生成 4 个实验子目录 + 1 个汇总 JSON：
   ```
   outputs/ablation_crafter_full/metrics.json
   outputs/ablation_crafter_no_icm/metrics.json
   outputs/ablation_crafter_no_episodic/metrics.json
   outputs/ablation_crafter_no_rnd/metrics.json
   outputs/ablation_summary_<timestamp>.json
   ```

3. `ablation_summary_*.json` 包含 4 组实验的对比字段：
   ```json
   {
     "experiments": [
       {"name": "crafter_full", "steps": 512, "final_ext_reward": ..., "final_int_reward": ..., "final_policy_loss": ...},
       {"name": "crafter_no_icm", "steps": 512, ...},
       {"name": "crafter_no_episodic", "steps": 512, ...},
       {"name": "crafter_no_rnd", "steps": 512, ...}
     ]
   }
   ```

4. `no_icm` 组的检查点中 `icm_net` 键不存在；`no_rnd` 组的 `rnd_net` 键不存在。

5. PowerShell 脚本 `run_all_ablation.ps1` 与 Python 脚本产出等效，无路径或参数传递错误。

### 失败判定 + 修复排查方向

| 失败现象 | 排查方向 |
|----------|----------|
| 某组实验未启动或中途崩溃 | 检查对应 YAML 配置文件中 `icm`/`rnd`/`episodic` 字段拼写与布尔值；查看 `scripts/run_ablation.py` 中实验列表是否硬编码了 4 组 |
| `no_icm` 组检查点仍含 `icm_net` | 排查 `scripts/train.py` 中检查点保存逻辑是否有条件判断 `if config.icm: state_dict['icm_net'] = ...` |
| 汇总 JSON 缺失或字段不全 | 检查 `run_ablation.py` 的 `summarize()` 函数是否正确聚合 4 组 metrics |
| PowerShell 脚本参数未传递 | 检查 `run_all_ablation.ps1` 中 `param()` 块与内部 `python scripts/run_ablation.py` 调用的参数拼接 |
| Wandb 项目下看不到 4 组 run | 确认 `--use-wandb` 传递到每组子实验，检查 wandb group/tag 命名 |

---

## TC-02: 训练中途中断、加载本地 checkpoint 断点续训完整验证

### 测试目的

验证训练过程中使用 Ctrl+C 或超时中断后，通过 `--resume` 参数加载本地 checkpoint 可正确恢复训练状态，`global_step` 从断点值继续递增，优化器状态、ICM/RND 网络权重、Episodic Memory 等全部正确恢复。

### 分步执行操作

```bash
# 步骤1: 启动初始训练，设置较短步数以便中途打断
python scripts/train.py --config experiments/crafter_full.yaml --total-steps 2048 --checkpoint-dir checkpoints/test_resume --checkpoint-interval 256
```

```bash
# 步骤2: 训练进行到约 512~1024 步时，手动 Ctrl+C 中断训练
# （或在另一个终端等待 30 秒后强制终止）
# 确认终端输出：[CHECKPOINT] Saved checkpoint at step <N> to checkpoints/test_resume/ckpt_<N>.pt
```

```bash
# 步骤3: 检查检查点文件存在
Get-ChildItem checkpoints\test_resume\ | Sort-Object Name
# 预期：ckpt_<N>.pt 文件存在
```

```bash
# 步骤4: 检查检查点内容结构
python -c "
import torch
ckpt = torch.load('checkpoints/test_resume/ckpt_<N>.pt', map_location='cpu')
print('Keys:', list(ckpt.keys()))
print('extra.step:', ckpt['extra']['step'])
print('Has actor_critic:', 'actor_critic' in ckpt)
print('Has ppo_optimizer:', 'ppo_optimizer' in ckpt)
print('Has icm_net:', 'icm_net' in ckpt)
print('Has rnd_net:', 'rnd_net' in ckpt)
print('extra.metrics:', ckpt['extra'].get('metrics', {}))
"
```

```bash
# 步骤5: 使用 --resume 断点续训
python scripts/train.py --config experiments/crafter_full.yaml --total-steps 2048 --resume checkpoints/test_resume/ckpt_<N>.pt --checkpoint-dir checkpoints/test_resume --checkpoint-interval 256
```

```bash
# 步骤6: 验证 global_step 延续
# 在训练日志中查找：[RESUME] Loaded checkpoint from ..., global_step=<N>
# 后续日志中 step 应从 <N> 继续递增，而非从 0 重新开始
```

### 预期合格输出

1. 中断时检查点文件 `ckpt_<N>.pt` 成功保存，`<N>` 为中断时的实际 global_step。

2. 检查点结构完整：
   ```
   Keys: ['actor_critic', 'ppo_optimizer', 'icm_net', 'rnd_net', 'extra']
   extra.step: <N>  (与文件名一致)
   extra.metrics: {'ext_reward_mean': ..., 'int_reward_mean': ..., 'policy_loss': ..., ...}
   ```

3. `--resume` 加载时控制台输出：
   ```
   [RESUME] Loaded checkpoint from checkpoints/test_resume/ckpt_<N>.pt
   [RESUME] global_step = <N>
   [RESUME] Restored actor_critic, ppo_optimizer, icm_net, rnd_net
   [RESUME] Continuing training from step <N> to 2048
   ```

4. 续训后日志中 `step` 从 `<N>` 继续递增至 2048，不出现 `step=0` 重置。

5. 续训后 loss 曲线与中断前连续（不出现突变跳跃），内在奖励值在合理范围内延续。

6. 续训结束后 `extra.step` 最终值为 2048。

### 失败判定 + 修复排查方向

| 失败现象 | 排查方向 |
|----------|----------|
| `--resume` 参数未被解析 | 检查 `scripts/train.py` 中 argparse 是否注册了 `--resume` 参数及类型为 str |
| 加载后 global_step 归零 | 排查 `train.py` 中 resume 逻辑是否将 `ckpt['extra']['step']` 赋值给 `global_step`，而非重新初始化为 0 |
| 优化器状态未恢复导致 loss 突变 | 检查是否执行了 `ppo_optimizer.load_state_dict(ckpt['ppo_optimizer'])`；注意 device 一致性（CPU→GPU 需 map_location） |
| ICM/RND 权重未恢复 | 检查 `icm_net.load_state_dict(ckpt['icm_net'])` 和 `rnd_net.load_state_dict(ckpt['rnd_net'])` 是否有条件判断或异常被吞 |
| 检查点文件损坏或无法加载 | 排查中断时机是否恰好在 torch.save 写入过程中；增加原子写入（先写临时文件再 rename） |
| 续训后 Episodic Memory 状态丢失 | 检查 Episodic Memory 是否需要序列化到检查点（LRU 缓存状态、FAISS 索引） |

---

## TC-03: 模型权重保存、离线评测脚本 evaluate.py 加载权重输出得分校验

### 测试目的

验证完整流程：训练 → 保存检查点 → `scripts/evaluate.py` 加载权重 → 在指定环境中运行评测 → 生成 JSON/MD 格式评测报告，且报告内容包含正确的指标字段。

### 分步执行操作

```bash
# 步骤1: 运行短训练以生成检查点
python scripts/train.py --config experiments/crafter_full.yaml --total-steps 1024 --checkpoint-dir checkpoints/test_eval --checkpoint-interval 1024
```

```bash
# 步骤2: 确认检查点文件存在
Get-ChildItem checkpoints\test_eval\
# 预期：ckpt_1024.pt
```

```bash
# 步骤3: 使用 evaluate.py 进行 Crafter 环境评测
python scripts/evaluate.py --checkpoint checkpoints/test_eval/ckpt_1024.pt --env crafter --n-episodes 5 --config experiments/crafter_full.yaml --output-dir outputs/test_eval_crafter
```

```bash
# 步骤4: 检查输出目录
Get-ChildItem -Recurse outputs\test_eval_crafter\
```

```bash
# 步骤5: 查看 JSON 评测报告内容
Get-Content outputs\test_eval_crafter\eval_results.json
```

```bash
# 步骤6: 查看 Markdown 评测报告内容
Get-Content outputs\test_eval_crafter\eval_report.md
```

```bash
# 步骤7: 交叉验证 Atari 环境（如 Montezuma）
python scripts/evaluate.py --checkpoint checkpoints/test_eval/ckpt_1024.pt --env atari --n-episodes 3 --config experiments/atari_montezuma_full.yaml --output-dir outputs/test_eval_atari
```

### 预期合格输出

1. 训练完成后检查点 `ckpt_1024.pt` 存在且可加载。

2. `evaluate.py` 控制台输出：
   ```
   [EVAL] Loading checkpoint: checkpoints/test_eval/ckpt_1024.pt
   [EVAL] Environment: crafter
   [EVAL] Running 5 episodes...
   [EVAL] Episode 1/5: reward=..., length=...
   ...
   [EVAL] Results saved to outputs/test_eval_crafter/
   ```

3. `outputs/test_eval_crafter/` 目录下生成：
   ```
   eval_results.json
   eval_report.md
   ```

4. `eval_results.json` 包含以下字段：
   ```json
   {
     "checkpoint": "checkpoints/test_eval/ckpt_1024.pt",
     "env": "crafter",
     "n_episodes": 5,
     "mean_reward": <float>,
     "std_reward": <float>,
     "mean_length": <float>,
     "episodes": [
       {"episode": 1, "reward": <float>, "length": <int>},
       ...
     ],
     "success_rate": <float>
   }
   ```

5. `eval_report.md` 为可读格式，包含摘要表格和每回合详情。

6. 评测过程中无 CUDA OOM 错误，显存峰值低于 2.2GB。

7. Atari 环境评测同样生成有效的 JSON/MD 报告。

### 失败判定 + 修复排查方向

| 失败现象 | 排查方向 |
|----------|----------|
| `evaluate.py` 无法加载检查点 | 检查 `--checkpoint` 路径是否正确；检查检查点 keys 是否匹配 evaluate.py 期望的结构 `{actor_critic, icm_net?, rnd_net?}` |
| 评测时环境创建失败 | 检查 `--env` 参数值是否在 `crafter/atari/minigrid` 枚举中；Crafter 需确认使用 `crafter.Env()` 直接创建而非 `gym.make` |
| JSON 报告缺失字段 | 排查 `evaluate.py` 中结果聚合逻辑，确保 `mean_reward`/`std_reward`/`episodes` 等字段均被写入 |
| MD 报告格式混乱 | 检查 `evaluate.py` 中 Markdown 模板生成逻辑，确认表格语法正确 |
| 评测时 reward 全为 0 或 NaN | 排查模型 forward 是否在 eval 模式下正确执行（`model.eval()`，关闭 dropout）；检查动作采样是否使用 argmax 而非随机采样 |
| Atari/Montezuma 环境不兼容 | 确认 `atari_montezuma_full.yaml` 中环境 ID 正确；Gym 版本兼容性（GymCompatWrapper 是否应用于 Atari） |

---

## TC-04: PyTorch 模型导出 ONNX 轻量化推理模型全流程验证

### 测试目的

验证 `scripts/export_onnx.py` 能将训练好的 PyTorch 模型导出为 ONNX 格式，导出后使用 ONNX Runtime 进行推理一致性校验（PyTorch vs ONNX 输出误差在阈值内），并验证动态 batch 维度支持。

### 分步执行操作

```bash
# 步骤1: 确保有训练好的检查点（复用 TC-03 的检查点或新建）
python scripts/train.py --config experiments/crafter_full.yaml --total-steps 512 --checkpoint-dir checkpoints/test_onnx --checkpoint-interval 512
```

```bash
# 步骤2: 导出 ONNX 模型（默认执行验证）
python scripts/export_onnx.py --checkpoint checkpoints/test_onnx/ckpt_512.pt --output onnx_exports/test_model.onnx --env crafter --opset 17
```

```bash
# 步骤3: 跳过验证导出（快速导出模式）
python scripts/export_onnx.py --checkpoint checkpoints/test_onnx/ckpt_512.pt --output onnx_exports/test_model_no_verify.onnx --env crafter --opset 17 --no-verify
```

```bash
# 步骤4: 确认 ONNX 文件已生成
Get-ChildItem onnx_exports\test_model*.onnx | Select-Object Name, Length
```

```bash
# 步骤5: 使用 ONNX Runtime 加载并验证推理一致性
python -c "
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession('onnx_exports/test_model.onnx')
print('Input names:', [i.name for i in sess.get_inputs()])
print('Input shapes:', [i.shape for i in sess.get_inputs()])
print('Output names:', [o.name for o in sess.get_outputs()])
print('Output shapes:', [o.shape for o in sess.get_outputs()])

# 动态 batch 测试：使用不同 batch size
for bs in [1, 4, 8, 16]:
    dummy = np.random.randn(bs, *sess.get_inputs()[0].shape[1:]).astype(np.float32)
    outputs = sess.run(None, {sess.get_inputs()[0].name: dummy})
    print(f'Batch={bs}: output shape={outputs[0].shape}, OK')
"
```

```bash
# 步骤6: PyTorch vs ONNX 数值一致性校验
python -c "
import torch
import onnxruntime as ort
import numpy as np
from models.actor_critic import ActorCritic

# 加载 PyTorch 模型
ckpt = torch.load('checkpoints/test_onnx/ckpt_512.pt', map_location='cpu')
model = ActorCritic(...)  # 按配置初始化
model.load_state_dict(ckpt['actor_critic'])
model.eval()

# 生成相同输入
dummy_input = torch.randn(1, 4, 84, 84)  # 按实际输入形状调整
with torch.no_grad():
    pt_output = model(dummy_input)

# ONNX 推理
sess = ort.InferenceSession('onnx_exports/test_model.onnx')
ort_output = sess.run(None, {sess.get_inputs()[0].name: dummy_input.numpy()})

# 比较输出
for i, (pt, ort_o) in enumerate(zip(pt_output, ort_output)):
    diff = np.abs(pt.numpy() - ort_o).max()
    print(f'Output[{i}]: max_diff={diff:.8f}, threshold=1e-4')
    assert diff < 1e-4, f'Output[{i}] mismatch: {diff}'
print('All outputs consistent!')
"
```

### 预期合格输出

1. `export_onnx.py` 控制台输出：
   ```
   [ONNX] Loading checkpoint: checkpoints/test_onnx/ckpt_512.pt
   [ONNX] Building model for env: crafter
   [ONNX] Exporting to onnx_exports/test_model.onnx (opset=17)
   [ONNX] Verifying ONNX model with ONNX Runtime...
   [ONNX] Verification passed. Max output diff: <value> (threshold: 1e-4)
   [ONNX] Export complete.
   ```

2. ONNX 文件生成成功，大小在合理范围（通常 5~50MB）。

3. `--no-verify` 模式跳过验证步骤，仅导出 ONNX 文件。

4. ONNX Runtime 能成功加载模型，输入/输出名称和形状清晰可读。

5. 动态 batch 测试通过：
   ```
   Batch=1: output shape=(1, ...), OK
   Batch=4: output shape=(4, ...), OK
   Batch=8: output shape=(8, ...), OK
   Batch=16: output shape=(16, ...), OK
   ```

6. PyTorch vs ONNX 数值一致性校验通过：
   ```
   Output[0]: max_diff=0.00001234, threshold=1e-4
   Output[1]: max_diff=0.00000876, threshold=1e-4
   All outputs consistent!
   ```

7. 所有输出 max_diff < 1e-4。

### 失败判定 + 修复排查方向

| 失败现象 | 排查方向 |
|----------|----------|
| 导出时报 `unsupported op` 错误 | 检查模型中是否有自定义算子不被 ONNX 支持；尝试提高 `--opset` 版本至 17 或更高；检查是否需要注册自定义 symbolic 方法 |
| ONNX Runtime 加载失败 | 检查 ONNX 文件是否完整（文件大小 > 0）；使用 `onnx.checker.check_model()` 验证模型格式；确认 onnxruntime 版本与 opset 兼容 |
| 动态 batch 报错 shape mismatch | 排查 `torch.onnx.export()` 是否设置 `dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}` |
| PyTorch vs ONNX 输出差异过大 | 检查模型是否含 BN/Dropout 层未切到 eval 模式；确认导出时 `model.eval()` 已调用；浮点精度问题可放宽阈值至 1e-3 |
| 导出的模型缺少 ICM/RND 子网络 | 确认 `export_onnx.py` 是否仅导出 ActorCritic 推理部分（推理时不需要 ICM/RND），如需导出全部需调整导出逻辑 |
| opset 版本不兼容 | 确认 PyTorch 与 ONNX opset 版本对应关系；PyTorch 2.x 通常支持 opset 17+ |

---

## TC-05: Episodic Memory LRU 自动淘汰机制有效性验证

### 测试目的

验证 Episodic Memory 模块在长轨迹训练中，向量库不会无限膨胀，容量上限严格保持 200 条（LRU 淘汰最久未访问的条目），且淘汰后 KNN 检索功能正常。

### 分步执行操作

```bash
# 步骤1: 编写 LRU 淘止单元测试脚本
python -c "
from memory.episodic_memory import EpisodicMemory
import numpy as np

# 初始化 Episodic Memory，容量上限 200
mem = EpisodicMemory(capacity=200, dim=288, k=5)

# 插入 250 条向量（超过容量上限 200）
for i in range(250):
    vec = np.random.randn(288).astype(np.float32)
    mem.add(vec, metadata={'step': i, 'episode': i // 50})

# 检查当前容量
current_size = mem.size()
print(f'After inserting 250 items: size={current_size}, capacity=200')
assert current_size == 200, f'LRU eviction failed: size={current_size}, expected=200'

# 验证被淘汰的是最久未访问的（step 0~49 应被淘汰）
# 检查剩余条目的 step 范围
remaining_steps = []
for i in range(mem.size()):
    entry = mem.get(i)
    if entry:
        remaining_steps.append(entry['metadata']['step'])
print(f'Remaining steps range: [{min(remaining_steps)}, {max(remaining_steps)}]')
assert min(remaining_steps) >= 50, 'LRU should have evicted oldest entries (step 0~49)'

# KNN 检索测试
query = np.random.randn(288).astype(np.float32)
distances, indices = mem.search(query, k=5)
print(f'KNN search: k=5, distances={distances}, indices={indices}')
assert len(indices) == 5, 'KNN should return 5 neighbors'
print('LRU eviction test PASSED')
"
```

```bash
# 步骤2: 在实际训练中验证（长轨迹场景）
python scripts/train.py --config experiments/crafter_full.yaml --total-steps 4096 --checkpoint-dir checkpoints/test_lru --checkpoint-interval 2048
```

```bash
# 步骤3: 训练后检查 Episodic Memory 状态
python -c "
import torch
ckpt = torch.load('checkpoints/test_lru/ckpt_2048.pt', map_location='cpu')
# 检查点中是否保存了 episodic memory 状态（如有）
if 'episodic_memory' in ckpt:
    mem_state = ckpt['episodic_memory']
    print(f'Episodic Memory size at step 2048: {mem_state.get(\"size\", \"N/A\")}')
    print(f'Episodic Memory capacity: {mem_state.get(\"capacity\", \"N/A\")}')
    assert mem_state.get('size', 0) <= 200, 'Memory exceeds capacity!'
else:
    print('Episodic Memory state not in checkpoint (may be in-memory only)')
"
```

```bash
# 步骤4: 监控训练过程中内存占用（确保不随步数线性增长）
python -c "
import psutil
import subprocess
import time

proc = subprocess.Popen(['python', 'scripts/train.py', '--config', 'experiments/crafter_full.yaml', '--total-steps', '2048', '--checkpoint-dir', 'checkpoints/test_lru_mem', '--checkpoint-interval', '512'])
samples = []
try:
    while proc.poll() is None:
        p = psutil.Process(proc.pid)
        mem_mb = p.memory_info().rss / 1024 / 1024
        samples.append(mem_mb)
        print(f'RSS: {mem_mb:.1f} MB')
        time.sleep(5)
finally:
    proc.wait()

# 内存不应持续线性增长
if len(samples) > 5:
    early_avg = sum(samples[:5]) / 5
    late_avg = sum(samples[-5:]) / 5
    growth_ratio = late_avg / early_avg
    print(f'Early avg RSS: {early_avg:.1f} MB, Late avg RSS: {late_avg:.1f} MB, Growth ratio: {growth_ratio:.2f}')
    assert growth_ratio < 2.0, f'Memory growth too high: {growth_ratio}'
    print('Memory stability test PASSED')
"
```

### 预期合格输出

1. 单元测试输出：
   ```
   After inserting 250 items: size=200, capacity=200
   Remaining steps range: [50, 249]
   KNN search: k=5, distances=[...], indices=[...]
   LRU eviction test PASSED
   ```

2. 插入 250 条后容量严格等于 200，最早插入的 50 条（step 0~49）被淘汰。

3. KNN 检索功能正常，返回 k=5 个最近邻及距离。

4. 训练过程中内存占用不随步数线性增长：
   ```
   Early avg RSS: ~XXX MB, Late avg RSS: ~XXX MB, Growth ratio: < 2.0
   Memory stability test PASSED
   ```

5. 训练 4096 步后，Episodic Memory 容量仍为 200（如检查点中保存了内存状态）。

### 失败判定 + 修复排查方向

| 失败现象 | 排查方向 |
|----------|----------|
| 插入 250 条后 size > 200 | 排查 `memory/episodic_memory.py` 中 LRU 淘汰逻辑：是否在 `add()` 时检查 `len(self) >= capacity` 后执行 `popitem(last=False)`（或等效 OrderedDict 操作） |
| 淘汰的不是最旧条目 | 检查 LRU 访问顺序更新逻辑：`search()`/`get()` 时是否更新访问时间戳或移动到 OrderedDict 末尾 |
| KNN 检索返回空或报错 | 排查 FAISS 索引重建逻辑：淘汰条目后是否同步更新 FAISS 索引；检查 `search()` 时索引是否有效 |
| 训练中内存持续线性增长 | 检查是否有其他内存泄漏（如未释放的 tensor 历史缓冲区）；使用 `torch.cuda.memory_summary()` 排查 GPU 内存；确认 Episodic Memory 的 numpy 数组是否被正确覆写而非追加 |
| FAISS 索引与 LRU 缓存不同步 | 确认 `add()`/`evict()` 时 FAISS 索引与 Python 端 LRU 容器同步更新；考虑使用 ID 映射维护一致性 |

---

## TC-06: Wandb 仪表盘 8 类监控指标完整记录

### 测试目的

验证启用 `--use-wandb` 后，Wandb 仪表盘完整记录 8 类监控指标：policy_loss, value_ext_loss, value_int_loss, entropy, ext_reward_mean, int_reward_mean, vram_peak_mb, clip_fraction，且数据点频率合理、无缺失。

### 分步执行操作

```bash
# 步骤1: 确认 Wandb 已登录
wandb login
# 或设置环境变量
$env:WANDB_API_KEY = "<your_api_key>"
```

```bash
# 步骤2: 启动带 Wandb 的训练
python scripts/train.py --config experiments/crafter_full.yaml --total-steps 1024 --use-wandb --checkpoint-dir checkpoints/test_wandb --checkpoint-interval 512
```

```bash
# 步骤3: 训练过程中访问 Wandb 仪表盘
# 浏览器打开控制台输出的 Wandb run URL
# 确认 8 个面板均有数据
```

```bash
# 步骤4: 训练后通过 Wandb API 验证指标记录
python -c "
import wandb
api = wandb.Api()

# 获取最新的 run
runs = api.runs('curiosity-ppo')
latest_run = runs[0]
print(f'Run ID: {latest_run.id}')
print(f'Run name: {latest_run.name}')

# 获取历史指标数据
history = latest_run.history(samples=100)
expected_metrics = [
    'policy_loss', 'value_ext_loss', 'value_int_loss',
    'entropy', 'ext_reward_mean', 'int_reward_mean',
    'vram_peak_mb', 'clip_fraction'
]

for metric in expected_metrics:
    if metric in history.columns:
        values = history[metric].dropna()
        print(f'{metric}: {len(values)} data points, range=[{values.min():.4f}, {values.max():.4f}]')
    else:
        print(f'MISSING: {metric} not found in wandb history!')

print('---')
# 验证所有指标都存在
missing = [m for m in expected_metrics if m not in history.columns]
if missing:
    print(f'FAILED: Missing metrics: {missing}')
else:
    print('All 8 metrics recorded. PASSED')
"
```

```bash
# 步骤5: 验证指标数值合理性
python -c "
import wandb
api = wandb.Api()
runs = api.runs('curiosity-ppo')
latest_run = runs[0]
history = latest_run.history(samples=100)

# policy_loss: 应为正数，通常在 0.01~10 范围
pl = history['policy_loss'].dropna()
assert len(pl) > 0 and pl.min() > -1, f'policy_loss out of range: {pl.describe()}'

# value_ext_loss / value_int_loss: 应为正数（MSE）
vl_ext = history['value_ext_loss'].dropna()
vl_int = history['value_int_loss'].dropna()
assert vl_ext.min() >= 0, f'value_ext_loss negative: {vl_ext.describe()}'
assert vl_int.min() >= 0, f'value_int_loss negative: {vl_int.describe()}'

# entropy: 应为正数
ent = history['entropy'].dropna()
assert ent.min() >= 0, f'entropy negative: {ent.describe()}'

# ext_reward_mean: 可正可负
# int_reward_mean: 训练初期约 60，逐步下降
int_r = history['int_reward_mean'].dropna()
print(f'int_reward_mean: first={int_r.iloc[0]:.2f}, last={int_r.iloc[-1]:.2f}')

# vram_peak_mb: 应小于 2200 (2.2GB)
vram = history['vram_peak_mb'].dropna()
assert vram.max() < 2200, f'VRAM exceeds 2.2GB: max={vram.max():.1f} MB'

# clip_fraction: 应在 [0, 1] 范围
cf = history['clip_fraction'].dropna()
assert cf.min() >= 0 and cf.max() <= 1, f'clip_fraction out of [0,1]: {cf.describe()}'

print('All metric value ranges valid. PASSED')
"
```

### 预期合格输出

1. 训练启动时 Wandb 控制台输出：
   ```
   [WANDB] Run initialized: curiosity-ppo/run-<id>
   [WANDB] Dashboard: https://wandb.ai/.../runs/<id>
   ```

2. Wandb 仪表盘 8 个面板均显示数据曲线：
   | 面板 | 指标名 | 预期范围 |
   |------|--------|----------|
   | 1 | policy_loss | > -1, 通常 0.01~10 |
   | 2 | value_ext_loss | >= 0 |
   | 3 | value_int_loss | >= 0 |
   | 4 | entropy | >= 0 |
   | 5 | ext_reward_mean | 任意（Crafter 初期可能为 0） |
   | 6 | int_reward_mean | 初期 ≈60，逐步下降 |
   | 7 | vram_peak_mb | < 2200 |
   | 8 | clip_fraction | [0, 1] |

3. Wandb API 验证输出：
   ```
   policy_loss: N data points, range=[...]
   value_ext_loss: N data points, range=[...]
   value_int_loss: N data points, range=[...]
   entropy: N data points, range=[...]
   ext_reward_mean: N data points, range=[...]
   int_reward_mean: N data points, range=[...]
   vram_peak_mb: N data points, range=[...]
   clip_fraction: N data points, range=[...]
   All 8 metrics recorded. PASSED
   ```

4. `int_reward_mean` 在训练初期约 60，随训练逐步平稳下降（非剧烈震荡）。

5. `vram_peak_mb` 全程低于 2200。

### 失败判定 + 修复排查方向

| 失败现象 | 排查方向 |
|----------|----------|
| 某些面板无数据 | 检查 `scripts/train.py` 中 `wandb.log()` 调用是否包含所有 8 个 key；确认指标在 log 频率（如每 N 步）都有值 |
| Wandb run 未创建 | 检查 `--use-wandb` 参数是否正确传递；确认 `wandb.init()` 调用位置在训练开始前；检查 WANDB_API_KEY |
| int_reward_mean 初始值异常 | 验证 ICM 初始损失 ≈2.83，RND Target/Predictor MSE 初始值合理；融合公式 r_int=η×ICM+r_epi×min(max(α,1),L) 参数正确 |
| vram_peak_mb 面板缺失 | 排查 VRAM 监控代码是否在每个 log 步调用 `torch.cuda.max_memory_allocated()` 或等效 API |
| clip_fraction 始终为 0 或 1 | 检查 PPO clip 计算逻辑：clip_fraction = mean(|ratio - 1| > eps) 的实现是否正确 |
| 指标记录频率不一致 | 确认所有指标在同一个 `wandb.log()` 调用中记录，或使用 `step=` 参数对齐时间步 |

---

## TC-07: 全训练周期显存峰值监控校验

### 测试目的

验证整个训练过程中 GPU 显存峰值不超过 2.2GB（2200MB）红线，使用 `check_vram_limit.py --wrap` 包裹训练命令进行监控，并在显存超限时触发告警。

### 分步执行操作

```bash
# 步骤1: 确认 check_vram_limit.py 存在且可用
python scripts/check_vram_limit.py --help
# 预期：显示 --wrap 参数说明
```

```bash
# 步骤2: 使用 check_vram_limit.py --wrap 包裹训练命令
python scripts/check_vram_limit.py --wrap "python scripts/train.py --config experiments/crafter_full.yaml --total-steps 1024 --checkpoint-dir checkpoints/test_vram --checkpoint-interval 512"
```

```bash
# 步骤3: 训练完成后查看 VRAM 监控报告
Get-Content outputs\vram_report_*.json
```

```bash
# 步骤4: 手动验证显存峰值（独立脚本）
python -c "
import torch
from models.actor_critic import ActorCritic
from models.icm import ICM
from models.rnd import RND

# 清空缓存
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# 模拟模型加载
device = 'cuda'
# (按实际配置初始化模型)

# 模拟前向+反向传播
# (执行几个训练步)

peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f'Peak VRAM: {peak_mb:.1f} MB')
print(f'Limit: 2200 MB (2.2 GB)')
print(f'Status: {\"PASS\" if peak_mb < 2200 else \"FAIL\"}')
assert peak_mb < 2200, f'VRAM exceeded limit: {peak_mb:.1f} MB > 2200 MB'
"
```

```bash
# 步骤5: 长时间训练显存稳定性验证
python scripts/check_vram_limit.py --wrap "python scripts/train.py --config experiments/crafter_full.yaml --total-steps 4096 --checkpoint-dir checkpoints/test_vram_long --checkpoint-interval 1024"
```

```bash
# 步骤6: 检查长训练 VRAM 报告
Get-Content outputs\vram_report_*.json | Select-Object -Last 1
```

### 预期合格输出

1. `check_vram_limit.py --wrap` 成功包裹并启动训练：
   ```
   [VRAM] Wrapping command: python scripts/train.py --config experiments/crafter_full.yaml ...
   [VRAM] Monitoring started. Limit: 2200 MB (2.2 GB)
   [VRAM] Current: XXX MB | Peak: XXX MB | Status: OK
   ...
   [VRAM] Training completed.
   [VRAM] Final peak: XXX MB | Limit: 2200 MB | PASS
   [VRAM] Report saved to outputs/vram_report_<timestamp>.json
   ```

2. `vram_report_*.json` 内容：
   ```json
   {
     "command": "python scripts/train.py --config experiments/crafter_full.yaml ...",
     "limit_mb": 2200,
     "peak_mb": <value>,
     "status": "PASS",
     "samples": [
       {"step": 0, "vram_mb": ...},
       {"step": 100, "vram_mb": ...},
       ...
     ],
     "exceeded": false
   }
   ```

3. `peak_mb` 值小于 2200，`status` 为 `"PASS"`，`exceeded` 为 `false`。

4. 长训练（4096 步）VRAM 峰值同样低于 2200MB，无显存泄漏迹象（峰值不随步数增长）。

5. 训练过程中无 `CUDA out of memory` 错误。

### 失败判定 + 修复排查方向

| 失败现象 | 排查方向 |
|----------|----------|
| VRAM 峰值超过 2200MB | 检查 batch_size 是否过大；ICM 4 层 CNN 中间激活是否未释放（使用 `torch.no_grad()` 或 detach）；RND Target 网络是否冻结且不计算梯度；Episodic Memory FAISS 是否误用 GPU |
| `CUDA out of memory` 错误 | 降低 batch_size；使用梯度累积（gradient accumulation）；检查是否有不必要的 tensor 保留在 GPU（如历史 buffer）；启用混合精度训练（`torch.cuda.amp`） |
| VRAM 随步数持续增长 | 排查内存泄漏：检查是否在循环中累积 tensor 列表而未清理；检查 Episodic Memory 是否将向量存在 GPU 而非 CPU；确认 `.detach()` 和 `.item()` 使用正确 |
| `check_vram_limit.py` 无法包裹命令 | 检查 `--wrap` 参数解析逻辑；确认子进程启动方式（subprocess.Popen）；PowerShell 下引号转义问题 |
| VRAM 监控报告缺失 | 检查 `check_vram_limit.py` 是否在训练结束后写入报告文件；确认输出目录存在且有写权限 |
| RTX3060 6GB 但峰值仍超限 | 确认是否有其他 GPU 进程占用显存（`nvidia-smi`）；检查 CUDA context 初始化开销；考虑使用 `torch.cuda.set_per_process_memory_fraction(0.35)` 限制 |

---

## TC-08: 三大环境 Crafter/MiniGrid/Montezuma 种子固定复现一致性测试

### 测试目的

验证在 `seed=42` 固定下，三大环境（Crafter、MiniGrid、Montezuma/Atari）的种子兼容性修复（GymCompatWrapper seed/options 拦截、crafter.Env() 直接创建绕过 gym.make TimeLimit 不兼容）生效，两次运行结果可复现一致。

### 分步执行操作

```bash
# 步骤1: Crafter 环境种子复现测试（第一次运行）
python scripts/train.py --config experiments/crafter_full.yaml --total-steps 512 --checkpoint-dir checkpoints/test_seed_crafter_run1 --checkpoint-interval 512
# 记录训练日志中的关键指标
```

```bash
# 步骤2: Crafter 环境种子复现测试（第二次运行，相同 seed）
python scripts/train.py --config experiments/crafter_full.yaml --total-steps 512 --checkpoint-dir checkpoints/test_seed_crafter_run2 --checkpoint-interval 512
```

```bash
# 步骤3: MiniGrid 环境种子复现测试（第一次运行）
python scripts/train.py --config experiments/minigrid_doorkey_full.yaml --total-steps 512 --checkpoint-dir checkpoints/test_seed_minigrid_run1 --checkpoint-interval 512
```

```bash
# 步骤4: MiniGrid 环境种子复现测试（第二次运行）
python scripts/train.py --config experiments/minigrid_doorkey_full.yaml --total-steps 512 --checkpoint-dir checkpoints/test_seed_minigrid_run2 --checkpoint-interval 512
```

```bash
# 步骤5: Montezuma/Atari 环境种子复现测试（第一次运行）
python scripts/train.py --config experiments/atari_montezuma_full.yaml --total-steps 512 --checkpoint-dir checkpoints/test_seed_atari_run1 --checkpoint-interval 512
```

```bash
# 步骤6: Montezuma/Atari 环境种子复现测试（第二次运行）
python scripts/train.py --config experiments/atari_montezuma_full.yaml --total-steps 512 --checkpoint-dir checkpoints/test_seed_atari_run2 --checkpoint-interval 512
```

```bash
# 步骤7: 对比两次运行的指标一致性
python -c "
import torch
import numpy as np

envs = {
    'crafter': ('checkpoints/test_seed_crafter_run1/ckpt_512.pt', 'checkpoints/test_seed_crafter_run2/ckpt_512.pt'),
    'minigrid': ('checkpoints/test_seed_minigrid_run1/ckpt_512.pt', 'checkpoints/test_seed_minigrid_run2/ckpt_512.pt'),
    'atari': ('checkpoints/test_seed_atari_run1/ckpt_512.pt', 'checkpoints/test_seed_atari_run2/ckpt_512.pt'),
}

for env_name, (path1, path2) in envs.items():
    ckpt1 = torch.load(path1, map_location='cpu')
    ckpt2 = torch.load(path2, map_location='cpu')

    # 对比 actor_critic 权重
    ac1 = ckpt1['actor_critic']
    ac2 = ckpt2['actor_critic']

    all_match = True
    for key in ac1:
        if key in ac2:
            diff = (ac1[key] - ac2[key]).abs().max().item()
            status = 'MATCH' if diff < 1e-6 else 'DIFFER'
            if diff >= 1e-6:
                all_match = False
            print(f'[{env_name}] {key}: max_diff={diff:.10f} -> {status}')

    # 对比 metrics
    m1 = ckpt1['extra'].get('metrics', {})
    m2 = ckpt2['extra'].get('metrics', {})
    for metric_key in m1:
        if metric_key in m2:
            v1, v2 = m1[metric_key], m2[metric_key]
            if isinstance(v1, (int, float)):
                diff = abs(v1 - v2)
                status = 'MATCH' if diff < 1e-4 else 'DIFFER'
                print(f'[{env_name}] metric {metric_key}: v1={v1}, v2={v2}, diff={diff} -> {status}')

    print(f'[{env_name}] Overall: {\"PASS\" if all_match else \"FAIL\"}')
    print('---')
"
```

```bash
# 步骤8: 验证 GymCompatWrapper seed 拦截功能
python -c "
from envs.gym_compat_wrapper import GymCompatWrapper
import numpy as np

# 模拟 wrapper 对 seed/options 的拦截
class MockEnv:
    def __init__(self):
        self.action_space = type('AS', (), {'n': 17})()
    def reset(self, seed=None, options=None):
        return np.zeros((4, 84, 84), dtype=np.float32), {}

class MockGymEnv:
    '''模拟旧版 gym.Env（不支持 seed/options 参数）'''
    def __init__(self):
        self.action_space = type('AS', (), {'n': 17})()
    def reset(self):
        return np.zeros((4, 84, 84), dtype=np.float32)

# 测试 wrapper 兼容新旧接口
env_new = GymCompatWrapper(MockEnv())
env_old = GymCompatWrapper(MockGymEnv())

# 新接口环境
obs1, _ = env_new.reset(seed=42, options={})
obs2, _ = env_new.reset(seed=42, options={})
print(f'New interface: obs match = {np.array_equal(obs1, obs2)}')

# 旧接口环境
obs3, _ = env_old.reset(seed=42, options={})
obs4, _ = env_old.reset(seed=42, options={})
print(f'Old interface: obs match = {np.array_equal(obs3, obs4)}')

assert np.array_equal(obs1, obs2), 'New interface seed not reproducible'
assert np.array_equal(obs3, obs4), 'Old interface seed not reproducible'
print('GymCompatWrapper seed test PASSED')
"
```

### 预期合格输出

1. 三大环境两次运行（seed=42）的检查点权重完全一致：
   ```
   [crafter] actor_critic.<key>: max_diff=0.0000000000 -> MATCH
   [crafter] metric ext_reward_mean: v1=..., v2=..., diff=0.0 -> MATCH
   [crafter] Overall: PASS
   ---
   [minigrid] ... Overall: PASS
   ---
   [atari] ... Overall: PASS
   ```

2. GymCompatWrapper seed 拦截测试通过：
   ```
   New interface: obs match = True
   Old interface: obs match = True
   GymCompatWrapper seed test PASSED
   ```

3. 三大环境均能成功初始化并完成 512 步训练，无环境创建报错。

4. Crafter 环境使用 `crafter.Env()` 直接创建（非 `gym.make`），无 TimeLimit 不兼容报错。

5. MiniGrid 和 Atari/Montezuma 环境的 seed/options 参数被 GymCompatWrapper 正确拦截。

### 失败判定 + 修复排查方向

| 失败现象 | 排查方向 |
|----------|----------|
| 两次运行权重不一致 | 检查 seed 是否在所有随机源中统一设置：`torch.manual_seed(42)`、`np.random.seed(42)`、`random.seed(42)`、环境 `env.reset(seed=42)`；检查 CUDA 非确定性操作是否设置了 `torch.use_deterministic_algorithms(True)` |
| Crafter 环境 `gym.make` 报 TimeLimit 不兼容 | 确认使用 `crafter.Env()` 直接创建而非 `gym.make(' CrafterTextWorld-v0')`；检查 wrapper 链是否包含不必要的 TimeLimit wrapper |
| 旧版 gym.Env 的 reset() 不接受 seed 参数 | 确认 GymCompatWrapper 在调用 `reset()` 时检测底层是否支持 `seed`/`options` 参数，不支持时去除这些参数调用 |
| MiniGrid seed 不生效 | 检查 MiniGrid 环境是否需要通过 `env.reset(seed=..., options=...)` 传递种子；确认 wrapper 传递链完整 |
| Atari/Montezuma 环境创建失败 | 检查 ALE（Arcade Learning Environment）是否正确安装；确认 ROM 文件路径配置；seed 需在 ALE 初始化时设置 |
| 非确定性 GPU 操作导致微小差异 | 设置 `torch.backends.cudnn.deterministic = True` 和 `torch.backends.cudnn.benchmark = False`；注意某些操作在 GPU 上无法完全确定，可放宽阈值至 1e-4 |

---

## TC-09: MiniGrid 轻量冒烟集成训练

### 测试目的

验证在 MiniGrid 环境下进行 1024 步轻量冒烟训练，训练循环不崩溃，检查点正常生成，各模块（ICM、RND、Episodic Memory、双价值头 PPO）在 MiniGrid 观测空间下正常工作。

### 分步执行操作

```bash
# 步骤1: 确认 MiniGrid 配置文件
Get-Content experiments/minigrid_doorkey_full.yaml
# 预期：包含 env: minigrid, env_id: MiniGrid-DoorKey-..., icm: true, rnd: true, episodic: true
```

```bash
# 步骤2: 启动 1024 步冒烟训练
python scripts/train.py --config experiments/minigrid_doorkey_full.yaml --total-steps 1024 --checkpoint-dir checkpoints/test_smoke_minigrid --checkpoint-interval 256
```

```bash
# 步骤3: 检查训练是否正常完成（无崩溃）
# 检查退出码
echo "Exit code: $LASTEXITCODE"
# 检查检查点文件
Get-ChildItem checkpoints\test_smoke_minigrid\
```

```bash
# 步骤4: 验证检查点内容
python -c "
import torch
ckpt = torch.load('checkpoints/test_smoke_minigrid/ckpt_1024.pt', map_location='cpu')
print('Keys:', list(ckpt.keys()))
print('Step:', ckpt['extra']['step'])
print('Metrics:', ckpt['extra'].get('metrics', {}))

# 验证各模块权重存在
assert 'actor_critic' in ckpt, 'Missing actor_critic'
assert 'ppo_optimizer' in ckpt, 'Missing ppo_optimizer'
assert 'icm_net' in ckpt, 'Missing icm_net'
assert 'rnd_net' in ckpt, 'Missing rnd_net'
print('All module weights present. PASSED')
"
```

```bash
# 步骤5: 验证训练日志关键指标
# 查找训练日志文件
Get-ChildItem checkpoints\test_smoke_minigrid\*.log -ErrorAction SilentlyContinue
# 或检查控制台输出中的关键信息
```

```bash
# 步骤6: 快速评测（验证训练后的模型可推理）
python scripts/evaluate.py --checkpoint checkpoints/test_smoke_minigrid/ckpt_1024.pt --env minigrid --n-episodes 3 --config experiments/minigrid_doorkey_full.yaml --output-dir outputs/test_smoke_minigrid_eval
```

### 预期合格输出

1. 训练正常完成 1024 步，退出码为 0，无异常堆栈。

2. 训练过程中控制台输出类似：
   ```
   [TRAIN] Env: minigrid (MiniGrid-DoorKey-...)
   [TRAIN] Config: experiments/minigrid_doorkey_full.yaml
   [TRAIN] Total steps: 1024
   [TRAIN] Step 256/1024 | policy_loss=... | ext_reward=... | int_reward=... | vram=XXX MB
   [TRAIN] Step 512/1024 | ...
   [TRAIN] Step 768/1024 | ...
   [TRAIN] Step 1024/1024 | Done.
   [CHECKPOINT] Saved checkpoint at step 1024
   [TRAIN] Training completed successfully.
   ```

3. 检查点文件生成：
   ```
   checkpoints/test_smoke_minigrid/ckpt_256.pt
   checkpoints/test_smoke_minigrid/ckpt_512.pt
   checkpoints/test_smoke_minigrid/ckpt_768.pt
   checkpoints/test_smoke_minigrid/ckpt_1024.pt
   ```

4. 检查点内容完整：`actor_critic`, `ppo_optimizer`, `icm_net`, `rnd_net`, `extra{step: 1024, metrics: {...}}`。

5. 训练日志中：
   - `policy_loss` 为有限数值（非 NaN/Inf）
   - `int_reward_mean` 初期约 60，逐步下降
   - `vram` 峰值低于 2200MB

6. 评测脚本成功运行，生成 `eval_results.json` 和 `eval_report.md`。

### 失败判定 + 修复排查方向

| 失败现象 | 排查方向 |
|----------|----------|
| 训练启动即崩溃 | 检查 MiniGrid 环境创建是否成功；观测空间维度是否与 ICM 4 层 CNN 输入匹配（可能需要预处理 wrapper 调整图像尺寸） |
| `int_reward_mean` 为 NaN 或 Inf | 排查 ICM forward 中除零问题（如动作 one-hot 编码错误）；RND MSE 计算中 Target/Predictor 输出维度是否匹配 |
| 检查点缺失 `icm_net` 或 `rnd_net` | 检查配置文件中 `icm: true` 和 `rnd: true` 是否生效；`train.py` 中条件保存逻辑 |
| 训练中 VRAM 超限 | MiniGrid 观测可能比 Crafter 小，但仍需检查 batch_size 和并行环境数 |
| 评测时环境不匹配 | 确认 `--env minigrid` 在 `evaluate.py` 中映射到正确的环境 ID；GymCompatWrapper 需应用于评测环境 |
| MiniGrid 观测空间格式不兼容 | MiniGrid 返回 `obs` 为 dict 或部分观测，检查预处理 wrapper 是否提取了图像部分并转换为 (C, H, W) 格式 |

---

## TC-10: Crafter 中等规模冒烟训练

### 测试目的

验证在 Crafter 环境下进行 1024 步中等规模冒烟训练，验证 Crafter 环境兼容性修复（`crafter.Env()` 直接创建绕过 `gym.make` 的 TimeLimit 不兼容）有效，训练循环完整运行，ICM(4层CNN→288维, 17维动作空间)、RND、Episodic Memory 全部正常工作。

### 分步执行操作

```bash
# 步骤1: 确认 Crafter 配置文件
Get-Content experiments/crafter_full.yaml
# 预期：包含 env: crafter, icm: true, rnd: true, episodic: true, action_dim: 17
```

```bash
# 步骤2: 确认 Crafter 环境可独立创建
python -c "
import crafter
env = crafter.Env()
print(f'Action space: {env.action_space}')
print(f'Action dim: {env.action_space.n}')
print(f'Observation shape: {env.observation_space.shape}')
obs = env.reset()
print(f'Reset obs shape: {obs.shape}')
assert env.action_space.n == 17, f'Expected 17 actions, got {env.action_space.n}'
print('Crafter env creation OK')
"
```

```bash
# 步骤3: 启动 1024 步 Crafter 冒烟训练
python scripts/train.py --config experiments/crafter_full.yaml --total-steps 1024 --checkpoint-dir checkpoints/test_smoke_crafter --checkpoint-interval 256
```

```bash
# 步骤4: 检查训练退出状态
echo "Exit code: $LASTEXITCODE"
Get-ChildItem checkpoints\test_smoke_crafter\
```

```bash
# 步骤5: 验证检查点完整性与 ICM 特征维度
python -c "
import torch

ckpt = torch.load('checkpoints/test_smoke_crafter/ckpt_1024.pt', map_location='cpu')
print('=== Checkpoint Structure ===')
print('Keys:', list(ckpt.keys()))
print('Step:', ckpt['extra']['step'])

# 验证 ICM 特征维度
icm_state = ckpt['icm_net']
# 找到特征层输出的维度（应为 288）
for key, val in icm_state.items():
    if 'weight' in key and val.dim() >= 2:
        print(f'ICM layer [{key}]: shape={val.shape}')

# 验证 ActorCritic 输出动作维度（应为 17）
ac_state = ckpt['actor_critic']
for key, val in ac_state.items():
    if 'action' in key.lower() or 'pi' in key.lower() or 'logit' in key.lower():
        print(f'AC action layer [{key}]: shape={val.shape}')

# 验证 metrics
metrics = ckpt['extra'].get('metrics', {})
print(f'\\n=== Metrics ===')
for k, v in metrics.items():
    print(f'{k}: {v}')

# 关键断言
assert ckpt['extra']['step'] == 1024, f'Step mismatch: {ckpt[\"extra\"][\"step\"]}'
assert 'icm_net' in ckpt, 'Missing ICM'
assert 'rnd_net' in ckpt, 'Missing RND'
print('\\nCrafter smoke test checkpoint validation PASSED')
"
```

```bash
# 步骤6: 验证 ICM 初始损失与最终损失
python -c "
import torch
# 查找训练日志中 ICM loss 变化
# 初始损失应约 2.83
ckpt = torch.load('checkpoints/test_smoke_crafter/ckpt_256.pt', map_location='cpu')
early_metrics = ckpt['extra'].get('metrics', {})
print(f'Early ICM loss (step 256): {early_metrics.get(\"icm_loss\", \"N/A\")}')

ckpt_final = torch.load('checkpoints/test_smoke_crafter/ckpt_1024.pt', map_location='cpu')
final_metrics = ckpt_final['extra'].get('metrics', {})
print(f'Final ICM loss (step 1024): {final_metrics.get(\"icm_loss\", \"N/A\")}')
print(f'Final int_reward_mean: {final_metrics.get(\"int_reward_mean\", \"N/A\")}')
"
```

```bash
# 步骤7: 快速评测验证模型可用性
python scripts/evaluate.py --checkpoint checkpoints/test_smoke_crafter/ckpt_1024.pt --env crafter --n-episodes 3 --config experiments/crafter_full.yaml --output-dir outputs/test_smoke_crafter_eval
```

```bash
# 步骤8: 检查评测结果
Get-Content outputs\test_smoke_crafter_eval\eval_results.json
```

### 预期合格输出

1. Crafter 环境独立创建成功：
   ```
   Action space: Discrete(17)
   Action dim: 17
   Observation shape: (64, 64, 3)
   Reset obs shape: (64, 64, 3)
   Crafter env creation OK
   ```

2. 训练正常完成 1024 步，退出码为 0。

3. 训练控制台输出：
   ```
   [TRAIN] Env: crafter
   [TRAIN] Using crafter.Env() directly (bypassing gym.make TimeLimit)
   [TRAIN] Config: experiments/crafter_full.yaml
   [TRAIN] ICM: enabled (4-layer CNN -> 288-dim, 17 actions)
   [TRAIN] RND: enabled (Target + Predictor, gamma=0.99)
   [TRAIN] Episodic Memory: enabled (CPU-FAISS KNN, LRU capacity=200)
   [TRAIN] Total steps: 1024
   [TRAIN] Step 256/1024 | policy_loss=... | value_ext_loss=... | value_int_loss=... | entropy=... | ext_reward=... | int_reward=... | vram=XXX MB | clip_fraction=...
   ...
   [TRAIN] Step 1024/1024 | Done.
   [TRAIN] Training completed successfully.
   ```

4. 检查点完整生成（4 个间隔检查点 + 最终检查点）。

5. ICM 特征维度验证：
   ```
   ICM layer [feature_extractor.<layer>.weight]: shape=[..., 288, ...]
   ```
   确认特征输出维度为 288。

6. ActorCritic 动作维度验证：
   ```
   AC action layer [actor_head.weight]: shape=[17, ...]
   ```
   确认输出维度为 17。

7. ICM 初始损失约 2.83：
   ```
   Early ICM loss (step 256): ~2.83
   Final ICM loss (step 1024): < 2.83 (应有所下降)
   Final int_reward_mean: < 60 (应低于初始约60，平稳下降)
   ```

8. 评测结果有效：
   ```json
   {
     "checkpoint": "checkpoints/test_smoke_crafter/ckpt_1024.pt",
     "env": "crafter",
     "n_episodes": 3,
     "mean_reward": <float>,
     "std_reward": <float>,
     ...
   }
   ```

9. 全程无 CUDA OOM，VRAM 峰值 < 2200MB。

10. 无 `gym.make` TimeLimit 不兼容报错（确认使用 `crafter.Env()` 直接创建）。

### 失败判定 + 修复排查方向

| 失败现象 | 排查方向 |
|----------|----------|
| `gym.make(' CrafterTextWorld-v0')` 报 TimeLimit 不兼容 | 确认 `envs/` 下 Crafter 环境工厂使用 `crafter.Env()` 直接创建；检查是否误用了 `gym.make`；如有 wrapper 链需移除 TimeLimit wrapper |
| ICM 特征维度不是 288 | 检查 ICM 配置中 `feature_dim: 288`；确认 4 层 CNN 最后一层输出通道数与全连接层维度匹配 |
| 动作维度不是 17 | 检查 `experiments/crafter_full.yaml` 中 `action_dim: 17`；确认 Crafter 默认 `action_space.n == 17` |
| ICM 初始损失远偏离 2.83 | 检查 ICM 前向模型和逆动力学模型的初始化；确认 loss 计算使用交叉熵（逆动力学）+ MSE（前向特征预测）；权重初始化方案 |
| `int_reward_mean` 初始值远偏离 60 | 检查融合公式 `r_int = η × ICM + r_epi × min(max(α, 1), L)` 中参数 η、r_epi、α、L 的初始值；确认 ICM 奖励和 episodic 奖励的归一化 |
| 训练中 `crafter.Env()` 报错 | 确认 crafter 包已正确安装（`pip install crafter`）；检查资源文件路径（如 textures, world assets）；Windows 下路径分隔符问题 |
| 评测时 reward 全为 0 | Crafter 冒烟训练 1024 步可能尚未学到有效策略，reward=0 不一定异常；检查 `int_reward_mean` 是否正常下降以确认训练有效 |
| VRAM 超过 2.2GB | Crafter 观测 (64,64,3) 较小，但 ICM 4 层 CNN 中间激活可能较大；检查 batch_size；确认 RND Target 网络梯度计算已关闭（`requires_grad=False`） |

---

## 附录 A: 测试通过判定矩阵

| 编号 | 测试场景 | 关键通过条件 | 优先级 |
|------|----------|-------------|--------|
| TC-01 | 消融批量启动 | 4 组实验全部完成，汇总 JSON 含 4 组对比数据 | P0 |
| TC-02 | 断点续训 | global_step 从断点延续，loss 曲线连续 | P0 |
| TC-03 | 离线评测 | JSON/MD 报告生成，字段完整有效 | P0 |
| TC-04 | ONNX 导出 | ONNX 文件生成，PyTorch vs ONNX diff < 1e-4，动态 batch 通过 | P0 |
| TC-05 | LRU 淘汰 | 容量上限 200，淘汰最旧条目，KNN 正常 | P1 |
| TC-06 | Wandb 8 指标 | 8 个面板均有数据，数值范围合理 | P1 |
| TC-07 | VRAM 监控 | 峰值 < 2200MB，无 OOM | P0 |
| TC-08 | 种子复现 | 三大环境两次运行权重一致 (diff < 1e-6) | P1 |
| TC-09 | MiniGrid 冒烟 | 1024 步不崩溃，检查点完整 | P2 |
| TC-10 | Crafter 冒烟 | 1024 步不崩溃，ICM 288维/17动作，兼容修复生效 | P2 |

## 附录 B: 快速回归命令

```bash
# 一键运行所有 P0 测试（核心功能验证）
python scripts/train.py --config experiments/crafter_full.yaml --total-steps 1024 --checkpoint-dir checkpoints/regression --checkpoint-interval 256
python scripts/run_ablation.py --env crafter --steps 512
python scripts/evaluate.py --checkpoint checkpoints/regression/ckpt_1024.pt --env crafter --n-episodes 3 --output-dir outputs/regression_eval
python scripts/export_onnx.py --checkpoint checkpoints/regression/ckpt_1024.pt --output onnx_exports/regression.onnx --env crafter --opset 17
python scripts/check_vram_limit.py --wrap "python scripts/train.py --config experiments/crafter_full.yaml --total-steps 512 --checkpoint-dir checkpoints/regression_vram"
```

```bash
# 一键运行所有冒烟测试
python scripts/train.py --config experiments/minigrid_doorkey_full.yaml --total-steps 1024 --checkpoint-dir checkpoints/smoke_minigrid --checkpoint-interval 256
python scripts/train.py --config experiments/crafter_full.yaml --total-steps 1024 --checkpoint-dir checkpoints/smoke_crafter --checkpoint-interval 256
