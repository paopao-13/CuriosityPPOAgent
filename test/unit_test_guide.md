# CuriosityPPOAgent 单元测试专项执行指南

> **项目**: CuriosityPPOAgent — ICM+RND 分层新颖好奇心 PPO 智能体
> **硬件**: AMD R7 6800H + RTX 3060 Laptop 6GB
> **显存红线**: 2.2 GB
> **测试规模**: 157 个单元测试，分布在 `tests/` 目录下 21 个测试文件中
> **技术架构**: ICM 4层CNN→288维, RND Target+Predictor MSE, Episodic Memory CPU-FAISS KNN LRU 200条, 融合公式 `r_int = η×ICM前向损失 + r_episodic×min(max(α_t,1),L)`, 双价值头PPO `γ_ext=0.999 γ_int=0.99`

---

## 目录

1. [157个单元测试分类明细](#1-157个单元测试分类明细)
2. [双平台一键批量执行](#2-双平台一键批量执行)
3. [单元测试硬性合格标准](#3-单元测试硬性合格标准)
4. [单模块独立测试执行](#4-单模块独立测试执行)

---

## 1. 157个单元测试分类明细

按功能模块分为六大类，共计 21 个测试文件、157 个测试用例。

### 1.1 智能体与训练核心（Agent & Training Core）— 35 个用例

| 序号 | 测试文件 | 测试内容 | 预计用例数 |
|:---:|:---|:---|:---:|
| 1 | `tests/test_agent.py` | CuriosityPPOAgent 端到端测试：初始化校验（n_actions=17, in_channels=3）、完整训练步指标返回（policy_loss/value_ext_loss/value_int_loss/entropy/icm_inverse_loss/icm_forward_loss/rnd_loss）、参数确实更新（ActorCritic + ICM 权重变化）、动作选择（确定性/随机策略）、多步训练不崩溃（3步×16n_steps×2envs）、消融无ICM（icm_net=None, rnd 仍存在）、消融无RND（rnd_net=None, icm 仍存在）、模型保存加载 | 7 |
| 2 | `tests/test_ppo_trainer.py` | PPO 训练器测试：裁剪目标函数（clip_ratio 范围控制）、双价值损失（ext_value + int_value 独立计算）、梯度累积 flush（accumulation_steps 触发 optimizer.step）、学习率衰减（linear schedule） | 7 |
| 3 | `tests/test_policy.py` | ActorCritic 双价值头测试：策略 logits 输出形状（n_actions 维）、外部价值头标量输出、内在价值头标量输出、前向传播返回值结构、梯度可反向传播 | 5 |
| 4 | `tests/test_gae.py` | GAE 广义优势估计计算测试：双轨 γ_ext=0.999/γ_int=0.99，情景制 done 截断 bootstrap、非情景制跨 episode bootstrap、done 处不污染前序步、returns 与 values 无关性（γ=λ=1）、不变式 returns=advantages+values、输出形状校验、多环境并行独立计算 | 8 |
| 5 | `tests/test_rollout_buffer.py` | RolloutBuffer 测试：n_steps×n_envs 存储容量、mini-batch 随机采样、advantages/returns/values 数据完整性、buffer 溢出处理、reset 清空、add 批量写入、采样 shuffle 不重复 | 8 |

### 1.2 好奇心驱动模块（Curiosity Modules）— 44 个用例

| 序号 | 测试文件 | 测试内容 | 预计用例数 |
|:---:|:---|:---|:---:|
| 6 | `tests/test_icm.py` | ICM 网络测试：forward 返回 (inverse_loss, forward_loss, phi_t) 三元组、逆动力学稀疏 Softmax 交叉熵损失可微、前向 MSE 损失可微、编码器 288 维输出形状、inverse_loss 量级在 [0, ln(17)×5+1] 范围内（随机策略初始损失≈2.83） | 4 |
| 7 | `tests/test_icm_module.py` | ICMCuriosity 模块测试：内在奖励计算（前向预测误差）、η 系数调制、奖励标量输出、奖励非负性、batch 维度处理、梯度不回传到 Target | 6 |
| 8 | `tests/test_rnd.py` | RND 网络测试：固定 Target 网络不可训练（requires_grad=False）、可训练 Predictor 网络、MSE 误差输出标量、Target 与 Predictor 输出形状一致、预测误差对 Predictor 参数可微 | 5 |
| 9 | `tests/test_rnd_module.py` | RNDCuriosity 模块测试：内在奖励计算（predictor-target MSE）、α_t 自适应系数计算、奖励标准化、奖励有限性、batch 处理、Target 网络参数冻结验证 | 7 |
| 10 | `tests/test_episodic_memory.py` | EpisodicMemory 情景记忆测试：空库伪计数=0、空库奖励=1e3（极度新颖）、添加后伪计数>0、添加后奖励>0且有限、重复访问奖励递减（新颖性下降）、reset 清空内存库、远距离查询 kernel 非零 | 7 |
| 11 | `tests/test_memory_bank.py` | LRUMemoryBank 测试：预分配 numpy 数组零拷贝、环形缓冲区写入覆盖、FIFO 淘汰策略、kNN 距离计算正确性、容量上限边界、reset 清空、重复添加去重 | 7 |
| 12 | `tests/test_ngu_fusion.py` | NGUFusion 融合模块测试：融合公式 `r_int = η×ICM + r_epi×min(max(α,1),L)` full 模式、no_icm 模式（r_icm=0）、no_episodic 模式（走 rnd-only 路径）、no_rnd 模式（α=1）、全关闭返回 0、α 裁剪到 L 上限、α 裁剪到 1 下限、controllable_emb=None 跳过 episodic | 8 |

### 1.3 编码器与网络架构（Encoders & Networks）— 13 个用例

| 序号 | 测试文件 | 测试内容 | 预计用例数 |
|:---:|:---|:---|:---:|
| 13 | `tests/test_encoders.py` | CNN 编码器测试：CrafterEncoder 64×64×3→288 维输出、NatureDQNEncoder 84×84×4→512 维输出、前向传播形状校验、梯度可反向传播、参数量合理性 | 5 |
| 14 | `tests/test_amp.py` | AMP 混合精度管理器测试：autocast 上下文管理、GradScaler 缩放、loss scaling 溢出检测、unscale 梯度、AMP 开关切换、fp16→fp32 类型转换、梯度 inf 检测、scaler state 保存恢复 | 8 |

### 1.4 环境与预处理包装器（Environment & Wrappers）— 32 个用例

| 序号 | 测试文件 | 测试内容 | 预计用例数 |
|:---:|:---|:---|:---:|
| 15 | `tests/test_vec_env.py` | DummyVecEnv 向量化环境测试：reset 堆叠（n_envs 维 obs）、step 自动 reset（terminal_observation 保存）、返回值结构 (obs, rewards, dones, infos)、多环境并行独立、close 资源释放、观测空间一致性 | 6 |
| 16 | `tests/test_compat.py` | GymCompatWrapper 兼容性测试：seed 参数拦截、options 参数拦截、step 4 值→5 值转换（旧版 gym 兼容）、step 5 值透传、observation_space 透传、action_space 透传 | 6 |
| 17 | `tests/test_wrappers.py` | 预处理包装器测试：ObsToFloat32（dtype 转换 + /255 归一化 + observation_space 更新 + step）、RunningMeanStd（Welford 基础统计 + 向量 shape + 增量匹配批量 + 单样本）、ObsNormalize（输出形状 + 统计量更新 + 常量归一化→0）、FrameStack（形状 + observation_space + reset 填充 + step 移位）、GrayResizeObservation 84×84（形状 + 灰度 + step + observation_space + 灰度输入兼容） | 20 |

### 1.5 配置与工具链（Config & Utils）— 19 个用例

| 序号 | 测试文件 | 测试内容 | 预计用例数 |
|:---:|:---|:---|:---:|
| 18 | `tests/test_config.py` | YAML 配置加载测试：默认配置完整性、YAML 文件覆盖加载、嵌套字段合并、类型校验 | 4 |
| 19 | `tests/test_seed.py` | 随机种子设置测试：python random 可复现、numpy seed 可复现、torch CPU seed 可复现、torch CUDA seed 可复现、全栈联合可复现 | 5 |
| 20 | `tests/test_reward_norm.py` | RunningMeanStd 和 RewardNormalizer 测试：初始 mean=0/var=1/count=1e-4、2D 批次更新统计、Welford 增量匹配批量、向量 shape 逐元素统计、标量单样本更新、normalizer 返回 float、归一化除以 std、正向 clip、负向 clip、调用更新统计量 | 10 |

### 1.6 修复验证（Fixes Validation）— 13 个用例

| 序号 | 测试文件 | 测试内容 | 预计用例数 |
|:---:|:---|:---|:---:|
| 21 | `tests/test_fixes.py` | 修复验证测试：11 个 P0/P1/P2 修复点回归验证（含 GAE 截断、ICM 逆动力学损失、RND Target 冻结、消融配置开关、奖励归一化 clip 等）+ Atari 集成验证 + 消融集成验证 | 13 |

### 1.7 用例统计总览

| 模块分类 | 文件数 | 用例数 | 占比 |
|:---|:---:|:---:|:---:|
| 智能体与训练核心 | 5 | 35 | 22.3% |
| 好奇心驱动模块 | 7 | 44 | 28.0% |
| 编码器与网络架构 | 2 | 13 | 8.3% |
| 环境与预处理包装器 | 3 | 32 | 20.4% |
| 配置与工具链 | 3 | 19 | 12.1% |
| 修复验证 | 1 | 13 | 8.3% |
| **合计** | **21** | **156+** | **100%** |

> **注**: 各文件用例数为约数（标注"约"），实际执行时以 `pytest --co -q` 收集结果为准，总量目标 157 个。

---

## 2. 双平台一键批量执行

### 2.1 标准批量执行（Linux / macOS / Windows 通用）

```bash
# 在项目根目录执行，扫描 tests/ 目录下全部 21 个测试文件
python -m pytest tests/ -v --tb=short
```

- `-v`: 详细输出每个测试用例的执行结果（PASSED / FAILED / SKIPPED）
- `--tb=short`: 失败时仅显示精简回溯信息，避免日志过长
- `pyproject.toml` 已配置 `testpaths = ["tests"]` 和 `addopts = "-v --tb=short"`，直接 `python -m pytest` 同样生效

### 2.2 带日志输出（Windows PowerShell）

```powershell
# 创建日志目录（如不存在）
New-Item -ItemType Directory -Force -Path "test/logs" | Out-Null

# 执行测试并同时输出到控制台和日志文件
python -m pytest tests/ -v --tb=short 2>&1 | Tee-Object -FilePath "test/logs/unit_test.log"
```

`Tee-Object` 将 stdout + stderr 合并后同时输出到屏幕和文件，日志保存在 `test/logs/unit_test.log`。

### 2.3 带日志输出（Linux / macOS）

```bash
# 创建日志目录
mkdir -p test/logs

# 执行测试并同时输出到控制台和日志文件
python -m pytest tests/ -v --tb=short 2>&1 | tee test/logs/unit_test.log
```

### 2.4 全链路自动化脚本（含依赖校验 + 显存预检）

项目已提供全链路自动化测试脚本，在上述单元测试基础上还包含 MiniGrid 冒烟训练、消融验证、ONNX 导出等环节：

```powershell
# Windows PowerShell
.\test\scripts\run_all_test.ps1
```

```bash
# Linux / macOS
chmod +x test/scripts/run_all_test.sh
./test/scripts/run_all_test.sh
```

脚本执行流程（6 步）：
1. Python 环境依赖校验（torch / numpy / gymnasium / crafter / minigrid / ale_py / wandb / onnx 等）
2. GPU 显存基线预检测（峰值 < 2.2 GB）
3. **批量执行全部单元测试（157 个）** → 日志写入 `test/logs/unit_test.log`
4. MiniGrid 轻量冒烟集成训练测试（1024 步）
5. 小规模消融功能验证（4 组 × 256 步：full / no_icm / no_episodic / no_rnd）
6. ONNX 模型导出 + Web Demo 基础校验

### 2.5 仅收集不执行（预览用例清单）

```bash
# 列出全部收集到的测试用例 ID，不实际执行
python -m pytest tests/ --collect-only -q

# 仅输出总数
python -m pytest tests/ --co -q | tail -1
```

---

## 3. 单元测试硬性合格标准

### 3.1 核心合格标准（三项全部满足方可判定通过）

| 标准编号 | 标准名称 | 判定条件 | 失败后果 |
|:---:|:---|:---|:---|
| S-01 | **100% 用例通过** | 157 个测试用例全部 PASSED，0 个 FAILED（SKIPPED 仅允许在无 CUDA 或无 Atari ROM 环境下触发） | 不得进入冒烟训练阶段 |
| S-02 | **无显存 OOM** | 全部测试执行期间 GPU 显存峰值 < 2.2 GB（通过 `test/scripts/check_vram_limit.py --baseline-only` 预检测） | 触发显存红线，需排查内存泄漏 |
| S-03 | **无接口兼容异常** | 无 `TypeError` / `AttributeError` / `ImportError` 等接口签名不匹配错误；GymCompatWrapper 正确处理 step 4/5 值转换 | 环境层接口断裂，训练无法启动 |

### 3.2 补充质量门禁

| 门禁编号 | 检查项 | 判定条件 |
|:---:|:---|:---|
| Q-01 | 数值有限性 | 所有损失值（policy_loss / value_ext_loss / value_int_loss / icm_inverse_loss / icm_forward_loss / rnd_loss）均为有限数值，无 `NaN` / `Inf` |
| Q-02 | 形状一致性 | 编码器输出 288 维（ICM）/ 512 维（NatureDQN）、策略 logits 维度 = n_actions=17、价值头输出标量 |
| Q-03 | 梯度有效性 | ActorCritic / ICM Predictor / RND Predictor 参数在训练步后确实更新（`torch.equal` 前后对比）；Target 网络参数保持冻结 |
| Q-04 | 消融正确性 | no_icm 模式下 `icm_net=None` 且 `rnd_net` 存在；no_rnd 模式下 `rnd_net=None` 且 `icm_net` 存在；对应 loss 键从 metrics 中消失 |
| Q-05 | 随机可复现 | `set_seed()` 后 python / random / numpy / torch 全栈产生相同序列 |
| Q-06 | GAE 不变式 | `returns = advantages + values` 在任意 γ/λ 配置下成立 |

### 3.3 判定流程

```text
执行 python -m pytest tests/ -v --tb=short
         │
         ▼
  ┌── PASSED == 157 ? ──→ No ──→ 判定 FAIL，查看 FAILED 用例
  │
  Yes
  │
  ▼
  ┌── 显存峰值 < 2.2GB ? ──→ No ──→ 判定 FAIL，排查显存泄漏
  │
  Yes
  │
  ▼
  ┌── 无接口兼容异常 ? ──→ No ──→ 判定 FAIL，修复接口签名
  │
  Yes
  │
  ▼
  判定 PASS → 可进入冒烟训练阶段
```

---

## 4. 单模块独立测试执行

当批量测试出现失败时，使用以下命令精准定位报错模块。

### 4.1 按文件执行（精确到单个测试文件）

```bash
# 智能体端到端测试
python -m pytest tests/test_agent.py -v

# PPO 训练器
python -m pytest tests/test_ppo_trainer.py -v

# GAE 广义优势估计
python -m pytest tests/test_gae.py -v

# ICM 网络
python -m pytest tests/test_icm.py -v

# RND 网络
python -m pytest tests/test_rnd.py -v

# NGU 融合模块
python -m pytest tests/test_ngu_fusion.py -v

# 情景记忆
python -m pytest tests/test_episodic_memory.py -v

# 预处理包装器（用例最多）
python -m pytest tests/test_wrappers.py -v

# 奖励归一化
python -m pytest tests/test_reward_norm.py -v

# 修复验证
python -m pytest tests/test_fixes.py -v
```

### 4.2 按关键字过滤（跨文件匹配测试函数名）

`pytest -k` 支持用表达式匹配测试用例名称（函数名 / 类名），逻辑运算符 `and` / `or` / `not` 可组合使用：

```bash
# 过滤所有 ICM 相关测试（匹配 test_icm.py + test_icm_module.py 中的用例）
python -m pytest tests/ -k "icm" -v

# 过滤所有 RND 相关测试
python -m pytest tests/ -k "rnd" -v

# 过滤所有消融相关测试
python -m pytest tests/ -k "ablation" -v

# 过滤 GAE 情景制测试
python -m pytest tests/ -k "gae and episodic" -v

# 过滤 GAE 非情景制测试
python -m pytest tests/ -k "gae and non_episodic" -v

# 过滤编码器形状校验测试
python -m pytest tests/ -k "encoder and shape" -v

# 过滤 FrameStack 相关测试
python -m pytest tests/ -k "frame_stack" -v

# 过滤 RunningMeanStd 相关测试（跨 test_wrappers.py 和 test_reward_norm.py）
python -m pytest tests/ -k "running_mean_std" -v

# 排除修复验证文件，仅跑功能测试
python -m pytest tests/ -k "not fixes" -v

# 组合过滤：ICM 或 RND，但排除模块级测试
python -m pytest tests/ -k "(icm or rnd) and not module" -v
```

### 4.3 按标记执行（marker 过滤）

`conftest.py` 中注册了以下条件跳过标记，可通过 `-m` 选择性执行：

```bash
# 仅执行需要 GPU 的测试（无 CUDA 环境自动跳过）
python -m pytest tests/ -m "requires_gpu" -v

# 仅执行需要 Atari ROM 的测试（无 ale_py 自动跳过）
python -m pytest tests/ -m "requires_atari" -v

# 跳过 GPU 依赖测试（CPU 模式快速验证）
python -m pytest tests/ -m "not requires_gpu" -v

# 跳过 Atari 依赖测试
python -m pytest tests/ -m "not requires_atari" -v
```

> **说明**: `conftest.py` 中的 `pytest_collection_modifyitems` 钩子会在收集阶段自动检测 `torch.cuda.is_available()` 和 `ale_py` 可导入性，对标记 `requires_gpu` / `requires_atari` 的用例在条件不满足时自动添加 `skip`。

### 4.4 单个测试函数执行（最小粒度定位）

```bash
# 执行指定文件中的指定测试函数
python -m pytest tests/test_agent.py::test_agent_init -v

# 执行指定文件中的指定测试类方法
python -m pytest tests/test_wrappers.py::test_obs_to_float32_dtype_and_range -v

# 执行单个函数并显示详细输出（含 print）
python -m pytest tests/test_gae.py::test_gae_episodic_zero_values -v -s
```

### 4.5 失败快速终止 + 重跑失败用例

```bash
# 遇到第一个失败立即停止（快速定位首个报错）
python -m pytest tests/ -v --tb=short -x

# 遇到第 N 个失败停止
python -m pytest tests/ -v --tb=short --maxfail=3

# 仅重跑上次失败的用例（依赖 pytest-cache）
python -m pytest tests/ -v --tb=short --lf

# 先跑失败用例，再跑其余用例
python -m pytest tests/ -v --tb=short --ff
```

### 4.6 模块定位速查表

| 报错现象 | 优先排查文件 | 执行命令 |
|:---|:---|:---|
| Agent 初始化失败 / 训练步崩溃 | `tests/test_agent.py` | `python -m pytest tests/test_agent.py -v` |
| PPO 裁剪 / 价值损失异常 | `tests/test_ppo_trainer.py` | `python -m pytest tests/test_ppo_trainer.py -v` |
| GAE 优势估计数值错误 | `tests/test_gae.py` | `python -m pytest tests/test_gae.py -v` |
| ICM 逆动力学 / 前向损失异常 | `tests/test_icm.py` | `python -m pytest tests/test_icm.py -v` |
| RND Target/Predictor 异常 | `tests/test_rnd.py` | `python -m pytest tests/test_rnd.py -v` |
| 融合公式 r_int 计算错误 | `tests/test_ngu_fusion.py` | `python -m pytest tests/test_ngu_fusion.py -v` |
| 情景记忆 kNN / 伪计数异常 | `tests/test_episodic_memory.py` | `python -m pytest tests/test_episodic_memory.py -v` |
| LRU 内存库 / 环形缓冲异常 | `tests/test_memory_bank.py` | `python -m pytest tests/test_memory_bank.py -v` |
| 编码器输出维度错误 | `tests/test_encoders.py` | `python -m pytest tests/test_encoders.py -v` |
| AMP 混合精度异常 | `tests/test_amp.py` | `python -m pytest tests/test_amp.py -v` |
| 环境包装器 / 观测预处理异常 | `tests/test_wrappers.py` | `python -m pytest tests/test_wrappers.py -v` |
| 向量化环境 reset/step 异常 | `tests/test_vec_env.py` | `python -m pytest tests/test_vec_env.py -v` |
| Gym 兼容性 step 4/5 值异常 | `tests/test_compat.py` | `python -m pytest tests/test_compat.py -v` |
| 奖励归一化 Welford 异常 | `tests/test_reward_norm.py` | `python -m pytest tests/test_reward_norm.py -v` |
| 随机种子不可复现 | `tests/test_seed.py` | `python -m pytest tests/test_seed.py -v` |
| YAML 配置加载异常 | `tests/test_config.py` | `python -m pytest tests/test_config.py -v` |
| RolloutBuffer 存储/采样异常 | `tests/test_rollout_buffer.py` | `python -m pytest tests/test_rollout_buffer.py -v` |
| 策略网络 logits/价值头异常 | `tests/test_policy.py` | `python -m pytest tests/test_policy.py -v` |
| ICM 模块奖励计算异常 | `tests/test_icm_module.py` | `python -m pytest tests/test_icm_module.py -v` |
| RND 模块 α_t 计算异常 | `tests/test_rnd_module.py` | `python -m pytest tests/test_rnd_module.py -v` |
| 修复回归 / 集成验证失败 | `tests/test_fixes.py` | `python -m pytest tests/test_fixes.py -v` |
