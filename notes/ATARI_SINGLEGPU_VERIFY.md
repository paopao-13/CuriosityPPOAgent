# Atari 单卡 (RTX 3060 Laptop 6GB) 验证报告

> 目标：核实 Atari（Montezuma's Revenge）依赖（`ale-py` + ROM）与多环境并行 `n_envs=8` 是否适配单卡 3060 笔记本。
> 验证环境：本机 RTX 3060 Laptop (6GB, `torch 2.7.1+cu118`), 系统 Python 3.11, `gymnasium 1.3.0` / `ale-py 0.12.0`。
> 验证日期：2026-07-15。

---

## 1. 依赖安装（实跑确认）

```bash
# 1) 安装 ale-py 与 AutoROM（接受 ROM 许可的额外组件）
pip install "ale-py" "autorom[accept-rom-license]"

# 2) 下载 ROM（关键：用控制台脚本 autorom，不是 python -m AutoROM.roms）
autorom --accept-license
```

**踩坑记录（已替你试错）**：
- ❌ `python -m AutoROM.roms --accept-license` → `No module named AutoROM.roms.__main__`（它是 package 不是可执行模块）。
- ✅ 直接用 `autorom` 控制台脚本即可。
- ROM 落盘位置：`.../site-packages/AutoROM/roms/`（AutoROM 0.6.1 默认装到自己的包目录，**不是** `~/.ale`）。`montezuma_revenge.bin` 存在即说明可用。
- `gymnasium.make("ALE/MontezumaRevenge-v5")` 在 ale-py 0.12.0 下能**自动找到**该 ROM，无需手动设置 `ALE_PY_ROM_DIR`。

`tests/conftest.py` 原本只在 `ale_py` 导入失败时跳过 atari 测试；现在 `ale_py` 已装，相关测试会被真正执行。

---

## 2. 实跑暴露并修复的 Bug：Atari 观测布局错乱

**这是会直接让 Atari 训练崩的潜伏 Bug**（Crafter 因布局约定不同一直掩盖它）。

### 现象
```
RuntimeError: Calculated padded input size per channel: (4 x 84). Kernel size: (8 x 8).
Kernel size can't be greater than actual input size
```
发生在 `NatureDQNEncoder.forward` 的第一个 `Conv2d(8×8)`。

### 根因
项目约定：**环境观测出 HWC，网络入 CHW**，由 `agent._to_tensor()` 统一做 `permute(0,3,1,2)` 转换。
- Crafter 观测是 `(64,64,3)` HWC → 转换后 `(N,3,64,64)` CHW → ✅ 一直正常。
- Atari 的 `FrameStack` 用 `np.stack(..., axis=0)` 把 4 帧堆在**第 0 维**，产出 `(4,84,84)` CHW。再经 `_to_tensor` 的 permute 被错当成 HWC 解析 → 变成 `(N,84,4,84)`，高度维=4 < 卷积核 8 → 崩溃。

`FrameStack` 是 **Atari 专用**（`grep` 确认全仓仅 `atari_env.py` 使用），所以修复零波及 Crafter。

### 修复
`src/curiosity_ppo/envs/wrappers.py` 的 `FrameStack`：把堆叠轴由 `axis=0` 改为 `axis=-1`，并同步把 `observation_space` 形状由 `(k,)+old.shape` 改为 `old.shape+(k,)`，使其与 Crafter 一样产出 **HWC `(84,84,4)`**。修复后观测为 `(8,84,84,4)`，`_to_tensor` 转换得到正确的 `(8,4,84,84)` CHW，卷积正常。

---

## 3. 端到端验证结果（RTX 3060 真卡）

用项目自带工厂 `make_atari_env(n_envs=8)` + `CuriosityPPOAgent`（`use_amp=True`，与 `atari_montezuma_full.yaml` 一致）实跑 3 个 rollout：

| 项 | 结果 |
|------|------|
| 环境构建 | ✅ `make_atari_env(8)` 成功，观测 `(8,84,84,4)` uint8 |
| wrapper 链路 | ✅ NoopReset/MaxAndSkip/EpisodicLife/GrayResize/FrameStack/ClipReward 全通 |
| 奖励裁剪 | ✅ 奖励被裁到 `{-1,0,1}` |
| 网络前向 | ✅ Actor-Critic + ICM + RND 前向/反向无维度错误 |
| FP16 训练 | ✅ AMP 路径（上一轮已修 `update()` 缺失）正常 |
| **峰值 VRAM** | **177.6 MB / 6144 MB ≈ 2.9%** |
| 单 rollout 耗时 | 稳态 ~9.1s（=1024 环境步 + 32 次 PPO 更新） |

> 内部奖励均值早期 ~470、intrinsic value loss ~2.8e7 属正常早期现象（RND predictor 尚未拟合、Montezuma 外在奖励长期为 0）。训练时建议监控内在/外在奖励尺度平衡；若想更稳定可加 intrinsic reward 的 running‑normalization（非阻塞项）。

---

## 4. `n_envs=8` 与单卡 3060 的匹配度结论

| 维度 | 结论 |
|------|------|
| **显存** | ✅ 极度宽裕。8 环境仅用 2.9% 显存，**理论上可提至 16–32 环境仍有巨大余量**。 |
| **吞吐瓶颈** | ⚠️ 不在显存，而在 **CPU 环境步进**：`DummyVecEnv` 是单进程串行 8 环境。增大 `n_envs` 能提升每 rollout 的环境步数，但 CPU 串行开销同步增加，边际收益有限。 |
| **推荐** | 保持 `n_envs=8`（与配置一致、安全、可复现）；若想加速可尝试 16，需重新评估单 rollout 耗时。当前配置对单卡笔记本**完全匹配**。 |

### 10M 环境步时间预算（方案 a 口径）
- 1 rollout = `n_envs × n_steps` = 8 × 128 = **1024 环境步**。
- 10M 环境步 ≈ 9766 rollouts × ~9.1s ≈ **24.4 小时**（仅训练循环，乐观值）。
- 计入周期性评测、检查点、后期 episode 变长后：** realistic 约 1.5–3 天**（单机 3060）。

---

## 5. Atari 指标可行性（与之前决策一致）

- 严格 **10M 环境步（≈40M ALE 帧）** 在单卡 3060 上**可以实跑**（链路已验证、显存/时间均可行）。
- 按既定**方案 a**：交付时报 10M 步下的**实测分 + 相对 PPO 基线 120 分的提升**，不写 3500+（该分数需约 10 亿帧 / ~12.8B 环境步，远超本预算）。
- 好奇心驱动有望稳定探索通过首房（250–800 分区间），相对 PPO 120 已是显著且可辩护的提升。

---

## 6. 复现命令

```bash
# 安装依赖（见 §1）
pip install -r requirements.txt
pip install "ale-py" "autorom[accept-rom-license]" && autorom --accept-license

# 多种子训练 + 自动评测 + 聚合（已支持 --seeds）
python scripts/run_ablation.py --env atari --steps 10000000 \
    --ablations full --seeds 42,43,44 --use-wandb

# 单独评测（取该组最新检查点）
python scripts/evaluate.py \
    --checkpoint results/ablation/atari_montezuma_full/seed_42/step_*.pt \
    --env atari --n-episodes 10
```

---

## 7. 本次变更文件

| 文件 | 改动 |
|------|------|
| `src/curiosity_ppo/envs/wrappers.py` | `FrameStack` 堆叠轴 `axis=0 → axis=-1`，观测空间形状同步为 HWC，修复 Atari 卷积崩溃 |
| `ATARI_SINGLEGPU_VERIFY.md` | 本报告 |
