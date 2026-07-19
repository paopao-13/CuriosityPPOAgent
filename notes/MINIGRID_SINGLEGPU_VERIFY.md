# MiniGrid 单卡验证报告（RTX 3060 Laptop 6G）

> 目标：确认第三个基准 MiniGrid DoorKey 在单卡 RTX 3060（6G）上能端到端跑通，
> 链路 `make_minigrid_env(8) → CuriosityPPOAgent(full, use_amp=True) → train() → 检查点`。
> 与 `ATARI_SINGLEGPU_VERIFY.md` 同构。

## 1. 环境依赖

- 系统 Python 3.11（torch 2.7.1+cu118，真卡 `cuda=True`）。
- `pip install minigrid` → **minigrid 3.1.0**。
- **关键**：项目 `minigrid_env.py` 用 `from minigrid.wrappers import ImgObsWrapper`，
  在 minigrid 3.1.0 + gymnasium 1.3.0 下**导入路径仍然有效，无需改代码**。
- ⚠️ 解释器陷阱：本机会话 `python` 默认指向 managed 3.13（空环境），
  **所有训练/验证必须用系统 3.11**（`C:\Users\jx\AppData\Local\Programs\Python\Python311\python.exe`）。

## 2. 观测布局（无需修改）

MiniGrid 链路：`gymnasium.make → ImgObsWrapper → ResizeObs(64×64) → ObsToFloat32 → DummyVecEnv`。

- 产出 `(64, 64, 3)` **HWC** float32，与项目约定（`env 出 HWC → _to_tensor permute(0,3,1,2) 转 CHW`）一致。
- **无 FrameStack**（MiniGrid 用图像观测 + Episodic Memory 提供时序，不堆叠帧），
  故**不受 Atari 那个 CHW/帧堆叠 Bug 影响**。验证实测 obs `(64,64,3)` 直接被卷积正确吃下。

## 3. 端到端实测（RTX 3060 真卡）

| 指标 | 值 |
|---|---|
| device / torch | cuda / 2.7.1+cu118 |
| obs 形状 | `(64, 64, 3)` HWC |
| 动作空间 | 7（Discrete） |
| 启用模块 | ICM + RND + Episodic + AMP 全开 |
| 3 个 rollout 耗时 | 43.1 s（~14.4 s/rollout = 2048 环境步） |
| **峰值 VRAM** | **280.0 MB / 6442 MB ≈ 4.3%** |
| 检查点 | `step_6144.pt`，含 ppo/icm/rnd **三优化器** |
| reload | `global_step=6144`，三优化器齐全（Bug C 修复在 MiniGrid 同样成立） |

### 训练质量观察（非阻塞）

- 内在奖励均值：rollout 1→2→3 = `24.7 → 11.2 → 9.3`，**单调递减**——
  这是 RND predictor 逐步拟合、探索趋稳的**正常早期现象**，无需改代码。
- 外在奖励前段为 0：DoorKey 是稀疏奖励任务，早期拿不到钥匙/开门属正常。
- 建议正式训练时监控内外奖励尺度平衡（必要时加 intrinsic reward running-normalization），但非阻塞。

## 4. 结论

✅ **MiniGrid DoorKey 在单卡 RTX 3060（6G）上完整跑通，零代码改动。**

- 显存极度宽裕（4.3%），瓶颈在 CPU 串行环境步进（`DummyVecEnv` 单进程 8 环境），非显存；保持 `n_envs=8` 即可复现。
- 1.5M 环境步（total_steps=1_500_000）时间预算 ≈ 1.5M / 2048 × 14.4s ≈ **2.6 小时/种子**（仅训练循环，乐观；
  计入评测/检查点 ≈ 3–4 小时/种子）。多种子（42/43/44）约 1 天可完成。

## 5. 复现命令

```bash
# 必须用系统 Python 3.11（勿用 managed 3.13）
PY="C:/Users/jx/AppData/Local/Programs/Python/Python311/python.exe"

# 单种子 full（严格 1.5M 环境步）
$PY scripts/train.py --config experiments/minigrid_doorkey_full.yaml \
    --total-steps 1500000 --checkpoint-dir results/checkpoints

# 多种子消融（自动按 seed 隔离目录 + 聚合 mean±std）
$PY scripts/run_ablation.py --ablations full,no_icm,no_episodic,no_rnd \
    --seeds 42,43,44
```

---

## 三基准单卡验证总览（截至本验证）

| 基准 | 观测 | 峰值 VRAM | 单卡结论 |
|---|---|---|---|
| Crafter | 64×64×3 HWC | ~208 MB（2.x GB 显存池占用口径见 README） | ✅ 单卡可跑 |
| Atari Montezuma | 84×84×4 HWC（FrameStack 已修） | 177.6 MB / 6G（2.9%） | ✅ 单卡可跑 |
| **MiniGrid DoorKey** | 64×64×3 HWC | **280.0 MB / 6G（4.3%）** | ✅ **单卡可跑** |

三个基准全部在 RTX 3060 Laptop 6G 单卡验证通过，架构与代码路径无遗留阻塞。
下一步：按 `RUN_PLAN.md` 后台起全套多种子实验，产出可复核的 `benchmark_report.json` 替换文档占位符。
