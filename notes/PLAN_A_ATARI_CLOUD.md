# 方案 A 详细实施计划：阿里云 PAI 云端重训 Atari Montezuma (Curiosity-PPO 10M 步)

> 适用对象：本地 16GB 宿主机 RAM 跑 Atari 10M 步必 OOM（`numpy._ArrayMemoryError` + Traceback），3.87M 进度已丢失。
> 目标：用云端 GPU 实例（内存 ≥32GB）从 0 重训到 10M 步，checkpoint 落 OSS 持久化、可断点续训。
> 仓库已备：`scripts/cloud_train_atari.sh`（自愈 watch-dog）、`requirements_atari.txt`（精简依赖）、`experiments/atari_montezuma_full.yaml`（ent_coef=0.02 抗坍缩）。

---

## 0. 可行性判定（先确认这件事能成）

| 风险点 | 本地现状 | 云端解法 | 是否成立 |
|---|---|---|---|
| 宿主机 RAM OOM | 16GB，numpy `obs_rms.update` 时崩 | A10/A100 实例内存 ≥30GB | ✅ 根因消除 |
| 进度丢失 | 一次崩丢 3.87M | checkpoint 间隔压到 10240 步 + 落 OSS | ✅ 最坏只丢 ~10k 步 |
| 熵坍缩 | 曾坍缩到 <0.3 | 用 ent_coef=0.02 配置 | ✅ 已验证 |
| 断点续训 | 不支持 | `agent.train()` 是 `while global_step<total_steps`，`--resume` 续跑 | ✅ 已验证（agent.py:627/708） |
| 成本 | — | 学生 300 元券 + DSW 750h 免费额度 | ✅ 预算充足 |

**成功标准（训练健康）**：`global_step` 达 10M；`entropy>0.3`（A10/A100 上应稳定 1+）；`icm_forward_loss<100`；日志无 `nan/inf/Traceback/MemoryError`。

---

## 1. 前置准备（本地，约 30 分钟）

| 步骤 | 操作 | 备注 |
|---|---|---|
| 1.1 | 阿里云官网完成**学生认证**（学信网） | 必备，否则无优惠 |
| 1.2 | 控制台「我的权益 / 优惠券」领 **300 元无门槛算力券** | 抵扣按量付费 |
| 1.3 | 开通 **PAI 人工智能平台**（免费） | 领 **DSW 750 计算时**试用（A10/V100/G6，3 个月内每月 250h）+ **DLC 100 CU·H** |
| 1.4 | 把 `curiosity-ppo` 推到 Git（GitHub / Gitee / 阿里云 Codeup） | 云端 `git clone` 拉取最稳；或打 zip 上传 |
| 1.5 | 确认本仓库已含 `scripts/cloud_train_atari.sh`、`requirements_atari.txt` | 本计划配套文件 |

**Gate 1**：学生认证通过 + 300 券到账 + PAI 已开通。未满足则不进阶段 2。

---

## 2. 创建实例（两条路由，按需选）

### 🅐 推荐路由 — PAI-DLC（无人值守长任务）
10M 步预估跑数小时到十几小时，DLC 任务在你**关电脑后继续跑**、崩溃可重投、日志与产出可直连 OSS。**最适合本次。**

### 🅑 调试路由 — PAI-DSW（交互式 Notebook）
先开 DSW 实例，在 Terminal 里验证前 1–2 万步**不 OOM**，确认无误后再转 DLC 或直接在 DSW 挂着跑。

### 实例规格抉择（关键）

| 规格 | 显存 | 实例内存 | 费用 | 备注 |
|---|---|---|---|---|
| **A10 (gn7i)** | 24GB | 通常 ≥30GB | **DSW 750h 免费额度内**（零成本） | 显存远超 Atari 需求，内存解除 OOM；算力较 A100 慢约 2×，但免费，优先选 |
| **A100 (gn8v/epn8)** | 40/80GB | ≥60GB | 抢占式按量，约数元/小时（300 券覆盖） | 更快，适合抢时间 |

> ⚠️ 选规格时**看「内存」列**：务必 ≥32GB。若只能选到小内存规格，启动脚本里设 `VEC_ENV=dummy` 即可避免本地那种 RAM OOM。

**Gate 2**：实例创建成功、能 `ssh`/进 Terminal、`nvidia-smi` 看到 GPU、内存 ≥30GB。

---

## 3. 环境初始化（DSW Terminal 或 DLC 启动命令）

```bash
# 3.1 拉代码（任选其一）
git clone <你的仓库地址> curiosity-ppo && cd curiosity-ppo
# 或：把本地 zip 上传到实例后 unzip curiosity-ppo.zip && cd curiosity-ppo

# 3.2 建隔离环境并装精简依赖（比全量快很多）
python -m venv .venv && source .venv/bin/activate
pip install -r requirements_atari.txt

# 3.3 Atari ROM 授权（首次需接受协议，requirements 已含 gymnasium[atari,accept-rom-license]）
python -c "import ale_py, gymnasium, torch; print('env ok, cuda=', torch.cuda.is_available())"
```

**Gate 3**：上面 `python -c` 打印 `env ok, cuda= True` 且无报错。`cuda=False` 说明 GPU 没挂上，停。

---

## 4. 挂 OSS（关键！否则实例释放即丢进度）

| 平台 | 做法 |
|---|---|
| **DSW** | 左侧「OSS」标签页挂载你的 Bucket 到 `/mnt/oss`，建目录 `/mnt/oss/atari_seed42` |
| **DLC** | 建任务时把「输出通道」指向 OSS Bucket 路径，容器内对应本地目录即持久化 |

训练把 checkpoint 写进这个挂载目录 → 实例释放/被抢占都不丢，可 `--resume` 续训。

**Gate 4**：`touch /mnt/oss/atari_seed42/.write_test && ls -l /mnt/oss/atari_seed42/` 能看到文件，说明挂载可写。

---

## 5. 启动训练（自愈脚本）

```bash
# 5.1 首次从 0 跑（CKPT_DIR 指向 OSS 挂载目录）
CKPT_DIR=/mnt/oss/atari_seed42 bash scripts/cloud_train_atari.sh

# 5.2 若被抢占/中断，脚本会自动从最新 ckpt 续训；也可手动指定：
# CKPT_DIR=/mnt/oss/atari_seed42 RESUME=/mnt/oss/atari_seed42/step_307200.pt \
#   bash scripts/cloud_train_atari.sh

# 5.3 小内存实例兜底：
# VEC_ENV=dummy CKPT_DIR=/mnt/oss/atari_seed42 bash scripts/cloud_train_atari.sh
```

**脚本行为（自愈 watch-dog）**：
- 崩溃（非 0 退出）→ 等待 10s，自动从 `CKPT_DIR` 最新 `step_*.pt` 续跑，最多重试 `MAX_RETRIES=20` 次；
- 日志出现 `nan/inf/Traceback/MemoryError/CUDA out of memory` → **立即放弃**（退出码 2），避免反复空转烧钱；
- 干净退出（0）→ 视为训练完成（到达 10M 步），正常结束；
- 日志经 `tee` 同时写 `train.log` 与 stdout（DLC 日志服务直接捕获 stdout，无需登录实例即可看进度）。

**Gate 5（小流量验证）**：启动后 `sleep 60` 后 `tail -n 5 /mnt/oss/atari_seed42/train.log`，应看到 `global_step` 在增长、`entropy` 正常（>0.3）、无 nan/inf。通过后再放手跑隔夜。

---

## 6. 监控（命令 + 阈值）

```bash
# 实时看进度（DLC 控制台日志页等同效果）
tail -f /mnt/oss/atari_seed42/train.log
```

| 监控项 | 健康阈值 | 命中即 |
|---|---|---|
| `global_step` | 持续递增 | 停更 >15min 且进程死 → 脚本会自动续训 |
| `entropy` | >0.3（A10/A100 上应 1+） | 长期 <0.3 且 checkpoint 停更 → 进程疑似死，脚本续训 |
| `icm_forward_loss` | 个位数以内 | >100 即发散，脚本会因 Traceback 放弃，需人工查 |
| `nan/inf/Traceback/MemoryError` | 无 | 出现即停（脚本自动 exit 2） |

**Gate 6**：`global_step` 稳定增长、`entropy` 不坍缩、无致命标记 → 可放心离开。

---

## 7. 完成与交付

- `global_step` 达 10M（≥98%）且脚本干净退出（exit 0）→ **Atari 重训完成，可评测**。
- 把最终 ckpt（OSS 上 `step_10000000.pt` 附近）保留，用于评测；minigrid / crafter 此前已完成，三者可一起出指标。
- 评测若仍需 GPU/大内存，建议在云端同实例跑 eval（本地 16GB 曾 OOM），或减小 eval 的 `n_envs`。

---

## 8. 成本核算（带假设，实际以控制台为准）

| 项 | 假设 | 估算 |
|---|---|---|
| 实例 | A10（免费额度内） | **0 元**（首选） |
| 实例 | A100 抢占式，约 3–5 元/小时 | 10M 步约 6–15h → **约 20–75 元** |
| 存储 OSS | 几个 GB ckpt | 几元/月，可忽略 |
| 总额度 | 学生 300 券 + DSW 750h | 完全覆盖，基本零成本 |

> 上表 A100 单价为假设值，**下单前在 PAI 控制台确认实时价**。优先用 A10 免费额度跑，零成本。

---

## 9. 故障预案

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 启动即 `CUDA out of memory` | 显存不够（极少见于 Atari） | 降 `n_envs`：编辑 yaml `n_envs` 或设更小；A10 24GB 基本不会 |
| 启动即 `numpy._ArrayMemoryError` | 实例内存 < 16GB | `VEC_ENV=dummy` 重跑；或换内存更大规格 |
| `torch.compile` 报错 | A10/A100 上偶发 | `USE_COMPILE=0` 重跑 |
| 日志 `entropy` 持续 <0.3 | 坍缩 | 已用 ent_coef=0.02 防御；若仍发生，人工停、调大 ent_coef |
| 进程被抢占中断 | 抢占式实例回收 | watch-dog 自动从 OSS 最新 ckpt 续训，无需干预 |
| `Traceback`/`nan` | 数值发散 | 脚本自动 exit 2；人工查 `train.log` 末尾，必要时清 OSS ckpt 从 0 重跑 |

---

## 10. 一句话执行流

> 领券 → 开 PAI-DSW 选 **A10（免费）** → 挂 OSS 到 `/mnt/oss` → `git clone` + `pip install -r requirements_atari.txt` → `CKPT_DIR=/mnt/oss/atari_seed42 bash scripts/cloud_train_atari.sh` → 看 `tail -f` 确认前 1 万步健康 → 放手跑。checkpoint 落 OSS，崩溃只丢 1 万步，10M 步约半天跑完，基本零成本。
