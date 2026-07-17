# 好奇心 PPO 项目 — 面试导向执行方案 (EXECUTION PLAN v2)

> 版本：v2（面试导向 / 量化可辩护 / 硬件可行 / 已补 4 处硬伤）
> 适用硬件：RTX 3060 Laptop 6GB / 主机 16GB RAM（实测显存峰值 ~2.2GB，瓶颈在主机 RAM 与单卡串行）
> 编写依据：仓库真实代码核查（checkpoint.py / agent.py / web/ 已读）
> 定位：目标岗位 = AI Agent 应用 / 模型开发；本项目的面试卖点是「端到端 RL 系统能力 + 方法论严谨性」，不是刷绝对分

---

## 0. 为什么重做方案（目标错位修正）

原目标（19.0% / 3500+ / 96.8 万步）是「论文复现绝对分」，与 AI Agent 应用岗的考察点（LLM 应用架构、RAG、Tool Calling、工程落地、成本/可观测性）**关联度低**。面试官不会因为 Atari 多跑了几百分就认为你更会写 Agent。

面试导向下，本项目的价值主张改为三条可讲述、可复现的证据链：

1. **我能独立跑通端到端 RL 系统**（算法实现 + 训练工程 + 可靠性兜底）。
2. **我用数学方法证明模块有效**（消融 + 显著性检验，而非堆分数）。
3. **我有工程素养**（原子 checkpoint、资源守卫、监控、可复现性）。

因此量化目标从「绝对分」改为「**相对 PPO-only 基线的统计显著增益 + 多种子可复现**」。

---

## 1. 已核实的代码事实（方案落点，非文档声明）

| 文件 / 位置 | 事实 | 影响 |
|---|---|---|
| `src/curiosity_ppo/utils/checkpoint.py:15` | `torch.save(checkpoint, path)` **非原子写** | 中途被 kill 会损坏/丢失 ckpt（已发生 3.87M 进度丢失事故） |
| `src/curiosity_ppo/ppo/agent.py:603` | 评测 `action = logits.argmax(dim=-1)` **贪婪 + 复用训练 env（seed=42）** | MiniGrid 假 0% 根因，算法未必失败 |
| `agent.py:584` `def evaluate(...)` | 训练内评测入口，返回 reward 非 success_rate | 缺成功率口径 |
| `web/package.json` | `onnxruntime-web` + React+Vite+TS | 是 **RL 网格可视化前端**，非 LLM 应用；模型位 `public/models/policy.onnx`（现仅 .gitkeep，未接真实模型） |
| `experiments/` | 有 `no_icm/no_episodic/no_rnd` ×3 基准，但**缺 `baseline`（三模块全关）配置** | 消融缺对照组 |

---

## 2. 量化目标（面试可辩护版，全部不强求精确绝对数）

| 基准 | 必达线 | 主指标 | 辅指标 | 证据链 |
|---|---|---|---|---|
| **MiniGrid DoorKey** | 解出任务 | **5-seed 平均成功率 ≥ 0.95** | 收敛步数中位数 + 95% CI | 修复评测协议后判定「算法成功」而非「评测假象」 |
| **Crafter** | 22 成就几何均值 **≥ 12–15%**，且 **≥ 同预算 PPO-only 基线** | 几何均值% | 相对基线提升（百分点）、新解锁成就数 | 好奇心模块在稀疏奖励下有统计显著增益 |
| **Atari Montezuma** | 10M 步跑完（单卡） | 10M 步实测均分 + **相对 PPO 120 分提升** | **首房通关率**（目标 ≥ 1/10 episode） | 系统端到端可跑通 + 方法有效，不追 3500+ |

> 核心叙事：「在三个稀疏奖励基准上，三重好奇心融合（ICM 过滤噪声 / RND 长期新颖 / Episodic 短期新颖）相对 PPO-only 带来统计显著（p<0.05）的探索增益，且训练工程在 6GB 显存 + 受限 RAM 的笔记本上稳定可复现。」

---

## 3. Phase 0 — 可靠性底座（必须先做，否则重跑也白跑）

### 3.1 原子化 checkpoint（`checkpoint.py`）
先写 `.tmp` 再 `os.replace`（POSIX 原子重命名），中途 kill 仅丢当前这一个临时文件、旧 ckpt 不损。

### 3.2 评测协议修正（修 MiniGrid 假 0%）
- `agent.evaluate()`：argmax → **随机采样**；返回 `(mean_reward, success_rate)`，MiniGrid 用 `info['is_success']` 判成功。
- `scripts/evaluate.py` 基准函数：随机采样 + **关 intrinsic reward**（评真策略，不评「带作弊器」）+ 算 success_rate / 22 成就几何均值 / 首房率。
- 评测用**独立 env + 独立 seed**，不复用训练 env。

### 3.3 资源守卫固化（`scripts/_queue_full_ablation.sh` 已具备，复用）
`--resume`（最新 ckpt 续跑）+ `wait_for_mem 3072`（RAM<3GB 等待）+ 单实例锁（防多份并发抢 RAM）+ `--checkpoint-interval 10240`（崩了最多丢 10k 步）。
**铁律**：禁 `kill -9` 父进程树；清理只杀明确 PID。

### 3.4 补 4 处前版硬伤（v2 新增，源自可行性校验）
1. **吞吐量基准门**：Atari 启动前先跑 **200k 步**测速，按实测 `steps/sec` 外推真实 ETA（之前估「数十小时」偏乐观，真实 2.5–5 天）。
2. **巡检自动化延期**：现有巡检 `valid_until=2026-07-18T12:00`（仅剩 ~33h），Atari 跑数天会监控失效 → 重建**无期限**巡检或明确「每 N 小时人工核 log」纪律。
3. **RAM 守卫封顶**：`wait_for_mem` 改为**封顶等待（如 20×30s）后 abort+报警**，杜绝「看似死实际在无限等」→ 误判崩溃 → kill 恶性循环。
4. **早停探针**：加熵坍缩 / 奖励停滞检测，长跑前 N 小时即报警，避免白烧数天。

### 3.5 RUN_MANIFEST + 磁盘守卫
启动即写 `RUN_MANIFEST.json`（config_hash/seed/start/total_steps/ckpt_dir）；`df` 可用 <10GB 暂停报警（历史磁盘曾 94%）。

---

## 4. 实验矩阵（Phase 1：对照组 + 好奇心组）

### 4.1 新建 PPO-only 基线（全关好奇心）
仓库缺 `baseline` 配置，新建 `experiments/{crafter,minigrid_doorkey,atari_montezuma}_baseline.yaml`（= 对应 full 但 `icm/rnd/episodic.enabled:false`）。

### 4.2 省 GPU 优化（关键）
PPO-only 对照**只在 Crafter 上跑一份**（最具判别力、最便宜）；MiniGrid/Atari 的纯 PPO 对照用**文献公认值**（Montezuma PPO≈120 分、DoorKey 纯 PPO 约 2.4M 步收敛），不各自自跑，省下数十小时单卡时间。

### 4.3 执行顺序（经守卫启动）
1. **先开 Atari 10M 重训**（耗时最长，后台长跑；先过 200k 步吞吐门）。
2. 用修复后协议**复评现有 MiniGrid/Crafter ckpt**，立刻区分「算法失败 vs 评测假象」。
3. 跑 Crafter baseline + 好奇心组，出 `benchmark_report.json`。

---

## 5. 统一评测 → `benchmark_report.json`（Phase 2）
`scripts/run_benchmarks.py`：三基准各 **5 seed（42–46）**，输出成功率/几何均值/均分 + std + CI。README 已引用此文件但缺失，补上即堵最大评审漏洞。字段示例：

```json
{
  "minigrid_doorkey": {"curiosity": {"success": 0.97, "ci95": [0.93,1.0], "conv_step_median": 920000},
                        "ppo_baseline": {"success": 0.55, "conv_step_median": 2400000}},
  "crafter":          {"curiosity": {"geom_mean": 0.15, "new_achievements": 11},
                        "ppo_baseline": {"geom_mean": 0.13, "new_achievements": 9}},
  "atari_montezuma":  {"curiosity": {"mean_score": 540, "first_room_rate": 0.3},
                        "ppo_baseline": {"mean_score": 120}}
}
```

---

## 6. 四组消融（Phase 3，你明确列的交付项）
`baseline / no_icm / no_episodic / no_rnd` × Crafter × 3 seed，结果填 `docs/ABLATION_REPORT.md`，证明三模块**各自独立增益**。

---

## 7. 交付物补齐（Phase 4）
- 演示视频（Crafter 最出片）：`scripts/record_video.py`
- 真实 ONNX 导出 → `web/public/models/policy.onnx` → 当前 `web/` 前端（onnxruntime-web）**真能推理**（现仅 .gitkeep，未接）
- Wandb：续跑加 `--use-wandb`，仪表盘截图进 `docs/WANDB_DASHBOARD.md`

---

## 8. 统计显著性（答辩级可复现）
多种子用 **Welch t-test + bootstrap CI** 报 p 值，结论统一为「好奇心版显著优于基线（p<0.05）」。

---

## 9. 里程碑与时间（硬件落地估算）

| 阶段 | 内容 | 耗时（单卡） | 风险 |
|---|---|---|---|
| Phase 0 | 原子写 + 评测协议 + 4 处硬伤修复 | 0.5 天 | 低（全可逆） |
| Phase 1 | Atari 10M（后台，关键路径）/ MiniGrid 1.5M / Crafter 1M + baseline | Atari 2.5–5 天；其余数小时 | 中（靠守卫缓解） |
| Phase 2–3 | 统一评测 + 四组消融 | 1–2 天 | 低 |
| Phase 4–5 | 视频/ONNX/web/wandb + 文档对齐 | 1 天 | 低 |
| **合计** | | **约 1–1.5 周墙钟（Atari 长跑占大头）** | |

---

## 10. 仍存在的真实风险（诚实告知）
1. **Crafter 衰减未根治**（eval 0.7→0.2）→ 先跑 200k 步小试验验证稳定再投全量。
2. **Atari 首房率可能仅「偶尔过」**（RND 原论文 ~1e8 帧才稳过，单卡 10M 步≈4e7 帧）→ 目标设 ≥1/10，否则可能不达。
3. **RAM 仍是天花板** → 严守 `wait_for_mem` 封顶守卫。

---

## 11. GitHub 作品集结构 + LLM Agent Demo 建议

### 11.1 本仓库（curiosity-ppo）=「深度」件
展示你**能从零实现并稳定跑通一个端到端 RL 系统**。README 讲清：方法（ICM+RND+Episodic+NGU）、实验设计（消融+显著性）、工程兜底（原子写/守卫/监控）、诚实的结果与局限。

### 11.2 LLM Agent Demo =「广度/应用」件（强烈建议，单独仓库）
`web/` 经核实是 **RL 网格可视化前端**（onnxruntime-web + 网格 env + argmax），**不是 LLM 应用**，不能直接改成 RAG+ToolCalling——两者后端架构根本不同（聊天/LLM 后端 vs 网格 ONNX 推理）。

正确做法：另起一个**独立小仓库** `llm-agent-demo`，复用 React+Vite 脚手架但后端换为 FastAPI + LLM：
- **RAG**：载入若干文档（你的笔记/代码）→ 嵌入 → top-k 检索 → 注入 prompt。
- **Tool Calling**：2–3 个工具（计算器 / 文件读取 / 搜索 mock），LLM 决定调用。
- **UI**：流式聊天界面。
- **成本**：用 OpenAI/Anthropic API 或本地 Ollama，demo 级成本≈0。

**为什么值得做（针对你的目标岗）**：
- 直接命中 AI Agent 应用岗的技能清单（LLM 集成、RAG、Tool Calling、Agent 循环、流式、成本/错误处理），而经典 RL 是间接相关。
- 两个仓库组合 = 「研究级 ML 系统能力 + 生产级 LLM 应用能力」的稀缺组合，且 LLM demo 是**可现场演示**的谈资，比 RL 训练曲线更有杀伤力。
- 风险低：架构清晰、易讲透，不像 RL 分数那样有「没达标」的尴尬。

**取舍**：LLM demo 是强加分项但额外约 2–4 天。建议**先按本方案做完 curiosity-ppo（优先）**，时间允许再补 `llm-agent-demo`。两者分开仓库，各自 README 叙事清晰。

---

## 12. 建议执行顺序
1. **Phase 0 代码修复**（原子写 + 评测协议 + 4 处硬伤）—— 可逆、安全、信息量大。
2. **Atari 先过 200k 步吞吐门** → 后台启动 10M 重训。
3. 修复协议**复评现有 MiniGrid/Crafter ckpt** → 区分算法失败 vs 评测假象。
4. Crafter baseline + 消融 → `benchmark_report.json` + `ABLATION_REPORT.md`。
5. 视频/ONNX（接真实模型到 `web/`）/Wandb/文档对齐。
6. （时间允许）另起 `llm-agent-demo` 仓库。

> 本计划为「规划文档」，**未启动任何训练/评测进程**（遵守「先别跑了」指令）。确认后从 Phase 0 开始执行。
