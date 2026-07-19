# Atari 云端续训 — 完整行动总表

> 目标：用已购的 **PAI-DSW 资源抵扣包（100 CU·H，¥0，代金券覆盖）**，从云端 checkpoint 续训 Montezuma 到 10M 步，并预留额度做消融实验。
> 当前状态：minigrid + crafter 已跑完可评测；Atari 云端 DSW 跑到 **676K/10M（6.8%）** 被欠费停，三 bug（cu121 torch / is_image / amp.py）已修好，数据在 7 天缓冲期内（释放截止 07-25）。

---

## 阶段一：现在要做 —— 启动续训

### 1.1 本机（Windows）推送修正代码【一次性，关键】
> DSW 实例 clone 的是 GitHub 旧版 `train.py`，可能不支持 `--resume`。先推送修正版。
```bash
cd <repo-root>
git add src scripts experiments docs
git commit -m "fix(atari): is_image CHW 判定 + amp 安全版 + resume 支持"
git push origin main
```

### 1.2 启动 DSW 实例（先核实资源包，再新建）
> ⚠️ 修正：PAI-DSW 无"资源包配"模式。正确做法：资源类型选「公共资源」（按量付费），DSW 资源包(100 CU·H)后台自动抵扣。
> 创建前先去 **费用中心 → 我的资源包** 确认那张 100 CU·H 包：① 适用产品=交互式建模(DSW) ② 地域与实例一致 ③ 状态可用 ④ 支持规格含 `ecs.gn7i-c8g1.2xlarge`（注意是 gn7i，不是旧实例用的 gn7r）。
- 阿里云 PAI-DSW 控制台 → **新建实例**（旧实例 `hi-atari-test-noss` 是 gn7r 规格、且按量未抵扣，直接弃用新建）
- 资源类型=**公共资源**，资源规格=**`ecs.gn7i-c8g1.2xlarge`（A10, 30GB）**，地域与资源包一致
- 实例变「运行中」后，费用自动从 100 CU·H 抵扣，不再扣现金余额

### 1.3 DSW 终端：拉取修正代码
```bash
cd /mnt/workspace/curiosity-ppo
git checkout -- . 2>/dev/null
git pull
python -c "import torch; print('cuda', torch.cuda.is_available())"   # 应打印 True
```

### 1.4 DSW 终端：续训（自动找最新 checkpoint）
```bash
CKPT=$(ls -t /mnt/workspace/atari_seed42/step_*.pt 2>/dev/null | head -1)
echo "Resuming from: $CKPT"
cd /mnt/workspace/curiosity-ppo && python scripts/train.py \
  --config experiments/atari_montezuma_full.yaml \
  --total-steps 10000000 \
  --resume "$CKPT" \
  --checkpoint-dir /mnt/workspace/atari_seed42 \
  --checkpoint-interval 10240 \
  2>&1 | tee -a /mnt/workspace/atari_seed42/train.log
```

---

## 阶段二：训练中 —— 放着别管

- **不用盯着**，训练进程在服务端跑，浏览器断连（Disconnected）不影响。
- 想看进度，开新 terminal session 跑：
  ```bash
  tail -5 /mnt/workspace/atari_seed42/train.log
  ```
- **预计耗时**：实测速率 ~1.23M 步/小时，剩 9.324M 步 → **约 8 小时**（保守 10h）。
- **预计跑完时间**：现在（07-18 约 02:00）起，约 **上午 10:00–12:00**。

---

## 阶段三：跑完后立刻做（按优先级）

> 第 3 步最省额度——每多开 1 分钟实例都在烧 100 CU·H。

1. **确认完成**
   ```bash
   tail -5 /mnt/workspace/atari_seed42/train.log
   ```
   最后一行 `global_step` ≈ 10000000，且无 `Traceback`/`NaN`/`OOM`。

2. **立刻评测（实例还开着、用 GPU，几分钟）**
   ```bash
   CKPT=$(ls -t /mnt/workspace/atari_seed42/step_*.pt 2>/dev/null | head -1)
   python scripts/evaluate.py --checkpoint "$CKPT" --env atari --n-episodes 10 \
     --config experiments/atari_montezuma_full.yaml \
     --output-dir /mnt/workspace/atari_seed42/eval
   ```
   输出 `benchmark_report.json` / `.md` = Montezuma 实际得分（训练时 `ext_reward=0` 是稀疏奖励假象，评测才出真分）。

3. **⚠️ 评测一跑完，立刻在 DSW 控制台点「停止」实例** —— 别让实例空转耗额度。

4. **备份数据（防 07-25 实例释放丢）**
   ```bash
   tar -czf /mnt/workspace/atari_seed42_backup.tar.gz -C /mnt/workspace atari_seed42
   ```
   然后在 DSW 文件浏览器下载该 `tar.gz` 到本地留底。

5. **（可选）录通关视频做作品集素材**
   ```bash
   python scripts/record_video.py --checkpoint "$CKPT" --env atari --output /mnt/workspace/atari_seed42/eval/video.mp4
   ```
   （参数以 `python scripts/record_video.py --help` 为准）

---

## 阶段四：之后做 —— 消融实验（预留额度）

仓库已含 Atari 全部消融配置：`atari_montezuma_{full, no_icm, no_episodic, no_rnd, full_ent002}.yaml`
- `no_icm` = PPO 基线（证好奇心增益）
- `no_episodic` / `no_rnd` = 模块贡献拆解
- `full_ent002` = 熵系数敏感性

**主线条已用 10M 跑完，消融只跑对比组（跳过 full）：**
```bash
python scripts/run_ablation.py --env atari --steps 3000000 --seeds 42 \
  --ablations no_icm,no_episodic,no_rnd
```
3 组 × 3M ≈ 9M 步 ≈ 7.3h，自动评测+聚合到 `results/ablation/`。
> 想更省：步数降到 2M，或只跑 `no_icm` 一组证核心结论。

---

## 额度总账（100 CU·H）

| 项目 | 耗时 | 占额度 |
|------|------|--------|
| 主线续训 10M | ~9h | 9%–18% |
| 消融 3 组 × 3M | ~7.3h | 7%–15% |
| **合计** | **~16h** | **~16%–32%** |

剩余 ~68–84h 可自由支配，额度充裕，不用为预算焦虑。

## 关键提醒
- DSW 按**运行时长**计费，跑完/不用时立刻「停止」实例。
- checkpoint 每 10240 步存一个，即使意外停机最多丢 ~10K 步（约 10 分钟），可再次 `--resume` 续上。
- 实例释放截止 07-25，重要数据务必在此之前备份到本地。
