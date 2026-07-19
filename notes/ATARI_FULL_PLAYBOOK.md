# Atari 训练完成后 → 评测 → 消融 → 收尾：全步骤总表

> 本文档是训练跑完 10M 步之后**所有操作的唯一参照**。按顺序做，遇到不懂的回来看对应步骤。
> ⏱️ 计费铁律：DSW 实例按运行时长烧资源包（约 7 CU·H/小时）。**任何不跑训练/评测的空档，立刻停实例。**
> 💰 当前额度（07-18 15:07 截图确认）：剩余 **92,863,000 / 100,000,000 ≈ 92.9 CU·H**。
> ⚠️ 实测速度 ~75万步/小时（非之前乐观估的123万），重算后：主线10M需~73.5 CU·H → 跑完剩~19.4 CU·H → **仅够1组 no_icm 消融(1-2M步)**，3组全做/多种子均不可行。

---

## 时间线总览

| 阶段 | 操作 | 实例状态 | 耗时 | 额度消耗 |
|------|------|---------|------|---------|
| 0 | 主线训练（已在跑） | 运行中 | ~7-8h（预计今晚 20-21 点完） | 已在扣 |
| 1 | 确认完成 + 评测主线 | 运行中 | ~10 分钟 | 可忽略 |
| 2 | 核心消融 3 组 × 3M | 运行中 | ~7h | ~49 CU·H |
| 3 | 备份 + 停实例 | 停 | - | 0 |
| 4 | （可选）下次启实例下载结果 | 启→停 | 10 分钟 | 可忽略 |

> ⚠️ 阶段 1 和 阶段 2 之间有"空档"风险：评测完如果不立刻决定，先停实例，明天再开跑消融。
> 不要评测完让实例空转等你想"什么时候跑消融"。

---

## 阶段 0：主线训练中（现在）

- 终端 `terminal/tty` 标签里应该持续刷 `Step: XXXXX | ...` 日志
- 不用盯着，浏览器断连不影响
- **预计今晚 20:00–21:00 跑完 10M**

---

## 阶段 1：训练完成 → 确认 + 评测主线

### 1.1 确认完成
```bash
tail -20 /mnt/workspace/atari_seed42/train.log
```
最后一行 `global_step` ≈ 10000000，且无 `Traceback` / `NaN` / `OOM` = 成功。

### 1.2 评测主线（GPU 上，几分钟）
```bash
LATEST=$(python -c "import glob,os; fs=sorted(glob.glob('/mnt/workspace/atari_seed42/step_*.pt'), key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0])); print(fs[-1])")
echo "Evaluating: $LATEST"
python scripts/evaluate.py \
  --config experiments/atari_montezuma_full.yaml \
  --checkpoint "$LATEST" \
  --eval-episodes 10 \
  2>&1 | tee /mnt/workspace/atari_seed42/eval.log

cat /mnt/workspace/atari_seed42/eval.log
```
产物：`benchmark_report.json / .md`（蒙特祖玛真实得分）。
⚠️ 训练时 `ext_reward_mean=0` 是稀疏奖励假象，**评测才出真分**。

### 1.3 决策点
- 如果你**现在就想接着跑消融** → 不certain停实例，直接进入阶段 2
- 如果**想先睡/明天再跑** → 先执行阶段 3 备份 + 停实例，明天从阶段 2 重新开始（重新启动实例后需要重装环境，见文末附录）

---

## 阶段 2：核心消融实验

> 主线条已用 10M 跑完，消融只跑**对比组**（跳过 full）。
> 仓库配置：`atari_montezuma_{no_icm, no_episodic, no_rnd}.yaml` 均存在。

### 2.1 推荐命令（单 seed 42，每组 3M 步）
```bash
# 确保在项目根目录
cd /mnt/workspace/curiosity-ppo
python scripts/run_ablation.py \
  --env atari \
  --steps 3000000 \
  --seeds 42 \
  --ablations no_icm,no_episodic,no_rnd
```
脚本自动：对每组跑 `train.py` → 训练完自动评测（10 episodes）→ 写 `results/ablation/atari_montezuma_<ablation>/seed_42/benchmark_report.json(.md)`

### 2.2 省额度三选一（按剩余额度选）

| 方案 | 命令改动 | 耗时 | 额度 | 何时用 |
|------|---------|------|------|--------|
| **A. 标准（推荐）** | 原命令 | ~7h | ~49 CU·H | 额度够、要完整对比 |
| **B. 减步数** | `--steps 2000000` | ~4.7h | ~33 CU·H | 想留余量 |
| **C. 只证核心** | `--ablations no_icm` | ~2.4h | ~17 CU·H | 只想证「ICM 好奇心有增益」 |

> 当前剩余 ~66 CU·H，方案 A 跑完剩 ~17 CU·H。**多种子（seed 42/43/44）本次做不了**（需 154 CU·H），报告如实写「单种子」。

### 2.3 消融跑完后检查
```bash
ls results/ablation/
# 应看到 atari_montezuma_no_icm / no_episodic / no_rnd 三个目录，各含 seed_42/benchmark_report.md
```

---

## 阶段 3：备份 + 停实例（保钱包 + 防 07-25 释放）

```bash
cd /mnt/workspace
tar czf atari_seed42_backup.tar.gz atari_seed42/
tar czf atari_ablation_backup.tar.gz results/ablation/
ls -lh atari_seed42_backup.tar.gz atari_ablation_backup.tar.gz
```
然后 **DSW 控制台 → 点「停止」实例**。

> 实例停了磁盘还在（释放截止 07-25）。下次启动实例后用 OSS / 文件浏览器把两个 tar.gz 下到本地。

---

## 阶段 4（可选）：下载结果到本地

下次启动实例后：
```bash
# 方法 1：用 ossutil 传 OSS 再从浏览器下
ossutil cp /mnt/workspace/atari_seed42_backup.tar.gz oss://你的bucket/
ossutil cp /mnt/workspace/atari_ablation_backup.tar.gz oss://你的bucket/

# 方法 2：直接在 DSW 文件浏览器里点下载
```
下载完立刻停实例。

---

## 最终结果对比表（写作品集用）

| 配置 | 模块状态 | Montezuma 得分（10 episodes 均值） |
|------|---------|----------------------------------|
| **full** | RND + ICM + Episodic | （主线评测） |
| no_icm | RND + Episodic（无 ICM 好奇心） | （消融） |
| no_episodic | RND + ICM（无情景记忆） | （消融） |
| no_rnd | ICM + Episodic（无 RND 长期调制） | （消融） |

**核心结论模板**（数字等跑出来再填，不要提前编）：
> "去掉 ICM 后得分从 X 降到 Y，证明内在好奇心模块贡献了 Z% 相对提升；
> 去掉 Episodic/RND 分别带来 ...，说明三者协同工作。"

---

## 附录：重新启动实例后的环境重建

如果阶段 1.3 你选择"先停"，明天重新启动实例后，**新容器是干净的**，需重装一遍：

```bash
cd /mnt/workspace/curiosity-ppo
git pull
pip install torch torchvision --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch/whl/cu121 --trusted-host mirrors.tuna.tsinghua.edu.cn
pip install -e .
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
python -c "import torch; print('cuda', torch.cuda.is_available())"   # 必须 True
```
然后直接进入阶段 2 跑消融（checkpoint 在持久盘 `/mnt/workspace/atari_seed42/` 不丢）。

---

## 命令速查卡（复制即用）

```bash
# === 阶段 1：评测主线 ===
LATEST=$(python -c "import glob,os; fs=sorted(glob.glob('/mnt/workspace/atari_seed42/step_*.pt'), key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0])); print(fs[-1])")
python scripts/evaluate.py --config experiments/atari_montezuma_full.yaml --checkpoint "$LATEST" --eval-episodes 10 2>&1 | tee /mnt/workspace/atari_seed42/eval.log

# === 阶段 2：核心消融（推荐方案 A）===
python scripts/run_ablation.py --env atari --steps 3000000 --seeds 42 --ablations no_icm,no_episodic,no_rnd

# === 阶段 3：备份 ===
cd /mnt/workspace && tar czf atari_seed42_backup.tar.gz atari_seed42/ && tar czf atari_ablation_backup.tar.gz results/ablation/ && ls -lh *.tar.gz
# 然后去控制台停实例
```
