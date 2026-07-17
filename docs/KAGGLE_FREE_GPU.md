# 免费 GPU 跑 Atari 实验 (Kaggle / Colab)

> 阿里云 DSW 欠费停机后, 学生零预算的替代方案.
> 目标: 在免费 T4/P100 上跑通 Atari Pong 1M 步, 验证 ICM 好奇心模块收益.

## 方案对比

| 渠道 | GPU | 免费额度 | 适合 |
|------|-----|---------|------|
| **Kaggle Notebooks** | T4 / P100 | 30h/周, 单次 ≤9h | ✅ 首选 |
| Google Colab | T4 | ~12h/周, 90min 不操作断连 | 备选 (长训练需自动 resume) |
| 本地 CPU | 无 | 无限但慢 | Pong 1M 步几小时也能出分 |

## 前置: 先把本地代码推到 GitHub (一次性)

本地 `agent.py` / `amp.py` / `train.py` 等都比 GitHub 新, 必须推送, 否则 Kaggle clone 会重蹈云端覆辙.

```bash
cd D:/简历/curiosity-ppo
git add -A
git commit -m "fix(atari): is_image CHW 判定 + amp 安全版 + Pong 轻量配置"
git push origin main
```

> 若 `git push` 要求登录, 用 GitHub 账号密码或 token (Settings → Developer settings → PAT).

## Kaggle 操作步骤

1. 打开 https://www.kaggle.com/code → **New Notebook**
2. 右侧 **Settings → Accelerator** 选 **GPU (T4 x2 或 P100)**
3. 依次跑以下 cell:

### Cell 1 — 克隆 + 安装 (推送后直接 clone 即可)

```bash
!git clone https://github.com/paopao-13/CuriosityPPOAgent.git
%cd CuriosityPPOAgent
!pip install -e .
!pip install "gymnasium[atari,accept-rom-license]" ale-py
```

> 若 clone 的是**旧版 GitHub** (未推送), 改用防御式补丁:
> ```bash
> !git clone https://github.com/paopao-13/CuriosityPPOAgent.git
> %cd CuriosityPPOAgent
> !pip install -e .
> !python scripts/patch_atari.py
> ```

### Cell 2 — 训练 Pong (免费 GPU 约 1-2 小时)

```bash
!python scripts/train.py \
  --config experiments/atari_pong_quick.yaml \
  --total-steps 1000000 \
  --checkpoint-dir results/pong_free \
  --checkpoint-interval 50000
```

### Cell 3 — 看进度 (训练跑一会儿后另开 cell)

```bash
!tail -5 results/pong_free/train.log
```

## 关键注意事项

- **Kaggle 单次最长 9 小时**, Pong 1M 步远小于此, 安全.
- **断连保护**: checkpoint 每 50000 步存一次, 即使被回收最多丢 ~50K 步.
- **别关浏览器 tab**: 训练在前台跑, 关了就停. 可最小化.
- **`use_amp: false`** 已在 `atari_pong_quick.yaml` 设好, 免费 GPU 不需要混合精度.
- 训练完用 `scripts/evaluate.py` 跑实际得分 (详见各基准 eval 脚本).

## 如果只想在本地免费跑

```bash
cd D:/简历/curiosity-ppo
python scripts/train.py --config experiments/atari_pong_quick.yaml --total-steps 1000000 --checkpoint-dir results/pong_local
```

CPU 上 Pong 1M 步约几小时, 零成本, 适合先验证 pipeline.
