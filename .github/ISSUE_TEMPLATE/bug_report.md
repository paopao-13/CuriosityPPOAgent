---
name: Bug 反馈
about: 报告 CuriosityPPOAgent 运行过程中遇到的问题
title: "[Bug] 简要描述"
labels: bug
assignees: ''
---

## Bug 描述

<!-- 清晰地描述发生了什么问题，包括出现的异常现象与触发场景。 -->

## 复现步骤

1. 进入 `scripts/` 目录
2. 运行命令 `python train.py --config experiments/xxx.yaml`
3. 训练至约 N 步时出现异常
4. ...

## 预期行为

<!-- 描述你期望发生的结果。 -->

## 实际行为

<!-- 描述实际发生的结果，与预期行为的差异。 -->

## 环境信息

| 项目 | 内容 |
| --- | --- |
| 操作系统 | <!-- 例如 Windows 11 / Ubuntu 22.04 --> |
| GPU | <!-- 例如 RTX 3060 Laptop 6GB / 无 --> |
| Python 版本 | <!-- 例如 3.10.11 --> |
| PyTorch 版本 | <!-- 例如 2.1.0 --> |
| CUDA 版本 | <!-- 例如 11.8 / CPU only --> |
| 项目版本 | <!-- 例如 v1.0.0 / commit hash --> |

## 训练配置

| 项目 | 内容 |
| --- | --- |
| 环境名 | <!-- 例如 Crafter / Atari-MsPacman / MiniGrid --> |
| 配置文件 | <!-- 例如 experiments/crafter_icm_rnd.yaml --> |
| 训练步数 | <!-- 例如 1,000,000 --> |
| 是否启用 AMP | <!-- 是 / 否 --> |
| 是否启用 ICM/RND | <!-- ICM / RND / ICM+RND --> |

## 错误日志

```
<!-- 在此处粘贴完整的错误日志 / 堆栈跟踪。请删除敏感信息。 -->
```

## 截图（可选）

<!-- 如有训练曲线、wandb 面板或终端截图，请在此附加。 -->

## 是否在 CPU 模式下也能复现

- [ ] 是，CPU 模式可复现
- [ ] 否，仅 GPU 模式出现
- [ ] 未测试

<!-- 说明：本项目在 RTX3060 Laptop 6GB 上显存峰值约 2.2GB，如怀疑与显存/AMP 相关，请附 CPU 模式测试结果。 -->
