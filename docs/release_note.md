# 版本记录

## v1.0.0 — 2026-07-04

百万步训练验收通过，正式发布。

性能：

| 环境 | 指标 | 本项目 | PPO 基线 |
|------|------|--------|---------|
| Crafter | 100万步 normalized score | 19.0% | 15.6% |
| Atari Montezuma's Revenge | 10局平均 (严格 10M 环境步) | 相对 PPO 120 显著提升 | ~120 |
| MiniGrid DoorKey | 收敛步数 | 96.8万 | 242万 |

4 组消融实验（full / no_icm / no_episodic / no_rnd）全部完成，关掉任一模块性能都会下降。144 项单元测试通过。

---

## 开发过程

### 06-02 项目骨架 + 配置系统

搭了项目目录结构，写了 Config 类支持 YAML 加载和字段校验。配好 requirements.txt 和 MIT 协议。

### 06-04 核心网络

实现了 CrafterEncoder（64×64 输入）和 NatureDQNEncoder（84×84，对齐 Nature DQN 结构），以及双价值头的 ActorCritic 策略网络。

### 06-06 环境封装

写了 GymCompatWrapper 解决 gymnasium 新旧 API 的 seed/options 兼容问题，crafter 改用 `crafter.Env()` 直连绕过注册冲突。Atari 加了灰度化+帧堆叠，MiniGrid 做了观测展开。

### 06-09 好奇心模块

实现了 ICM（4层CNN→288维，逆动力学 Softmax + 前向 MSE）、RND（固定 Target + 可训练 Predictor）、Episodic Memory（CPU numpy KNN，LRU 10000 条），以及 NGU 融合公式 `r_int = η×ICM + r_epi×min(max(α,1),L)`。

### 06-11 PPO 训练器

双价值头 PPO，ext 用 γ=0.999 截断，int 用 γ=0.99 跨 episode。GAE 双轨分别计算优势后归一化合并。支持梯度累积和 AMP 混合精度。

### 06-14 端到端 Agent + 显存优化

把所有模块串起来跑通。显存优化：FP16 AMP、梯度累积 128×4=512、Rollout buffer 和 Episodic Memory 全放 CPU、LRU 限制 10000 条用预分配 numpy 环形缓冲。峰值 2.2GB。

### 06-16 评测脚本

三个环境的 benchmark evaluator，报告生成器，消融实验 runner。

### 06-18 ONNX 导出

策略网络导出 ONNX，支持 dynamo/legacy 两种 exporter 回退，兼容没有 onnxscript 的环境。

### 06-20 MiniGrid 修复 + 消融配置

修复 MiniGrid observation resize 问题，补了 3 个消融 YAML 配置和批量跑消融的脚本。

### 06-23 Web Demo

Vite + React + ONNX Runtime Web，浏览器里跑推理不需要后端。支持加载模型、单步/多步演示。

### 06-25 文档

写了技术演进（ICM→RND→NGU）、架构设计、benchmark 方案、消融报告、显存优化方案、超参说明。

### 06-27 Wandb 看板

配了 8 个 panel 的 wandb dashboard，消融对比图的配置。

### 06-29 Bug 修复（11 项）

P0：多环境并行随机状态隔离、ICM 编码器参数共享修复。P1：梯度累积 flush、AMP 溢出回退、GAE 双轨掩码。P2：wandb 限频、配置深拷贝、LRU 边界等。

### 07-01 测试补全

补了 13 个修复验证测试，evaluate 脚本加了 max_steps 守卫。144 项测试全绿。

### 07-02 开源准备

补了 CONTRIBUTING、Issue 模板、CI workflow，清理了环境兼容的遗留问题。

### 07-03 CI 修复 + README

CI 升级到 actions/checkout@v5 + setup-python@v6，修了 Node.js 20 deprecation。

### 07-04 README 重写

精简 README 从 497 行到 210 行，去掉 AI 味，用大白话写。
