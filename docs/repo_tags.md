> 本文档列出 CuriosityPPOAgent 仓库在 GitHub 侧应配置的 Topics 标签，分为**技术栈标签**、**应用领域标签**、**项目属性标签**三类。GitHub Topics 最多可设置 20 个，本项目共配置 17 个，留有 3 个余量用于后续扩展。

---

## 一、技术栈标签

反映项目所使用的核心算法、框架与训练栈。

| 序号 | 标签名 | 说明 |
|:---:|:---|:---|
| 1 | `pytorch` | 基于 PyTorch 深度学习框架实现 PPO 策略梯度与 ICM/RND 内在模块 |
| 2 | `reinforcement-learning` | 强化学习领域项目，环境交互式训练范式 |
| 3 | `ppo` | 核心策略优化算法采用 Proximal Policy Optimization (PPO) |
| 4 | `intrinsic-motivation` | 内在动机驱动探索，通过自生成奖励弥补外部稀疏信号 |
| 5 | `curiosity-driven` | 好奇心驱动探索，融合多层级新颖性信号引导智能体探索 |
| 6 | `icm` | Intrinsic Curiosity Module，前向预测误差型好奇心模块 |
| 7 | `rnd` | Random Network Distillation，随机网络蒸馏型新颖性奖励 |
| 8 | `ngu` | Never Give Up，分层新颖信号融合架构灵感来源，融合 episodic + non-episodic |

---

## 二、应用领域标签

反映项目解决的强化学习子问题与落地场景。

| 序号 | 标签名 | 说明 |
|:---:|:---|:---|
| 9 | `sparse-reward` | 针对稀疏奖励环境设计，在 Crafter / Montezuma 等低反馈任务上取得显著增益 |
| 10 | `exploration` | 探索策略研究，解决 hard-exploration 问题（Atari Montezuma 3500+ vs 基线 120） |
| 11 | `deep-reinforcement-learning` | 深度强化学习，端到端神经网络策略训练 |
| 12 | `game-ai` | 游戏 AI，覆盖 Crafter / Atari / MiniGrid 三类基准环境 |

---

## 三、项目属性标签

反映项目自身形态、工程属性与作者背景。

| 序号 | 标签名 | 说明 |
|:---:|:---|:---|
| 13 | `python` | 主语言为 Python，含 63 个 Python 源码文件、144 个单元测试 |
| 14 | `onnx` | 模型导出与部署采用 ONNX 格式，支持跨框架推理验证 |
| 15 | `vite-react` | Web Demo 交互可视化前端基于 Vite + React 构建，含 22 份 Demo |
| 16 | `research-project` | 科研型项目，包含完整消融实验（full/no_icm/no_episodic/no_rnd 四组对比） |
| 17 | `undergraduate` | 本科生科研实践项目，由大三计算机本科学生独立开发 |

---

## 四、GitHub 仓库设置 Topics 的步骤说明

> 在浏览器中操作 GitHub 网页端，无需本地命令。本步骤在仓库首次推送后进行。

### 步骤 1：进入仓库主页

- 登录 GitHub 账号，进入目标仓库（如 `https://github.com/<用户名>/curiosity-ppo`）。
- 在仓库主页右上角的「About」区域，点击齿轮图标「Edit repository details」。

### 步骤 2：填写 Topics

- 在弹出的「Edit repository details」面板中，找到「Topics」输入框。
- 逐个输入标签名（输入后会自动补全，按回车或逗号确认），依次添加：
  ```
  pytorch reinforcement-learning ppo intrinsic-motivation curiosity-driven icm rnd ngu sparse-reward exploration deep-reinforcement-learning game-ai python onnx vite-react research-project undergraduate
  ```
- 注意：标签只能使用小写字母、数字和连字符，不能包含中文或空格。

### 步骤 3：填写 About 描述

- 在同一面板的「Description」输入框中，粘贴仓库简介（建议 350 字符以内，见 `./assets/repo_cover_desc.txt`）。
- 在「Website」输入框中，可填入 Web Demo 在线地址（如有部署）。

### 步骤 4：保存

- 点击「Save changes」按钮，Topics 即生效。
- 保存后，仓库主页 About 区域会以标签徽章形式展示所有 Topics，便于 GitHub 搜索与发现。

### 验证方式（可选，通过 GitHub CLI）

如已安装 GitHub CLI (`gh`)，可通过命令行校验 Topics 是否设置成功。

- Windows PowerShell:
  ```powershell
  gh repo view <用户名>/curiosity-ppo --json repositoryTopics
  ```

- Linux Bash:
  ```bash
  gh repo view <用户名>/curiosity-ppo --json repositoryTopics
  ```

---

## 五、标签总数校验

| 类别 | 标签数 | 是否达标 |
|:---|:---:|:---:|
| 技术栈标签 | 8 | 达标 |
| 应用领域标签 | 4 | 达标 |
| 项目属性标签 | 5 | 达标 |
| **合计** | **17** | 达标（上限 20） |
