# 贡献指南

首先，感谢你对 CuriosityPPOAgent 项目的关注与支持！本项目是一个面向强化学习入门学习者的 ICM + RND 分层好奇心驱动 PPO 教学级实现，欢迎任何形式的贡献——无论是修复 Bug、新增功能、完善文档，还是优化训练性能，都同样宝贵。

以下指南将帮助你顺利参与本项目开发。

---

## 目录

- [开发环境准备](#开发环境准备)
- [代码规范](#代码规范)
- [提交规范](#提交规范)
- [分支策略](#分支策略)
- [Pull Request 流程](#pull-request-流程)
- [测试要求](#测试要求)
- [项目结构说明](#项目结构说明)
- [联系方式](#联系方式)

---

## 开发环境准备

### 1. 克隆仓库

```powershell
# Windows PowerShell
git clone https://github.com/<your-username>/curiosity-ppo.git
cd curiosity-ppo
```

```bash
# Linux Bash
git clone https://github.com/<your-username>/curiosity-ppo.git
cd curiosity-ppo
```

### 2. 创建虚拟环境并安装依赖

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .[dev]
```

```bash
# Linux Bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .[dev]
```

### 3. 验证环境

运行全部单元测试，确认 144 项测试全部通过：

```powershell
# Windows PowerShell
python -m pytest tests\ -v --tb=short
```

```bash
# Linux Bash
python -m pytest tests/ -v --tb=short
```

### 4. 安装 pre-commit 钩子（可选但推荐）

```powershell
# Windows PowerShell
pip install pre-commit
pre-commit install
```

```bash
# Linux Bash
pip install pre-commit
pre-commit install
```

---

## 代码规范

### 格式化

本项目使用 [black](https://github.com/psf/black) 进行代码格式化，配置如下：

- 行宽限制：100 字符
- 目标 Python 版本：3.10+

```powershell
# Windows PowerShell — 格式化全部源码
black src\ tests\ scripts\ benchmarks\ --line-length 100
```

```bash
# Linux Bash — 格式化全部源码
black src/ tests/ scripts/ benchmarks/ --line-length 100
```

### 类型标注

所有新增 Python 代码必须包含完整的类型标注（Type Hints），使用 `mypy` 进行静态类型检查：

```powershell
# Windows PowerShell
mypy src\curiosity_ppo\ --ignore-missing-imports
```

```bash
# Linux Bash
mypy src/curiosity_ppo/ --ignore-missing-imports
```

**类型标注示例**：

```python
from typing import Optional
import torch
import numpy as np

def compute_intrinsic_reward(
    obs: torch.Tensor,
    icm_forward_loss: torch.Tensor,
    episodic_reward: torch.Tensor,
    alpha_t: float,
    eta: float = 1.0,
    l_cap: float = 5.0,
) -> torch.Tensor:
    """计算融合内在奖励。

    Args:
        obs: 观测张量，形状 (B, C, H, W)。
        icm_forward_loss: ICM 前向预测损失。
        episodic_reward: Episodic Memory 新颖度奖励。
        alpha_t: 可学习融合权重。
        eta: ICM 损失全局系数。
        l_cap: 融合上限截断值。

    Returns:
        融合内在奖励 r_int。
    """
    weight = min(max(alpha_t, 1.0), l_cap)
    r_int = eta * icm_forward_loss + episodic_reward * weight
    return r_int
```

### 代码风格要点

- 函数和类必须有 **docstring**（Google 风格）
- 导入顺序：标准库 → 第三方库 → 本项目模块，各组之间空一行
- 常量使用 `UPPER_SNAKE_CASE`，类名使用 `PascalCase`，函数和变量使用 `snake_case`
- 避免使用魔法数字，关键超参数应从 YAML 配置文件读取

---

## 提交规范

本项目遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范，提交消息格式如下：

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Type 类型说明

| Type | 说明 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat(curiosity): 新增 NGU 自适应融合权重 α_t` |
| `fix` | Bug 修复 | `fix(ppo): 修复 GAE 优势估计的梯度截断问题` |
| `docs` | 文档更新 | `docs: 补充 VRAM 优化方案的显存分解表` |
| `refactor` | 代码重构（不改变功能） | `refactor(icm): 将编码器抽取为独立模块` |
| `test` | 测试相关 | `test(episodic): 新增 LRU 淘汰策略的边界测试` |
| `chore` | 构建/工具/依赖等杂项 | `chore: 升级 PyTorch 至 2.1` |
| `perf` | 性能优化 | `perf(amp): 优化 FP16 混合精度的 GradScaler 逻辑` |
| `ci` | CI 配置变更 | `ci: 添加 GitHub Actions 单元测试工作流` |

### Scope 可选范围

`curiosity` / `networks` / `ppo` / `envs` / `utils` / `web` / `scripts` / `benchmarks` / `config` / `docs` / `tests`

### 提交示例

```bash
# Linux Bash
git add src/curiosity_ppo/curiosity/ngu_fusion.py
git commit -m "feat(curiosity): 实现 ICM+Episodic 分层好奇心融合公式 r_int = η×ICM + r_episodic×min(max(α_t,1),L)"
```

```powershell
# Windows PowerShell
git add src\curiosity_ppo\curiosity\ngu_fusion.py
git commit -m "feat(curiosity): 实现 ICM+Episodic 分层好奇心融合公式"
```

---

## 分支策略

本项目采用 `main` + `feature` 分支模型：

| 分支 | 用途 | 命名规则 |
|------|------|---------|
| `main` | 稳定发布分支，始终可运行 | — |
| `feature/*` | 新功能开发 | `feature/add-rnd-module` |
| `fix/*` | Bug 修复 | `fix/gae-truncation` |
| `docs/*` | 文档更新 | `docs/update-readme` |
| `refactor/*` | 代码重构 | `refactor/extract-encoder` |

### 分支操作流程

```powershell
# Windows PowerShell — 从 main 创建新分支
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

```bash
# Linux Bash — 从 main 创建新分支
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

**注意**：不要直接向 `main` 分支提交代码，所有变更必须通过 Pull Request 合并。

---

## Pull Request 流程

### 1. 确保代码通过全部检查

提交 PR 前，请确保以下检查全部通过：

```powershell
# Windows PowerShell
black src\ tests\ scripts\ benchmarks\ --check --line-length 100
mypy src\curiosity_ppo\ --ignore-missing-imports
python -m pytest tests\ -v --tb=short
```

```bash
# Linux Bash
black src/ tests/ scripts/ benchmarks/ --check --line-length 100
mypy src/curiosity_ppo/ --ignore-missing-imports
python -m pytest tests/ -v --tb=short
```

### 2. 推送分支并创建 PR

```powershell
# Windows PowerShell
git push origin feature/your-feature-name
```

```bash
# Linux Bash
git push origin feature/your-feature-name
```

然后在 GitHub 上创建 Pull Request，PR 标题应遵循 Conventional Commits 格式。

### 3. PR 描述模板

```markdown
## 变更说明
<!-- 简要描述本次变更的内容和目的 -->

## 变更类型
- [ ] 新功能 (feat)
- [ ] Bug 修复 (fix)
- [ ] 文档更新 (docs)
- [ ] 代码重构 (refactor)
- [ ] 测试新增 (test)
- [ ] 其他 (chore)

## 测试情况
- [ ] 新增/修改的代码已配套单元测试
- [ ] 全部 144 项单元测试通过
- [ ] `black --check` 格式检查通过
- [ ] `mypy` 类型检查通过

## 关联 Issue
<!-- 如有关联 Issue，请填写 closes #123 -->
```

### 4. Code Review

- 至少需要 1 位维护者审核通过后方可合并
- 审核重点：功能正确性、代码规范、测试覆盖率、显存影响
- 如果 PR 涉及训练逻辑变更，请在 PR 描述中附上训练曲线或评测结果

---

## 测试要求

### 基本要求

- **所有新增代码必须配套单元测试**，测试文件放在 `tests/` 目录，命名规则为 `test_<module_name>.py`
- 测试框架使用 [pytest](https://docs.pytest.org/)
- 测试覆盖率目标：核心模块（curiosity / ppo / networks）不低于 90%

### 运行测试

```powershell
# Windows PowerShell — 运行全部测试
python -m pytest tests\ -v --tb=short

# 运行特定模块测试
python -m pytest tests\test_icm.py tests\test_rnd.py -v

# 生成覆盖率报告
python -m pytest tests\ --cov=src/curiosity_ppo --cov-report=html
```

```bash
# Linux Bash — 运行全部测试
python -m pytest tests/ -v --tb=short

# 运行特定模块测试
python -m pytest tests/test_icm.py tests/test_rnd.py -v

# 生成覆盖率报告
python -m pytest tests/ --cov=src/curiosity_ppo --cov-report=html
```

### 测试编写示例

```python
import pytest
import torch
from curiosity_ppo.curiosity.ngu_fusion import NGUFusion


class TestNGUFusion:
    """NGU 分层融合模块测试。"""

    def test_fusion_formula_basic(self):
        """测试基本融合公式 r_int = η×ICM + r_episodic×min(max(α_t,1),L)。"""
        fusion = NGUFusion(eta=1.0, l_cap=5.0)
        icm_loss = torch.tensor([0.5])
        r_episodic = torch.tensor([2.0])
        alpha_t = 3.0

        r_int = fusion.compute(icm_loss, r_episodic, alpha_t)

        expected = 1.0 * 0.5 + 2.0 * min(max(3.0, 1.0), 5.0)
        assert torch.allclose(r_int, torch.tensor([expected]))

    def test_alpha_clamped_to_lower_bound(self):
        """测试 α_t 低于下界 1 时被截断为 1。"""
        fusion = NGUFusion(eta=1.0, l_cap=5.0)
        r_int = fusion.compute(
            torch.tensor([0.0]), torch.tensor([1.0]), alpha_t=0.5
        )
        assert torch.allclose(r_int, torch.tensor([1.0]))

    def test_alpha_clamped_to_upper_bound(self):
        """测试 α_t 高于上界 L 时被截断为 L。"""
        fusion = NGUFusion(eta=1.0, l_cap=5.0)
        r_int = fusion.compute(
            torch.tensor([0.0]), torch.tensor([1.0]), alpha_t=10.0
        )
        assert torch.allclose(r_int, torch.tensor([5.0]))
```

---

## 项目结构说明

```
curiosity-ppo/
├── src/curiosity_ppo/     # 核心源码 (curiosity / networks / ppo / envs / utils)
├── scripts/               # 训练、评测、消融实验脚本
├── experiments/           # YAML 配置文件 (7个: 全局 + 3环境 + 4消融)
├── benchmarks/            # 基准评测脚本
├── web/                   # Web Demo (Vite + React + TypeScript + ONNX Runtime Web)
├── docs/                  # 技术文档 (10篇)
├── tests/                 # 单元测试 (144项, pytest)
├── test/                  # 集成测试与运行日志
├── results/               # 训练结果 (checkpoints / onnx, 已 gitignore)
├── requirements.txt       # Python 依赖
├── pyproject.toml         # 项目元数据与工具配置
└── .env.example           # 环境变量模板
```

详细目录结构请参考 [README.md](./README.md#项目结构) 中的项目结构章节。

### 模块职责

| 模块路径 | 职责 |
|---------|------|
| `curiosity_ppo.curiosity` | ICM / RND / Episodic Memory / NGU 融合 / 奖励归一化 |
| `curiosity_ppo.networks` | CNN 编码器 / ICM 网络 / RND 网络 / Actor-Critic 策略网络 |
| `curiosity_ppo.ppo` | PPO 训练器 / GAE 优势估计 / Rollout 缓冲区 / Agent 封装 |
| `curiosity_ppo.envs` | Crafter / Atari / MiniGrid 环境封装与向量化 |
| `curiosity_ppo.utils` | FP16 AMP / VRAM 管理 / Checkpoint / Logger / Seed |

---

## 联系方式

- **GitHub Issues**：[提交 Issue](https://github.com/<your-username>/curiosity-ppo/issues) —— Bug 报告、功能请求、问题讨论
- **GitHub Discussions**：[发起讨论](https://github.com/<your-username>/curiosity-ppo/discussions) —— 技术交流、复现问题、学习笔记分享
- **Email**：curiosityppo@example.com

---

再次感谢你的贡献！每一个 Star、Fork、Issue 和 PR 都是对本项目的巨大支持。

> 本项目基于 MIT 开源协议，贡献者提交的代码同样遵循 MIT 协议。
