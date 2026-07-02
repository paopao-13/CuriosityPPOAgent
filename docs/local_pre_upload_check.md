# CuriosityPPOAgent 上传前本地实操校验核对手册

> 项目全称：CuriosityPPOAgent ICM+RND 分层新颖信号融合好奇心驱动 PPO 智能体
> 版本覆盖：v0.1 ~ v1.0（12 次 Git 提交）
> 资产规模：63 个 Python 源码 · 7 个 yaml 配置 · 10 篇文档 · 22 份 Web Demo · 144 项单元测试
> 硬件基准：AMD R7 6800H + RTX3060 Laptop 6GB，显存峰值 2.2GB
> 开源协议：MIT

---

## 1. 文档前置强制说明

### 1.1 校验核心目的

本手册的设立基于三条不可妥协的核心目的，每一项均直接关联面试场景下的项目可信度：

| 序号 | 校验核心目的 | 详细说明 | 面试加分价值 |
|------|-------------|---------|-------------|
| 目的一 | 规避线上仓库代码运行报错 | 确保 clone 仓库后可直接 `pip install -r requirements.txt` 并运行训练/评测/导出全流程，杜绝 ImportError、ModuleNotFound、路径不存在等低级错误 | 面试官 clone 仓库后能一键跑通，直接形成"工程能力强、交付质量高"的第一印象，避免因报错产生负面技术评价 |
| 目的二 | 降低面试官负面观感 | 确保仓库无冗余大文件、无个人磁盘路径泄露、无 IDE 缓存残留、无调试断点遗留，README 数据与代码输出一致 | 仓库整洁度是面试官判断开发者职业素养的隐性指标，干净规范的仓库结构能显著提升"可信赖工程师"形象 |
| 目的三 | 防止复现失败质疑 | 确保量化指标（Crafter 19.0%、Atari 3500+、MiniGrid 96.8 万步）可在本地复现，ICM 逆损失≈2.83=ln17、内在奖励≈60 等关键数值有代码佐证 | 面试官若当场要求复现某项指标，能立即给出运行命令并得到符合区间的结果，彻底消除"数据造假"的潜在质疑 |

> **面试加分价值总结**：本手册将"上传前校验"从可选操作提升为强制流程，向面试官展示的是一种具备工程化质量管控意识的成熟开发者形象——不是写完代码就推，而是推之前完成体系化自检。

### 1.2 硬性红线规则

以下红线规则在执行任何 Git 推送操作前必须全部满足，**任意一项不达标，禁止执行 `git push` 远程仓库操作**：

| 红线编号 | 红线规则 | 合格判定标准 | 违反后果 | 面试加分价值 |
|---------|---------|-------------|---------|-------------|
| RED-01 | 144 项单元测试全部通过 | `pytest` 退出码为 0，0 报错 0 跳过 | 代码质量存疑，面试官可质疑核心算法正确性 | 展示测试驱动开发意识，144 项全通过是算法可靠性的硬证据 |
| RED-02 | 全仓库零硬编码绝对路径 | 绝对路径检索命令返回空，无 C:/ D:/ 等匹配项 | 面试官 clone 后无法运行，直接判定项目不可复现 | 体现环境无关性意识，证明代码具备跨机器可移植性 |
| RED-03 | 无大体积冗余文件进入版本控制 | `.gitignore` 生效，checkpoint/wandb/video/node_modules 均被拦截 | 仓库膨胀至数百 MB，clone 缓慢，专业度骤降 | 展示对版本控制仓库体积的管控能力 |
| RED-04 | README 量化数据与代码输出一致 | 训练/评测输出数值落在文档标注的理论区间内 | 数据造假嫌疑，面试中最致命的信任崩塌 | 数据一致性是技术诚信的基石，通过校验即通过信任关 |
| RED-05 | 12 份 Github 配套文件全部就位且路径正确 | 目录安放核对总表 12 行全部打勾 | 仓库规范残缺，开源协作能力受质疑 | 完整的开源配套文件展示标准化工程交付能力 |

> **强制执行声明**：以上五条红线为"一票否决制"。若任一红线未达标，即使其余全部通过，也必须先修复后再推送。这一机制本身即向面试官传递"严格质量门禁"的工程治理信号。

---

## 2. 第一部分：12 份 Github 配套文件目录安放核对总表

以下为项目已生成的 12 份 Github 配套文件，上传前须逐一核对存放路径与合格标准。所有路径均相对于 `<项目根目录>`。

| 序号 | 文件名称 | 目标存放相对路径 | 校验合格标准 |
|------|---------|----------------|-------------|
| 1 | README.md | `<项目根目录>/README.md` | 存在于根目录；包含项目简介、架构图、快速开始、量化结果、引用协议；Mermaid 图语法正确 |
| 2 | LICENSE | `<项目根目录>/LICENSE` | 存在于根目录；内容为 MIT 协议全文；年份与作者信息正确 |
| 3 | .gitignore | `<项目根目录>/.gitignore` | 存在于根目录；包含 Python/IDE/权重/缓存/日志/视频/wandb/node_modules 屏蔽规则 |
| 4 | CONTRIBUTING.md | `<项目根目录>/CONTRIBUTING.md` | 存在于根目录；包含开发环境、提交规范、分支策略、PR 流程说明 |
| 5 | bug_report.md | `<项目根目录>/.github/ISSUE_TEMPLATE/bug_report.md` | 存在于 `.github/ISSUE_TEMPLATE/` 目录下；包含 Bug 描述、复现步骤、预期/实际行为、环境信息模板字段 |
| 6 | feature_request.md | `<项目根目录>/.github/ISSUE_TEMPLATE/feature_request.md` | 存在于 `.github/ISSUE_TEMPLATE/` 目录下；包含功能动机、方案描述、替代方案、补充说明模板字段 |
| 7 | test.yml | `<项目根目录>/.github/workflows/test.yml` | 存在于 `.github/workflows/` 目录下；YAML 语法合法；包含 checkout、setup-python、install、pytest 步骤 |
| 8 | release_note.md | `<项目根目录>/docs/release_note.md` | 存在于 `docs/` 目录下；覆盖 v0.1~v1.0 共 12 个版本变更记录 |
| 9 | repo_tags.md | `<项目根目录>/docs/repo_tags.md` | 存在于 `docs/` 目录下；包含 12 个 Git Tag 与版本对应关系 |
| 10 | repo_cover_desc.txt | `<项目根目录>/assets/repo_cover_desc.txt` | 存在于 `assets/` 目录下；包含仓库封面图描述文本 |
| 11 | git_upload_step.md | `<项目根目录>/docs/git_upload_step.md` | 存在于 `docs/` 目录下；包含完整的 Git 上传操作步骤 |
| 12 | repo_self_checklist.md | `<项目根目录>/docs/repo_self_checklist.md` | 存在于 `docs/` 目录下；包含 6 组 24 项自检条目 |

### 一键核对命令

在 `<项目根目录>` 下执行以下 PowerShell 命令，批量验证 12 份文件是否全部就位：

```powershell
$files = @(
    "README.md",
    "LICENSE",
    ".gitignore",
    "CONTRIBUTING.md",
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/feature_request.md",
    ".github/workflows/test.yml",
    "docs/release_note.md",
    "docs/repo_tags.md",
    "assets/repo_cover_desc.txt",
    "docs/git_upload_step.md",
    "docs/repo_self_checklist.md"
)
$missing = @()
foreach ($f in $files) {
    if (Test-Path $f) {
        Write-Host "[OK]   $f" -ForegroundColor Green
    } else {
        Write-Host "[MISS] $f" -ForegroundColor Red
        $missing += $f
    }
}
if ($missing.Count -eq 0) {
    Write-Host "`n=== 12 份配套文件全部就位 ===" -ForegroundColor Green
} else {
    Write-Host "`n=== 缺失 $($missing.Count) 份文件，请补齐后再上传 ===" -ForegroundColor Red
}
```

> **面试加分价值**：12 份配套文件覆盖了开源项目的完整生命周期（文档、协议、贡献指南、Issue 模板、CI、发布说明、Tag 管理），向面试官展示的是"具备开源项目治理经验"而非"只会写算法代码"的差异化竞争力。

---

## 3. 第二部分：本地工程前置清理实操步骤（必执行）

> **执行前提**：以下所有命令均在 `<项目根目录>` 下以 PowerShell 执行。本部分为红线 RED-03 的实操落地。

### 3.1 .gitignore 屏蔽规则校验

#### 3.1.1 核心屏蔽规则清单

`.gitignore` 文件必须包含以下屏蔽规则，确保权重、缓存、日志、视频等大体积或环境敏感文件不进入版本控制：

| 屏蔽类别 | 具体规则模式 | 说明 |
|---------|-------------|------|
| Python 编译缓存 | `__pycache__/`、`*.pyc`、`*.pyo` | 防止字节码缓存入库 |
| 训练 checkpoint 权重 | `*.pth`、`*.pt`、`*.ckpt`、`checkpoints/` | 权重文件动辄数百 MB，必须拦截 |
| wandb 完整日志 | `wandb/`、`*.wandb` | 在线实验日志目录体积庞大 |
| 训练日志 | `logs/`、`*.log` | 文本日志不纳入版本控制 |
| 演示视频 | `videos/`、`*.mp4`、`*.avi`、`*.gif` | 视频文件体积大，用 Web Demo 替代 |
| IDE 缓存 | `.idea/`、`.vscode/`、`*.swp` | 环境相关，不应入库 |
| 前端依赖 | `node_modules/`、`dist/` | 22 份 Web Demo 的前端依赖 |
| 操作系统文件 | `.DS_Store`、`Thumbs.db` | 跨平台兼容性 |
| 虚拟环境 | `venv/`、`env/`、`.env` | 环境隔离，不入库 |

#### 3.1.2 一键检索验证命令

执行以下命令，验证权重/缓存/日志/视频是否被 `.gitignore` 正确拦截：

```powershell
# 检查 .gitignore 是否包含关键屏蔽规则
$patterns = @("__pycache__", "\.pth", "\.pt", "wandb", "\.log", "videos", "\.mp4", "node_modules", "\.idea", "\.vscode", "venv")
Write-Host "=== .gitignore 屏蔽规则校验 ===" -ForegroundColor Cyan
foreach ($p in $patterns) {
    $found = Select-String -Path ".gitignore" -Pattern $p -Quiet
    if ($found) {
        Write-Host "[OK]   规则已配置: $p" -ForegroundColor Green
    } else {
        Write-Host "[MISS] 规则缺失:   $p" -ForegroundColor Red
    }
}

# 检查是否有本应被忽略的文件已进入 Git 暂存区
Write-Host "`n=== 检查暂存区是否有违规文件 ===" -ForegroundColor Cyan
$staged = git ls-files --cached
$violations = $staged | Where-Object { $_ -match '\.(pth|pt|ckpt|mp4|avi|log)$' -or $_ -match '(__pycache__|wandb/|node_modules/|\.idea/|\.vscode/)' }
if ($violations) {
    Write-Host "[WARN] 以下违规文件已在暂存区，需移除:" -ForegroundColor Red
    $violations | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
} else {
    Write-Host "[OK] 暂存区无违规文件" -ForegroundColor Green
}
```

#### 3.1.3 合格判定标准

- `.gitignore` 包含上述全部屏蔽规则，无缺失项
- `git ls-files --cached` 结果中无 `.pth`/`.pt`/`.mp4`/`.log` 等违规文件
- `git status` 显示的未跟踪文件中不包含 checkpoint、wandb、node_modules 等目录

> **面试加分价值**：规范的 `.gitignore` 是版本控制基本功。面试官浏览仓库时若发现 `__pycache__` 或 `.pth` 文件入库，会直接质疑开发者的工程规范意识。本项校验确保仓库"瘦身且干净"。

### 3.2 大体积冗余文件清理清单

上传前必须清理以下大体积冗余文件，确保仓库体积控制在合理范围内：

| 序号 | 清理对象 | 典型路径 | 清理操作 | 预估释放体积 | 面试加分价值 |
|------|---------|---------|---------|-------------|-------------|
| 1 | 训练 checkpoint 权重 | `checkpoints/*.pth`、`checkpoints/*.pt` | 删除或迁移至本地非 Git 目录 | 200MB~2GB | 避免仓库膨胀，展示对模型文件管理的认知 |
| 2 | wandb 完整日志 | `wandb/` 目录 | 删除整个目录，保留 `wandb_summary.json` 摘要即可 | 50MB~500MB | 实验日志不应全量入库，摘要足够说明问题 |
| 3 | 长演示视频 | `videos/*.mp4`、`videos/*.gif` | 删除视频文件，改用 22 份 Web Demo 的 HTML 交互演示 | 100MB~1GB | 用轻量 Web Demo 替代视频，展示前端可视化能力 |
| 4 | IDE 缓存 | `.idea/`、`.vscode/` | 删除目录，确保已被 `.gitignore` 拦截 | 5MB~50MB | 环境无关性，避免暴露个人配置 |
| 5 | 前端 node_modules | `**/node_modules/` | 删除所有 `node_modules/` 目录，保留 `package.json` | 100MB~500MB | 依赖不应入库，`npm install` 可恢复 |
| 6 | Python 缓存 | `**/__pycache__/` | 递归删除所有 `__pycache__` 目录 | 5MB~20MB | 字节码缓存无意义入库 |
| 7 | 虚拟环境 | `venv/`、`env/` | 删除虚拟环境目录，保留 `requirements.txt` | 500MB~2GB | 环境隔离是基本规范 |

#### 一键清理命令

```powershell
# 安全清理（仅删除 .gitignore 已拦截的文件，不影响源码）
Write-Host "=== 大体积冗余文件清理 ===" -ForegroundColor Cyan

# 1. 清理 Python 缓存
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Write-Host "[OK] 已清理 __pycache__" -ForegroundColor Green

# 2. 清理 node_modules（保留 package.json）
Get-ChildItem -Recurse -Directory -Filter "node_modules" | Remove-Item -Recurse -Force
Write-Host "[OK] 已清理 node_modules" -ForegroundColor Green

# 3. 清理 IDE 缓存
Remove-Item -Recurse -Force ".idea" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force ".vscode" -ErrorAction SilentlyContinue
Write-Host "[OK] 已清理 IDE 缓存" -ForegroundColor Green

# 4. 清理训练权重（请确认已备份到本地非 Git 目录）
Remove-Item -Recurse -Force "checkpoints" -ErrorAction SilentlyContinue
Write-Host "[OK] 已清理 checkpoints" -ForegroundColor Green

# 5. 清理 wandb 完整日志（保留摘要）
Remove-Item -Recurse -Force "wandb" -ErrorAction SilentlyContinue
Write-Host "[OK] 已清理 wandb 日志" -ForegroundColor Green

# 6. 清理演示视频
Remove-Item -Recurse -Force "videos" -ErrorAction SilentlyContinue
Write-Host "[OK] 已清理 videos" -ForegroundColor Green

Write-Host "`n=== 清理完成，请执行 git status 确认仓库体积 ===" -ForegroundColor Cyan
```

#### 合格判定标准

- `git status` 未跟踪文件列表中不出现 checkpoint、wandb、node_modules、videos 等目录
- 仓库总体积（不含 `.git`）控制在 50MB 以内（主要由 Python 源码 + yaml 配置 + 文档 + Web Demo HTML 构成）

> **面试加分价值**：精简的仓库体积意味着面试官能快速 clone 并浏览。冗余文件清理体现的是"对仓库受众体验的考量"——不浪费他人带宽与时间。

### 3.3 全局硬编码绝对路径一键检索

#### 核心目的

扫描全部 Python 源码与配置文件，排查是否存在本地磁盘绝对路径（如 `C:\Users\`、`D:\`、`/home/` 等），确保代码具备跨机器可移植性。此项对应红线 RED-02。

#### 一键检索 PowerShell 命令

```powershell
# 全局硬编码绝对路径检索
Write-Host "=== 全局硬编码绝对路径检索 ===" -ForegroundColor Cyan

# 匹配 Windows 绝对路径（C:\ D:\ E:\ 等）和 Linux 绝对路径（/home/ /Users/）
$pattern = '(?i)([A-Z]:\\)|(\\/home\\/)|(\\/Users\\/)|(\\\\\\\\.{2,}\\\\)'
$extensions = @("*.py", "*.yaml", "*.yml", "*.json", "*.md", "*.txt", "*.ps1", "*.cfg", "*.ini")

$found = $false
foreach ($ext in $extensions) {
    $matches = Get-ChildItem -Recurse -File -Include $ext -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\\.git\\' -and $_.FullName -notmatch 'node_modules' } |
        Select-String -Pattern $pattern -ErrorAction SilentlyContinue
    if ($matches) {
        $found = $true
        foreach ($m in $matches) {
            Write-Host "[WARN] $($m.RelativePath):$($m.LineNumber) -> $($m.Line.Trim())" -ForegroundColor Red
        }
    }
}

if (-not $found) {
    Write-Host "`n[OK] 检索结果为空，无硬编码绝对路径匹配项" -ForegroundColor Green
    Write-Host "=== 全局绝对路径校验通过 ===" -ForegroundColor Green
} else {
    Write-Host "`n[FAIL] 发现硬编码绝对路径，请替换为相对路径或环境变量" -ForegroundColor Red
}
```

#### 合格判定标准

- **检索结果为空，无任何匹配项**——这是红线 RED-02 的硬性要求
- 所有路径均使用相对路径（如 `./configs/`、`../data/`）或通过环境变量/配置文件注入
- 若必须使用绝对路径（如 ONNX 导出输出目录），应通过命令行参数或配置文件传入，而非硬编码在源码中

#### 常见违规示例与修复

| 违规示例 | 修复方案 |
|---------|---------|
| `MODEL_PATH = "D:\\项目\\checkpoints\\model.pth"` | `MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "model.pth")` |
| `DATA_DIR = "C:/Users/jx/data/crafter"` | `DATA_DIR = os.environ.get("CRAFTER_DATA_DIR", "./data/crafter")` |
| `log_file = "/home/user/logs/train.log"` | `log_file = os.path.join(PROJECT_ROOT, "logs", "train.log")` |

> **面试加分价值**：零硬编码绝对路径是代码可移植性的直接证明。面试官 clone 仓库后若因路径问题无法运行，会立即质疑开发者的工程经验。此项校验通过，即意味着"任何机器都能跑"的可复现承诺。

---

## 4. 第三部分：四层核心功能实操校验

> **统一格式说明**：以下每条校验均按"执行命令 → 分步操作 → 合格判定标准 → 失败快速排查方案"四段式组织。

### 4.1 单元测试校验

#### 执行命令

```powershell
# 在 <项目根目录> 下执行
python -m pytest tests/ -v --tb=short
```

#### 分步操作

1. 打开 PowerShell，切换至 `<项目根目录>`
2. 激活项目虚拟环境（如 `.\venv\Scripts\Activate.ps1`）
3. 确认 `pytest` 已安装：`pip install pytest`（若未安装）
4. 执行上述命令，等待全部 144 项测试用例运行完毕
5. 检查终端输出的测试统计行：`===== 144 passed in Xs =====`

#### 合格判定标准

| 判定项 | 合格标准 | 对应红线 |
|--------|---------|---------|
| 测试总数 | 144 项全部被收集执行 | RED-01 |
| 通过数 | 144 项全部通过（0 failed） | RED-01 |
| 报错数 | 0 报错 | RED-01 |
| 跳过数 | 0 跳过（skipped）或仅有明确标注的预期跳过 | RED-01 |
| 退出码 | 退出码为 0 | RED-01 |

#### 失败快速排查方案

| 故障现象 | 可能原因 | 排查与修复 |
|---------|---------|-----------|
| `ModuleNotFoundError: No module named 'xxx'` | 依赖未安装 | 执行 `pip install -r requirements.txt` 安装全部依赖 |
| `ImportError: cannot import name 'xxx'` | 模块路径变更或循环导入 | 检查 `tests/` 中的 import 路径是否与源码目录结构一致 |
| `FileNotFoundError: configs/xxx.yaml` | 配置文件路径错误 | 确认在 `<项目根目录>` 下执行，或检查测试中的相对路径 |
| `RuntimeError: CUDA out of memory` | 显存不足 | 确认无其他 GPU 进程占用；测试应使用 CPU 模式或 mock 数据 |
| 个别测试 assertion 失败 | 算法逻辑变更未同步更新测试 | 逐个检查失败测试，确认是代码回归还是测试期望值需更新 |
| `OSError: [WinError 126]` 动态链接库缺失 | CUDA/CUDNN 环境问题 | 确认 PyTorch CUDA 版本与驱动匹配，或强制使用 CPU 模式 |

> **面试加分价值**：144 项单元测试全通过是项目质量最硬的证据。面试官看到 `144 passed` 会立即建立对算法正确性的信任，这比任何口头解释都更有说服力。

### 4.2 README 全流程复现校验

#### 执行命令

```powershell
# 完整执行：训练 → 消融 → 评测 → ONNX 导出整套命令

# 步骤1: 训练（短步数验证，非完整训练）
python scripts/train.py --env crafter --steps 10000 --config configs/crafter.yaml

# 步骤2: 消融实验（快速验证4组配置可启动）
python scripts/run_ablation.py --env crafter --steps 10000
# 或使用 PowerShell 脚本一键运行
.\scripts\run_all_ablation.ps1 -Steps 10000

# 步骤3: 评测
python scripts/evaluate.py --env crafter --checkpoint checkpoints/best.pth --episodes 10

# 步骤4: ONNX 导出
python scripts/export_onnx.py --checkpoint checkpoints/best.pth --output models/agent.onnx
```

#### 分步操作

1. **训练验证**：执行 `scripts/train.py`，使用短步数（10000 步）验证训练流程可正常启动
   - 观察 ICM 前向损失初始值：应≈2.83（=ln17，17 维动作空间的均匀分布熵）
   - 观察内在奖励初始值：应≈60（融合公式 `r_int = η×ICM前向损失 + r_episodic×min(max(α_t,1),L)` 的初始输出区间）
   - 确认 FP16 AMP 正常启用，显存峰值不超过 2.2GB
   - 确认梯度累积 128×4=512 等效 batch size 生效

2. **消融验证**：执行 `scripts/run_ablation.py` 或 `scripts/run_all_ablation.ps1`
   - 确认 4 组消融配置均可启动：full / no_icm / no_episodic / no_rnd
   - 每组配置运行短步数，确认无崩溃

3. **评测验证**：执行 `scripts/evaluate.py`
   - 确认能加载 checkpoint 并完成评测回合
   - 观察评测指标输出格式是否符合 README 标注

4. **ONNX 导出验证**：执行 `scripts/export_onnx.py`
   - 确认导出成功，生成 `.onnx` 文件
   - 确认导出日志无 warning/error

#### 合格判定标准

| 判定项 | 合格标准 | 理论依据 |
|--------|---------|---------|
| 训练启动 | 无运行崩溃，正常输出日志 | 代码可执行性 |
| ICM 逆损失初始值 | ≈2.83（允许 ±0.2 浮动） | 17 维动作空间均匀分布的交叉熵=ln(17)≈2.833 |
| 内在奖励初始值 | ≈60（允许 ±10 浮动） | 融合公式 `r_int = η×ICM前向损失 + r_episodic×min(max(α_t,1),L)` 初始输出 |
| 消融 4 组启动 | full / no_icm / no_episodic / no_rnd 均无崩溃 | 消融配置正确性 |
| 评测执行 | 正常加载权重并完成评测回合 | 评测流程完整性 |
| ONNX 导出 | 成功生成 `.onnx` 文件，无报错 | 模型部署能力 |
| 显存峰值 | ≤2.2GB（RTX3060 Laptop 6GB） | FP16 AMP + 梯度累积 + CPU 缓存卸载生效 |

#### 失败快速排查方案

| 故障现象 | 可能原因 | 排查与修复 |
|---------|---------|-----------|
| ICM 逆损失初始值偏差大 | 动作空间维度配置错误 | 检查 `configs/crafter.yaml` 中 `action_dim` 是否为 17；确认 ICM 前向网络输出维度匹配 |
| 内在奖励为 0 或 NaN | 融合公式系数 η/α_t 配置异常 | 检查配置文件中 `eta`、`alpha` 参数；确认 Episodic Memory LRU 200 向量初始化正常 |
| 显存 OOM | FP16 AMP 未生效或 batch size 过大 | 确认 `use_amp: true`；确认梯度累积步数配置；检查 CPU 缓存卸载开关 |
| 消融某组崩溃 | 消融配置文件缺失或字段错误 | 检查 4 组消融 yaml 配置的完整性，对比 `full` 配置确认差异字段 |
| ONNX 导出失败 | 动态维度或自定义算子不支持 | 检查模型是否有动态 batch 维；尝试 `opset_version` 调整；确认无不支持的算子 |
| 评测加载权重失败 | checkpoint 路径或格式错误 | 确认 `--checkpoint` 参数指向有效 `.pth` 文件；检查 `state_dict` key 匹配 |

> **面试加分价值**：README 全流程可复现是开源项目的最高诚信证明。面试官若要求"现场跑一下"，能立即执行并得到 ICM 逆损失≈2.83 的理论一致结果，直接消除一切数据造假质疑。ICM 逆损失=ln17 这一理论推导更是展示数学功底的高光时刻。

### 4.3 Mermaid 架构图渲染校验

#### 执行命令

无需命令行执行，使用在线渲染工具校验。

#### 分步操作

1. 打开浏览器，访问 Mermaid 在线渲染网址：**https://mermaid.live**
2. 在 `<项目根目录>/README.md` 中找到 Mermaid 架构图代码块（以 ` ```mermaid ` 标记）
3. 复制完整的 Mermaid 代码内容
4. 粘贴到 mermaid.live 左侧编辑器面板
5. 观察右侧渲染面板的输出结果

#### 合格判定标准

| 判定项 | 合格标准 |
|--------|---------|
| 语法校验 | 无语法报错（编辑器下方无红色错误提示） |
| 图形渲染 | 右侧面板完整展示架构图，无空白或断裂 |
| 数据流完整性 | 图中包含完整数据流：观测输入 → ICM(4层CNN→288维) → RND → Episodic Memory(LRU 200) → 融合公式 → PPO → 动作输出 |
| 融合公式展示 | 图中标注融合公式 `r_int = η×ICM前向损失 + r_episodic×min(max(α_t,1),L)` |
| 节点连接 | 所有节点之间的连线方向正确，无悬空节点 |
| Github 兼容 | 语法兼容 Github 原生 Mermaid 渲染（Github 使用 mermaid v9.x） |

#### 失败快速排查方案

| 故障现象 | 可能原因 | 排查与修复 |
|---------|---------|-----------|
| `Parse error` 语法报错 | Mermaid 语法版本不兼容 | 检查是否使用了过新/过旧的语法特性；将 `graph` 改为 `flowchart` 或反之；检查特殊字符是否需转义 |
| 图形渲染为空白 | 代码块标记错误 | 确认 README 中使用 ` ```mermaid ` 而非 ` ```Mermaid `（大小写敏感） |
| 节点文字溢出 | 节点标签过长 | 使用 `<br/>` 换行或缩短标签文字 |
| 箭头方向错误 | 箭头语法错误 | 确认使用 `-->`（有向）、`-.->`（虚线有向）等正确语法 |
| Github 不渲染 | Github Mermaid 版本限制 | 避免使用 `subgraph` 嵌套过深；避免使用 Github 不支持的图表类型 |
| 融合公式显示异常 | LaTeX 公式在 Mermaid 中不支持 | 使用纯文本表示公式，如 `r_int = eta * ICM_loss + r_episodic * min(max(alpha_t, 1), L)` |

> **面试加分价值**：一张清晰渲染的架构图是面试官理解项目的最快入口。Mermaid 语法正确、数据流完整、融合公式可见，意味着面试官在 30 秒内就能 grasp 项目核心技术点。这是"降低理解门槛"即"提升技术沟通能力"的体现。

### 4.4 Github Actions CI 脚本本地模拟校验

#### 执行命令

```powershell
# 方式1: 直接验证 YAML 语法合法性
python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml')); print('[OK] YAML 语法合法')"

# 方式2: 本地模拟 CI 测试步骤（手动执行 test.yml 中的核心步骤）
# 模拟 install 步骤
pip install -r requirements.txt
pip install pytest

# 模拟 test 步骤
python -m pytest tests/ -v --tb=short
```

#### 分步操作

1. **YAML 语法校验**：执行方式 1 命令，确认 `.github/workflows/test.yml` 语法合法
2. **依赖安装模拟**：执行 `pip install -r requirements.txt`，模拟 CI 中的 install 步骤
3. **测试执行模拟**：执行 `python -m pytest tests/ -v`，模拟 CI 中的 test 步骤
4. **检查 test.yml 关键字段**：确认以下字段配置正确

   | 检查项 | 合格标准 |
   |--------|---------|
   | `name` | CI 工作流名称有意义（如 `CI` / `Test`） |
   | `on` | 触发条件包含 `push` 和 `pull_request`，分支为 `main` |
   | `runs-on` | 使用 `ubuntu-latest`（避免使用 windows/macos 增加排队时间） |
   | `python-version` | 指定明确版本（如 `3.10`），避免使用 `3.x` 通配 |
   | `install` 步骤 | 包含 `pip install -r requirements.txt` |
   | `test` 步骤 | 包含 `pytest` 命令 |
   | `cache` | 配置 pip 缓存以加速 CI |

5. **可选：使用 act 工具本地完整模拟**（需安装 [act](https://github.com/nektos/act)）

   ```powershell
   # 安装 act 后执行（可选，非必须）
   act -W .github/workflows/test.yml
   ```

#### 合格判定标准

| 判定项 | 合格标准 |
|--------|---------|
| YAML 语法 | `yaml.safe_load` 无异常，语法合法 |
| 测试脚本启动 | `pytest` 可正常启动，无语法报错 |
| 依赖安装 | `pip install -r requirements.txt` 全部成功 |
| 测试通过 | 144 项测试全部通过（与 4.1 一致） |
| 触发条件 | `on.push.branches` 包含 `main`，`on.pull_request.branches` 包含 `main` |
| 运行环境 | `runs-on: ubuntu-latest`，`python-version` 明确 |

#### 失败快速排查方案

| 故障现象 | 可能原因 | 排查与修复 |
|---------|---------|-----------|
| `yaml.YAMLError` | YAML 缩进或语法错误 | 检查缩进使用空格而非 Tab；检查冒号后空格；使用 YAML 校验工具修复 |
| CI 中 pip install 失败 | `requirements.txt` 包含平台相关依赖 | 确认依赖跨平台兼容；移除 Windows 专用包；使用 `sys.platform` 条件安装 |
| CI 中 pytest 失败但本地通过 | 环境差异（OS/Python 版本/路径） | 确认无硬编码绝对路径；确认无平台相关 API 调用；检查路径分隔符 |
| CI 中找不到测试文件 | `tests/` 目录未被提交 | 确认 `tests/` 不在 `.gitignore` 中；执行 `git add tests/` |
| CI 超时 | 测试执行时间过长 | 优化测试用例执行速度；或拆分为多个 job 并行执行 |

> **面试加分价值**：Github Actions CI 配置正确意味着推送后自动运行测试，面试官在仓库页面看到绿色对勾 CI 徽章，会立即产生"这个开发者重视持续集成"的专业印象。CI 是现代化工程实践的标配能力证明。

---

## 5. 第四部分：24 项分层自检勾选清单

> **说明**：本清单将 `repo_self_checklist.md` 的 6 组 24 项重新组织为 4 大分类，每类 5 项共 20 项，另加 4 项补充校验，合计 24 项。上传前须逐项勾选确认。

### 5.1 文件合规校验（5 项）

| 序号 | 自检项 | 可复制验证命令 | 合格判定 | 勾选 |
|------|--------|--------------|---------|------|
| F-01 | 12 份 Github 配套文件全部就位 | 见第二部分一键核对命令 | 12 份文件全部 `[OK]` | ☐ |
| F-02 | `.gitignore` 屏蔽规则完整 | 见 3.1.2 一键检索命令 | 所有关键规则 `[OK]`，暂存区无违规文件 | ☐ |
| F-03 | 仓库无大体积冗余文件 | `git ls-files \| Measure-Object` 检查文件数；`git count-objects -vH` 检查体积 | 无 .pth/.mp4/.wandb 入库，仓库体积 <50MB | ☐ |
| F-04 | 项目根目录结构规范 | `Get-ChildItem -Directory \| Select-Object Name` | 包含 src/ tests/ configs/ scripts/ docs/ assets/ 等标准目录 | ☐ |
| F-05 | 文档与代码版本一致 | `git log --oneline -n 12` 对照 `docs/release_note.md` | 12 次提交记录与 release_note 一一对应 | ☐ |

### 5.2 代码运行校验（5 项）

| 序号 | 自检项 | 可复制验证命令 | 合格判定 | 勾选 |
|------|--------|--------------|---------|------|
| C-01 | 144 项单元测试全部通过 | `python -m pytest tests/ -v --tb=short` | `144 passed`，退出码 0 | ☐ |
| C-02 | 训练脚本可正常启动 | `python scripts/train.py --env crafter --steps 1000` | 无崩溃，正常输出训练日志 | ☐ |
| C-03 | 评测脚本可正常执行 | `python scripts/evaluate.py --env crafter --checkpoint checkpoints/best.pth --episodes 5` | 正常加载权重并完成评测 | ☐ |
| C-04 | ONNX 导出脚本可正常执行 | `python scripts/export_onnx.py --checkpoint checkpoints/best.pth --output models/agent.onnx` | 成功生成 .onnx 文件 | ☐ |
| C-05 | 消融脚本 4 组均可启动 | `python scripts/run_ablation.py --env crafter --steps 1000` | full/no_icm/no_episodic/no_rnd 四组均无崩溃 | ☐ |

### 5.3 量化数据统一校验（5 项）

| 序号 | 自检项 | 可复制验证命令 | 合格判定 | 勾选 |
|------|--------|--------------|---------|------|
| Q-01 | ICM 逆损失初始值≈2.83=ln17 | `python scripts/train.py --env crafter --steps 100` 观察日志 | 初始 ICM forward loss ≈2.83（±0.2） | ☐ |
| Q-02 | 内在奖励初始值≈60 | 同上，观察 intrinsic reward 日志 | 初始 r_int ≈60（±10） | ☐ |
| Q-03 | Crafter 成绩 19.0% 对标基线 15.6% | 检查 README 与 `docs/release_note.md` 中的数值 | README/release_note/代码输出三处数值一致 | ☐ |
| Q-04 | Atari 成绩 3500+ 对标基线 120 | 同上交叉核对 | 三处数值一致，3500+ > 120 基线 | ☐ |
| Q-05 | MiniGrid 96.8 万步对标基线 242 万 | 同上交叉核对 | 三处数值一致，96.8 万 < 242 万（步数更少更优） | ☐ |

### 5.4 开源规范校验（5 项）

| 序号 | 自检项 | 可复制验证命令 | 合格判定 | 勾选 |
|------|--------|--------------|---------|------|
| O-01 | MIT 协议文件正确 | `Get-Content LICENSE \| Select-Object -First 3` | 第一行包含 `MIT License`，年份与作者正确 | ☐ |
| O-02 | CONTRIBUTING.md 内容完整 | `Get-Content CONTRIBUTING.md` | 包含环境搭建、提交规范、分支策略、PR 流程 | ☐ |
| O-03 | Issue 模板字段完整 | `Get-Content .github/ISSUE_TEMPLATE/bug_report.md` | 包含 Bug 描述/复现步骤/预期行为/环境信息字段 | ☐ |
| O-04 | CI 工作流配置正确 | `python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yml'))"` | YAML 合法，包含 push/PR 触发、install、pytest 步骤 | ☐ |
| O-05 | 全仓库零硬编码绝对路径 | 见 3.3 一键检索命令 | 检索结果为空，无匹配项 | ☐ |

### 5.5 补充校验（4 项）

| 序号 | 自检项 | 可复制验证命令 | 合格判定 | 勾选 |
|------|--------|--------------|---------|------|
| S-01 | README Mermaid 图渲染正常 | 复制至 https://mermaid.live 渲染 | 无语法报错，完整展示全数据流 | ☐ |
| S-02 | 显存峰值≤2.2GB | 训练时观察 `nvidia-smi` 或日志中的 GPU memory | 峰值显存 ≤2.2GB（FP16 AMP + 梯度累积生效） | ☐ |
| S-03 | 22 份 Web Demo HTML 可正常打开 | `Get-ChildItem -Recurse -Filter "*.html" \| Measure-Object` 并抽查打开 | 22 份 HTML 文件存在，浏览器打开无报错 | ☐ |
| S-04 | Git 提交历史规范 | `git log --oneline` | 12 次提交 v0.1~v1.0，提交信息符合规范 | ☐ |

### 24 项总览汇总

| 分类 | 项数 | 编号范围 | 核心关注点 |
|------|------|---------|-----------|
| 文件合规校验 | 5 项 | F-01 ~ F-05 | 文件就位、结构规范、版本一致 |
| 代码运行校验 | 5 项 | C-01 ~ C-05 | 测试通过、脚本可执行 |
| 量化数据统一校验 | 5 项 | Q-01 ~ Q-05 | 理论数值一致、三处交叉核对 |
| 开源规范校验 | 5 项 | O-01 ~ O-05 | 协议、贡献指南、CI、路径 |
| 补充校验 | 4 项 | S-01 ~ S-04 | 渲染、显存、Demo、提交历史 |
| **合计** | **24 项** | **F/C/Q/O/S** | **全部勾选后方可推送** |

> **面试加分价值**：24 项分层自检清单本身就是一种工程化质量管控的展示。面试官看到如此体系化的自检流程，会认为这位开发者具备"交付前验收"的成熟工程习惯——这是 senior 级别开发者的典型特征。

---

## 6. 第五部分：校验失败分层故障排查指南

> 以下按四大故障分类提供可落地的修复步骤。每类故障均标注典型现象、根因分析与分步修复方案。

### 6.1 故障分类一：目录文件放错

#### 典型现象

- Github 页面显示仓库根目录缺少 README.md
- Issue 模板在 Github 仓库页面不显示
- CI 工作流未自动触发
- `docs/` 目录下文件散落在根目录

#### 根因分析

| 错误类型 | 根因 | 影响 |
|---------|------|------|
| 文件放在错误层级 | 如 `README.md` 放在了 `src/README.md` 而非根目录 | Github 首页不显示 README |
| Issue 模板路径错误 | 如放在 `.github/bug_report.md` 而非 `.github/ISSUE_TEMPLATE/bug_report.md` | Github 不识别 Issue 模板 |
| CI 路径错误 | 如放在 `.github/workflows/ci.yaml` 但文件名与触发配置不匹配 | CI 不触发 |
| 目录结构混乱 | docs/assets/scripts 目录未按标准组织 | 仓库可读性差 |

#### 分步修复方案

1. **执行目录核对**：运行第二部分的一键核对命令，定位缺失或放错的文件
2. **使用 git mv 移动文件**（保留 Git 历史）：

   ```powershell
   # 示例：将放错的 README.md 移回根目录
   git mv src/README.md ./README.md

   # 示例：将 Issue 模板移到正确目录
   git mv .github/bug_report.md .github/ISSUE_TEMPLATE/bug_report.md
   ```

3. **创建缺失目录**：

   ```powershell
   New-Item -ItemType Directory -Force -Path ".github/ISSUE_TEMPLATE"
   New-Item -ItemType Directory -Force -Path ".github/workflows"
   New-Item -ItemType Directory -Force -Path "assets"
   ```

4. **重新核对**：再次执行一键核对命令，确认 12 份文件全部 `[OK]`
5. **提交修复**：`git add -A && git commit -m "fix: 修正 Github 配套文件目录结构"`

> **面试加分价值**：正确的目录结构是开源项目的"门面"。文件放对位置意味着开发者理解 Github 平台规范（Issue 模板路径、CI 工作流路径等），这是开源协作经验的基础证明。

### 6.2 故障分类二：代码运行报错

#### 典型现象

- `ModuleNotFoundError: No module named 'xxx'`
- `ImportError: cannot import name 'xxx' from 'xxx'`
- `FileNotFoundError: [Errno 2] No such file or directory: 'configs/xxx.yaml'`
- `KeyError: 'xxx'` 配置字段缺失
- `RuntimeError: shape mismatch` 张量维度不匹配

#### 根因分析

| 错误类型 | 根因 | 影响 |
|---------|------|------|
| 依赖缺失 | `requirements.txt` 未包含某依赖或版本不匹配 | 代码无法导入 |
| 导入路径错误 | 源码目录结构变更后 import 未同步更新 | 模块加载失败 |
| 配置文件路径错误 | 使用了硬编码相对路径但执行目录不对 | 文件找不到 |
| 配置字段缺失 | yaml 配置文件缺少必要字段 | 参数读取失败 |
| 张量维度不匹配 | ICM/RND 网络结构与动作空间维度不一致 | 前向传播报错 |

#### 分步修复方案

1. **依赖修复**：

   ```powershell
   # 重新安装全部依赖
   pip install -r requirements.txt

   # 若某依赖缺失，手动安装并更新 requirements.txt
   pip install <missing-package>
   pip freeze > requirements.txt
   ```

2. **导入路径修复**：
   - 在 `<项目根目录>` 下创建或检查 `setup.py` / `pyproject.toml`，确保包可被 `pip install -e .` 安装
   - 检查 `__init__.py` 文件是否存在于每个 Python 包目录
   - 确认 `tests/` 中的 import 使用包路径而非文件路径

3. **配置路径修复**：
   - 确保所有脚本在 `<项目根目录>` 下执行
   - 在脚本入口添加 `sys.path` 修正：

     ```python
     import os, sys
     PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
     sys.path.insert(0, PROJECT_ROOT)
     ```

4. **配置字段修复**：
   - 对比 `configs/` 下的 yaml 文件与代码中的 `config['xxx']` 读取
   - 确保所有必要字段都有默认值

5. **张量维度修复**：
   - 确认 ICM: 4 层 CNN → 288 维特征，动作空间 17 维
   - 确认 RND: 预测网络与目标网络输入维度一致
   - 确认 PPO: 策略网络输出维度 = 动作空间维度（17）

> **面试加分价值**：代码零报错运行是项目可复现性的基石。面试官 clone 后能直接跑通，意味着开发者已充分考虑了环境依赖与路径配置——这是"站在使用者角度思考"的工程同理心体现。

### 6.3 故障分类三：显存 OOM

#### 典型现象

- `RuntimeError: CUDA out of memory. Tried to allocate X MiB`
- 训练过程中显存持续增长不释放
- 显存峰值超过 2.2GB 基准线
- 多环境并行时显存溢出

#### 根因分析

| 错误类型 | 根因 | 影响 |
|---------|------|------|
| FP16 AMP 未生效 | `use_amp: false` 或 AMP 代码有误 | 显存占用翻倍 |
| 梯度累积配置错误 | 累积步数配置过小，等效 batch size 不足 | 显存峰值偏高 |
| CPU 缓存卸载未启用 | `cpu_offload: false` | 模型全部驻留 GPU |
| Episodic Memory 未限制 | LRU 容量过大或未启用 | 向量缓存膨胀 |
| 显存碎片化 | 长时间训练未清理缓存 | 显存碎片导致 OOM |
| 其他进程占用 | 浏览器/其他程序占用 GPU 显存 | 可用显存不足 |

#### 分步修复方案

1. **确认显存优化配置**：检查 `configs/crafter.yaml`（及其他环境配置）中以下字段：

   | 配置项 | 正确值 | 作用 |
   |--------|--------|------|
   | `use_amp` | `true` | 启用 FP16 混合精度训练 |
   | `gradient_accumulation_steps` | `4`（micro_batch=128，等效 512） | 减少单次显存占用 |
   | `cpu_offload` | `true` | 将不活跃参数卸载至 CPU |
   | `episodic_memory_size` | `200` | LRU 缓存上限 200 向量 |

2. **清理 GPU 缓存**：

   ```powershell
   # 检查 GPU 显存占用
   nvidia-smi

   # 在 Python 代码中定期清理缓存
   # torch.cuda.empty_cache()
   ```

3. **关闭其他 GPU 进程**：
   - 关闭浏览器硬件加速（Chrome/Edge 会占用 GPU 显存）
   - 关闭其他正在运行的训练/推理进程
   - 使用 `nvidia-smi` 确认 GPU 空闲

4. **降低 batch size**：
   - 若仍 OOM，将 micro_batch_size 从 128 降至 64
   - 相应增加 gradient_accumulation_steps 至 8（保持等效 512）

5. **强制 CPU 模式**（仅用于调试）：

   ```powershell
   # 调试时使用 CPU 模式排除 GPU 问题
   $env:CUDA_VISIBLE_DEVICES = "-1"
   python scripts/train.py --env crafter --steps 100 --device cpu
   ```

> **面试加分价值**：在 RTX3060 Laptop 6GB 显存上实现峰值仅 2.2GB 的显存优化（FP16 AMP + 梯度累积 + CPU 卸载 + LRU 缓存），本身就是面试中的技术亮点。能清晰说明每项优化的原理与效果，展示的是"在资源约束下解决工程问题"的实战能力。

### 6.4 故障分类四：本地绝对路径违规

#### 典型现象

- 面试官 clone 仓库后执行报错 `FileNotFoundError: 'D:\项目\xxx'`
- 代码中硬编码了 `C:\Users\jx\` 等个人路径
- 配置文件中包含 `/home/username/` 等 Linux 个人路径
- CI 中因路径不存在而失败

#### 根因分析

| 错误类型 | 根因 | 影响 |
|---------|------|------|
| 硬编码绝对路径 | 开发时为方便直接写死本地路径 | 跨机器无法运行 |
| 配置文件硬编码路径 | yaml/json 中写死本地数据目录 | 配置不可移植 |
| 日志中泄露路径 | 训练日志输出中包含本地绝对路径 | 暴露个人磁盘信息 |
| IDE 配置泄露 | `.idea/` 或 `.vscode/` 中包含本地路径 | 配置文件入库暴露路径 |

#### 分步修复方案

1. **全局检索定位**：执行 3.3 节的一键检索命令，列出所有匹配项及行号

2. **逐项替换修复**：

   ```python
   # 修复前（硬编码绝对路径）
   DATA_DIR = "D:\\项目\\data\\crafter"
   CHECKPOINT_DIR = "C:/Users/jx/checkpoints"

   # 修复后（相对路径 + 项目根目录）
   import os
   PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   DATA_DIR = os.path.join(PROJECT_ROOT, "data", "crafter")
   CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
   ```

3. **配置文件修复**：将 yaml 中的绝对路径改为相对路径

   ```yaml
   # 修复前
   data_dir: "D:/项目/data/crafter"

   # 修复后
   data_dir: "./data/crafter"
   ```

4. **环境变量注入**（可选，更灵活）：

   ```python
   # 通过环境变量覆盖，默认使用相对路径
   DATA_DIR = os.environ.get("CRAFTER_DATA_DIR", os.path.join(PROJECT_ROOT, "data", "crafter"))
   ```

5. **IDE 配置清理**：确认 `.idea/` 和 `.vscode/` 已被 `.gitignore` 拦截，若已入库则移除：

   ```powershell
   git rm -r --cached .idea/
   git rm -r --cached .vscode/
   git commit -m "fix: 移除 IDE 配置缓存，消除绝对路径泄露"
   ```

6. **重新检索确认**：再次执行 3.3 节检索命令，确认结果为空

7. **全流程复测**：执行 4.2 节 README 全流程复现校验，确认修复后代码可正常运行

> **面试加分价值**：零绝对路径是代码可移植性的底线。此项校验通过，意味着面试官在任何机器上 clone 后都能直接运行——这是"可复现性"承诺的技术兑现，也是消除"只能在作者机器上跑"质疑的最有力证据。

---

## 附录：校验执行总流程速查

> 按以下顺序依次执行，全部通过后方可执行 `git push`。

```
步骤1 → 第二部分：12 份配套文件目录核对（一键核对命令）
         ↓ 全部 [OK]
步骤2 → 3.1 .gitignore 屏蔽规则校验（一键检索命令）
         ↓ 规则完整，暂存区无违规
步骤3 → 3.2 大体积冗余文件清理（一键清理命令）
         ↓ 仓库体积 <50MB
步骤4 → 3.3 全局硬编码绝对路径检索（一键检索命令）
         ↓ 检索结果为空 [RED-02 通过]
步骤5 → 4.1 单元测试校验（pytest 144 项）
         ↓ 144 passed [RED-01 通过]
步骤6 → 4.2 README 全流程复现校验（训练→消融→评测→ONNX）
         ↓ ICM 逆损失≈2.83，内在奖励≈60 [RED-04 通过]
步骤7 → 4.3 Mermaid 架构图渲染校验（mermaid.live）
         ↓ 无语法报错，数据流完整
步骤8 → 4.4 Github Actions CI 本地模拟校验
         ↓ YAML 合法，测试可启动
步骤9 → 第四部分：24 项分层自检勾选清单
         ↓ 24 项全部勾选
步骤10 → 执行 git push（所有红线通过）
```

---

> **最终声明**：本手册的每一项校验均直接映射面试场景中的可信度构建。严格执行本手册不仅是技术合规要求，更是向面试官展示"工程化质量管控意识"的核心手段。**校验全通过的项目，才是值得上传的项目。**

*文档版本：v1.0 | 对应 Git Tag：v1.0 | 最后更新：2026-07-14*
