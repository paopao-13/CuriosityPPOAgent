===== FILE: ./docs/pre_git_push_local_check.md =====

# CuriosityPPOAgent 推送前本地全量实操校验手册

> **项目名称**：CuriosityPPOAgent ICM+RND 分层新颖信号融合好奇心驱动 PPO 智能体
>
> **适用版本**：v1.0（12 次 Git 迭代 v0.1 ~ v1.0）
>
> **硬件环境**：AMD Ryzen 7 6800H + NVIDIA RTX 3060 Laptop 6GB（显存峰值 2.2GB）

---

## 前置说明

### 校验目的

本手册用于在执行 `git push` 到 GitHub 远程仓库之前，对本地工程进行系统化、全量化的实操校验，确保推送内容无缺陷、可复现、合规范。核心目标如下：

1. **规避线上仓库代码运行报错**：推送前在本地完整执行训练、消融、评测、ONNX 导出等全套命令链，拦截一切运行时异常，确保面试官克隆仓库后可直接复现，不会遇到 ImportError、ModuleNotFoundError、FileNotFoundError 等低级错误。

2. **降低面试官负面观感**：线上仓库是求职者的技术名片。若推送后出现残留的本地绝对路径（`D:\xxx`）、大文件未屏蔽导致 clone 缓慢、CI 红叉不灭、README 命令不可复现等问题，将严重损害技术可信度。本手册通过 24 项分层自检将此类风险降至零。

3. **防止复现失败质疑**：本项目核心卖点为 Crafter 100 万步 19.0%（基线 15.6%）、Atari Montezuma 3500+（基线 120）、MiniGrid 96.8 万步（基线 242 万，2.5x 加速）等量化指标。若面试官按 README 逐步复现时因代码缺陷导致指标偏差，将直接引发数据造假质疑。本手册通过量化数据统一校验确保所有文档中的数字一致且可追溯。

> **面试加分点**：展示对工程质量的系统性把控能力——不依赖线上 CI 事后发现，而是在推送前完成本地全量自检。这体现了"提交即合格"的工程纪律，是中高级开发者的核心素养。

### 硬性规则

| 规则编号 | 规则内容 | 约束级别 |
|:---:|---|:---:|
| R-01 | **校验不通过，禁止推送。** 任何一项自检未通过，必须修复后重新校验，不允许"先推后修" | 强制 |
| R-02 | 24 项分层自检清单必须逐项勾选，全部通过后方可执行 `git push` | 强制 |
| R-03 | 校验过程中发现的问题及修复措施须记录至提交日志，保证可追溯 | 强制 |
| R-04 | 本手册须在每次重大推送前完整执行一遍，不可跳项 | 强制 |

### 面试加分意义

| 维度 | 加分体现 |
|---|---|
| 工程纪律 | 推送前全量自检 = 零缺陷交付意识，区别于"推上去看 CI 报错再修"的初级习惯 |
| 复现严谨性 | 训练→消融→评测→ONNX 全链路本地跑通 = 数据可信、可复现，杜绝造假质疑 |
| 开源规范 | 12 份配套文件齐备 + MIT 协议 + CI/CD + Issue 模板 = 完整开源工程素养 |
| 细节把控 | 绝对路径扫描、大文件清理、.gitignore 验证 = 对线上仓库整洁度的极致追求 |
| 量化思维 | 144 用例 100% 通过 + 指标数据跨文档统一校验 = 用数据说话的工程实证能力 |

> **面试加分点**：本手册本身就是面试时展示工程严谨性的有力证据——"我在推送前用 24 项自检清单逐项验证，确保线上零缺陷"远胜于口头承诺"我代码质量很高"。

---

## 第一部分：12 份仓库配套文件目录核对总表

在推送前，须确认以下 12 份 GitHub 配套文件均已存在且内容完整。使用下方 PowerShell 命令一键核验。

### 一键核验命令

```powershell
# 进入项目根目录
cd <项目根目录>

# 定义期望文件清单
$expectedFiles = @(
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

Write-Host "===== 仓库配套文件目录核对 =====" -ForegroundColor Cyan
$allExist = $true
foreach ($f in $expectedFiles) {
    if (Test-Path $f) {
        $size = (Get-Item $f).Length
        Write-Host "[OK]   $f  ($size bytes)" -ForegroundColor Green
    } else {
        Write-Host "[MISS] $f" -ForegroundColor Red
        $allExist = $false
    }
}
Write-Host "-----------------------------------"
if ($allExist) {
    Write-Host "结果：12/12 文件全部存在，核对通过" -ForegroundColor Green
} else {
    Write-Host "结果：存在缺失文件，请补全后重新校验" -ForegroundColor Red
}
```

### 文件目录核对总表

| 序号 | 文件名称 | 存放路径 | 合格标准 |
|:---:|---|---|---|
| 1 | README.md | `./README.md` | 项目根目录；包含项目简介、安装步骤、训练/评测/ONNX 导出命令、核心指标（Crafter 19.0%、Atari 3500+、MiniGrid 96.8 万步）、Mermaid 架构图、MIT 协议声明 |
| 2 | LICENSE | `./LICENSE` | 项目根目录；标准 MIT 协议全文，版权年份及持有者信息正确 |
| 3 | .gitignore | `./.gitignore` | 项目根目录；包含 Python、IDE、权重文件（*.pt/*.pth）、wandb 日志、视频输出、__pycache__、虚拟环境等屏蔽规则 |
| 4 | CONTRIBUTING.md | `./CONTRIBUTING.md` | 项目根目录；包含开发环境配置、代码规范、提交规范、分支策略、PR 流程说明 |
| 5 | bug_report.md | `./.github/ISSUE_TEMPLATE/bug_report.md` | GitHub Issue 模板目录；包含 Bug 描述、复现步骤、期望行为、实际行为、环境信息（OS/GPU/Python 版本）字段 |
| 6 | feature_request.md | `./.github/ISSUE_TEMPLATE/feature_request.md` | GitHub Issue 模板目录；包含功能描述、使用场景、期望方案、替代方案字段 |
| 7 | test.yml | `./.github/workflows/test.yml` | GitHub Actions 工作流目录；触发条件为 push/PR，包含 Python 环境安装、依赖安装、pytest 执行步骤 |
| 8 | release_note.md | `./docs/release_note.md` | 文档目录；包含 v0.1~v1.0 共 12 个版本迭代记录，每个版本有变更摘要和新增功能 |
| 9 | repo_tags.md | `./docs/repo_tags.md` | 文档目录；包含仓库标签列表（如 reinforcement-learning、ppo、curiosity-driven、icm、rnd、crafter、atari 等） |
| 10 | repo_cover_desc.txt | `./assets/repo_cover_desc.txt` | 资源目录；包含仓库封面图描述文本，可用于社交分享或仓库简介卡片 |
| 11 | git_upload_step.md | `./docs/git_upload_step.md` | 文档目录；包含 Git 初始化、远程仓库关联、分支管理、推送操作的完整步骤指南 |
| 12 | repo_self_checklist.md | `./docs/repo_self_checklist.md` | 文档目录；包含仓库自检清单，覆盖文件完整性、代码质量、文档一致性等检查项 |

> **面试加分点**：12 份配套文件覆盖了开源项目的全部标准组件（README、LICENSE、.gitignore、CONTRIBUTING、Issue 模板、CI/CD、发布说明、标签、封面、推送指南、自检清单），展示了完整的开源工程治理体系，远超多数求职者的"只有一个 README"水平。

---

## 第二部分：本地工程清理流程

### 2.1 .gitignore 屏蔽校验

推送前须确认 .gitignore 规则有效拦截了所有不应进入版本库的文件类别。

#### .gitignore 屏蔽规则检索命令

```powershell
# 进入项目根目录
cd <项目根目录>

Write-Host "===== .gitignore 屏蔽规则校验 =====" -ForegroundColor Cyan

# 检查 .gitignore 文件是否存在
if (-not (Test-Path ".gitignore")) {
    Write-Host "[FAIL] .gitignore 文件不存在！" -ForegroundColor Red
    exit 1
}

Write-Host "`n--- 读取 .gitignore 内容 ---" -ForegroundColor Yellow
Get-Content .gitignore

Write-Host "`n--- 必须包含的屏蔽规则校验 ---" -ForegroundColor Yellow

$requiredPatterns = @{
    "权重文件 (*.pt)"       = "\*\.pt"
    "权重文件 (*.pth)"      = "\*\.pth"
    "wandb 日志"            = "wandb"
    "视频输出 (*.mp4)"      = "\*\.mp4"
    "Python 缓存"           = "__pycache__"
    "IDE 配置 (.idea)"      = "\.idea"
    "IDE 配置 (.vscode)"    = "\.vscode"
    "虚拟环境"              = "venv|\.venv|env"
    "ONNX 模型 (*.onnx)"    = "\*\.onnx"
    "日志文件 (*.log)"      = "\*\.log"
}

$gitignoreContent = Get-Content .gitignore -Raw
$allPatternsOk = $true

foreach ($name in $requiredPatterns.Keys) {
    $pattern = $requiredPatterns[$name]
    if ($gitignoreContent -match $pattern) {
        Write-Host "[OK]   $name - 规则存在" -ForegroundColor Green
    } else {
        Write-Host "[MISS] $name - 规则缺失，请添加" -ForegroundColor Red
        $allPatternsOk = $false
    }
}

Write-Host "`n--- 实际文件拦截验证 ---" -ForegroundColor Yellow
# 验证是否有应被屏蔽但已被 Git 跟踪的文件
$trackedFiles = git ls-files
$sensitivePatterns = @("*.pt", "*.pth", "*.mp4", "*.onnx", "*.log", "wandb/", "__pycache__")
$violations = @()

foreach ($f in $trackedFiles) {
    foreach ($sp in $sensitivePatterns) {
        $spClean = $sp -replace "[\*/]", ""
        if ($f -like "*$spClean*") {
            $violations += $f
        }
    }
}

if ($violations.Count -eq 0) {
    Write-Host "[OK]   无敏感文件被 Git 跟踪" -ForegroundColor Green
} else {
    Write-Host "[FAIL] 以下敏感文件已被 Git 跟踪，需执行 git rm --cached 移除：" -ForegroundColor Red
    foreach ($v in $violations) {
        Write-Host "       - $v" -ForegroundColor Red
    }
}

Write-Host "`n--- Git 仓库状态 ---" -ForegroundColor Yellow
$status = git status --porcelain
$ignoredCheck = git status --ignored --porcelain
Write-Host "未跟踪/已修改文件数: $($status.Count)"
Write-Host "`n===== .gitignore 校验完成 =====" -ForegroundColor Cyan
```

**合格标准**：
- .gitignore 文件存在且包含全部 10 类必须屏蔽规则
- `git ls-files` 输出中不包含任何 .pt/.pth/.mp4/.onnx/.log/wandb/__pycache__ 文件
- 若发现已跟踪的敏感文件，执行 `git rm --cached <文件路径>` 移除后重新校验

> **面试加分点**：.gitignore 屏蔽校验体现了对版本库洁净度的严格管控——权重文件（可达数百 MB）误推会导致 clone 耗时剧增，__pycache__ 误推暴露本地 Python 版本信息，这些细节直接反映工程素养。

### 2.2 大文件清理清单

推送前须确认仓库中不存在超过 10MB 的大文件（文档、配置、源码除外）。

#### 大文件扫描命令

```powershell
# 进入项目根目录
cd <项目根目录>

Write-Host "===== 大文件清理扫描 =====" -ForegroundColor Cyan
Write-Host "扫描 Git 已跟踪文件中超过 1MB 的文件...`n" -ForegroundColor Yellow

# 列出已跟踪文件中超过 1MB 的文件
$largeFiles = git ls-files | ForEach-Object {
    $filePath = $_
    if (Test-Path $filePath) {
        $sizeMB = [math]::Round((Get-Item $filePath).Length / 1MB, 2)
        if ($sizeMB -ge 1.0) {
            [PSCustomObject]@{
                File   = $filePath
                SizeMB = $sizeMB
            }
        }
    }
}

if ($largeFiles) {
    Write-Host "发现大文件清单：" -ForegroundColor Red
    $largeFiles | Sort-Object SizeMB -Descending | Format-Table -AutoSize
    Write-Host "[FAIL] 存在超过 1MB 的已跟踪文件，请确认是否必要" -ForegroundColor Red
} else {
    Write-Host "[OK] 未发现超过 1MB 的已跟踪文件" -ForegroundColor Green
}

# 扫描工作区中未被 Git 跟踪的大文件（应被 .gitignore 拦截）
Write-Host "`n--- 扫描工作区大文件（未被 Git 跟踪的）---" -ForegroundColor Yellow
$allLargeFiles = Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Length -ge 10MB } |
    Select-Object FullName, @{N='SizeMB';E={[math]::Round($_.Length/1MB,2)}}

if ($allLargeFiles) {
    Write-Host "工作区大文件（确认已被 .gitignore 拦截）：" -ForegroundColor Yellow
    $allLargeFiles | Format-Table -AutoSize
} else {
    Write-Host "[OK] 工作区无超过 10MB 的大文件" -ForegroundColor Green
}
```

#### 大文件清理清单表

| 类别 | 典型文件/目录 | 处理方式 | 合格标准 |
|---|---|---|---|
| 模型权重 | `*.pt`、`*.pth`、`checkpoints/` | 添加至 .gitignore；已跟踪的执行 `git rm --cached` | Git 不跟踪任何权重文件 |
| 训练日志 | `wandb/`、`runs/`、`*.log` | 添加至 .gitignore | Git 不跟踪任何日志目录/文件 |
| 视频输出 | `*.mp4`、`*.avi`、`videos/` | 添加至 .gitignore | Git 不跟踪任何视频文件 |
| IDE 缓存 | `.idea/`、`.vscode/`（可选保留配置）、`*.code-workspace` | 添加至 .gitignore | Git 不跟踪 IDE 临时缓存 |
| 依赖目录 | `node_modules/`、`__pycache__/`、`*.egg-info/` | 添加至 .gitignore | Git 不跟踪任何依赖/缓存目录 |
| 数据集 | `*.hdf5`、`*.npz`、`datasets/` | 添加至 .gitignore；README 说明下载方式 | Git 不跟踪原始数据集 |
| ONNX 模型 | `*.onnx`、`onnx_models/` | 添加至 .gitignore | Git 不跟踪导出模型（提供导出脚本即可） |

> **面试加分点**：大文件清理清单展示了"仓库只存代码和配置，不存产物"的最佳实践。权重文件动辄数百 MB，若误推到 GitHub 会导致仓库臃肿、clone 慢、超过文件大小限制（GitHub 单文件 100MB 硬限制）。这份清单证明你对 Git 版本管理的边界有清晰认知。

### 2.3 全局硬编码本地路径一键扫描

推送前须确认仓库中不存在任何硬编码的本地绝对路径（如 `D:\`、`C:\Users\`、`D:/简历/` 等），这些路径在其他机器上会导致 FileNotFoundError。

#### 本地路径一键扫描命令

```powershell
# 进入项目根目录
cd <项目根目录>

Write-Host "===== 硬编码本地绝对路径扫描 =====" -ForegroundColor Cyan
Write-Host "扫描所有已跟踪的 Python/YAML/Markdown 文件...`n" -ForegroundColor Yellow

# 定义本地路径正则模式（匹配 Windows 绝对路径）
$pathPatterns = @(
    '[A-Z]:\\',           # Windows 反斜杠路径如 D:\
    '[A-Z]:/',            # Windows 正斜杠路径如 D:/
    '\\Users\\',          # Users 目录路径
    '/home/',             # Linux 用户目录路径
    '/Users/'             # macOS 用户目录路径
)

# 获取所有已跟踪的文本文件
$trackedTextFiles = git ls-files | Where-Object {
    $_ -match '\.(py|yaml|yml|md|txt|json|cfg|ini|toml|sh|bat)$'
}

$violations = @()

foreach ($file in $trackedTextFiles) {
    if (Test-Path $file) {
        $content = Get-Content $file -Raw -ErrorAction SilentlyContinue
        if ($content) {
            $lineNum = 0
            foreach ($line in ($content -split "`n")) {
                $lineNum++
                foreach ($pattern in $pathPatterns) {
                    if ($line -match $pattern) {
                        # 排除注释中的路径示例（如 # 路径示例：D:\xxx）
                        $trimmedLine = $line.Trim()
                        if ($trimmedLine -notmatch '^\s*#' -and $trimmedLine -notmatch '^\s*<!--' -and $trimmedLine -notmatch '示例|example|占位|placeholder') {
                            $violations += [PSCustomObject]@{
                                File    = $file
                                Line    = $lineNum
                                Content = $line.Trim()
                            }
                        }
                    }
                }
            }
        }
    }
}

if ($violations.Count -eq 0) {
    Write-Host "[OK] 未发现硬编码本地绝对路径" -ForegroundColor Green
} else {
    Write-Host "[FAIL] 发现 $($violations.Count) 处硬编码本地绝对路径：" -ForegroundColor Red
    $violations | Format-Table -AutoSize -Wrap
    Write-Host "`n修复建议：" -ForegroundColor Yellow
    Write-Host "  1. 将硬编码路径替换为相对路径（如 ./configs/、../data/）"
    Write-Host "  2. 或使用环境变量/配置文件管理路径"
    Write-Host "  3. 或使用 <项目根目录> 占位符"
}

Write-Host "`n===== 本地路径扫描完成 =====" -ForegroundColor Cyan
```

**合格标准**：扫描结果为"未发现硬编码本地绝对路径"，0 处匹配。若发现匹配，须逐条替换为相对路径或配置文件管理后重新扫描。

> **面试加分点**：硬编码本地路径是开源项目最常见的低级错误之一——面试官克隆仓库后因 `D:\简历\curiosity-ppo\data\` 路径不存在而直接报错，会立刻对代码质量产生负面判断。一键扫描命令展示了你对这个问题的系统性防范意识。

---

## 第三部分：四大核心功能校验

### 3.1 单元测试全量运行（144 用例零报错）

#### 执行命令

```powershell
# 进入项目根目录
cd <项目根目录>

# 激活虚拟环境（根据实际环境名调整）
# conda activate curiosity-ppo
# 或
# .\venv\Scripts\Activate.ps1

# 全量运行单元测试，输出详细日志
python -m pytest tests/ -v --tb=long --html=tests/test_report.html --self-contained-html 2>&1 | Tee-Object -FilePath tests/pytest_full_log.txt
```

#### 操作步骤

| 步骤 | 操作 | 说明 |
|:---:|---|---|
| 1 | 进入项目根目录 | `cd <项目根目录>` |
| 2 | 激活虚拟环境 | 确保已安装 `requirements.txt` 中的全部依赖 |
| 3 | 执行 pytest 全量运行 | 使用 `-v` 参数输出每个用例的详细结果 |
| 4 | 检查测试通过率 | 关注终端输出末尾的 `passed`/`failed` 统计 |
| 5 | 查看测试覆盖率（可选） | `python -m pytest tests/ --cov=src --cov-report=term-missing` |
| 6 | 保存测试报告 | HTML 报告和文本日志均保存至 `tests/` 目录（已被 .gitignore 屏蔽） |

#### 合格判定

| 判定项 | 合格标准 | 判定方式 |
|---|---|---|
| 用例总数 | 144 | 终端输出 `=== 144 passed ===` 或 `collected 144 items` |
| 通过率 | 100%（144/144 passed） | 终端输出 `0 failed` |
| 错误数 | 0 errors | 终端输出无 `error` 标记 |
| 警告数 | 允许 DeprecationWarning，但无 RuntimeError/AssertionError | 查看 `--tb=long` 回溯信息 |
| 测试报告 | HTML 报告成功生成 | `tests/test_report.html` 文件存在且可打开 |

#### 故障排查

| 故障现象 | 可能原因 | 修复方案 |
|---|---|---|
| `ModuleNotFoundError: No module named 'xxx'` | 虚拟环境未激活或依赖未安装 | 激活虚拟环境后执行 `pip install -r requirements.txt` |
| `FileNotFoundError: configs/xxx.yaml` | 工作目录不在项目根目录 | 确保在项目根目录执行 pytest |
| `RuntimeError: CUDA out of memory` | 显存不足（RTX 3060 6GB） | 确认测试用例使用 mock 数据而非真实 GPU 推理；检查是否有未释放的 CUDA 张量 |
| `AssertionError` 在 ICM 损失测试中 | ICM 逆损失初始化值偏差 | 确认 `ln(17) ≈ 2.833` 作为初始损失期望值；检查动作空间维度是否为 17 |
| 部分用例 `failed` 但非代码问题 | 测试数据缓存过期或随机种子不一致 | 在 `conftest.py` 中固定随机种子 `torch.manual_seed(42)`，清理缓存后重试 |

> **面试加分点**：144 项单元测试 100% 通过是本项目最硬核的量化交付证据。推送前在本地完整跑通，意味着面试官克隆后执行 `pytest` 即可复现 144 passed 的结果，这是"代码真的能跑"的最直接证明。

### 3.2 README 全套复现命令测试

本环节须按 README 中记录的命令链逐步执行，验证训练→消融→评测→ONNX 导出全链路可在本地完整跑通，并校验关键指标值。

#### 执行命令

```powershell
# 进入项目根目录
cd <项目根目录>

# ========== 步骤 1: 训练命令（短步数冒烟测试）==========
# 使用小步数快速验证训练流程可跑通（非完整训练）
python scripts/train.py --config configs/crafter.yaml --steps 10000 --device cuda --amp --grad-accum 4

# ========== 步骤 2: 消融实验命令（4 组）==========
python scripts/train.py --config configs/crafter.yaml --ablation full --steps 5000 --device cuda
python scripts/train.py --config configs/crafter.yaml --ablation no_icm --steps 5000 --device cuda
python scripts/train.py --config configs/crafter.yaml --ablation no_episodic --steps 5000 --device cuda
python scripts/train.py --config configs/crafter.yaml --ablation no_rnd --steps 5000 --device cuda

# ========== 步骤 3: 评测命令 ==========
python scripts/evaluate.py --config configs/crafter.yaml --checkpoint checkpoints/best.pt --episodes 10

# ========== 步骤 4: ONNX 导出命令 ==========
python scripts/export_onnx.py --config configs/crafter.yaml --checkpoint checkpoints/best.pt --output exports/model.onnx
```

#### 操作步骤

| 步骤 | 操作 | 验证重点 |
|:---:|---|---|
| 1 | 执行训练命令（冒烟测试，10000 步） | 训练流程无报错，日志正常输出 loss/reward |
| 2 | 检查 ICM 逆损失初始值 | 初始 loss ≈ 2.83（= ln(17)，17 维动作空间） |
| 3 | 检查内在奖励值 | 内在奖励 r_int ≈ 60（训练初期范围） |
| 4 | 执行 4 组消融实验 | full/no_icm/no_episodic/no_rnd 均可正常启动并运行 |
| 5 | 执行评测命令 | 评测脚本可加载 checkpoint 并输出指标 |
| 6 | 执行 ONNX 导出命令 | 成功生成 .onnx 文件，可用 `onnxruntime` 加载验证 |
| 7 | 校验显存使用 | 训练期间显存峰值 ≤ 2.2GB（RTX 3060 6GB 上限内） |

#### 合格判定

| 判定项 | 合格标准 | 判定方式 |
|---|---|---|
| 训练命令 | `python scripts/train.py` 正常执行无报错 | 终端无 Traceback，正常输出训练日志 |
| ICM 逆损失 | 初始 loss ≈ 2.83（ln(17)） | 查看训练日志第一轮 `icm_loss` 字段 |
| 内在奖励 | r_int ≈ 60（训练初期） | 查看训练日志 `intrinsic_reward` 字段 |
| 消融实验 | 4 组（full/no_icm/no_episodic/no_rnd）均可启动 | 每组运行至少 1000 步无报错 |
| 评测命令 | `python scripts/evaluate.py` 正常执行 | 输出评测指标（reward/success_rate） |
| ONNX 导出 | `python scripts/export_onnx.py` 正常执行 | `exports/model.onnx` 文件生成且大小 > 0 |
| 显存峰值 | ≤ 2.2GB | 训练期间通过 `nvidia-smi` 监控 |
| FP16 AMP | 正常启用无精度异常 | 训练日志中出现 `AMP enabled` 标记 |
| 梯度累积 | 128×4 等效 512 正常工作 | 训练日志中出现 `grad_accum=4` 标记 |

#### 关键指标校验命令

```powershell
# 校验 ICM 逆损失初始值（期望 ≈ 2.83 = ln(17)）
python -c "import math; print(f'ln(17) = {math.log(17):.4f}')"
# 预期输出：ln(17) = 2.8332

# 验证 ONNX 模型可加载
python -c "
import onnxruntime as ort
import numpy as np
session = ort.InferenceSession('exports/model.onnx')
inputs = {inp.name: np.random.randn(1, 4, 64, 64).astype(np.float32) for inp in session.get_inputs()}
outputs = session.run(None, inputs)
print(f'ONNX 模型加载成功，输出形状: {[o.shape for o in outputs]}')
"

# 监控显存（训练期间另开终端执行）
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 5
```

#### 故障排查

| 故障现象 | 可能原因 | 修复方案 |
|---|---|---|
| `CUDA out of memory` | FP16 AMP 未启用或 batch_size 过大 | 确认 `--amp` 参数已传入；减小 batch_size 或增加 `--grad-accum` |
| ICM 损失初始值偏离 2.83 | 动作空间维度配置错误 | 检查 `configs/crafter.yaml` 中 `action_dim` 是否为 17 |
| 内在奖励异常（0 或极大值） | 融合公式参数 η/α 配置错误 | 检查配置中 `eta`（ICM 权重系数）和 `alpha`（episodic 系数）取值 |
| 消融实验 `no_icm` 模式报错 | 代码未正确处理 ICM 模块缺失 | 检查 `src/agent.py` 中 ablation 模式的条件分支逻辑 |
| ONNX 导出失败 | 模型包含动态控制流或不支持的算子 | 检查模型 forward 方法中是否有 if/for 条件分支；使用 `torch.onnx.export` 的 `dynamic_axes` 参数 |
| 评测时 checkpoint 加载失败 | checkpoint 路径错误或 state_dict 键不匹配 | 确认 `--checkpoint` 路径正确；检查模型结构是否与训练时一致 |

> **面试加分点**：README 全套复现命令本地跑通是防止"复现失败质疑"的核心防线。ICM 逆损失 ≈ 2.83（= ln(17)）这一数学推导值的验证，展示了你对好奇心驱动机制底层原理的深入理解——初始时模型对 17 维动作完全不确定，逆损失等于均匀分布的熵 ln(17)，这是 ICM 理论的直接体现。

### 3.3 Mermaid 架构图在线渲染校验

README 中的 Mermaid 架构图须在 mermaid.live 在线编辑器中正确渲染，确保面试官在 GitHub 上查看时图形显示正常。

#### 执行步骤

| 步骤 | 操作 | 说明 |
|:---:|---|---|
| 1 | 从 README.md 中提取 Mermaid 代码块 | 定位 ` ```mermaid ` 和 ` ``` ` 之间的内容 |
| 2 | 打开浏览器访问 mermaid.live | 在线 Mermaid 渲染编辑器 |
| 3 | 将 Mermaid 代码粘贴至左侧编辑区 | 确保完整复制，无截断 |
| 4 | 查看右侧渲染结果 | 图形应完整显示所有节点和连线 |
| 5 | 检查 GitHub 预览效果 | 在 GitHub 仓库页面查看 README.md 渲染效果 |

#### 合格判定

| 判定项 | 合格标准 | 判定方式 |
|---|---|---|
| 语法正确性 | mermaid.live 无语法错误提示 | 编辑器右侧无红色错误标记 |
| 节点完整性 | 所有模块节点（ICM/RND/Episodic Memory/PPO/融合层）均渲染 | 逐一核对图中节点是否与设计一致 |
| 连线正确性 | 数据流向连线正确（如 ICM 前向损失→融合层→内在奖励→PPO） | 核对箭头方向和标签 |
| GitHub 渲染 | GitHub 仓库页面 README 中 Mermaid 图正常显示 | 推送后在 GitHub 页面确认 |
| 文字可读性 | 节点标签文字无截断、无乱码 | 检查中文标签是否正常显示 |

#### 故障排查

| 故障现象 | 可能原因 | 修复方案 |
|---|---|---|
| mermaid.live 报 `Parse error` | Mermaid 语法版本不兼容 | 检查是否使用了旧版语法；统一使用 mermaid v9+ 语法 |
| 图形节点缺失 | 代码块被截断或不完整 | 重新从 README 复制完整 Mermaid 代码 |
| GitHub 不渲染 Mermaid | 代码块语言标记错误 | 确认使用 ` ```mermaid ` 而非 ` ```mermaid.js ` |
| 中文标签乱码 | 编码问题或特殊字符冲突 | 将含特殊字符的标签用引号包裹，如 `["中文标签"]` |
| 图形过于拥挤 | 节点过多导致布局混乱 | 使用 `subgraph` 分组，优化布局方向（LR/TD） |

> **面试加分点**：Mermaid 架构图在线渲染校验体现了对文档可视化质量的追求。一张清晰、正确渲染的架构图能让面试官在 10 秒内理解项目的模块组成和数据流向，远胜于纯文字描述。确保 GitHub 渲染无误说明你关注最终交付效果。

### 3.4 GitHub Actions CI 脚本本地语法校验

推送前须在本地验证 `.github/workflows/test.yml` 的语法正确性和逻辑完整性，避免推送后 CI 红叉。

#### 执行命令

```powershell
# 进入项目根目录
cd <项目根目录>

# ========== 步骤 1: YAML 语法校验 ==========
python -c "
import yaml, sys
try:
    with open('.github/workflows/test.yml', 'r', encoding='utf-8') as f:
        workflow = yaml.safe_load(f)
    print('[OK] YAML 语法校验通过')
    print(f'  工作流名称: {workflow.get(\"name\", \"未设置\")}')
    print(f'  触发条件: {list(workflow.get(\"on\", {}).keys()) if isinstance(workflow.get(\"on\"), dict) else workflow.get(\"on\")}')
    jobs = workflow.get('jobs', {})
    print(f'  Job 数量: {len(jobs)}')
    for job_name, job_config in jobs.items():
        steps = job_config.get('steps', [])
        print(f'  Job [{job_name}] 步骤数: {len(steps)}')
        for i, step in enumerate(steps):
            name = step.get('name', step.get('uses', step.get('run', '未命名')))
            print(f'    Step {i+1}: {name}')
except yaml.YAMLError as e:
    print(f'[FAIL] YAML 语法错误: {e}')
    sys.exit(1)
except FileNotFoundError:
    print('[FAIL] test.yml 文件不存在')
    sys.exit(1)
"

# ========== 步骤 2: 必须包含的步骤校验 ==========
python -c "
import yaml
with open('.github/workflows/test.yml', 'r', encoding='utf-8') as f:
    workflow = yaml.safe_load(f)

requiredSteps = [
    'actions/checkout',      # 代码检出
    'actions/setup-python',  # Python 环境安装
    'pip install',            # 依赖安装
    'pytest'                  # 测试执行
]

allSteps = []
for job in workflow.get('jobs', {}).values():
    for step in job.get('steps', []):
        allSteps.append(str(step.get('uses', '')) + str(step.get('run', '')))
allText = ' '.join(allSteps)

print('===== 必须步骤校验 =====')
allOk = True
for req in requiredSteps:
    if req in allText:
        print(f'[OK]   {req}')
    else:
        print(f'[MISS] {req}')
        allOk = False
if allOk:
    print('结果：全部必须步骤校验通过')
else:
    print('结果：存在缺失步骤，请补全')
"

# ========== 步骤 3: 本地模拟 CI 测试（可选）==========
# 使用 act 工具在本地运行 GitHub Actions（需安装 act）
# act -W .github/workflows/test.yml
```

#### 操作步骤

| 步骤 | 操作 | 说明 |
|:---:|---|---|
| 1 | YAML 语法校验 | 使用 Python yaml 库解析 test.yml，确认无语法错误 |
| 2 | 工作流结构校验 | 确认包含 name、on（触发条件）、jobs（至少一个 job） |
| 3 | 必须步骤校验 | 确认包含 checkout、setup-python、pip install、pytest 四个核心步骤 |
| 4 | Python 版本校验 | 确认 setup-python 中指定的版本与项目兼容（推荐 3.9+） |
| 5 | 本地模拟运行（可选） | 安装 `act` 工具后执行 `act -W .github/workflows/test.yml` |
| 6 | 确认 pytest 命令一致性 | CI 中的 pytest 命令须与本地测试命令一致 |

#### 合格判定

| 判定项 | 合格标准 | 判定方式 |
|---|---|---|
| YAML 语法 | `yaml.safe_load()` 无异常 | Python 脚本输出 `[OK] YAML 语法校验通过` |
| 工作流名称 | name 字段已设置 | 脚本输出工作流名称 |
| 触发条件 | 包含 push 和 pull_request | 脚本输出触发条件列表 |
| Job 数量 | ≥ 1 | 脚本输出 Job 数量 |
| 核心步骤 | 4 项必须步骤全部存在 | checkout + setup-python + pip install + pytest |
| Python 版本 | 3.9 或以上 | setup-python 的 python-version 字段 |
| pytest 命令 | 与本地一致 | CI 中 pytest 命令路径与参数和本地匹配 |

#### 故障排查

| 故障现象 | 可能原因 | 修复方案 |
|---|---|---|
| `yaml.YAMLError` | YAML 缩进或语法错误 | 检查缩进（YAML 不允许 Tab）；使用在线 YAML 校验器排查 |
| `actions/checkout` 缺失 | 未配置代码检出步骤 | 在 steps 开头添加 `- uses: actions/checkout@v4` |
| `setup-python` 版本过旧 | 指定了不支持的 Python 版本 | 更新为 `python-version: '3.10'` 或更高 |
| pip install 步骤缺失 | 未安装项目依赖 | 添加 `- run: pip install -r requirements.txt` |
| pytest 步骤缺失 | 未配置测试执行 | 添加 `- run: python -m pytest tests/ -v` |
| CI 中 pytest 失败但本地通过 | CI 环境缺少系统依赖或数据文件 | 确认 CI 中包含数据下载步骤；或确认测试不依赖外部数据文件 |

> **面试加分点**：CI 脚本本地语法校验体现了"推送前消灭红叉"的前置意识。多数求职者推送后才依赖 CI 发现问题，而你在本地就能确保 CI 通过，这展示了 DevOps 思维和对持续集成的深入理解。

---

## 第四部分：24 项分层自检勾选清单

以下 24 项自检清单分为 4 大类，每类 6 项。推送前须逐项执行验证命令并勾选，全部 24 项通过后方可推送。

### 第一类：文件合规校验（6 项）

| 序号 | 自检项 | 验证命令 | 合格标准 | 勾选 |
|:---:|---|---|---|:---:|
| F-01 | 12 份配套文件全部存在 | `git ls-files \| Select-String "README\|LICENSE\|\.gitignore\|CONTRIBUTING\|bug_report\|feature_request\|test\.yml\|release_note\|repo_tags\|repo_cover_desc\|git_upload_step\|repo_self_checklist"` | 12 个文件全部匹配 | [ ] |
| F-02 | 无敏感文件被 Git 跟踪 | `git ls-files \| Select-String "\.pt$\|\.pth$\|\.mp4$\|\.onnx$\|\.log$"` | 无匹配输出 | [ ] |
| F-03 | 无硬编码本地绝对路径 | 执行 2.3 节扫描命令 | 0 处匹配 | [ ] |
| F-04 | .gitignore 包含全部屏蔽规则 | 执行 2.1 节校验命令 | 10 类规则全部存在 | [ ] |
| F-05 | 无超过 10MB 的已跟踪大文件 | 执行 2.2 节扫描命令 | 无匹配输出 | [ ] |
| F-06 | 文档目录结构完整 | `Get-ChildItem -Recurse -Directory \| Select-Object Name` | 包含 docs/、configs/、scripts/、tests/、src/、assets/ 目录 | [ ] |

### 第二类：代码运行校验（6 项）

| 序号 | 自检项 | 验证命令 | 合格标准 | 勾选 |
|:---:|---|---|---|:---:|
| C-01 | 单元测试 144 用例全通过 | `python -m pytest tests/ -v` | `144 passed, 0 failed` | [ ] |
| C-02 | 训练脚本可正常启动 | `python scripts/train.py --config configs/crafter.yaml --steps 1000 --device cuda --amp` | 无 Traceback，正常输出训练日志 | [ ] |
| C-03 | 评测脚本可正常执行 | `python scripts/evaluate.py --config configs/crafter.yaml --episodes 5` | 无 Traceback，输出评测指标 | [ ] |
| C-04 | ONNX 导出脚本可正常执行 | `python scripts/export_onnx.py --config configs/crafter.yaml --output exports/test.onnx` | 成功生成 .onnx 文件 | [ ] |
| C-05 | 4 组消融实验均可启动 | 依次执行 `--ablation full/no_icm/no_episodic/no_rnd` | 4 组均无报错启动 | [ ] |
| C-06 | 显存峰值 ≤ 2.2GB | 训练期间执行 `nvidia-smi --query-gpu=memory.used --format=csv` | 显存使用 ≤ 2300MB | [ ] |

### 第三类：量化数据统一校验（6 项）

| 序号 | 自检项 | 验证命令 | 合格标准 | 勾选 |
|:---:|---|---|---|:---:|
| D-01 | Crafter 指标 19.0% 跨文档一致 | `Select-String -Path README.md,docs/release_note.md,docs/repo_self_checklist.md -Pattern "19\.0"` | 所有文档中 Crafter 指标均为 19.0%（基线 15.6%） | [ ] |
| D-02 | Atari Montezuma 指标 3500+ 一致 | `Select-String -Path README.md,docs/release_note.md -Pattern "3500"` | 所有文档中 Atari 指标均为 3500+（基线 120） | [ ] |
| D-03 | MiniGrid 指标 96.8 万步一致 | `Select-String -Path README.md,docs/release_note.md -Pattern "96\.8"` | 所有文档中 MiniGrid 指标均为 96.8 万步（基线 242 万，2.5x） | [ ] |
| D-04 | ICM 逆损失 ≈ 2.83 一致 | `Select-String -Path README.md,docs/release_note.md -Pattern "2\.83"` | 所有文档中 ICM 初始损失均为 ≈2.83（= ln(17)） | [ ] |
| D-05 | 融合公式跨文档一致 | `Select-String -Path README.md,docs/release_note.md -Pattern "r_int"` | 融合公式 `r_int = eta*ICM + r_episodic*min(max(alpha_t,1),L)` 一致 | [ ] |
| D-06 | 显存优化参数跨文档一致 | `Select-String -Path README.md -Pattern "FP16\|AMP\|512\|LRU\|200"` | FP16 AMP、梯度累积 128x4=512、LRU 200 向量等参数一致 | [ ] |

### 第四类：开源规范校验（6 项）

| 序号 | 自检项 | 验证命令 | 合格标准 | 勾选 |
|:---:|---|---|---|:---:|
| O-01 | MIT LICENSE 内容完整 | `Get-Content LICENSE \| Select-String "MIT\|Permission\|without restriction"` | 包含标准 MIT 协议全文 | [ ] |
| O-02 | README 包含协议声明 | `Select-String -Path README.md -Pattern "MIT\|License"` | README 中明确声明 MIT 协议 | [ ] |
| O-03 | CONTRIBUTING.md 存在且非空 | `(Get-Item CONTRIBUTING.md).Length -gt 100` | 文件大小 > 100 字节 | [ ] |
| O-04 | Issue 模板完整 | `Test-Path .github/ISSUE_TEMPLATE/bug_report.md; Test-Path .github/ISSUE_TEMPLATE/feature_request.md` | 两个模板均存在 | [ ] |
| O-05 | CI 工作流语法正确 | 执行 3.4 节 YAML 校验命令 | YAML 语法校验通过，4 项核心步骤齐全 | [ ] |
| O-06 | requirements.txt 存在且非空 | `Test-Path requirements.txt; (Get-Item requirements.txt).Length -gt 50` | 文件存在且大小 > 50 字节 | [ ] |

### 自检结果汇总

```powershell
# 自检结果汇总模板（手动填写后保存）
Write-Host "===== 24 项自检结果汇总 =====" -ForegroundColor Cyan
Write-Host "文件合规校验（F-01 ~ F-06）: __/6 通过"
Write-Host "代码运行校验（C-01 ~ C-06）: __/6 通过"
Write-Host "量化数据统一校验（D-01 ~ D-06）: __/6 通过"
Write-Host "开源规范校验（O-01 ~ O-06）: __/6 通过"
Write-Host "-----------------------------------"
Write-Host "总计: __/24 通过"
Write-Host ""
Write-Host "判定规则：24/24 通过 -> 允许推送；否则 -> 禁止推送，修复后重新校验"
```

> **面试加分点**：24 项分层自检清单是本手册的核心交付物。将推送前的检查从"凭感觉"升级为"逐项勾选"的标准化流程，这种 checklist-driven 的工作方式是中高级工程师的标志。面试时展示这份清单，可以直接证明你具备系统化的工程质量管控能力。

---

## 第五部分：校验失败分层故障修复指南

### 5.1 目录文件放错类故障

| 故障编号 | 故障现象 | 根因分析 | 修复步骤 | 验证方式 |
|:---:|---|---|---|---|
| E-01 | 文件不在预期目录（如 test.yml 不在 .github/workflows/ 下） | 手动创建文件时目录层级错误 | 1. 确认正确路径<br>2. `Move-Item <当前路径> <正确路径>`<br>3. `git add <正确路径>` | 执行第一部分一键核验命令，12/12 通过 |
| E-02 | .gitignore 未拦截敏感文件 | .gitignore 规则缺失或语法错误 | 1. 打开 .gitignore 添加缺失规则<br>2. 若文件已被跟踪：`git rm --cached <文件>`<br>3. `git commit -m "fix: remove tracked sensitive files"` | 执行 2.1 节校验命令，无敏感文件被跟踪 |
| E-03 | docs/ 目录缺失关键文档 | 文档创建后未提交或误删 | 1. 确认缺失文档列表<br>2. 按文档模板重新创建<br>3. `git add docs/<文件名>.md` | 执行第一部分核对总表，全部文件存在 |
| E-04 | 配置文件路径与代码引用不一致 | 代码中硬编码了错误的配置路径 | 1. 在代码中搜索配置文件引用路径<br>2. 统一为相对路径 `configs/xxx.yaml`<br>3. 重新运行训练脚本验证 | `python scripts/train.py --config configs/crafter.yaml --steps 100` 正常启动 |

### 5.2 代码运行报错类故障

| 故障编号 | 故障现象 | 根因分析 | 修复步骤 | 验证方式 |
|:---:|---|---|---|---|
| E-05 | `ModuleNotFoundError: No module named 'xxx'` | 虚拟环境未激活或依赖未安装 | 1. 激活虚拟环境：`conda activate curiosity-ppo` 或 `.\venv\Scripts\Activate.ps1`<br>2. 安装依赖：`pip install -r requirements.txt`<br>3. 确认安装成功：`pip list \| Select-String xxx` | 重新执行报错命令，无 ModuleNotFoundError |
| E-06 | `ImportError: cannot import name 'xxx' from 'src.xxx'` | 模块重构后 import 路径未更新 | 1. 确认被引用模块的实际路径<br>2. 更新 import 语句<br>3. 运行 `python -c "from src.xxx import yyy"` 验证 | import 语句无报错 |
| E-07 | `KeyError: 'xxx'` 读取配置文件时 | YAML 配置键名与代码引用不一致 | 1. 打开配置文件查看实际键名<br>2. 统一代码中的键名引用<br>3. 重新运行 | 配置读取无 KeyError |
| E-08 | `AssertionError` 在单元测试中 | 测试断言期望值与实际值不符 | 1. 查看 `--tb=long` 回溯信息<br>2. 确认期望值是否合理<br>3. 若为代码 bug：修复源码；若为测试期望值过时：更新测试 | `python -m pytest tests/ -v` 全部通过 |
| E-09 | `RuntimeError: Expected all tensors on same device` | 张量设备不一致（CPU/CUDA 混用） | 1. 在模型 forward 方法中统一 `.to(device)`<br>2. 确认输入数据与模型在同一设备<br>3. 检查 Episodic Memory LRU 缓存的张量设备 | 训练 1000 步无 RuntimeError |
| E-10 | ONNX 导出 `TracerWarning` 或导出失败 | 模型包含动态控制流或条件分支 | 1. 将 forward 中的 if/for 改为 torch.where/torch.stack<br>2. 使用 `torch.onnx.export` 的 `dynamic_axes` 参数<br>3. 验证导出模型：`onnxruntime.InferenceSession` | ONNX 文件成功生成且可加载 |

### 5.3 显存 OOM 类故障

| 故障编号 | 故障现象 | 根因分析 | 修复步骤 | 验证方式 |
|:---:|---|---|---|---|
| E-11 | `CUDA out of memory` 训练时 | batch_size 过大或 AMP 未启用 | 1. 确认 `--amp` 参数已传入启用 FP16<br>2. 减小 batch_size（如从 256 降至 128）<br>3. 增加 `--grad-accum 4`（等效 batch_size 不变） | 训练 1000 步无 OOM，显存峰值 ≤ 2.2GB |
| E-12 | `CUDA out of memory` 评测时 | 评测时未启用 AMP 或 batch_size 过大 | 1. 评测脚本中添加 `torch.no_grad()`<br>2. 启用 FP16 推理<br>3. 减小评测 batch_size | 评测 10 episodes 无 OOM |
| E-13 | 显存碎片化导致间歇性 OOM | 长时间训练显存碎片累积 | 1. 在训练循环中添加 `torch.cuda.empty_cache()`<br>2. 每 N 步执行一次显存清理<br>3. 确认 Episodic Memory LRU 上限为 200 向量 | 长时间训练无间歇性 OOM |
| E-14 | CPU 缓存卸载未生效导致显存超限 | CPU offload 配置缺失 | 1. 确认配置中 `cpu_offload: true`<br>2. 检查 `src/memory.py` 中 CPU offload 逻辑<br>3. 确认非活跃模型层正确卸载至 CPU | 显存峰值 ≤ 2.2GB |

### 5.4 本地绝对路径违规类故障

| 故障编号 | 故障现象 | 根因分析 | 修复步骤 | 验证方式 |
|:---:|---|---|---|---|
| E-15 | 代码中存在 `D:\` 或 `C:\Users\` 等绝对路径 | 开发时硬编码了本地路径 | 1. 执行 2.3 节扫描命令定位所有匹配<br>2. 逐条替换为相对路径（如 `./configs/`、`./data/`）<br>3. 或使用 `pathlib.Path(__file__).parent` 动态构建路径 | 执行 2.3 节扫描，0 处匹配 |
| E-16 | 配置文件中存在绝对路径 | YAML 配置中 data_dir/checkpoint_dir 硬编码 | 1. 打开配置文件定位绝对路径<br>2. 替换为相对路径：`data_dir: ./data/`<br>3. 或使用环境变量：`data_dir: ${PROJECT_ROOT}/data/` | 配置文件中无绝对路径 |
| E-17 | 文档中存在本地路径（如 README 中的示例命令） | 文档编写时使用了本地环境路径 | 1. 搜索文档中的绝对路径<br>2. 替换为相对路径或使用 `<项目根目录>` 占位符<br>3. 确保示例命令可在任何机器上复现 | 文档中无绝对路径 |
| E-18 | .gitignore 中存在本地路径 | .gitignore 中误写了本地虚拟环境绝对路径 | 1. 打开 .gitignore<br>2. 将绝对路径替换为通配规则（如 `venv/`、`.venv/`）<br>3. 确认规则在所有机器上通用 | .gitignore 中无绝对路径 |

> **面试加分点**：分层故障修复指南将常见问题按类别归纳，每类故障都提供了"现象→根因→修复步骤→验证方式"的完整闭环。这体现了系统化的问题排查思维——不是遇到报错就慌，而是有标准化的诊断和修复流程。面试时若被问及"遇到 XX 问题怎么解决"，你可以直接引用这份指南中的案例，展示结构化的问题解决能力。

---

## 附录：推送前最终确认命令

在 24 项自检全部通过后，执行以下最终确认命令，确认无误后方可推送。

```powershell
# 进入项目根目录
cd <项目根目录>

Write-Host "========== 推送前最终确认 ==========" -ForegroundColor Cyan

Write-Host "`n[1/5] Git 状态检查" -ForegroundColor Yellow
git status

Write-Host "`n[2/5] 待推送提交检查" -ForegroundColor Yellow
git log origin/main..HEAD --oneline 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  (首次推送或远程分支不存在，检查本地提交日志)" -ForegroundColor Yellow
    git log --oneline -5
}

Write-Host "`n[3/5] 待推送文件清单" -ForegroundColor Yellow
git diff --stat origin/main..HEAD 2>$null
if ($LASTEXITCODE -ne 0) {
    git diff --stat --cached
    git diff --stat
}

Write-Host "`n[4/5] 最终敏感文件确认" -ForegroundColor Yellow
$finalCheck = git ls-files | Select-String "\.pt$|\.pth$|\.mp4$|\.onnx$|\.log$|wandb|__pycache__"
if ($finalCheck) {
    Write-Host "[STOP] 仍有敏感文件被跟踪，禁止推送！" -ForegroundColor Red
    $finalCheck
} else {
    Write-Host "[OK] 无敏感文件被跟踪" -ForegroundColor Green
}

Write-Host "`n[5/5] 确认推送" -ForegroundColor Yellow
Write-Host "请确认以下事项：" -ForegroundColor White
Write-Host "  [ ] 24 项自检清单全部通过"
Write-Host "  [ ] 12 份配套文件全部存在"
Write-Host "  [ ] 144 项单元测试全部通过"
Write-Host "  [ ] README 复现命令全部跑通"
Write-Host "  [ ] 无硬编码本地绝对路径"
Write-Host "  [ ] 无敏感文件被 Git 跟踪"
Write-Host "  [ ] CI 工作流语法正确"
Write-Host ""
$confirm = Read-Host "全部确认无误？输入 YES 执行推送"
if ($confirm -eq "YES") {
    Write-Host "执行推送..." -ForegroundColor Green
    git push origin main
} else {
    Write-Host "推送已取消。请完成自检后重试。" -ForegroundColor Red
}

Write-Host "`n========== 最终确认完成 ==========" -ForegroundColor Cyan
```

> **面试加分点**：最终确认命令将推送操作从"直接 git push"升级为"5 步确认 + 人工 YES 确认"的安全流程。这种"最后防线"的设计展示了你对线上仓库质量的终极把控——即使前面所有步骤都通过了，推送瞬间仍有一次人工二次确认，确保万无一失。这种工程纪律性是高级开发者的典型特征。

---

> **文档版本**：v1.0
>
> **最后更新**：2026-07-14
>
> **适用项目**：CuriosityPPOAgent ICM+RND 分层新颖信号融合好奇心驱动 PPO 智能体
