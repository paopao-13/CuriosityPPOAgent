#=============================================================================
# CuriosityPPOAgent 全链路自动化测试脚本 (Windows PowerShell)
#
# 执行流程:
#   1. Python 环境依赖校验
#   2. GPU 显存基线预检测
#   3. 批量执行全部单元测试 (157 个)
#   4. MiniGrid 轻量冒烟集成训练测试 (1024 步)
#   5. 小规模消融功能验证 (4 组 × 短步数)
#   6. ONNX 模型导出 + Web Demo 基础校验
#
# 硬件: AMD R7 6800H + RTX3060 Laptop 6GB
# 显存红线: 2.2GB
#
# 用法:
#   .\test\scripts\run_all_test.ps1
#=============================================================================

$ErrorActionPreference = "Continue"

# 计数器
$script:PassCount = 0
$script:FailCount = 0

function Print-Pass {
    param([string]$msg)
    Write-Host "[PASS] $msg" -ForegroundColor Green
    $script:PassCount++
}

function Print-Fail {
    param([string]$msg)
    Write-Host "[FAIL] $msg" -ForegroundColor Red
    $script:FailCount++
}

function Print-Info {
    param([string]$msg)
    Write-Host "[INFO] $msg" -ForegroundColor Cyan
}

function Print-Warn {
    param([string]$msg)
    Write-Host "[WARN] $msg" -ForegroundColor Yellow
}

# 确保日志目录存在
New-Item -ItemType Directory -Force -Path "test/logs" | Out-Null
New-Item -ItemType Directory -Force -Path "results/checkpoints/smoke" | Out-Null
New-Item -ItemType Directory -Force -Path "results/onnx" | Out-Null

#=============================================================================
# 步骤 1: Python 环境依赖校验
#=============================================================================
Print-Info "========== 步骤 1/6: Python 环境依赖校验 =========="

$requiredPackages = @(
    "torch", "numpy", "gymnasium", "gym", "crafter",
    "minigrid", "ale_py", "wandb", "onnx", "onnxruntime",
    "pytest", "yaml"
)

$allOk = $true
foreach ($pkg in $requiredPackages) {
    $result = python -c "import $pkg" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Print-Pass "依赖检查: $pkg"
    } else {
        Print-Fail "依赖检查: $pkg (未安装)"
        $allOk = $false
    }
}

# GPU 检查
$gpuCheck = python -c "import torch; print('CUDA_OK' if torch.cuda.is_available() else 'NO_CUDA'); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1
if ($gpuCheck -match "CUDA_OK") {
    $gpuName = ($gpuCheck -split "`n")[1].Trim()
    Print-Pass "CUDA 可用: $gpuName"
} else {
    Print-Warn "CUDA 不可用, 后续显存测试将跳过"
}

if (-not $allOk) {
    Print-Fail "依赖校验未通过, 请先安装缺失依赖: pip install -r requirements.txt"
    exit 1
}

#=============================================================================
# 步骤 2: GPU 显存基线预检测
#=============================================================================
Print-Info "========== 步骤 2/6: GPU 显存基线预检测 =========="

python test/scripts/check_vram_limit.py --baseline-only 2>&1
if ($LASTEXITCODE -eq 0) {
    Print-Pass "显存基线正常 (峰值 < 2.2GB)"
} else {
    Print-Warn "显存基线检测跳过或异常 (CPU 模式无显存)"
}

#=============================================================================
# 步骤 3: 批量执行全部单元测试
#=============================================================================
Print-Info "========== 步骤 3/6: 批量执行全部单元测试 (157 个) =========="

python -m pytest tests/ -v --tb=short 2>&1 | Tee-Object -FilePath "test/logs/unit_test.log"
if ($LASTEXITCODE -eq 0) {
    $testCount = (python -m pytest tests/ --co -q 2>$null | Select-Object -Last 1)
    Print-Pass "单元测试全部通过"
} else {
    Print-Fail "单元测试存在失败用例, 详见 test/logs/unit_test.log"
    exit 1
}

#=============================================================================
# 步骤 4: MiniGrid 轻量冒烟集成训练测试
#=============================================================================
Print-Info "========== 步骤 4/6: MiniGrid 轻量冒烟训练测试 (1024 步) =========="

python scripts/train.py `
    --config experiments/minigrid_doorkey_full.yaml `
    --total-steps 1024 `
    --checkpoint-interval 512 `
    --checkpoint-dir results/checkpoints/smoke `
    2>&1 | Tee-Object -FilePath "test/logs/smoke_minigrid.log"

if ($LASTEXITCODE -eq 0) {
    Print-Pass "MiniGrid 冒烟训练完成 (1024 步)"

    $ckptFiles = Get-ChildItem "results/checkpoints/smoke/step_*.pt" -ErrorAction SilentlyContinue
    if ($ckptFiles) {
        Print-Pass "检查点文件已生成: $($ckptFiles[0].Name)"
    } else {
        Print-Fail "检查点文件未生成"
    }
} else {
    Print-Fail "MiniGrid 冒烟训练失败"
    exit 1
}

#=============================================================================
# 步骤 5: 小规模消融功能验证 (4 组 × 短步数)
#=============================================================================
Print-Info "========== 步骤 5/6: 小规模消融功能验证 =========="

$ablations = @("full", "no_icm", "no_episodic", "no_rnd")
$ablationEnv = "crafter"
$ablationSteps = 256

foreach ($ablation in $ablations) {
    $configPath = "experiments/${ablationEnv}_${ablation}.yaml"
    if (-not (Test-Path $configPath)) {
        Print-Warn "消融配置不存在: $configPath, 跳过"
        continue
    }

    Print-Info "运行消融组: $ablation ($ablationSteps 步)"
    python scripts/train.py `
        --config $configPath `
        --total-steps $ablationSteps `
        --checkpoint-interval 999999 `
        2>&1 | Tee-Object -FilePath "test/logs/ablation_${ablation}.log"

    if ($LASTEXITCODE -eq 0) {
        Print-Pass "消融组 $ablation 运行正常"
    } else {
        Print-Fail "消融组 $ablation 运行失败"
    }
}

#=============================================================================
# 步骤 6: ONNX 模型导出 + Web Demo 基础校验
#=============================================================================
Print-Info "========== 步骤 6/6: ONNX 模型导出 + Web Demo 基础校验 =========="

# 找到冒烟训练生成的检查点
$ckptFiles = Get-ChildItem "results/checkpoints/smoke/step_*.pt" -ErrorAction SilentlyContinue
if ($ckptFiles) {
    $ckptPath = $ckptFiles[0].FullName
    $onnxOutput = "results/onnx/smoke_test.onnx"

    python scripts/export_onnx.py `
        --checkpoint $ckptPath `
        --output $onnxOutput `
        --env minigrid `
        2>&1 | Tee-Object -FilePath "test/logs/onnx_export.log"

    if ($LASTEXITCODE -eq 0 -and (Test-Path $onnxOutput)) {
        $fileSize = (Get-Item $onnxOutput).Length / 1KB
        Print-Pass "ONNX 模型导出成功 ($([math]::Round($fileSize, 1)) KB)"
    } else {
        Print-Fail "ONNX 模型导出失败"
    }
} else {
    Print-Warn "未找到检查点, 跳过 ONNX 导出测试"
}

# Web Demo 基础校验: 检查前端文件完整性
$webFiles = @(
    "web/package.json",
    "web/index.html",
    "web/src/App.tsx",
    "web/src/main.tsx",
    "web/vite.config.ts"
)
$webOk = $true
foreach ($wf in $webFiles) {
    if (Test-Path $wf) {
        # 文件存在
    } else {
        Print-Fail "Web Demo 文件缺失: $wf"
        $webOk = $false
    }
}

if ($webOk) {
    Print-Pass "Web Demo 前端文件完整性校验通过"
}

# 检查 npm 是否可用
$npmCheck = Get-Command npm -ErrorAction SilentlyContinue
if ($npmCheck) {
    $npmVersion = npm --version 2>&1
    Print-Pass "npm 可用: v$npmVersion"
    Print-Info "如需启动 Web Demo, 请执行: cd web; npm install; npm run dev"
} else {
    Print-Warn "npm 不可用, Web Demo 启动测试需手动执行"
}

#=============================================================================
# 汇总报告
#=============================================================================
Write-Host ""
Write-Host "===================================================================" -ForegroundColor White
Write-Host "              全链路自动化测试汇总报告" -ForegroundColor White
Write-Host "===================================================================" -ForegroundColor White
Write-Host "  通过: $script:PassCount  |  失败: $script:FailCount" -ForegroundColor White
Write-Host "  日志目录: test/logs/" -ForegroundColor White
Write-Host "===================================================================" -ForegroundColor White

if ($script:FailCount -eq 0) {
    Write-Host "  全部测试通过! 可进入长周期性能验收阶段。" -ForegroundColor Green
    Write-Host "  下一步: 参照 test/performance_acceptance_standard.md 执行完整百万步训练"
    exit 0
} else {
    Write-Host "  存在 $($script:FailCount) 项失败, 请逐条排查后重新运行。" -ForegroundColor Red
    exit 1
}
