# 一键运行四组消融实验 (PowerShell)
# 用法: .\scripts\run_all_ablation.ps1
# 可选参数: -Steps 1000000 -Env crafter

param(
    [string]$Env = "crafter",
    [int]$Steps = 1000000,
    [switch]$UseWandb
)

$ablations = @("full", "no_icm", "no_episodic", "no_rnd")

Write-Host "`n=== Curiosity PPO 消融实验 ===" -ForegroundColor Cyan
Write-Host "环境: $Env | 总步数: $Steps | Wandb: $UseWandb`n"

foreach ($ablation in $ablations) {
    $configPath = "experiments/${Env}_${ablation}.yaml"
    
    if (-not (Test-Path $configPath)) {
        Write-Host "[SKIP] 配置文件不存在: $configPath" -ForegroundColor Yellow
        continue
    }
    
    Write-Host "`n{'='*60}" -ForegroundColor Green
    Write-Host "  运行消融: $ablation" -ForegroundColor Green
    Write-Host "{'='*60}`n" -ForegroundColor Green
    
    $cmdArgs = @("scripts/train.py", "--config", $configPath, "--total-steps", $Steps)
    if ($UseWandb) {
        $cmdArgs += "--use-wandb"
    }
    
    python @cmdArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] $ablation 实验失败" -ForegroundColor Red
    } else {
        Write-Host "[OK] $ablation 实验完成" -ForegroundColor Green
    }
}

Write-Host "`n=== 消融实验全部完成 ===" -ForegroundColor Cyan
Write-Host "结果保存在 results/ 目录"
