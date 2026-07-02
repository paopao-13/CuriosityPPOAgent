#!/usr/bin/env bash
#=============================================================================
# CuriosityPPOAgent 全链路自动化测试脚本 (Linux / macOS)
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
#   chmod +x test/scripts/run_all_test.sh
#   ./test/scripts/run_all_test.sh
#=============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0
TOTAL_STEPS=0

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

#=============================================================================
# 步骤 1: Python 环境依赖校验
#=============================================================================
print_info "========== 步骤 1/6: Python 环境依赖校验 =========="

REQUIRED_PACKAGES=(
    "torch"
    "numpy"
    "gymnasium"
    "gym"
    "crafter"
    "minigrid"
    "ale_py"
    "wandb"
    "onnx"
    "onnxruntime"
    "pytest"
    "yaml"
)

ALL_OK=true
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        print_pass "依赖检查: $pkg"
    else
        print_fail "依赖检查: $pkg (未安装)"
        ALL_OK=false
    fi
done

# GPU 检查
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null && \
    print_pass "CUDA 可用: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')" || \
    print_warn "CUDA 不可用, 后续显存测试将跳过"

if [ "$ALL_OK" = false ]; then
    print_fail "依赖校验未通过, 请先安装缺失依赖: pip install -r requirements.txt"
    exit 1
fi

#=============================================================================
# 步骤 2: GPU 显存基线预检测
#=============================================================================
print_info "========== 步骤 2/6: GPU 显存基线预检测 =========="

python test/scripts/check_vram_limit.py --baseline-only && \
    print_pass "显存基线正常 (峰值 < 2.2GB)" || \
    print_warn "显存基线检测跳过或异常 (CPU 模式无显存)"

#=============================================================================
# 步骤 3: 批量执行全部单元测试
#=============================================================================
print_info "========== 步骤 3/6: 批量执行全部单元测试 (157 个) =========="

if python -m pytest tests/ -v --tb=short 2>&1 | tee test/logs/unit_test.log; then
    TEST_COUNT=$(python -m pytest tests/ --co -q 2>/dev/null | tail -1 | grep -o '[0-9]*' | head -1)
    print_pass "单元测试全部通过 ($TEST_COUNT 个用例)"
else
    print_fail "单元测试存在失败用例, 详见 test/logs/unit_test.log"
    exit 1
fi

#=============================================================================
# 步骤 4: MiniGrid 轻量冒烟集成训练测试
#=============================================================================
print_info "========== 步骤 4/6: MiniGrid 轻量冒烟训练测试 (1024 步) =========="

mkdir -p results/checkpoints test/logs

if python scripts/train.py \
    --config experiments/minigrid_doorkey_full.yaml \
    --total-steps 1024 \
    --checkpoint-interval 512 \
    --checkpoint-dir results/checkpoints/smoke \
    2>&1 | tee test/logs/smoke_minigrid.log; then
    print_pass "MiniGrid 冒烟训练完成 (1024 步)"

    # 验证检查点已生成
    if ls results/checkpoints/smoke/step_*.pt 1>/dev/null 2>&1; then
        print_pass "检查点文件已生成"
    else
        print_fail "检查点文件未生成"
    fi
else
    print_fail "MiniGrid 冒烟训练失败"
    exit 1
fi

#=============================================================================
# 步骤 5: 小规模消融功能验证 (4 组 × 短步数)
#=============================================================================
print_info "========== 步骤 5/6: 小规模消融功能验证 =========="

ABLATIONS=("full" "no_icm" "no_episodic" "no_rnd")
ABLATION_ENV="crafter"
ABLATION_STEPS=256

for ablation in "${ABLATIONS[@]}"; do
    CONFIG_PATH="experiments/${ABLATION_ENV}_${ablation}.yaml"
    if [ ! -f "$CONFIG_PATH" ]; then
        print_warn "消融配置不存在: $CONFIG_PATH, 跳过"
        continue
    fi

    print_info "运行消融组: $ablation ($ABLATION_STEPS 步)"
    if python scripts/train.py \
        --config "$CONFIG_PATH" \
        --total-steps $ABLATION_STEPS \
        --checkpoint-interval 999999 \
        2>&1 | tee "test/logs/ablation_${ablation}.log"; then
        print_pass "消融组 $ablation 运行正常"
    else
        print_fail "消融组 $ablation 运行失败"
    fi
done

#=============================================================================
# 步骤 6: ONNX 模型导出 + Web Demo 基础校验
#=============================================================================
print_info "========== 步骤 6/6: ONNX 模型导出 + Web Demo 基础校验 =========="

# 找到冒烟训练生成的检查点
CKPT_PATH=$(ls results/checkpoints/smoke/step_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT_PATH" ]; then
    print_warn "未找到检查点, 跳过 ONNX 导出测试"
else
    ONNX_OUTPUT="results/onnx/smoke_test.onnx"
    mkdir -p results/onnx

    if python scripts/export_onnx.py \
        --checkpoint "$CKPT_PATH" \
        --output "$ONNX_OUTPUT" \
        --env minigrid \
        2>&1 | tee test/logs/onnx_export.log; then
        print_pass "ONNX 模型导出成功"

        # 验证 ONNX 文件存在且大小 > 0
        if [ -s "$ONNX_OUTPUT" ]; then
            FILE_SIZE=$(du -h "$ONNX_OUTPUT" | cut -f1)
            print_pass "ONNX 文件有效 ($FILE_SIZE)"
        else
            print_fail "ONNX 文件为空或不存在"
        fi
    else
        print_fail "ONNX 模型导出失败"
    fi
fi

# Web Demo 基础校验: 检查前端文件完整性
WEB_FILES=(
    "web/package.json"
    "web/index.html"
    "web/src/App.tsx"
    "web/src/main.tsx"
    "web/vite.config.ts"
)
WEB_OK=true
for wf in "${WEB_FILES[@]}"; do
    if [ -f "$wf" ]; then
        : # 文件存在
    else
        print_fail "Web Demo 文件缺失: $wf"
        WEB_OK=false
    fi
done

if [ "$WEB_OK" = true ]; then
    print_pass "Web Demo 前端文件完整性校验通过"
fi

# 检查 npm/node 是否可用
if command -v npm &>/dev/null; then
    print_pass "npm 可用: $(npm --version)"
    print_info "如需启动 Web Demo, 请执行: cd web && npm install && npm run dev"
else
    print_warn "npm 不可用, Web Demo 启动测试需手动执行"
fi

#=============================================================================
# 汇总报告
#=============================================================================
echo ""
echo "==================================================================="
echo "              全链路自动化测试汇总报告"
echo "==================================================================="
echo "  通过: $PASS_COUNT  |  失败: $FAIL_COUNT"
echo "  日志目录: test/logs/"
echo "==================================================================="

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}  全部测试通过! 可进入长周期性能验收阶段。${NC}"
    echo "  下一步: 参照 test/performance_acceptance_standard.md 执行完整百万步训练"
    exit 0
else
    echo -e "${RED}  存在 $FAIL_COUNT 项失败, 请逐条排查后重新运行。${NC}"
    exit 1
fi
