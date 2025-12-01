<#
.SYNOPSIS
    完整模型分析流程脚本

.DESCRIPTION
    整合所有分析脚本，对训练好的模型执行完整的分析流程：
    1. 错误分析
    2. 校准分析
    3. 阈值扫描
    4. GradCAM 可视化
    5. 推理基准测试

.PARAMETER Checkpoint
    模型检查点路径

.PARAMETER DataRoot
    数据根目录

.PARAMETER Split
    评估的数据集划分 (val/test)

.PARAMETER OutputDir
    输出目录

.EXAMPLE
    .\scripts\complete_analysis.ps1 -Checkpoint runs/model_efficientnet_b2/best_model.pt
    .\scripts\complete_analysis.ps1 -Checkpoint runs/best/best_model.pt -Split test
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Checkpoint,
    
    [string]$DataRoot = "data",
    
    [ValidateSet("val", "test")]
    [string]$Split = "test",
    
    [string]$OutputDir = "reports/complete_analysis"
)

# 颜色输出函数
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function Write-Step {
    param([int]$StepNum, [string]$Description)
    Write-Host ""
    Write-Host "[Step $StepNum] $Description" -ForegroundColor Yellow
    Write-Host ("-" * 50)
}

# 检查文件是否存在
if (-not (Test-Path $Checkpoint)) {
    Write-ColorOutput "Error: Checkpoint not found: $Checkpoint" "Red"
    exit 1
}

if (-not (Test-Path $DataRoot)) {
    Write-ColorOutput "Error: Data root not found: $DataRoot" "Red"
    exit 1
}

# 创建输出目录
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$analysisDir = "$OutputDir/$timestamp"
New-Item -ItemType Directory -Force -Path $analysisDir | Out-Null

Write-Header "COMPLETE MODEL ANALYSIS"
Write-Host "  Checkpoint: $Checkpoint"
Write-Host "  Data: $DataRoot ($Split split)"
Write-Host "  Output: $analysisDir"
Write-Host ""

$startTime = Get-Date
$stepResults = @{}

# ============================================
# Step 1: 模型评估
# ============================================
Write-Step 1 "Model Evaluation"
try {
    python src/eval.py --ckpt $Checkpoint --data_root $DataRoot --split $Split `
        --output "$analysisDir/evaluation"
    $stepResults["evaluation"] = "OK"
    Write-ColorOutput "  Evaluation completed!" "Green"
} catch {
    $stepResults["evaluation"] = "FAILED"
    Write-ColorOutput "  Evaluation failed: $_" "Red"
}

# ============================================
# Step 2: 错误分析
# ============================================
Write-Step 2 "Error Analysis"
try {
    python scripts/error_analysis.py --ckpt $Checkpoint --data_root $DataRoot --split $Split `
        --output "$analysisDir/error_analysis"
    $stepResults["error_analysis"] = "OK"
    Write-ColorOutput "  Error analysis completed!" "Green"
} catch {
    $stepResults["error_analysis"] = "FAILED"
    Write-ColorOutput "  Error analysis failed: $_" "Red"
}

# ============================================
# Step 3: 校准分析
# ============================================
Write-Step 3 "Calibration Analysis"
try {
    python scripts/calibration_analysis.py --ckpt $Checkpoint --data_root $DataRoot --split $Split `
        --output "$analysisDir/calibration"
    $stepResults["calibration"] = "OK"
    Write-ColorOutput "  Calibration analysis completed!" "Green"
} catch {
    $stepResults["calibration"] = "FAILED"
    Write-ColorOutput "  Calibration analysis failed: $_" "Red"
}

# ============================================
# Step 4: 阈值扫描
# ============================================
Write-Step 4 "Threshold Sweep"
try {
    python scripts/threshold_sweep.py --ckpt $Checkpoint --data_root $DataRoot --split $Split `
        --output "$analysisDir/threshold_sweep"
    $stepResults["threshold_sweep"] = "OK"
    Write-ColorOutput "  Threshold sweep completed!" "Green"
} catch {
    $stepResults["threshold_sweep"] = "FAILED"
    Write-ColorOutput "  Threshold sweep failed: $_" "Red"
}

# ============================================
# Step 5: GradCAM 可视化
# ============================================
Write-Step 5 "GradCAM Visualization"
try {
    python scripts/gradcam_evaluation.py --ckpt $Checkpoint --data_root $DataRoot --split $Split `
        --output "$analysisDir/gradcam" --num_samples 20
    $stepResults["gradcam"] = "OK"
    Write-ColorOutput "  GradCAM visualization completed!" "Green"
} catch {
    $stepResults["gradcam"] = "FAILED"
    Write-ColorOutput "  GradCAM visualization failed: $_" "Red"
}

# ============================================
# Step 6: 推理基准测试
# ============================================
Write-Step 6 "Inference Benchmark"
try {
    python scripts/benchmark_inference.py --ckpt $Checkpoint --output "$analysisDir/benchmark"
    $stepResults["benchmark"] = "OK"
    Write-ColorOutput "  Inference benchmark completed!" "Green"
} catch {
    $stepResults["benchmark"] = "FAILED"
    Write-ColorOutput "  Inference benchmark failed: $_" "Red"
}

# ============================================
# 汇总报告
# ============================================
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Header "ANALYSIS COMPLETE"

Write-Host "`nResults Summary:" -ForegroundColor Cyan
Write-Host ("-" * 40)

foreach ($step in $stepResults.Keys | Sort-Object) {
    $status = $stepResults[$step]
    $color = if ($status -eq "OK") { "Green" } else { "Red" }
    $symbol = if ($status -eq "OK") { "[OK]" } else { "[FAIL]" }
    Write-Host "  $symbol $step" -ForegroundColor $color
}

Write-Host ""
Write-Host "Duration: $($duration.TotalMinutes.ToString('F1')) minutes"
Write-Host "Output Directory: $analysisDir"

# 生成摘要文件
$summaryPath = "$analysisDir/ANALYSIS_SUMMARY.md"
@"
# Complete Analysis Summary

**Checkpoint:** $Checkpoint  
**Data Split:** $Split  
**Date:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")  
**Duration:** $($duration.TotalMinutes.ToString('F1')) minutes

## Results

| Analysis | Status |
|----------|--------|
$(foreach ($step in $stepResults.Keys | Sort-Object) {
    $status = $stepResults[$step]
    "| $step | $status |"
})

## Output Files

- ``evaluation/`` - Model evaluation metrics
- ``error_analysis/`` - Error analysis and failure cases
- ``calibration/`` - Probability calibration analysis
- ``threshold_sweep/`` - Threshold optimization
- ``gradcam/`` - GradCAM visualizations
- ``benchmark/`` - Inference performance benchmark
"@ | Out-File -FilePath $summaryPath -Encoding UTF8

Write-Host "`nSummary saved to: $summaryPath" -ForegroundColor Cyan
Write-Host ""

