# Complete Project Analysis Script
# 一键完成项目所有分析任务

param(
    [string]$DataRoot = "data",
    [string]$BestModel = "runs/model_efficientnet_b2/best.pt",
    [switch]$SkipTraining,
    [switch]$QuickMode
)

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Pneumonia Detection Project - Complete Analysis Pipeline" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Step 1: 环境验证
Write-Host "[1/7] Verifying Environment..." -ForegroundColor Yellow
python scripts/verify_environment.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Environment check failed!" -ForegroundColor Red
    exit 1
}
python scripts/verify_dataset_integrity.py
Write-Host ""

# Step 2: 分析所有已有实验
Write-Host "[2/7] Analyzing All Experiments..." -ForegroundColor Yellow
python scripts/analyze_all_experiments.py --runs_dir runs --output_dir reports/comprehensive
Write-Host ""

# Step 3: 验证集评估(带阈值扫描)
Write-Host "[3/7] Evaluating on Validation Set (Threshold Sweep)..." -ForegroundColor Yellow
python -m src.eval --ckpt "$BestModel" --data_root "$DataRoot" --split val --model "$ModelName" --threshold_sweep --report reports/best_model_val.json
Write-Host ""

# Step 4: 测试集评估
Write-Host "[4/7] Evaluating on Test Set..." -ForegroundColor Yellow
python -m src.eval --ckpt "$BestModel" --data_root "$DataRoot" --split test --model "$ModelName" --threshold_sweep --report reports/best_model_test.json
Write-Host ""

# Step 5: 校准分析
Write-Host "[5/7] Running Calibration Analysis..." -ForegroundColor Yellow
python scripts/calibration_analysis.py --ckpt "$BestModel" --data_root "$DataRoot" --model "$ModelName" --output_dir reports/calibration --split val
Write-Host ""

# Step 6: 错误分析
Write-Host "[6/7] Running Error Analysis..." -ForegroundColor Yellow
python scripts/error_analysis.py --ckpt "$BestModel" --data_root "$DataRoot" --model "$ModelName" --split val --output_dir reports/error_analysis --max_samples 20
Write-Host ""

# Step 7: 生成可视化对比图表
Write-Host "[7/7] Generating Comparison Plots..." -ForegroundColor Yellow
python scripts/plot_metrics.py --csv "runs/model_efficientnet_b2/metrics.csv" --output "reports/plots"
Write-Host ""

# 生成最终报告摘要
Write-Host ""
Write-Host "=" * 80 -ForegroundColor Green
Write-Host "Analysis Complete! Generated Reports:" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Green
Write-Host "📊 Experiment Comparison: reports/comprehensive/" -ForegroundColor White
Write-Host "🎯 Best Model (Val): reports/best_model_val.json" -ForegroundColor White
Write-Host "📈 Test Set Results: reports/best_model_test.json" -ForegroundColor White
Write-Host "📉 Calibration: reports/calibration/" -ForegroundColor White
Write-Host "❌ Error Analysis: reports/error_analysis/" -ForegroundColor White
Write-Host "📊 Plots: reports/plots/" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Review failure modes in: reports/error_analysis/failure_modes.json" -ForegroundColor White
Write-Host "  2. Check calibration metrics in: reports/calibration/" -ForegroundColor White
Write-Host "  3. Update MODEL_CARD.md with latest findings" -ForegroundColor White
Write-Host "  4. Prepare presentation slides using generated plots" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Ready for Project Submission!" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Green
