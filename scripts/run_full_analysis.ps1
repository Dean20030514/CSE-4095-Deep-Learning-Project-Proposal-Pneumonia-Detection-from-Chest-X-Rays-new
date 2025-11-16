# Quick Analysis Runner
# Runs all analysis tools on the best model in sequence
# Usage: .\scripts\run_full_analysis.ps1

param(
    [string]$ModelCheckpoint = "runs/model_efficientnet_b2/best.pt",
    [string]$Split = "val",
    [string]$OutputBaseDir = "reports/full_analysis"
)

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Pneumonia X-ray - Full Analysis  " -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if checkpoint exists
if (-not (Test-Path $ModelCheckpoint)) {
    Write-Host "Error: Checkpoint not found at $ModelCheckpoint" -ForegroundColor Red
    Write-Host "Please specify correct checkpoint path with -ModelCheckpoint parameter" -ForegroundColor Yellow
    exit 1
}

Write-Host "Model: $ModelCheckpoint" -ForegroundColor Green
Write-Host "Split: $Split" -ForegroundColor Green
Write-Host "Output: $OutputBaseDir" -ForegroundColor Green
Write-Host ""

# Step 1: Experiment Comparison
Write-Host "[1/5] Running experiment comparison analysis..." -ForegroundColor Yellow
python scripts/analyze_all_experiments.py `
    --runs_dir runs `
    --output_dir "$OutputBaseDir/experiment_comparison" `
    --top_n 10

if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Experiment comparison failed" -ForegroundColor Red
}
Write-Host ""

# Step 2: Threshold Sweep
Write-Host "[2/5] Running threshold sweep analysis..." -ForegroundColor Yellow
python scripts/threshold_sweep.py `
    --ckpt $ModelCheckpoint `
    --split $Split `
    --output_dir "$OutputBaseDir/threshold_analysis"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Threshold sweep failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: Calibration Analysis
Write-Host "[3/5] Running calibration analysis..." -ForegroundColor Yellow
if ($Split -eq "val") {
    # Fit temperature scaling on validation set
    python scripts/calibration_analysis.py `
        --ckpt $ModelCheckpoint `
        --split $Split `
        --fit_temperature `
        --output_dir "$OutputBaseDir/calibration"
} else {
    # Don't fit on test set
    python scripts/calibration_analysis.py `
        --ckpt $ModelCheckpoint `
        --split $Split `
        --output_dir "$OutputBaseDir/calibration"
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Calibration analysis failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 4: Error Analysis
Write-Host "[4/5] Running error analysis..." -ForegroundColor Yellow
python scripts/error_analysis.py `
    --ckpt $ModelCheckpoint `
    --split $Split `
    --output_dir "$OutputBaseDir/error_analysis" `
    --max_samples 20

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Error analysis failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 5: Standard Evaluation with Threshold Sweep
Write-Host "[5/5] Running standard evaluation with threshold sweep..." -ForegroundColor Yellow
python src/eval.py `
    --ckpt $ModelCheckpoint `
    --split $Split `
    --threshold_sweep `
    --report "$OutputBaseDir/evaluation_report.json"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Evaluation failed" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Summary
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Analysis Complete!  " -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved to: $OutputBaseDir" -ForegroundColor Green
Write-Host ""
Write-Host "Generated outputs:" -ForegroundColor White
Write-Host "  - Experiment comparison: $OutputBaseDir/experiment_comparison/" -ForegroundColor Gray
Write-Host "  - Threshold analysis: $OutputBaseDir/threshold_analysis/" -ForegroundColor Gray
Write-Host "  - Calibration analysis: $OutputBaseDir/calibration/" -ForegroundColor Gray
Write-Host "  - Error analysis: $OutputBaseDir/error_analysis/" -ForegroundColor Gray
Write-Host "  - Evaluation report: $OutputBaseDir/evaluation_report.json" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review generated plots and reports" -ForegroundColor White
Write-Host "  2. Update Model Card with findings" -ForegroundColor White
Write-Host "  3. Prepare presentation materials" -ForegroundColor White
Write-Host ""
