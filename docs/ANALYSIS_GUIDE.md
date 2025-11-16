# Analysis Toolkit Guide

Complete guide for all analysis tools in the project.

---

## 🚀 Quick Start

### Run Complete Analysis (Recommended)

```powershell
# Validation set (for model tuning)
.\scripts\run_full_analysis.ps1 -Split val

# Test set (final evaluation, run once only)
.\scripts\run_full_analysis.ps1 -Split test

# Custom checkpoint
.\scripts\run_full_analysis.ps1 -ModelCheckpoint "runs/my_model/best.pt"
```

**Time:** ~10-15 minutes  
**Output:** `reports/full_analysis/`

---

## 📊 Individual Tools

### 1. Experiment Comparison

Compares all experiments, ranks by performance, generates visualizations.

```powershell
python scripts/analyze_all_experiments.py --runs_dir runs --output_dir reports/experiments
```

**Outputs:**

- `experiment_summary.csv` - Full comparison table
- `EXPERIMENT_SUMMARY.md` - Markdown report
- `macro_recall_comparison.png` - Ranking bar chart
- `metrics_heatmap.png` - Multi-metric heatmap
- `recall_precision_scatter.png` - Trade-off visualization
- Training curves for top models

---

### 2. Threshold Sweep

Finds optimal thresholds for different clinical scenarios.

```powershell
python scripts/threshold_sweep.py \
  --ckpt runs/model_efficientnet_b2/best.pt \
  --split val \
  --output_dir reports/threshold_analysis
```

**What it does:**

- Tests 38 thresholds (0.05 to 1.0, step 0.025)
- Computes Precision, Recall, Specificity, F1, Youden's Index
- Identifies optimal thresholds for 5 modes:
  - **MAX_RECALL**: Maximize sensitivity (medical screening)
  - **BALANCED_F1**: Optimal F1-score
  - **MAX_YOUDEN**: Best sensitivity + specificity balance
  - **HIGH_PRECISION**: Minimize false alarms
  - **MIN_MISS**: Recall ≥99%, minimize FP

**Outputs:**

- `threshold_sweep_results.json` - All results + optimal thresholds
- `threshold_metrics_curve.png` - Metrics vs threshold
- `precision_recall_curve.png` - PR curve with key points
- `youden_index_curve.png` - Optimal operating point

**Recommendations:**

- Medical screening: Use MIN_MISS or MAX_RECALL
- Balanced use: Use BALANCED_F1 or MAX_YOUDEN
- Confirmatory testing: Use HIGH_PRECISION

---

### 3. Calibration Analysis

Evaluates how well predicted probabilities match actual correctness.

```powershell
# Validation set (with temperature scaling)
python scripts/calibration_analysis.py \
  --ckpt runs/model_efficientnet_b2/best.pt \
  --split val \
  --fit_temperature

# Test set (evaluation only, no fitting)
python scripts/calibration_analysis.py \
  --ckpt runs/model_efficientnet_b2/best.pt \
  --split test
```

**Metrics:**

- ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error)
- Brier Score

**Outputs:**

- `calibration_report.json` - Metrics and scores
- `reliability_diagram_before.png` - Pre-calibration
- `reliability_diagram_after.png` - Post-temperature scaling
- `confidence_histogram.png` - Confidence distribution
- `per_class_calibration.png` - Per-class reliability

⚠️ **Important:** Only fit temperature scaling on validation set!

---

### 4. Error Analysis

Deep dive into false positives and false negatives.

```powershell
python scripts/error_analysis.py \
  --ckpt runs/model_efficientnet_b2/best.pt \
  --split val \
  --output_dir reports/error_analysis \
  --max_samples 20
```

**Features:**

- Collects all FP and FN cases
- Categorizes by confidence level
- Provides clinical implications
- Generates error galleries
- Recommends next steps

**Failure Categories:**

- **FP-1:** High confidence false positives (MAJOR)
- **FP-2:** Low confidence false positives (MINOR)
- **FN-1:** High confidence false negatives (⚠️ CRITICAL)
- **FN-2:** Low confidence false negatives (MAJOR)

**Outputs:**

- `failure_modes.json` - Detailed analysis + recommendations
- `FP_gallery.png` - False positive images
- `FN_gallery.png` - False negative images
- `error_summary.png` - Breakdown pie charts

---

## 📁 Output Structure

After running `run_full_analysis.ps1`:

```text
reports/full_analysis/
├── experiment_comparison/
│   ├── experiment_summary.csv
│   ├── EXPERIMENT_SUMMARY.md
│   └── *.png (plots)
├── threshold_analysis/
│   ├── threshold_sweep_results.json
│   └── *.png (curves)
├── calibration/
│   ├── calibration_report.json
│   └── *.png (diagrams)
├── error_analysis/
│   ├── failure_modes.json
│   └── *.png (galleries)
└── evaluation_report.json
```

---

## 🎯 For Reports/Presentations

### Essential Figures

1. **Table 1:** `experiment_summary.csv` - Model comparison
2. **Figure 1:** `macro_recall_comparison.png` - Rankings
3. **Figure 2:** `precision_recall_curve.png` - Trade-offs
4. **Figure 3:** `reliability_diagram_before.png` - Calibration
5. **Figure 4:** `FN_gallery.png` - Error examples

### Key Metrics

From `evaluation_report.json`:

- Accuracy, Precision, Recall, F1 (per-class and macro)
- ROC-AUC, PR-AUC
- Confusion matrix
- Optimal thresholds for different scenarios

---

## ⚙️ Advanced Options

### Experiment Comparison

```powershell
# Top 5 only
python scripts/analyze_all_experiments.py --top_n 5

# Custom runs directory
python scripts/analyze_all_experiments.py --runs_dir my_experiments
```

### Threshold Sweep

```powershell
# Test set evaluation
python scripts/threshold_sweep.py --ckpt <model> --split test

# Edit script to change threshold resolution (np.arange step size)
```

### Calibration

```powershell
# More bins for finer resolution
python scripts/calibration_analysis.py --ckpt <model> --n_bins 15

# Only on validation set for temperature fitting
python scripts/calibration_analysis.py --ckpt <model> --split val --fit_temperature
```

### Error Analysis

```powershell
# More samples in gallery
python scripts/error_analysis.py --ckpt <model> --max_samples 30

# Test set
python scripts/error_analysis.py --ckpt <model> --split test
```

---

## 🔧 Troubleshooting

### Out of Memory

```powershell
# Analysis tools use small batch sizes, but if OOM:
# - Close other GPU processes
# - Use CPU (automatic fallback)
# - Reduce --max_samples in error_analysis
```

### No Experiments Found

```powershell
# Check runs directory
ls runs/

# Verify metrics.csv exists
ls runs/*/metrics.csv
```

### Missing Dependencies

```powershell
pip install seaborn matplotlib scikit-learn pandas tqdm
```

---

## ⏱️ Time Estimates

- Experiment comparison: ~1 minute
- Threshold sweep: ~2 minutes
- Calibration analysis: ~2 minutes
- Error analysis: ~3 minutes
- **Full analysis:** ~10-15 minutes

---

## 📚 Best Practices

1. **Always analyze validation set first**, then test set last
2. **Never tune on test set** - it's for final evaluation only
3. **Temperature scaling:** Only fit on validation set
4. **Threshold tuning:** Find optimal on validation, apply to test
5. **Document everything:** Save all reports for reproducibility

---

## 🎓 Workflow Recommendations

### During Development (Validation Set)

```powershell
# Run after each major experiment
.\scripts\run_full_analysis.ps1 -Split val

# Review results and iterate
```

### Final Evaluation (Test Set)

```powershell
# Run ONCE on best model
.\scripts\run_full_analysis.ps1 -Split test -ModelCheckpoint "runs/best_model/best.pt"

# Use for final report
```

---

For detailed tool implementation, see `CODE_OPTIMIZATION_SUMMARY.md`.
