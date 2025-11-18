# Scripts Directory Documentation

> **20 script tools** | Organized by function | Optimized and streamlined

---

## 🚀 Training-Related (2 scripts)

| Script | Function | Usage |
|--------|----------|-------|
| `automated_full_training.ps1` | Automated batch training (optimized) | `.\scripts\automated_full_training.ps1` |
| `create_all_training_configs.py` | Generate config files (GPU optimized) | `python scripts/create_all_training_configs.py` |

---

## 📊 Analysis-Related (10 scripts)

### Comprehensive Analysis

| Script | Function | Usage |
|--------|----------|-------|
| `analyze_all_experiments.py` | Compare all experiments (models, augmentation) | `python scripts/analyze_all_experiments.py` |
| `run_full_analysis.ps1` | Complete analysis pipeline | `.\scripts\run_full_analysis.ps1 -Split test` |
| `generate_project_report.py` | Generate project report | `python scripts/generate_project_report.py` |
| `plot_metrics.py` | Plot metrics charts | `python scripts/plot_metrics.py` |

### Deep Analysis

| Script | Function | Usage |
|--------|----------|-------|
| `error_analysis.py` | Error analysis | `python scripts/error_analysis.py --ckpt <model>` |
| `calibration_analysis.py` | Calibration analysis | `python scripts/calibration_analysis.py --ckpt <model>` |
| `threshold_sweep.py` | Threshold sweeping | `python scripts/threshold_sweep.py --ckpt <model>` |
| `gradcam_evaluation.py` | GradCAM visualization | `python scripts/gradcam_evaluation.py --ckpt <model>` |
| `domain_shift_analysis.py` | Domain shift analysis | `python scripts/domain_shift_analysis.py` |
| `label_noise_detection.py` | Label noise detection | `python scripts/label_noise_detection.py` |

---

## 🛠️ Utility Scripts (5 scripts)

| Script | Function | Usage |
|--------|----------|-------|
| `verify_environment.py` | Verify environment | `python scripts/verify_environment.py` |
| `verify_dataset_integrity.py` | Verify dataset | `python scripts/verify_dataset_integrity.py` |
| `create_optimal_dataset.py` | Create optimized dataset | `python scripts/create_optimal_dataset.py` |
| `download_sample_data.py` | Download sample data | `python scripts/download_sample_data.py` |
| `run_tests.ps1` / `run_tests.sh` | Run test suite | `.\scripts\run_tests.ps1 -Coverage` |

---

## 🎨 Demo-Related (2 scripts)

| Script | Function | Usage |
|--------|----------|-------|
| `demo_presentation.py` | Demo presentation | `python scripts/demo_presentation.py` |
| `project_dashboard.py` | Project dashboard | `python scripts/project_dashboard.py` |

---

## 🔄 Common Workflows

### Workflow 1: Complete Training and Analysis

```powershell
# 1. Auto-train all configs
.\scripts\automated_full_training.ps1

# 2. Comparative analysis
python scripts/analyze_all_experiments.py

# 3. Deep analysis of best model
python scripts/error_analysis.py --ckpt runs/model_efficientnet_b2/best_model.pt
python scripts/calibration_analysis.py --ckpt runs/model_efficientnet_b2/best_model.pt

# 4. Generate report
python scripts/generate_project_report.py
```

### Workflow 2: Quick Validation

```powershell
# 1. Verify environment
python scripts/verify_environment.py
python scripts/verify_dataset_integrity.py

# 2. Quick test
python src/train.py --config src/configs/quick_test_resnet18.yaml

# 3. Evaluate results
python src/eval.py --ckpt runs/quick_test_resnet18/best_model.pt --split val
```

### Workflow 3: Single Model Analysis

```powershell
# Specify model checkpoint
$ckpt = "runs/model_efficientnet_b2/best_model.pt"

# Complete analysis
python scripts/error_analysis.py --ckpt $ckpt
python scripts/calibration_analysis.py --ckpt $ckpt
python scripts/threshold_sweep.py --ckpt $ckpt
python scripts/gradcam_evaluation.py --ckpt $ckpt
```

---

## 📝 Script Dependencies

```
automated_full_training.ps1
└── Calls src/train.py (multiple times)
    └── Uses src/configs/*.yaml

analyze_all_experiments.py
└── Reads runs/*/metrics_history.csv
    └── Generates reports/comprehensive/

error_analysis.py
├── Loads checkpoint
├── Runs model inference
└── Generates reports/error_analysis/

run_full_analysis.ps1
├── Calls analyze_all_experiments.py
├── Calls threshold_sweep.py
├── Calls calibration_analysis.py
└── Calls error_analysis.py
```

---

## 🎯 Script Usage Recommendations

### High Frequency (Recommended to Master)

1. `verify_environment.py` - After every environment change
2. `automated_full_training.ps1` - Batch training
3. `analyze_all_experiments.py` - Experiment comparison
4. `run_tests.ps1` - After code changes

### Medium Frequency

5. `error_analysis.py` - During model tuning
6. `plot_metrics.py` - When preparing reports
7. `threshold_sweep.py` - When optimizing decision thresholds

### Low Frequency

8. `create_optimal_dataset.py` - Dataset preparation
9. `domain_shift_analysis.py` - Advanced research
10. `label_noise_detection.py` - Data quality check

---

## 💡 Best Practices

### Before Training

```powershell
# 1. Verify environment
python scripts/verify_environment.py

# 2. Verify data
python scripts/verify_dataset_integrity.py

# 3. Verify configuration
python src/utils/config_validator.py <config>
```

### During Training

```powershell
# Monitor training
Get-Content runs/<exp>/train.log -Wait -Tail 20

# Monitor GPU
nvidia-smi -l 5
```

### After Training

```powershell
# 1. Analyze experiments
python scripts/analyze_all_experiments.py

# 2. Deep analysis of best model
python scripts/error_analysis.py --ckpt <best_model>
python scripts/calibration_analysis.py --ckpt <best_model>

# 3. Generate report
python scripts/generate_project_report.py
```

---

**Total Scripts:** 20 (optimized and streamlined)  
**Organization Clarity:** ⭐⭐⭐⭐⭐  
**Ease of Use:** ⭐⭐⭐⭐⭐
