# Scripts Directory Documentation

> **26 Python scripts** | **1 PowerShell manager** | All scripts can be run via project.ps1

---

## Unified Project Manager (project.ps1)

**One script to run them all!** The `project.ps1` script can execute all 26 Python scripts.

### Basic Commands

```powershell
.\scripts\project.ps1 -Quick          # Quick start (~10 min)
.\scripts\project.ps1 -Train          # Batch training (15 configs)
.\scripts\project.ps1 -Analyze        # Model analysis (6 scripts)
.\scripts\project.ps1 -All            # Core workflow (11 scripts)
.\scripts\project.ps1 -Demo           # Start demo app
.\scripts\project.ps1 -Test           # Run test suite
```

### Advanced Commands (NEW)

```powershell
.\scripts\project.ps1 -Complete       # Run ALL 26 scripts
.\scripts\project.ps1 -Advanced       # Advanced analysis (7 scripts)
.\scripts\project.ps1 -Benchmark      # Performance benchmarks (3 scripts)
.\scripts\project.ps1 -Visualize      # Visualization tools (4 scripts)
```

### Common Options

```powershell
-SkipValidation      # Skip environment checks
-SkipTraining        # Skip training phase
-NoDemo              # Don't launch demo at end
-Model <path>        # Specify model checkpoint
-Config <path>       # Specify config file
-ContinueOnError     # Continue if a script fails
-ExportModels        # Export to ONNX/TorchScript
```

### Script Coverage Summary

| Command | Scripts Covered | Description |
|---------|-----------------|-------------|
| `-All` | 11 | Core workflow (train + analyze + report) |
| `-Complete` | **26** | **All scripts** (full pipeline) |
| `-Advanced` | 7 | Uncertainty, noise detection, cross-val |
| `-Benchmark` | 3 | Inference benchmark, Optuna, auto-optimize |
| `-Visualize` | 4 | Metrics plots, augmentation viz, dashboard |

---

## Script Categories

### 1. Environment & Validation (2 scripts)

| Script | Function | Covered By |
|--------|----------|------------|
| `verify_environment.py` | Check Python/PyTorch/GPU | `-Quick`, `-All`, `-Complete` |
| `verify_dataset_integrity.py` | Verify dataset | `-Quick`, `-All`, `-Complete` |

### 2. Training (2 scripts)

| Script | Function | Covered By |
|--------|----------|------------|
| `create_all_training_configs.py` | Generate configs | `-Complete` |
| `src/train.py` | Model training | `-Train`, `-All`, `-Complete` |

### 3. Core Analysis (6 scripts)

| Script | Function | Covered By |
|--------|----------|------------|
| `analyze_all_experiments.py` | Compare experiments | `-Analyze`, `-All`, `-Complete` |
| `threshold_sweep.py` | Threshold optimization | `-Analyze`, `-All`, `-Complete` |
| `calibration_analysis.py` | Calibration analysis | `-Analyze`, `-All`, `-Complete` |
| `error_analysis.py` | Error analysis | `-Analyze`, `-All`, `-Complete` |
| `gradcam_evaluation.py` | GradCAM visualization | `-Analyze`, `-All`, `-Complete` |
| `src/eval.py` | Model evaluation | `-Analyze`, `-All`, `-Complete` |

### 4. Advanced Analysis (7 scripts)

| Script | Function | Covered By |
|--------|----------|------------|
| `uncertainty_estimation.py` | MC Dropout uncertainty | `-Advanced`, `-Complete` |
| `domain_shift_analysis.py` | Domain shift analysis | `-Advanced`, `-Complete` |
| `label_noise_detection.py` | Label noise detection | `-Advanced`, `-Complete` |
| `ensemble_evaluation.py` | Model ensemble | `-Advanced`, `-Complete` |
| `find_optimal_lr.py` | LR range test | `-Advanced`, `-Complete` |
| `cross_validation.py` | K-fold cross validation | `-Advanced`, `-Complete` |
| `generate_project_report.py` | Generate report | `-Advanced`, `-All`, `-Complete` |

### 5. Benchmarking (3 scripts)

| Script | Function | Covered By |
|--------|----------|------------|
| `benchmark_inference.py` | Inference speed test | `-Benchmark`, `-Complete` |
| `optuna_hyperparameter_search.py` | Optuna HPO | `-Benchmark`, `-Complete` |
| `auto_optimize_hyperparams.py` | Auto optimization | `-Benchmark`, `-Complete` |

### 6. Visualization (4 scripts)

| Script | Function | Covered By |
|--------|----------|------------|
| `plot_metrics.py` | Training curves | `-Visualize`, `-Complete` |
| `visualize_augmentations.py` | Augmentation viz | `-Visualize`, `-Complete` |
| `demo_presentation.py` | Demo presentation | `-Visualize`, `-Complete` |
| `project_dashboard.py` | Project dashboard | `-Visualize`, `-Complete` |

### 7. Utility (2 scripts)

| Script | Function | Covered By |
|--------|----------|------------|
| `create_optimal_dataset.py` | Dataset preparation | Manual |
| `download_sample_data.py` | Download samples | Manual |

---

## Quick Reference

### Run Everything

```powershell
# Complete pipeline: all 26 scripts
.\scripts\project.ps1 -Complete -NoDemo

# With training skip (if models exist)
.\scripts\project.ps1 -Complete -SkipTraining
```

### Common Workflows

```powershell
# Workflow 1: First-time setup
.\scripts\project.ps1 -Quick

# Workflow 2: Full training + analysis
.\scripts\project.ps1 -All -NoDemo

# Workflow 3: Analysis only (models exist)
.\scripts\project.ps1 -Analyze -Model runs/model_efficientnet_b2/best_model.pt

# Workflow 4: Advanced research
.\scripts\project.ps1 -Advanced -Model runs/model_efficientnet_b2/best_model.pt

# Workflow 5: Generate visualizations
.\scripts\project.ps1 -Visualize
```

### Manual Script Execution

```powershell
# Individual script examples
python scripts/verify_environment.py
python scripts/analyze_all_experiments.py
python scripts/error_analysis.py --ckpt runs/model_efficientnet_b2/best_model.pt
python scripts/benchmark_inference.py --ckpt runs/model_efficientnet_b2/best_model.pt
```

---

## Script Dependencies

```
project.ps1 -Complete
├── verify_environment.py
├── verify_dataset_integrity.py
├── create_all_training_configs.py
├── src/train.py (15 configs)
├── analyze_all_experiments.py
├── threshold_sweep.py
├── calibration_analysis.py
├── error_analysis.py
├── gradcam_evaluation.py
├── src/eval.py
├── uncertainty_estimation.py
├── domain_shift_analysis.py
├── label_noise_detection.py
├── ensemble_evaluation.py
├── find_optimal_lr.py
├── cross_validation.py
├── generate_project_report.py
├── benchmark_inference.py
├── optuna_hyperparameter_search.py
├── auto_optimize_hyperparams.py
├── plot_metrics.py
├── visualize_augmentations.py
├── demo_presentation.py
└── project_dashboard.py
```

---

## Best Practices

### Before Running

```powershell
# Always verify environment first
python scripts/verify_environment.py
```

### During Long Runs

```powershell
# Use -ContinueOnError to avoid stopping on failures
.\scripts\project.ps1 -Complete -ContinueOnError

# Check logs
Get-Content logs/project_*.log -Tail 50
```

### Output Locations

| Output Type | Location |
|-------------|----------|
| Training | `runs/<experiment>/` |
| Analysis | `reports/analysis_<timestamp>/` |
| Advanced | `reports/advanced_<timestamp>/` |
| Benchmark | `reports/benchmark_<timestamp>/` |
| Visualizations | `reports/visualization_<timestamp>/` |
| Logs | `logs/project_<timestamp>.log` |

---

**Total Scripts:** 26 Python + 1 PowerShell manager
**One-Click Coverage:** 100% (via `-Complete`)
**Organization:** Fully automated

