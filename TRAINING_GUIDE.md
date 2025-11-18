# Training Guide

> **Unified training documentation** - Consolidates all training-related information  
> Replaces: TRAINING_COMMANDS.md, TRAINING_PLAN.md, QUICK_START_RETRAINING.md, RETRAINING_GUIDE.md

---

## 🚀 Quick Start (5 minutes)

### 1. Verify Environment
```powershell
python scripts/verify_environment.py
python scripts/verify_dataset_integrity.py
```

### 2. Quick Test (3 epochs, ~10 minutes)
```powershell
python src/train.py --config src/configs/quick_test_resnet18.yaml
```

### 3. Best Configuration Training
```powershell
# CPU: ~10-20 hours
# GPU: ~2-4 hours
python src/train.py --config src/configs/final_model.yaml
```

---

## 📋 Complete Training Workflow

### Step 1: Environment Setup

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify environment
python scripts/verify_environment.py

# 3. Verify dataset
python scripts/verify_dataset_integrity.py
```

### Step 2: Choose Training Method

#### 🔥 Method A: Automated Batch Training (Recommended)

```powershell
# Train all configs (~10-15 hours)
.\scripts\automated_full_training.ps1

# Train key configs only (~4-6 hours)
.\scripts\automated_full_training.ps1 -HighPriorityOnly
```

#### ⚡ Method B: Manual Single Training

```powershell
# Baseline model
python src/train.py --config src/configs/baseline_resnet18.yaml

# Best model
python src/train.py --config src/configs/model_efficientnet_b2.yaml

# Final model
python src/train.py --config src/configs/final_model.yaml
```

### Step 3: Monitor Training

```powershell
# View log in real-time
Get-Content runs/experiment_name/train.log -Wait -Tail 20

# View metrics
Import-Csv runs/experiment_name/metrics_history.csv | Format-Table

# Monitor GPU
nvidia-smi -l 5
```

---

## 📊 Available Configurations (17 total)

### 🎯 Recommended Configs

| Config | Model | Resolution | Purpose |
|--------|-------|------------|---------|
| **final_model.yaml** | EfficientNet-B2 | 512px | ⭐ Final production model |
| **model_efficientnet_b2.yaml** | EfficientNet-B2 | 384px | ⭐ Best performance |
| **lr_0.0005.yaml** | EfficientNet-B2 | 384px | ⭐ Optimal learning rate |
| **quick_test_resnet18.yaml** | ResNet18 | 224px | 🔧 Quick test |

### 🔬 Experimental Configs

**Architecture Comparison:**
- `baseline_resnet18.yaml` - ResNet18 baseline
- `baseline_efficientnet.yaml` - EfficientNet baseline
- `model_resnet50.yaml` - ResNet50
- `model_densenet121.yaml` - DenseNet121
- `model_efficientnet_b0.yaml` - EfficientNet-B0

**Learning Rate Experiments:**
- `lr_0.0001.yaml` - LR=0.0001
- `lr_0.0005.yaml` - LR=0.0005 ⭐
- `lr_0.001.yaml` - LR=0.001

**Data Augmentation Experiments:**
- `aug_light.yaml` - Light augmentation
- `aug_medium.yaml` - Medium augmentation ⭐
- `aug_aggressive.yaml` - Aggressive augmentation

**Special Purpose:**
- `medical_screening_optimized.yaml` - Medical screening (maximize recall)
- `full_resnet18.yaml` - Complete training

---

## 🎮 Training Parameters

### Command Line Parameter Override

```powershell
python src/train.py \
  --config <config_file> \
  --epochs <num_epochs> \
  --batch_size <batch_size> \
  --lr <learning_rate> \
  --model <model_name> \
  --augment_level <augmentation_level>
```

### Examples

```powershell
# Quick test
python src/train.py \
  --config src/configs/baseline_resnet18.yaml \
  --epochs 3 \
  --batch_size 16

# Custom training
python src/train.py \
  --config src/configs/model_efficientnet_b2.yaml \
  --lr 0.0003 \
  --augment_level heavy
```

---

## 📈 Expected Performance

| Model | Macro Recall | Val Accuracy | Training Time (GPU) |
|-------|--------------|--------------|---------------------|
| EfficientNet-B2 @ 384px | 98.26% | 98.30% | ~1.5h |
| ResNet18 | 97.63% | 98.47% | ~1h |
| DenseNet121 | 97.60% | 97.62% | ~1.5h |

---

## 🔧 Performance Optimization

### GPU-Accelerated Training (Recommended) ✅

**Current Environment: CUDA 13.0 + RTX 5070 Laptop GPU**

```yaml
# GPU optimized config (recommended)
batch_size: 16-32    # GPU can handle larger batches
amp: true            # Mixed precision (faster, saves VRAM)
num_workers: 4-8     # Parallel data loading
img_size: 384-512    # Can use higher resolution
```

**Training Speed:**
- Per epoch: 5-10 seconds (GPU) vs 1-2 minutes (CPU)
- 100 epochs: ~10-20 minutes (GPU) vs 2-3 hours (CPU)
- **Speedup: 10-15x** ⚡

### If GPU Out of Memory (OOM)

```yaml
batch_size: 8        # Reduce batch size
img_size: 224        # Lower resolution
amp: true            # Keep mixed precision
```

### If System Out of Memory

```yaml
batch_size: 4  # Minimum value
img_size: 224
num_workers: 2
```

---

## 📊 Post-Training Analysis

### Single Experiment Analysis

```powershell
# Evaluate validation set
python src/eval.py \
  --ckpt runs/model_efficientnet_b2/best_model.pt \
  --data_root data \
  --split val

# Threshold sweep
python scripts/threshold_sweep.py \
  --ckpt runs/model_efficientnet_b2/best_model.pt

# Error analysis
python scripts/error_analysis.py \
  --ckpt runs/model_efficientnet_b2/best_model.pt
```

### Batch Analysis

```powershell
# Compare all experiments
python scripts/analyze_all_experiments.py

# Generate visualizations
python scripts/plot_metrics.py

# Complete analysis report
python scripts/generate_project_report.py
```

---

## 🎯 Training Strategy Recommendations

### 🔥 If Time is Limited (4-6 hours)

Train 5 key experiments:
1. baseline_resnet18.yaml
2. model_efficientnet_b2.yaml
3. lr_0.0005.yaml
4. aug_medium.yaml
5. final_model.yaml

```powershell
.\scripts\automated_full_training.ps1 -HighPriorityOnly
```

### ⚡ If Ample Time Available (10-15 hours)

Train all 15 configurations:

```powershell
.\scripts\automated_full_training.ps1
```

### 🎓 If Learning/Demo Purpose

Train only quick test:

```powershell
python src/train.py --config src/configs/quick_test_resnet18.yaml
```

---

## 🐛 FAQ

### Q: How to resume interrupted training?

```powershell
# Continue training (if supported)
python src/train.py --config xxx.yaml --resume runs/exp/last_model.pt
```

### Q: How to modify configuration?

Edit YAML file directly or use command line parameter override:

```powershell
python src/train.py --config xxx.yaml --epochs 50 --lr 0.0003
```

### Q: GPU out of memory?

```powershell
# Reduce batch size
python src/train.py --config xxx.yaml --batch_size 4
```

### Q: How to use pretrained models?

All configs use ImageNet pretrained weights by default (`pretrained: true`).

---

## 📂 Training Outputs

Each training run generates:

```
runs/<experiment_name>/
├── best_model.pt         # Best model checkpoint
├── last_model.pt         # Last epoch checkpoint
├── metrics_history.csv   # Training metrics
├── train.log             # Training log
└── config.yaml           # Config copy
```

---

## ✅ Training Completion Checklist

- [ ] All experiments completed
- [ ] Each experiment has `best_model.pt`
- [ ] Each experiment has `metrics_history.csv`
- [ ] Ran `analyze_all_experiments.py`
- [ ] EfficientNet-B2 achieves ≥98% Macro Recall
- [ ] Tested best model
- [ ] Backed up important models

---

## 🚀 Next Steps

After training completes:

1. **Evaluate test set**
```powershell
python src/eval.py --ckpt runs/best/best_model.pt --split test
```

2. **Run demo application**
```powershell
streamlit run src/app/streamlit_app.py
```

3. **Generate report**
```powershell
python scripts/generate_project_report.py
```

---

**Need help?** Check [docs/README.md](docs/README.md) for documentation index or run `pytest tests/ -v` for tests
