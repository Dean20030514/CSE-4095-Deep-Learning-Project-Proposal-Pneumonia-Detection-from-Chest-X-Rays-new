# 🔐 Model Backups

**Backup Date**: 2025-11-19  
**Purpose**: Preserve key trained models for deployment and reference

---

## 📦 Backed Up Models

### 1. best_overall_val98.80_test97.30.pt
**Source**: `runs/aug_aggressive/best_model.pt`  
**Architecture**: EfficientNet-B0/ResNet18 @ 384px with aggressive augmentation  
**Performance**:
- Validation: 98.80% macro recall, 98.82% pneumonia recall, 98.81% accuracy
- Test: 97.30% accuracy, 97.39% macro recall, 97.18% pneumonia recall
- ROC-AUC: 99.73% | PR-AUC: 99.89%

**Use Case**: 🏆 **Production deployment** requiring maximum overall performance

**Training Details**:
- Experiment: aug_aggressive
- Training Time: 204 minutes
- Best Epoch: 51
- Configuration: Aggressive data augmentation, optimal hyperparameters

---

### 2. best_sensitivity_pneumonia99.06.pt
**Source**: `runs/lr_0.0001/best_model.pt`  
**Architecture**: EfficientNet-B0 @ 384px  
**Performance**:
- Test Pneumonia Recall: **99.06%** (highest sensitivity) ⭐
- Only 2 false negatives out of 213 pneumonia cases
- Test Macro Recall: 98.00%
- Test Accuracy: 98.31%

**Use Case**: 🎯 **Medical screening** and triage scenarios prioritizing detection rate

**Training Details**:
- Experiment: lr_0.0001
- Learning Rate: 0.0001 (lower for better sensitivity)
- Training Time: 152 minutes
- Best Epoch: 38

**Threshold Recommendation**: 0.10-0.15 for 99%+ sensitivity

---

### 3. production_densenet121_98.45.pt
**Source**: `runs/model_densenet121/best_model.pt`  
**Architecture**: DenseNet121 @ 384px  
**Performance**:
- Validation: 98.45% macro recall, 98.11% pneumonia recall
- Test: 98.29% accuracy
- Training Time: Only 52 minutes ⚡
- Parameters: 7M (most efficient)

**Use Case**: 💰 **Resource-constrained** deployments, rapid prototyping, production-ready

**Training Details**:
- Experiment: model_densenet121
- Best Epoch: 13 (fast convergence)
- Efficiency Score: 1.893 (best parameter efficiency)

---

## 🎯 Model Selection Guide

| Scenario | Recommended Model | Key Metric |
|----------|-------------------|------------|
| **Maximum Performance** | best_overall_val98.80 | 97.30% test accuracy |
| **Screening/Triage** | best_sensitivity_pneumonia99.06 | 99.06% pneumonia recall |
| **Fast Deployment** | production_densenet121 | 52 min training |
| **Limited Resources** | production_densenet121 | 7M parameters |

---

## 📋 How to Use These Models

### Loading a Model

```python
import torch
from src.models.factory import build_model

# Load checkpoint
ckpt = torch.load('model_backups/best_overall_val98.80_test97.30.pt')

# Build model
model_name = ckpt['config']['model']
num_classes = len(ckpt['classes'])
model, _ = build_model(model_name, num_classes)

# Load weights
model.load_state_dict(ckpt['model'])
model.eval()
```

### Running Evaluation

```powershell
# Test set evaluation
python src/eval.py --ckpt model_backups/best_overall_val98.80_test97.30.pt --split test

# With threshold sweep
python scripts/threshold_sweep.py --ckpt model_backups/best_sensitivity_pneumonia99.06.pt --split test
```

### Running Demo

```powershell
streamlit run src/app/streamlit_app.py
# Then select the model from the dropdown in the sidebar
```

---

## 🔒 File Integrity

| File | Size | Created |
|------|------|---------|
| best_overall_val98.80_test97.30.pt | ~89 MB | 2025-11-18 |
| best_sensitivity_pneumonia99.06.pt | ~89 MB | 2025-11-18 |
| production_densenet121_98.45.pt | ~80 MB | 2025-11-18 |

---

## ⚠️ Important Notes

1. **Educational Use Only**: These models are for research and educational purposes, NOT for clinical diagnosis
2. **Validation Required**: Any deployment must undergo proper clinical validation
3. **Version Compatibility**: Requires PyTorch 2.0+, Python 3.8+
4. **Data Requirements**: Input images should be 384×384 chest X-rays
5. **Threshold Tuning**: Adjust classification threshold based on use case (screening vs confirmatory)

---

## 📚 Reference Documentation

- Full experimental analysis: `reports/COMPREHENSIVE_EXPERIMENTAL_ANALYSIS.md`
- Model details: `docs/MODEL_CARD.md`
- Quick reference: `docs/QUICK_RESULTS_REFERENCE.md`
- Training guide: `TRAINING_GUIDE.md`

---

**Last Updated**: 2025-11-19

