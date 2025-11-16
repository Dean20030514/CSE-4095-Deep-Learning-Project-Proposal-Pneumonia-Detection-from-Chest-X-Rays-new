# Pneumonia Detection from Chest X-Rays

> **Deep Learning Project - Binary Classification for Medical Triage**  
> ⚠️ **Educational Use Only** - Not for clinical diagnosis

---

## 📊 Project Status

**🏆 Best Model:** EfficientNet-B2 @ 384px  
**📈 Performance:** 98.26% Macro Recall | 98.30% Accuracy | 98.35% Pneumonia Recall  
**✅ Status:** All experiments completed, production-ready

---

## ⚡ Quick Start

### 1. Setup (5 min)

```powershell
# Option A: Conda (Recommended)
conda env create -f environment.yml
conda activate cxr

# Option B: pip + venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Verify

```powershell
python scripts/verify_environment.py
python scripts/verify_dataset_integrity.py
```

### 3. Train (15 min)

```powershell
# Best config
python -m src.train --config src/configs/balanced_training.yaml

# Quick test (2 epochs)
python -m src.train --config src/configs/colab_friendly.yaml --epochs 2
```

### 4. Evaluate

```powershell
python -m src.eval --ckpt runs/model_efficientnet_b2/best.pt --split val
```

### 5. Demo

```powershell
streamlit run src/app/streamlit_app.py -- --ckpt runs/model_efficientnet_b2/best.pt
```

---

## 📂 Project Structure

```
├── data/                    # Dataset (5,891 images)
├── src/                    # Core code
│   ├── train.py           # Training script
│   ├── eval.py            # Evaluation
│   ├── configs/           # YAML configs
│   └── utils/             # Helpers + metrics
├── scripts/               # Analysis tools
│   ├── analyze_all_experiments.py
│   ├── threshold_sweep.py
│   ├── calibration_analysis.py
│   └── error_analysis.py
├── runs/                  # Experiment outputs
└── docs/                  # Documentation
```

---

## 🎯 Key Features

- **Multiple Architectures:** ResNet, EfficientNet, DenseNet
- **Class Imbalance Handling:** Weighted sampling + Focal Loss
- **Explainability:** Grad-CAM visualization
- **Comprehensive Analysis:** Threshold sweep, calibration, error analysis
- **Production Ready:** Streamlit demo app

---

## 📈 Top Models

| Rank | Model | Macro Recall | Accuracy | Pneumonia Recall | Time |
|------|-------|--------------|----------|------------------|------|
| 🥇 | EfficientNet-B2 | 98.26% | 98.30% | 98.35% | 14 min |
| 🥈 | ResNet18 | 97.63% | 97.79% | 99.53% | 25 min |
| 🥉 | DenseNet121 | 97.60% | 97.62% | 96.93% | 20 min |

**Key Findings:**
- Best overall: EfficientNet-B2 (fast convergence + balanced)
- Highest sensitivity: ResNet18 (99.53% pneumonia recall)
- Most efficient: DenseNet121 (8M params)
- Optimal LR: 0.0005 | Best augmentation: Medium

---

## 🔬 Analysis Tools

### One-Click Analysis

```powershell
# Validation set (tuning)
.\scripts\run_full_analysis.ps1 -Split val

# Test set (final evaluation)
.\scripts\run_full_analysis.ps1 -Split test
```

**Generates:**
- Experiment comparison + rankings
- Threshold sweep (5 clinical modes)
- Calibration analysis (ECE, Brier score)
- Error analysis (FP/FN galleries + failure modes)

### Individual Tools

```powershell
python scripts/analyze_all_experiments.py
python scripts/threshold_sweep.py --ckpt runs/model_efficientnet_b2/best.pt
python scripts/calibration_analysis.py --ckpt runs/model_efficientnet_b2/best.pt
python scripts/error_analysis.py --ckpt runs/model_efficientnet_b2/best.pt
```

---

## 📖 Documentation

- **[Analysis Guide](docs/ANALYSIS_GUIDE.md)** - Complete toolkit documentation
- **[Model Card](MODEL_CARD.md)** - Model specifications
- **[Playbook](docs/PLAYBOOK.md)** - Full implementation guide
- **[Code Summary](docs/CODE_OPTIMIZATION_SUMMARY.md)** - Optimization details
- **[Changelog](docs/CHANGELOG.md)** - Project history

---

## 🎓 Dataset

**Source:** Kaggle Chest X-Ray Images (Pneumonia)  
**Split:** 85/10/5 (train/val/test) - stratified, seed=42

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,399 | 3,608 | 5,007 (85%) |
| Val | 164 | 424 | 588 (10%) |
| Test | 83 | 213 | 296 (5%) |
| **Total** | **1,646** | **4,245** | **5,891** |

**Size:** 1.19 GB (fully deduplicated)

---

## 🚀 Training Configurations

### Available Configs

- `balanced_training.yaml` - **Recommended** (best overall)
- `colab_friendly.yaml` - Low-resource option
- `final_model.yaml` - EfficientNet-B2 @ 512px (production)
- `medical_screening.yaml` - Max recall optimization
- `ensemble_*.yaml` - For ensemble learning

### Custom Training

```powershell
python -m src.train --config <path> \
  --epochs 25 \
  --lr 0.0005 \
  --batch_size 16 \
  --augment_level medium
```

---

## 📊 Metrics Tracked

**Primary:** Pneumonia Recall (sensitivity)  
**Secondary:** Accuracy, Precision, F1, Macro Recall  
**Advanced:** ROC-AUC, PR-AUC, MCC, Cohen's Kappa, ECE

---

## ⚠️ Medical Disclaimer

> **Educational and research purposes only.**  
> **NOT approved for clinical diagnosis or treatment decisions.**  
> Always consult qualified healthcare professionals.

### Ethical Considerations

- False negatives (missed pneumonia) > false positives
- Use low thresholds for screening/triage
- Requires validation on local data before deployment
- May not generalize to all populations

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU)
- 8GB RAM (16GB recommended)
- GPU with 6GB+ VRAM (optional)

---

## 📝 Citation

```
Pneumonia Detection from Chest X-Rays using Deep Learning
Deep Learning Course Project, 2025
```

---

## 📄 License

Educational use only. Dataset from Kaggle.

---

**Last Updated:** 2025-11-16 | **Status:** ✅ Complete
