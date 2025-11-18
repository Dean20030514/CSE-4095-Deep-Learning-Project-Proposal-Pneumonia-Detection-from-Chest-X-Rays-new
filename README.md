# Pneumonia Detection from Chest X-Rays

> **Deep Learning Project - Binary Classification for Medical Triage**  
> ⚠️ **Educational Use Only** - Not for clinical diagnosis

---

## 🎯 Navigation Guide

### 👨‍🏫 For Instructors / TAs

**Start here to evaluate the project:**

1. **📄 [docs/FINAL_PROJECT_REPORT.md](docs/FINAL_PROJECT_REPORT.md)** - Complete academic report with all results, analysis, and conclusions
2. **📋 [docs/EXECUTIVE_SUMMARY_EN.md](docs/EXECUTIVE_SUMMARY_EN.md)** - Executive summary of the project
3. **🎤 [docs/PRESENTATION_SCRIPT.md](docs/PRESENTATION_SCRIPT.md)** - Presentation script with speaker notes
4. **🎨 [docs/PRESENTATION_SLIDES_OUTLINE.md](docs/PRESENTATION_SLIDES_OUTLINE.md)** - Detailed presentation structure
5. **📊 [reports/comprehensive/EXPERIMENT_SUMMARY.md](reports/comprehensive/EXPERIMENT_SUMMARY.md)** - Full experimental comparison table
6. **🎴 [docs/MODEL_CARD.md](docs/MODEL_CARD.md)** - Model documentation with limitations and ethical considerations

### 🧑‍💻 For Students Reproducing the Project

**Follow this learning path:**

1. **📖 [docs/PLAYBOOK.md](docs/PLAYBOOK.md)** - Comprehensive implementation playbook
2. **🔬 [docs/ANALYSIS_GUIDE.md](docs/ANALYSIS_GUIDE.md)** - How to run all analysis scripts

### 🔬 For Researchers Extending the Work

**Technical deep-dives:**

1. **📈 [reports/calibration/calibration_report.json](reports/calibration/calibration_report.json)** - Model calibration analysis
2. **🔍 [reports/error_analysis/failure_modes.json](reports/error_analysis/failure_modes.json)** - Detailed error analysis

---

## 📊 Project Status

**🏆 Best Model:** EfficientNet-B2 @ 384px  
**📈 Performance:** 98.26% Macro Recall | 98.30% Accuracy | 98.35% Pneumonia Recall  
**✅ Status:** All experiments completed, production-ready

---

## ⚡ Quick Start

**🎮 GPU Acceleration Enabled: CUDA 13.0 + RTX 5070** ⚡

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

### 3. Train

```powershell
# Quick test (3 epochs, ~10 minutes) 🔧
python src/train.py --config src/configs/quick_test_resnet18.yaml

# Best performance (98.26% macro recall) ⭐
python src/train.py --config src/configs/model_efficientnet_b2.yaml

# Final production model (highest quality) 🏆
python src/train.py --config src/configs/final_model.yaml
```

📖 **Detailed Training Guide:** [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

### 4. Evaluate

```powershell
python src/eval.py --ckpt runs/model_efficientnet_b2/best_model.pt --data_root data --split val
```

### 5. Demo

```powershell
streamlit run src/app/streamlit_app.py
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

| Rank | Model | Macro Recall | Accuracy | Pneumonia Recall | Time (GPU) |
|------|-------|--------------|----------|------------------|------------|
| 🥇 | EfficientNet-B2 | 98.26% | 98.30% | 98.35% | ~20 min ⚡ |
| 🥈 | ResNet18 | 97.63% | 97.79% | 99.53% | ~15 min ⚡ |
| 🥉 | DenseNet121 | 97.60% | 97.62% | 96.93% | ~18 min ⚡ |

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

### Core Documentation (by priority)

1. **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** ⭐ - Complete Training Guide
2. **[docs/FINAL_PROJECT_REPORT.md](docs/FINAL_PROJECT_REPORT.md)** - Complete Technical Report
3. **[docs/MODEL_CARD.md](docs/MODEL_CARD.md)** - Model Documentation
4. **[docs/PLAYBOOK.md](docs/PLAYBOOK.md)** - Implementation Guide
5. **[docs/ANALYSIS_GUIDE.md](docs/ANALYSIS_GUIDE.md)** - Analysis Tools
6. **[docs/EXECUTIVE_SUMMARY_EN.md](docs/EXECUTIVE_SUMMARY_EN.md)** - Executive Summary

### Configuration Files

- **[pyproject.toml](pyproject.toml)** - Project configuration (includes pytest settings)
- **[.pre-commit-config.yaml](.pre-commit-config.yaml)** - Git hooks
- **[LICENSE](LICENSE)** - MIT License + Medical Disclaimer

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

### Available Configs (17 total)

**Recommended configs:**
- `model_efficientnet_b2.yaml` ⭐ - EfficientNet-B2 @ 384px (best: 98.26% macro recall)
- `final_model.yaml` 🏆 - EfficientNet-B2 @ 512px (final production)
- `lr_0.0005.yaml` - Optimal learning rate experiment
- `quick_test_resnet18.yaml` 🔧 - Quick test (3 epochs, ~10 min)

**All configs:** See [src/configs/README.md](src/configs/README.md) for complete list

### Custom Training

```powershell
python src/train.py --config <path> \
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

## 🧪 Testing

### Run Tests

```powershell
# Basic tests
pytest tests/ -v

# Coverage report
pytest tests/ --cov=src --cov-report=html

# Using scripts (recommended)
.\scripts\run_tests.ps1 -Coverage -Lint  # Windows
bash scripts/run_tests.sh                # Linux/Mac
```

**Test Status:** ✅ 31/31 tests passed (100%)

📖 **Test Documentation:** See [tests/README.md](tests/README.md)

---

## 🛠️ Development

### Install Development Dependencies

```powershell
pip install -r requirements-dev.txt
```

### Code Quality Tools

```powershell
# Format code
black src/ tests/
isort src/ tests/

# Check code
flake8 src/ tests/
mypy src/

# Enable Git hooks (auto-check)
pre-commit install
pre-commit run --all-files
```

### Configuration Validation

```powershell
python src/utils/config_validator.py src/configs/final_model.yaml
```

---

## 🛠️ Requirements

- Python 3.8+ (3.13+ recommended)
- PyTorch 2.9+
- **CUDA 13.0 support** ✅ GPU acceleration available
- 8GB RAM (16GB recommended)
- GPU with 8GB+ VRAM (tested on RTX 5070)

---

## 📝 Citation

```text
Pneumonia Detection from Chest X-Rays using Deep Learning
Deep Learning Course Project, 2025
```

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

⚠️ **Important Notice:** This software is for educational and research purposes only, **NOT for clinical diagnosis**. See the medical disclaimer in the LICENSE file.

---

**Last Updated:** 2025-11-18 | **Status:** ✅ Production-Ready | **GPU:** CUDA 13.0 + RTX 5070 ⚡
