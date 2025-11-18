# Experiment Comparison Report

> ⚠️ **Note:** This report is from 2025-11-16. Run `python scripts/analyze_all_experiments.py` to generate an updated report with all completed experiments.

**Generated:** 2025-11-16 18:09:17

**Total Experiments:** 14 (more experiments available, needs update)

---

## 🏆 Top 3 Models by Macro Recall

### 🥇 Rank 1: model_efficientnet_b2

- **Macro Recall:** 0.9826
- **Val Accuracy:** 0.9830
- **Pneumonia Recall:** 0.9835
- **Pneumonia Precision:** 0.9929
- **Normal Recall:** 0.9817
- **Best Epoch:** 4
- **Training Time:** ~0.0 minutes

### 🥈 Rank 2: lr_0.0005

- **Macro Recall:** 0.9826
- **Val Accuracy:** 0.9830
- **Pneumonia Recall:** 0.9835
- **Pneumonia Precision:** 0.9929
- **Normal Recall:** 0.9817
- **Best Epoch:** 14
- **Training Time:** ~0.0 minutes

### 🥉 Rank 3: aug_aggressive

- **Macro Recall:** 0.9821
- **Val Accuracy:** 0.9796
- **Pneumonia Recall:** 0.9764
- **Pneumonia Precision:** 0.9952
- **Normal Recall:** 0.9878
- **Best Epoch:** 8
- **Training Time:** ~0.0 minutes

---

## 📊 Complete Experiment Comparison

| Experiment            |   Best Epoch |   Val Accuracy |   Macro Recall |   Macro F1 |   Pneumonia Recall |   Pneumonia Precision |   Pneumonia F1 |   Normal Recall |   Normal Precision |   Val Loss |   Train Time (min) |
|:----------------------|-------------:|---------------:|---------------:|-----------:|-------------------:|----------------------:|---------------:|----------------:|-------------------:|-----------:|-------------------:|
| model_efficientnet_b2 |            4 |         0.9830 |         0.9826 |     0.9790 |             0.9835 |                0.9929 |         0.9882 |          0.9817 |                nan |     0.0666 |             0.0000 |
| lr_0.0005             |           14 |         0.9830 |         0.9826 |     0.9790 |             0.9835 |                0.9929 |         0.9882 |          0.9817 |                nan |     0.0773 |             0.0000 |
| aug_aggressive        |            8 |         0.9796 |         0.9821 |     0.9750 |             0.9764 |                0.9952 |         0.9857 |          0.9878 |                nan |     0.0742 |             0.0000 |
| aug_light             |            8 |         0.9796 |         0.9821 |     0.9750 |             0.9764 |                0.9952 |         0.9857 |          0.9878 |                nan |     0.0689 |             0.0000 |
| aug_medium            |            6 |         0.9813 |         0.9814 |     0.9770 |             0.9811 |                0.9928 |         0.9870 |          0.9817 |                nan |     0.0637 |             0.0000 |
| lr_0.001              |           11 |         0.9813 |         0.9796 |     0.9769 |             0.9835 |                0.9905 |         0.9870 |          0.9756 |                nan |     0.0771 |             0.0000 |
| baseline_efficientnet |            6 |         0.9728 |         0.9793 |     0.9670 |             0.9646 |                0.9976 |         0.9808 |          0.9939 |                nan |     0.0866 |             0.0000 |
| model_resnet18        |           13 |         0.9847 |         0.9763 |     0.9808 |             0.9953 |                0.9837 |         0.9894 |          0.9573 |                nan |     0.0687 |             0.0000 |
| model_densenet121     |            4 |         0.9762 |         0.9760 |     0.9707 |             0.9764 |                0.9904 |         0.9834 |          0.9756 |                nan |     0.0856 |             0.0000 |
| full_resnet18         |            6 |         0.9728 |         0.9755 |     0.9668 |             0.9693 |                0.9928 |         0.9809 |          0.9817 |                nan |     0.0059 |             0.0000 |
| model_resnet50        |           13 |         0.9779 |         0.9753 |     0.9727 |             0.9811 |                0.9881 |         0.9846 |          0.9695 |                nan |     0.0753 |             0.0000 |
| model_efficientnet_b0 |            6 |         0.9762 |         0.9741 |     0.9706 |             0.9788 |                0.9881 |         0.9834 |          0.9695 |                nan |     0.0620 |             0.0000 |
| lr_0.0001             |            8 |         0.9779 |         0.9735 |     0.9726 |             0.9835 |                0.9858 |         0.9847 |          0.9634 |                nan |     0.0570 |             0.0000 |
| baseline_resnet18     |            9 |         0.9728 |         0.9662 |     0.9662 |             0.9811 |                0.9811 |         0.9811 |          0.9512 |                nan |     0.0799 |             0.0000 |

---

## 🔑 Key Findings

1. **Best Overall Model:** model_efficientnet_b2
   - Achieves 0.9826 macro recall
   - Pneumonia sensitivity: 0.9835

2. **Fastest Training:** model_efficientnet_b2
   - Training time: ~0.0 minutes
   - Macro recall: 0.9826

3. **Highest Pneumonia Recall:** model_resnet18
   - Pneumonia recall: 0.9953
   - Minimizes false negatives (critical for medical screening)

---

**Note:** All metrics are based on validation set performance at the best epoch (selected by macro recall).
