# Experiment Comparison Report

**Generated:** 2025-11-19 07:03:49

**Total Experiments:** 15

---

## 🏆 Top 3 Models by Macro Recall

### 🥇 Rank 1: aug_aggressive

- **Macro Recall:** 0.9880
- **Val Accuracy:** 0.9881
- **Pneumonia Recall:** 0.9882
- **Pneumonia Precision:** 0.9952
- **Normal Recall:** 0.9878
- **Best Epoch:** 51
- **Training Time:** ~204.0 minutes

### 🥈 Rank 2: model_densenet121

- **Macro Recall:** 0.9845
- **Val Accuracy:** 0.9830
- **Pneumonia Recall:** 0.9811
- **Pneumonia Precision:** 0.9952
- **Normal Recall:** 0.9878
- **Best Epoch:** 13
- **Training Time:** ~52.0 minutes

### 🥉 Rank 3: aug_light

- **Macro Recall:** 0.9840
- **Val Accuracy:** 0.9796
- **Pneumonia Recall:** 0.9741
- **Pneumonia Precision:** 0.9976
- **Normal Recall:** 0.9939
- **Best Epoch:** 13
- **Training Time:** ~52.0 minutes

---

## 📊 Complete Experiment Comparison

| Experiment                |   Best Epoch |   Val Accuracy |   Macro Recall |   Macro F1 |   Pneumonia Recall |   Pneumonia Precision |   Pneumonia F1 |   Normal Recall |   Normal Precision |   Val Loss |   Train Time (min) |
|:--------------------------|-------------:|---------------:|---------------:|-----------:|-------------------:|----------------------:|---------------:|----------------:|-------------------:|-----------:|-------------------:|
| aug_aggressive            |           51 |         0.9881 |         0.9880 |     0.9853 |             0.9882 |                0.9952 |         0.9917 |          0.9878 |                nan |     0.0075 |           204.0000 |
| model_densenet121         |           13 |         0.9830 |         0.9845 |     0.9791 |             0.9811 |                0.9952 |         0.9881 |          0.9878 |                nan |     0.0041 |            52.0000 |
| aug_light                 |           13 |         0.9796 |         0.9840 |     0.9751 |             0.9741 |                0.9976 |         0.9857 |          0.9939 |                nan |     0.0106 |            52.0000 |
| model_efficientnet_b0     |           27 |         0.9847 |         0.9838 |     0.9811 |             0.9858 |                0.9929 |         0.9893 |          0.9817 |                nan |     0.0037 |           108.0000 |
| full_resnet18             |           10 |         0.9813 |         0.9833 |     0.9770 |             0.9788 |                0.9952 |         0.9869 |          0.9878 |                nan |     0.0029 |            40.0000 |
| final_efficientnet_b2_512 |           38 |         0.9796 |         0.9821 |     0.9750 |             0.9764 |                0.9952 |         0.9857 |          0.9878 |                nan |     0.0041 |           152.0000 |
| aug_medium                |           27 |         0.9813 |         0.9814 |     0.9770 |             0.9811 |                0.9928 |         0.9870 |          0.9817 |                nan |     0.0033 |           108.0000 |
| model_efficientnet_b2     |           20 |         0.9830 |         0.9807 |     0.9789 |             0.9858 |                0.9905 |         0.9882 |          0.9756 |                nan |     0.0036 |            80.0000 |
| lr_0.0001                 |           38 |         0.9847 |         0.9800 |     0.9809 |             0.9906 |                0.9882 |         0.9894 |          0.9695 |                nan |     0.0065 |           152.0000 |
| lr_0.001                  |           30 |         0.9813 |         0.9796 |     0.9769 |             0.9835 |                0.9905 |         0.9870 |          0.9756 |                nan |     0.0096 |           120.0000 |
| model_resnet18            |            6 |         0.9745 |         0.9786 |     0.9689 |             0.9693 |                0.9952 |         0.9821 |          0.9878 |                nan |     0.0053 |            24.0000 |
| lr_0.0005                 |            9 |         0.9762 |         0.9760 |     0.9707 |             0.9764 |                0.9904 |         0.9834 |          0.9756 |                nan |     0.0047 |            36.0000 |
| model_resnet50            |            8 |         0.9762 |         0.9760 |     0.9707 |             0.9764 |                0.9904 |         0.9834 |          0.9756 |                nan |     0.0051 |            32.0000 |
| baseline_efficientnet     |            6 |         0.9779 |         0.9753 |     0.9727 |             0.9811 |                0.9881 |         0.9846 |          0.9695 |                nan |     0.0831 |            24.0000 |
| baseline_resnet18         |           11 |         0.9694 |         0.9676 |     0.9624 |             0.9717 |                0.9856 |         0.9786 |          0.9634 |                nan |     0.0922 |            44.0000 |

---

## 🔑 Key Findings

1. **Best Overall Model:** aug_aggressive
   - Achieves 0.9880 macro recall
   - Pneumonia sensitivity: 0.9882

2. **Fastest Training:** model_resnet18
   - Training time: ~24.0 minutes
   - Macro recall: 0.9786

3. **Highest Pneumonia Recall:** lr_0.0001
   - Pneumonia recall: 0.9906
   - Minimizes false negatives (critical for medical screening)

---

**Note:** All metrics are based on validation set performance at the best epoch (selected by macro recall).
