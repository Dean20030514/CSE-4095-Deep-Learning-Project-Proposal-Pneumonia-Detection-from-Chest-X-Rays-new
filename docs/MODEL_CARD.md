# Model Card: Pneumonia Detection from Chest X-Rays

## Model Details

**Model Name**: Pneumonia-Detection-Multi-Architecture  
**Version**: 3.0 (15-Experiment Comprehensive Analysis)  
**Date**: 2025-11-19  
**Developers**: CSE-4095 Deep Learning Team  
**Course**: CSE-4095 Deep Learning  

### Model Architecture
Based on **15 systematic experiments**, we provide multiple optimized models for different scenarios:

- **🏆 Champion Model (Highest Performance)**: aug_aggressive  
  - Architecture: EfficientNet-B0/ResNet18 @ 384px with aggressive augmentation
  - **Validation**: 98.80% macro recall, 98.82% pneumonia recall, 98.81% accuracy
  - **Test**: 97.39% macro recall, 97.18% pneumonia recall, 97.30% accuracy
  - Best for: Production deployment requiring maximum overall performance

- **🎯 Medical Screening (Highest Sensitivity)**: lr_0.0001  
  - Architecture: EfficientNet-B0 @ 384px
  - **Test**: 99.06% pneumonia recall (minimizes false negatives)
  - Best for: Screening/triage scenarios prioritizing detection rate

- **💰 Best Balance (Efficiency)**: model_densenet121  
  - Architecture: DenseNet121 @ 384px
  - **Validation**: 98.45% macro recall, 98.11% pneumonia recall
  - Training time: Only 52 minutes
  - Parameters: 7M (most parameter-efficient)
  - Best for: Resource-constrained deployments

- **Input Size**: 384×384 pixels (standardized across all models)
- **Pretrained**: ImageNet weights (transfer learning)
- **Framework**: PyTorch 2.9+ with torch.amp (CUDA 13.0 support)
- **Loss Function**: Weighted Cross-Entropy (optimal for class imbalance)
- **Data Augmentation**: Albumentations pipeline with multiple levels tested

### Model Selection Rationale
After **15 comprehensive experiments** including:
- **5 CNN architectures** (ResNet18/50, EfficientNet-B0/B2, DenseNet121)
- **3 learning rates** (0.0001, 0.0005, 0.001)
- **3 augmentation strategies** (light, medium, aggressive)
- **Total training time**: 1,400+ minutes across 300+ epochs

**Key Findings**:
- **Aggressive augmentation**: +0.4-0.8% performance gain (98.80% validation macro recall)
- **Lower learning rate (0.0001)**: Achieves 99.06% pneumonia recall (best for medical screening)
- **DenseNet121**: Best efficiency with 98.45% in only 52 minutes
- **Architecture matters more than size**: DenseNet121 (7M params) outperforms ResNet50 (25.6M params)

---

## Intended Use

### Primary Use Case
This model is designed for **educational research and demonstration purposes only**. It classifies chest X-ray images into:
- **NORMAL**: No pneumonia detected
- **PNEUMONIA**: Pneumonia detected

### Appropriate Clinical Scenarios (Educational Context)

This model demonstrates potential applications in the following **non-clinical** educational scenarios:

#### ✅ **Suitable Use Cases (Educational/Research Only)**

1. **Primary Care Triage Simulation**
   - **Context**: Educational simulation of outpatient clinics or urgent care settings
   - **Use**: Pre-screening to prioritize patients who may need urgent radiologist review
   - **Benefit**: Demonstrates how AI could help reduce radiologist workload in teaching scenarios
   - **Threshold**: Screening mode (high sensitivity, threshold ~0.15) to minimize missed cases

2. **Emergency Department Education**
   - **Context**: Teaching tool for emergency medicine training
   - **Use**: Rapid preliminary assessment simulation for time-critical cases
   - **Benefit**: Shows how AI could flag potential pneumonia for immediate physician attention
   - **Requirement**: Always paired with clinical judgment simulation exercises

3. **Resource-Limited Setting Research**
   - **Context**: Academic research on AI deployment in areas with radiologist shortages
   - **Use**: Studying feasibility of AI-assisted screening in underserved regions
   - **Benefit**: Demonstrates potential to extend diagnostic reach in educational models
   - **Limitation**: Requires validation on diverse populations before any real deployment

4. **Quality Control Demonstration**
   - **Context**: Medical imaging quality assurance education
   - **Use**: Secondary check system demonstration for missed findings
   - **Benefit**: Teaching about how AI can serve as a safety net in double-reading workflows

5. **Student Learning Tool**
   - **Context**: Radiology and medical student education
   - **Use**: Interactive learning aid for pattern recognition training
   - **Benefit**: Provides immediate feedback and visual explanations (Grad-CAM)

#### ❌ **Inappropriate Use Cases**

1. **Definitive Diagnosis**
   - ❌ Making final diagnostic decisions without radiologist confirmation
   - ❌ Determining treatment plans based solely on model output
   - ❌ Ruling out pneumonia in symptomatic patients with negative predictions

2. **High-Stakes Clinical Decisions**
   - ❌ ICU patient management or critical care decisions
   - ❌ Surgical planning or invasive procedure guidance
   - ❌ Legal/forensic medical determinations
   - ❌ Insurance claims or disability assessments

3. **Unsupervised Screening**
   - ❌ Population-wide screening without physician oversight
   - ❌ Direct-to-consumer health assessment applications
   - ❌ Automated reporting systems without human verification

4. **Pediatric vs Adult Mismatch**
   - ❌ Using pediatric-trained model on adult patients without validation
   - ❌ Applying to elderly patients with complex comorbidities
   - ❌ Neonatal or premature infant imaging

5. **Complex Cases**
   - ❌ Patients with immunocompromise (HIV, chemotherapy, etc.)
   - ❌ Post-operative or post-transplant monitoring
   - ❌ Suspected COVID-19 or other atypical pneumonias without specific training

### Clinical Integration Considerations (Educational Framework)

**Recommended Workflow (Teaching Scenario)**:
1. Image acquisition and quality check
2. Model inference with confidence display
3. **Mandatory** physician review and final interpretation
4. Documentation of both AI and physician findings
5. Feedback loop for continuous learning

**Decision Thresholds by Scenario**:
- **Screening Mode** (threshold ~0.15): Maximize sensitivity, acceptable false positive rate
- **Balanced Mode** (threshold ~0.50): Equal weighting of precision and recall
- **Confirmation Mode** (threshold ~0.75): High precision for reducing false alarms

**Warning Indicators**:
- Borderline confidence (40-60% probability) → Flag for expert review
- High confidence mismatch with clinical presentation → Investigate further
- Poor image quality detected → Manual quality assessment required

### Intended Users
- Students learning deep learning and medical imaging
- Researchers studying AI in healthcare
- Educators demonstrating ML applications

### Out-of-Scope Uses

This model should **NOT** be used in the following scenarios:

❌ **NOT** for clinical diagnosis or treatment decisions  
❌ **NOT** a replacement for radiologist interpretation  
❌ **NOT** approved as a medical device by FDA, CE, or other regulatory bodies  
❌ **NOT** for deployment in real healthcare settings without proper clinical validation  
❌ **NOT** for forensic, legal, or insurance determinations  
❌ **NOT** for direct-to-consumer health applications without physician oversight  
❌ **NOT** on patient populations significantly different from training data (e.g., adults if trained on pediatric data)  
❌ **NOT** for COVID-19 or other specific pathogen detection without targeted training

### Regulatory and Safety Considerations

**Medical Device Status**: This model is **NOT** a medical device and has not undergone regulatory review.

**Required Safeguards for Any Future Clinical Use Would Include**:

- Prospective clinical validation studies
- IRB/Ethics committee approval
- Regulatory clearance (FDA 510(k), CE marking, etc.)
- Clinical workflow integration testing
- Continuous monitoring and performance tracking
- Physician oversight and final decision authority
- Clear labeling of AI-generated outputs
- Liability and insurance considerations
- Patient consent for AI-assisted diagnosis

**Performance Monitoring**: If ever deployed in a clinical setting, the following should be tracked:

- Diagnostic accuracy compared to ground truth
- False positive/negative rates by subpopulation
- Physician override frequency and reasons
- Patient outcomes correlation
- System uptime and reliability
- Image quality rejection rates

---

## Training Data

### Dataset
- **Source**: [e.g., Kaggle Pneumonia Dataset, NIH ChestX-ray14]
- **Size**: 
  - Training: 5,216 images (1,341 NORMAL + 3,875 PNEUMONIA)
  - Validation: [N images]
  - Test: [N images]
- **Patient Demographics**: [Age range, if known]
- **Acquisition**: [e.g., Pediatric patients, Guangzhou Women and Children's Medical Center]

### Data Splits
- **Strategy**: Patient-level split (no patient appears in multiple splits)
- **Class Balance**: Training data is imbalanced (74.3% PNEUMONIA, 25.7% NORMAL)
- **Mitigation**: Weighted loss function + weighted random sampling

### Preprocessing
- Resize to [size]×[size] pixels
- Normalization: ImageNet mean/std
- Data augmentation (training only):
  - Horizontal flip (50%)
  - Rotation (±10-15°)
  - Brightness/contrast adjustment (±20%)
  - [Other augmentations]

---

## Performance Metrics

### Overall Performance - Top 5 Models (Validation Set)

| Rank | Experiment | Macro Recall | Pneumonia Recall | Val Accuracy | Training Time |
|------|------------|--------------|------------------|--------------|---------------|
| 🥇 | **aug_aggressive** | **98.80%** | 98.82% | 98.81% | 204 min |
| 🥈 | model_densenet121 | 98.45% | 98.11% | 98.30% | 52 min |
| 🥉 | aug_light | 98.40% | 97.41% | 97.96% | 52 min |
| 4 | model_efficientnet_b0 | 98.38% | 98.58% | 98.47% | 108 min |
| 5 | full_resnet18 | 98.33% | 97.88% | 98.13% | 40 min |

### Test Set Performance (Final Evaluation - aug_aggressive)

| Metric | Value | Notes |
|--------|-------|-------|
| **Overall Accuracy** | **97.30%** | Unbiased final evaluation |
| **Macro Recall** | **97.39%** | Primary KPI |
| **Pneumonia Recall (Sensitivity)** | **97.18%** | Detected 207/213 cases |
| **Normal Recall (Specificity)** | **97.59%** | Detected 81/83 cases |
| **ROC-AUC** | **99.73%** | Excellent discrimination |
| **PR-AUC** | **99.89%** | Outstanding precision-recall tradeoff |
| **False Positives** | 2 (2.41%) | Minimal false alarms |
| **False Negatives** | 6 (2.82%) | Acceptable missed cases |

**Confusion Matrix (Test Set)**:
```
                 Predicted
Actual       NORMAL  PNEUMONIA
NORMAL         81       2        (97.59% recall)
PNEUMONIA       6     207        (97.18% recall)
```

### Scenario-Specific Model Recommendations

#### 1. Maximum Performance (Production Deployment)
**Model**: `aug_aggressive`
- Validation: 98.80% macro recall, 98.82% pneumonia recall
- Test: 97.30% accuracy, 97.18% pneumonia recall
- Training: 204 minutes (acceptable for best performance)

#### 2. Medical Screening (Minimize False Negatives)
**Model**: `lr_0.0001`
- **Pneumonia Recall: 99.06%** (highest sensitivity)
- Only 2 false negatives out of 213 pneumonia cases
- Optimal for: Triage, screening programs, high-risk populations

#### 3. Balanced Efficiency (Production-Ready)
**Model**: `model_densenet121`
- 98.45% macro recall in only 52 minutes
- 7M parameters (most efficient)
- Optimal for: Resource-constrained settings, rapid deployment

#### 4. Rapid Prototyping (Development)
**Model**: `model_resnet18`
- 97.86% macro recall in 24 minutes
- Fast iteration cycles
- Optimal for: Research, experimentation, quick testing

### Architecture Comparison (Validation Set)

| Architecture | Macro Recall | Parameters | Time | Efficiency Score |
|--------------|--------------|------------|------|------------------|
| EfficientNet-B0 | 98.38% | 5.3M | 108 min | 0.911 |
| **DenseNet121** | **98.45%** | **7M** | **52 min** | **1.893** 🏆 |
| ResNet18 | 97.86% | 11.2M | 24 min | **4.078** ⚡ |
| ResNet50 | 97.60% | 25.6M | 32 min | 3.050 |
| EfficientNet-B2 | 98.07% | 9M | 80 min | 1.226 |

### Hyperparameter Analysis

#### Learning Rate Impact
| LR | Macro Recall | Pneumonia Recall | Best Epoch |
|----|--------------|------------------|------------|
| 0.0001 | 98.00% | **99.06%** ⭐ | 38 |
| **0.0005** | **98.45%** 🏆 | 98.11% | 9 |
| 0.001 | 97.96% | 98.35% | 30 |

**Finding**: LR=0.0001 achieves highest pneumonia recall (99.06%) but requires longer training. LR=0.0005 provides best overall balance.

#### Data Augmentation Impact
| Level | Macro Recall | Val Accuracy | Training Time |
|-------|--------------|--------------|---------------|
| **Aggressive** | **98.80%** 🏆 | 98.81% | 204 min |
| Medium | 98.14% | 98.13% | 108 min |
| Light | 98.40% | 97.96% | 52 min |

**Finding**: Aggressive augmentation provides +0.4-0.8% performance gain, strong regularization effect prevents overfitting.

### Threshold Analysis (Test Set)

| Mode | Threshold | Recall | Precision | F1 | Use Case |
|------|-----------|--------|-----------|----|----|
| **Screening** | 0.10 | **99.06%** | 97.24% | 98.14% | Maximize detection |
| **Balanced** | 0.15 | 99.06% | 98.14% | **98.60%** | General use |
| **Confirmatory** | 0.525 | 97.18% | **99.52%** | 98.34% | Minimize false positives |

**Key Insight**: Lowering threshold to 0.10-0.15 can achieve 99.06% pneumonia recall (only 2 missed cases) while maintaining 97-98% precision.

---

## Training Configuration

### Hyperparameters (EfficientNet-B0 Primary Model)
- **Epochs**: 25 (early stopped at epoch 19)
- **Batch Size**: 16
- **Learning Rate**: 5.0e-4 (optimal, determined via grid search: 0.0001, 0.0005, 0.001)
- **Weight Decay**: 1.0e-4
- **Optimizer**: AdamW
- **Scheduler**: CosineAnnealingLR (T_max=25)
- **Loss Function**: Weighted Cross-Entropy (optimal for balanced performance)
- **Mixed Precision**: Yes (torch.amp with CUDA)
- **Random Seed**: 42
- **Data Augmentation**: Albumentations - Medium level (optimal)
  - Horizontal flip: 50%
  - Rotation: ±10-15°
  - Brightness/contrast: ±15-20%
  - CLAHE: clip_limit=2.0, p=0.3
  - Gaussian blur: p=0.2
  - ShiftScaleRotate: p=0.5

### Training Details
- **Device**: NVIDIA GPU (CUDA enabled)
- **Training Time**: ~23 minutes (19 epochs before early stopping)
- **Early Stopping**: Triggered at epoch 19 (patience=5, no improvement after epoch 14)
- **Best Checkpoint**: Saved at epoch 14 with 98.26% macro recall
- **Best Epoch**: 14 (macro_recall = 0.9826, pneumonia_recall = 0.9835, normal_recall = 0.9817)
- **Hyperparameter Tuning**: 3 learning rates tested (0.0001, 0.0005, 0.001) - 0.0005 optimal
- **Augmentation Tuning**: 3 levels tested (light, medium, aggressive) - medium optimal

---

## Explainability

### Grad-CAM Visualization
The model uses Grad-CAM to highlight regions of interest in chest X-rays:
- ✅ **Typical behavior**: Activations focus on lung fields
- ⚠️ **Caution**: Heatmaps may highlight artifacts, device shadows, or text overlays

**Example Interpretations**:
- PNEUMONIA cases: Model attends to opacified/infiltrated regions
- NORMAL cases: Model shows diffuse attention across clear lung fields

[Include 2-3 Grad-CAM example images here]

---

## Limitations & Risks

### Known Limitations

1. **Class Imbalance**: Training data is skewed toward PNEUMONIA (74%), which may bias predictions
2. **Dataset Scope**: Trained primarily on pediatric chest X-rays; generalization to adult populations uncertain
3. **Image Quality**: Performance degrades with low-quality, overexposed, or heavily artifacted images
4. **Subtle Cases**: May miss early-stage or minimal pneumonia
5. **Bacterial vs Viral**: Cannot distinguish between bacterial and viral pneumonia subtypes

### Failure Modes

**False Negatives (Missed Pneumonia)**:
- Subtle/early infiltrates
- Low-contrast images
- Overlapping structures (e.g., heart shadow)

**False Positives (Over-diagnosis)**:
- Device shadows or wires misclassified as infiltrates
- Motion artifacts
- Normal anatomical variants

**Confidence Miscalibration**:
- Model may be overconfident in borderline cases
- Temperature scaling applied: [Yes/No]

### Ethical Considerations
- **Bias**: Dataset may not represent diverse patient populations
- **Harm Potential**: Incorrect predictions could delay treatment (FN) or cause unnecessary anxiety (FP)
- **Transparency**: Model decisions must be reviewed by qualified clinicians

---

## Robustness Testing

### Stress Tests (Optional)
- **Gaussian Noise**: [Performance at σ=X]
- **Blur**: [Performance at kernel size X]
- **Contrast Reduction**: [Performance at -X%]

### External Validation (Optional)
- **Dataset**: [e.g., CheXpert subset]
- **Performance**: [Metrics on external data]
- **Domain Shift**: [Observed degradation]

---

## Calibration

**Expected Calibration Error (ECE)**: [X.XX]  
**Maximum Calibration Error (MCE)**: [X.XX]  
**Brier Score**: [X.XX]

[Include reliability diagram if available]

---

## Model Updates & Maintenance

### Version History
- **v1.0** ([Date]): Initial release for course project

### Future Improvements
- [ ] Expand to multi-class (Normal/Bacterial/Viral)
- [ ] Train on larger, more diverse datasets
- [ ] Improve calibration via temperature scaling
- [ ] Add external validation on public benchmarks
- [ ] Investigate adversarial robustness

---

## References

1. Rajpurkar et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.
2. Irvin et al. (2019). CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels.
3. Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
4. Lin et al. (2017). Focal Loss for Dense Object Detection.

---

## Contact & Code

**Developers**: [Your Names / Emails]  
**Repository**: [GitHub URL]  
**License**: [e.g., MIT, Apache 2.0]  

---

**Disclaimer**: This model is for educational purposes only. It has not been validated for clinical use and should not be used for medical diagnosis or treatment decisions. Always consult a qualified healthcare professional for medical advice.
