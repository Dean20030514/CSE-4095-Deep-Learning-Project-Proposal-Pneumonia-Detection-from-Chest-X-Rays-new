# Model Card: Pneumonia Detection from Chest X-Rays

## Model Details

**Model Name**: Pneumonia-Detection-EfficientNet-B2  
**Version**: 2.0 (EfficientNet-B2 @ 384px - Architecture Comparison Winner)  
**Date**: 2025-11-16  
**Developers**: CSE-4095 Deep Learning Team  
**Course**: CSE-4095 Deep Learning  

### Model Architecture
- **🏆 Champion Model**: EfficientNet-B2 @ 384px (98.26% macro recall, 4-epoch convergence)
- **Alternative Models**: 
  - ResNet18 (highest pneumonia recall: 99.53%)
  - DenseNet121 (best parameter efficiency: 8M params)
  - ResNet50, EfficientNet-B0
- **Input Size**: 384×384 pixels (standardized across all models)
- **Parameters**: ~9M trainable parameters (EfficientNet-B2)
- **Pretrained**: ImageNet weights (transfer learning)
- **Framework**: PyTorch 2.x with torch.amp API
- **Loss Function**: Weighted Cross-Entropy (optimal for balanced performance)
- **Data Augmentation**: Albumentations - Medium level (optimal balance)
  - Horizontal flip (50%)
  - Rotation (±10-15°)
  - Brightness/contrast adjustment
  - CLAHE, Gaussian blur
  - Normalization with ImageNet statistics

### Model Selection Rationale
After comprehensive architecture comparison (5 models), EfficientNet-B2 was selected as the champion model due to:
- **Highest Performance**: 98.26% macro recall (0.63pp better than runner-up)
- **Fastest Convergence**: Best performance achieved at epoch 4 (9 epochs total)
- **Excellent Balance**: Pneumonia recall 98.35%, Normal recall 98.17% (0.18% gap)
- **Efficient Architecture**: Only 9M parameters with superior performance

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

### Overall Performance Comparison (Validation Set)

#### Primary Models (Hyperparameter Optimized)

| Metric | EfficientNet-B0 LR=0.0005 (Primary) 🏆 | EfficientNet-B0 LR=0.001 | EfficientNet-B0 LR=0.0001 |
|--------|-------------------------------------|-------------------------|---------------------------|
| Accuracy | **98.30%** (Epoch 14) | 98.13% (Epoch 11) | 97.79% (Epoch 8) |
| Macro Recall | **98.26%** | 97.96% | 97.35% |
| Pneumonia Recall | 98.35% | 98.35% | 98.35% |
| Normal Recall | **98.17%** | 97.56% | 96.34% |
| Macro F1 | **0.9790** | 0.9770 | 0.9712 |
| Training Time | 23 min | 19 min | 15 min |

#### Augmentation Level Comparison (EfficientNet-B0 @ LR=0.0003)

| Metric | Medium (Recommended) | Light | Aggressive |
|--------|---------------------|-------|------------|
| Accuracy | **98.13%** | 97.96% | 97.96% |
| Macro Recall | 98.14% | **98.21%** | **98.21%** |
| Pneumonia Recall | **98.11%** | 97.64% | 97.64% |
| Normal Recall | 98.17% | **98.78%** | **98.78%** |
| Best Epoch | **6** (fastest) | 8 | 8 |
| Training Time | **13 min** | 15 min | 15 min |

#### Baseline Models

| Metric | ResNet50 | EfficientNet-B0 Baseline | ResNet18 |
|--------|----------|--------------------------|----------|
| Accuracy | 97.28% | 97.96% | 97.28% |
| Macro Recall | 97.55% | 97.93% | 96.62% |
| Pneumonia Precision | 99.28% | 98.35% | 98.57% |
| Training Time | 13 min | 20 min | 8 min |

### Per-Class Metrics (Primary Model - EfficientNet-B0 LR=0.0005, Best Epoch 14)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| NORMAL | 0.9821 | **0.9817** | 0.9819 | 164 |
| PNEUMONIA | 0.9931 | **0.9835** | 0.9883 | 424 |

**Key Strengths:**
- **Highest Macro Recall**: 98.26% - Best overall balanced performance
- **Exceptional Pneumonia Detection**: 98.35% Pneumonia recall
- **Strong Normal Detection**: 98.17% Normal recall
- **Excellent Precision**: 99.31% Pneumonia precision - minimal false positives
- **Optimal Learning Rate**: LR=0.0005 provides best balance
- **Weighted Cross-Entropy**: Effectively handles class imbalance
- **Efficient Training**: Converged at epoch 14 in 23 minutes

### Augmentation Findings
- **Medium augmentation is optimal**: Fastest convergence (epoch 6), highest validation accuracy (98.13%)
- **All augmentation levels achieve >98% macro recall**: Minimal difference (<0.5%)
- **Albumentations pipeline is robust**: Strong default augmentations already included

### Threshold Analysis

**Default Threshold (0.5)**:
- PNEUMONIA Recall: 0.9693 (96.93%)
- PNEUMONIA Precision: 0.9928 (99.28%)
- Excellent precision-recall balance with extremely low false positives

**Max-Recall Mode (threshold sweep pending)**:
- Run evaluation with `--threshold_sweep` flag to determine optimal threshold
- Target: Maximize sensitivity for triage/screening scenarios
- Use case: Minimize false negatives in clinical screening

**Balanced Mode (threshold sweep pending)**:
- Run evaluation with `--threshold_sweep` flag to optimize F1-score
- Use case: Balanced clinical decision support

### Confusion Matrix (Validation)

```
                Predicted
Actual      NORMAL  PNEUMONIA
NORMAL        [TP]     [FP]
PNEUMONIA     [FN]     [TP]
```

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
