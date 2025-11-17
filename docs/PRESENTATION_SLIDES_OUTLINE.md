# Presentation Slide Deck Outline

**Title:** Pneumonia Detection from Chest X-Rays: A Deep Learning Approach  
**Course:** CSE-4095 Deep Learning  
**Total Slides:** 20 (+ backup slides)  
**Duration:** 7-8 minutes

---

## Slide Structure & Content

### Slide 1: Title Slide
**Layout:** Title + Subtitle + Image  
**Content:**
- **Title:** Pneumonia Detection from Chest X-Rays
- **Subtitle:** A Deep Learning Approach to Medical Screening
- **Author:** [Your Name]
- **Course:** CSE-4095 Deep Learning, Fall 2025
- **Visual:** Chest X-ray image (1 normal, 1 pneumonia side-by-side)

**Notes:** Keep simple, professional. Use high-quality medical images.

---

### Slide 2: Motivation & Global Context
**Layout:** Text + Statistics Visualization  
**Content:**
- **Title:** Why Pneumonia Detection Matters
- **Key Points:**
  - Pneumonia: 15% of child deaths globally (WHO)
  - Resource constraints: Shortage of radiologists
  - Diagnostic delays in low-resource settings
  - AI potential: Triage support, second reader system
- **Visual:** 
  - WHO statistics infographic
  - World map showing pneumonia burden
  - Icon: Stethoscope + AI chip

**Notes:** Establish clinical urgency and AI's role (support, not replacement).

---

### Slide 3: Dataset Challenge
**Layout:** Problem Statement + Visual Comparison  
**Content:**
- **Title:** Dataset Quality: The Foundation of Trust
- **Original Dataset Issues:**
  - ❌ Duplicate images (inflates metrics)
  - ❌ Random splits (data leakage risk)
  - ❌ Severe imbalance (3:1 ratio)
  - ❌ Poor validation set (8 images!)
- **Visual:**
  - Before/After comparison table
  - Diagram showing "same patient in train & val" problem
  - Red X marks on problematic aspects

**Notes:** Emphasize that dataset quality beats model complexity.

---

### Slide 4: Optimal Dataset Construction
**Layout:** Solution + Comparison Table  
**Content:**
- **Title:** Rebuilding the Dataset: Patient-Level Splits
- **Our Solutions:**
  - ✅ De-duplication (perceptual hashing)
  - ✅ Patient-level separation (no leakage)
  - ✅ Stratified sampling (85/10/5)
  - ✅ Class balance maintained
- **Table:**
  | Split | Total | Normal | Pneumonia | %Pneumonia |
  |-------|-------|--------|-----------|------------|
  | Train | 4,683 | 1,170  | 3,513     | 75.0%      |
  | Val   | 589   | 148    | 441       | 74.9%      |
  | Test  | 296   | 83     | 213       | 71.9%      |
- **Visual:** Flowchart: Raw Data → De-dup → Patient Group → Stratify → Clean Splits

**Notes:** Show the process, highlight patient-level as key innovation.

---

### Slide 5: Model Architecture Comparison
**Layout:** Comparison Table + Architecture Icons  
**Content:**
- **Title:** Neural Network Architectures: 5 Candidates
- **Comparison Table:**
  | Model | Parameters | Input Size | Key Strength |
  |-------|-----------|------------|--------------|
  | EfficientNet-B0 | 5.3M | 224×224 | Efficiency |
  | **EfficientNet-B2** ⭐ | **9.2M** | **384×384** | **High-res + balance** |
  | ResNet-18 | 11.7M | 224×224 | Fast training |
  | ResNet-50 | 25.6M | 224×224 | Deep features |
  | DenseNet-121 | 8.0M | 224×224 | Dense connectivity |
- **Visual:**
  - Architecture icons/logos
  - Highlight EfficientNet-B2 with gold star
  - Size vs. performance scatter plot

**Notes:** Explain why B2 @ 384px: resolution matters for X-rays.

---

### Slide 6: Training Strategy
**Layout:** Flowchart + Key Parameters  
**Content:**
- **Title:** Training Pipeline: Transfer Learning + Optimization
- **Flowchart:**
  ```
  ImageNet Pre-trained Weights
         ↓
  Fine-tune All Layers (LR=5e-4)
         ↓
  Weighted Cross-Entropy (1.33 on pneumonia)
         ↓
  Data Augmentation (flip, rotate, jitter)
         ↓
  Early Stopping (patience=5, monitor macro recall)
         ↓
  Best Model @ Epoch 4
  ```
- **Key Hyperparameters:**
  - Optimizer: AdamW (weight decay 1e-4)
  - Batch Size: 32
  - Max Epochs: 30 (early stop ~10-15)
  - Learning Rate Schedule: ReduceLROnPlateau
- **Visual:** Pipeline diagram with icons

**Notes:** Emphasize early stopping on macro recall (balanced metric).

---

### Slide 7: Evaluation Metrics & Clinical Priority
**Layout:** Pyramid/Hierarchy Diagram  
**Content:**
- **Title:** Not All Metrics Are Equal: Medical Priorities
- **Metrics Hierarchy (Pyramid):**
  ```
        ▲ ROC-AUC, PR-AUC (threshold-independent)
       ███
      █████ Macro Recall (class balance)
     ███████
    █████████ Pneumonia Recall (MOST CRITICAL)
   ███████████
  ```
- **Why Pneumonia Recall?**
  - False Negative (FN) = Missed sick patient (dangerous!)
  - False Positive (FP) = Extra review (acceptable cost)
  - Target: >95% recall
- **Visual:** Pyramid with color gradient (red=critical, orange=important, green=nice-to-have)

**Notes:** Explain medical context: FN >> FP in harm.

---

### Slide 8: Experimental Results - Top 3 Models
**Layout:** Podium/Ranking Visual + Table  
**Content:**
- **Title:** 14 Experiments: Top 3 Models
- **Podium Visual:**
  ```
      🥇
     ███  🥈      🥉
    █████ ███     ███
   ███████████   █████
   EfficientNet  ResNet-18  DenseNet
      B2                      121
  ```
- **Comparison Table:**
  | Rank | Model | Macro Recall | Val Acc | Pneu Recall | Normal Recall | Epoch |
  |------|-------|--------------|---------|-------------|---------------|-------|
  | 🥇 | **EfficientNet-B2** | **98.26%** | 98.30% | 98.35% | 98.17% | 4 |
  | 🥈 | ResNet-18 | 97.63% | 98.47% | **99.53%** | 95.73% | 13 |
  | 🥉 | DenseNet-121 | 97.60% | 97.62% | 97.64% | 97.56% | 4 |
- **Key Insight:** B2 = best balance, ResNet-18 = max sensitivity (trade-off)

**Notes:** Highlight that 99.53% recall comes at cost of normal recall.

---

### Slide 9: Test Set Performance
**Layout:** Large Metrics + Confusion Matrix  
**Content:**
- **Title:** Champion Model: EfficientNet-B2 Test Results
- **Key Metrics (Large Font):**
  - Accuracy: **96.62%**
  - Pneumonia Recall: **96.71%** (206/213 caught)
  - Normal Recall: **96.39%** (80/83 identified)
  - ROC-AUC: **99.64%**
  - PR-AUC: **99.86%**
  - MCC: **0.918**
- **Confusion Matrix:**
  ```
              Predicted
          Normal  Pneumonia
  Actual
  Normal     80       3  (FP)
  Pneumonia   7     206  (TP)
             (FN)
  ```
- **Visual:** Color-coded matrix (green=correct, red=errors)

**Notes:** Point out 7 FN vs. 3 FP—asymmetric costs.

---

### Slide 10: Threshold Optimization
**Layout:** Three Scenarios + Metrics Table  
**Content:**
- **Title:** Threshold Tuning: Matching Clinical Needs
- **Three Operating Points:**

  **Scenario 1: Screening Mode (Threshold=0.10)**
  - 🎯 Goal: Catch everything
  - Recall: **99.53%** (1 miss / 213)
  - FP: 6 (acceptable in ER triage)
  - Use Case: Emergency room, mass screening

  **Scenario 2: Balanced Mode (Threshold=0.50)**
  - 🎯 Goal: Reasonable trade-off
  - Recall: **96.71%** (7 miss / 213)
  - FP: 3 (minimal workload)
  - Use Case: Routine outpatient screening

  **Scenario 3: High Precision Mode (Threshold=0.75)**
  - 🎯 Goal: Minimize false alarms
  - Recall: **93.43%** (14 miss / 213)
  - FP: 2 (very low)
  - Use Case: Resource-limited settings

- **Visual:** Three icons (ER, clinic, rural hospital) linked to thresholds

**Notes:** Emphasize context-dependent threshold selection.

---

### Slide 11: Error Analysis - False Negatives
**Layout:** Gallery + Categorization  
**Content:**
- **Title:** What Did We Miss? False Negative Analysis
- **7 False Negatives Breakdown:**
  - Subtle infiltrates (3): Low-contrast, early-stage
  - Image quality (2): Blur, underexposure
  - Atypical presentation (1): Upper lobe (not typical lower)
  - Borderline case (1): Confidence ~0.48
- **Visual:**
  - 4 example X-ray images (2×2 grid)
  - Each with label: "Subtle infiltrate, Conf=0.35"
  - Red arrows pointing to missed regions

**Notes:** Connect to clinical reality—radiologists also struggle with these.

---

### Slide 12: Error Analysis - False Positives
**Layout:** Gallery + Categorization  
**Content:**
- **Title:** False Alarms: Why Did We Overdiagnose?
- **3 False Positives Breakdown:**
  - Thymus shadow (1): Normal pediatric anatomy
  - Image artifacts (1): Grid lines, exposure
  - Vascular congestion (1): Cardiac-related
- **Visual:**
  - 3 example X-ray images
  - Yellow arrows pointing to misleading regions
  - Side annotation: "Thymus (normal in kids)"

**Notes:** Teach that pediatric anatomy differs from adult.

---

### Slide 13: Grad-CAM Visualization
**Layout:** 4-Panel Comparison  
**Content:**
- **Title:** Where Does the Model Look? Grad-CAM Explainability
- **4 Examples (2×2 grid):**

  **Top-Left: True Positive**
  - Original X-ray
  - Heatmap overlay (red on bilateral lower lobes)
  - Label: "✅ Pneumonia detected, Conf=0.98"
  - Note: "Clinically valid focus"

  **Top-Right: True Negative**
  - Original X-ray
  - Heatmap (even distribution)
  - Label: "✅ Normal identified, Conf=0.93"
  - Note: "Checks entire lung, finds nothing"

  **Bottom-Left: False Negative**
  - Original X-ray
  - Heatmap (focused on heart border)
  - Label: "❌ Missed pneumonia, Conf=0.48"
  - Note: "Distracted by cardiac silhouette"

  **Bottom-Right: False Positive**
  - Original X-ray
  - Heatmap (on thymus)
  - Label: "❌ False alarm, Conf=0.64"
  - Note: "Mistakes normal anatomy"

- **Visual:** Side-by-side original + heatmap for each

**Notes:** Explain that Grad-CAM builds trust—we see model's "reasoning."

---

### Slide 14: Calibration Analysis
**Layout:** Calibration Curve + Table  
**Content:**
- **Title:** Can We Trust the Confidence? Calibration Check
- **Calibration Table:**
  | Confidence Bin | Model Says | Actually Is | Count | Status |
  |----------------|------------|-------------|-------|--------|
  | 0.0 - 0.5 | 0.42 | 0.14 | 7 | ⚠️ Overconfident |
  | 0.5 - 0.7 | 0.61 | 0.67 | 3 | ✅ Good |
  | 0.7 - 0.9 | 0.83 | 0.89 | 18 | ✅ Good |
  | **0.9 - 1.0** | **0.97** | **0.98** | **268** | ✅ Excellent |
- **Expected Calibration Error (ECE):** 0.025 (excellent, <0.05 is good)
- **Visual:** Calibration curve (diagonal=perfect, our curve close to diagonal)
- **Takeaway:** 90% of predictions are high-confidence and reliable

**Notes:** Explain that low-confidence (<0.6) should trigger human review.

---

### Slide 15: Limitations - Dataset
**Layout:** Warning Icon + Bullet List  
**Content:**
- **Title:** ⚠️ Critical Limitations: Know What This Model Can't Do
- **Dataset Limitations:**
  - ❌ Single-center data (Guangzhou hospital only)
  - ❌ Pediatric-only (ages 1-5, NOT for adults)
  - ❌ Single imaging protocol (one X-ray machine)
  - ❌ Binary classification (misses TB, cancer, effusions)
  - ❌ No metadata (age, sex, symptoms missing)
- **Visual:**
  - Red warning triangle
  - Map highlighting single location
  - Icons: Child (yes), Adult (no/crossed out)

**Notes:** Own the limitations—shows scientific maturity.

---

### Slide 16: Limitations - Model & Ethics
**Layout:** Prohibited Use Cases + Checklist  
**Content:**
- **Title:** Ethical Deployment: What's Allowed & Prohibited
- **Model Limitations:**
  - 7 false negatives (3.3% miss rate) ⚠️
  - Sensitive to image quality (blur, exposure)
  - No uncertainty quantification (single model)
  - Not validated on external datasets

- **Intended Use (✅):**
  - Educational demonstration
  - Triage support (prioritize urgent cases)
  - Second reader in double-reading workflow

- **Prohibited Use (❌):**
  - Standalone diagnostic tool (no human review)
  - Deployment on adults (trained on kids only)
  - Use without local validation
  - Legal/reimbursement evidence

- **Visual:**
  - Green checkmarks for allowed uses
  - Red X marks for prohibited uses

**Notes:** Emphasize IRB approval + clinical trials needed for real deployment.

---

### Slide 17: What We Deliver - Transparency
**Layout:** Repository Structure + Badges  
**Content:**
- **Title:** Open Science: Full Transparency & Reproducibility
- **Deliverables:**
  - ✅ Full source code (train, eval, analysis)
  - ✅ Model card with limitations
  - ✅ Error analysis & failure modes
  - ✅ Calibration analysis
  - ✅ Grad-CAM visualizations
  - ✅ Automated analysis scripts
  - ✅ Comprehensive documentation (8+ guides)
- **Visual:**
  - GitHub repo structure diagram
  - Folder icons: `src/`, `scripts/`, `reports/`, `docs/`
  - Badges: "Reproducible", "Open Source", "Documented"

**Notes:** Show that this is production-grade, not just a Kaggle notebook.

---

### Slide 18: Future Work - Roadmap
**Layout:** Timeline with Three Phases  
**Content:**
- **Title:** Future Directions: From Prototype to Clinic
- **Timeline:**

  **Short-Term (Next Semester):**
  - External validation (ChestX-ray14, MIMIC-CXR)
  - Multi-label classification (multiple diseases)
  - Ensemble methods (uncertainty quantification)
  - Radiologist benchmark comparison

  **Medium-Term (Research Paper):**
  - Multi-center study (3-5 hospitals, diverse populations)
  - Subgroup analysis (age, sex, disease severity)
  - Prospective clinical trial (IRB-approved)
  - Cost-effectiveness analysis

  **Long-Term (Clinical Deployment):**
  - FDA regulatory approval (510(k) or De Novo)
  - Integration with hospital PACS/EHR systems
  - Real-world performance monitoring
  - Generalization (adults, TB, COVID-19)

- **Visual:** Roadmap with three columns, icons for each milestone

**Notes:** Show ambition but acknowledge current project is foundation.

---

### Slide 19: Key Takeaways
**Layout:** 4 Bullet Points with Icons  
**Content:**
- **Title:** Key Lessons Learned

  1. **Data Quality > Model Complexity**
     - Icon: Database with checkmark
     - Cleaning dataset improved performance more than architecture tuning

  2. **Medical AI Has Different Priorities**
     - Icon: Heart + AI
     - Recall > accuracy, calibration > raw metrics, explainability > black box

  3. **Error Analysis Guides Improvement**
     - Icon: Magnifying glass
     - Understanding failures leads to targeted fixes (quality checks, pediatric variants)

  4. **Ethics & Transparency Are Non-Negotiable**
     - Icon: Balance scale
     - Acknowledge limitations, validate on diverse data, never replace human judgment

**Notes:** These are generalizable lessons for any medical AI project.

---

### Slide 20: Thank You + Contact
**Layout:** Simple Text + Visual  
**Content:**
- **Title:** Thank You!
- **Subtitle:** Questions & Discussion
- **Contact:**
  - Name: [Your Name]
  - Email: [your.email@university.edu]
  - GitHub: [github.com/yourusername/repo]
- **Visual:**
  - QR code linking to GitHub repo
  - Professional photo (optional)
  - University logo

**Notes:** Keep slide up during Q&A. Have backup slides ready.

---

## Backup Slides (Not Presented, For Q&A)

### B1: Detailed Architecture Diagram
- EfficientNet-B2 layer-by-layer breakdown
- Feature maps visualization

### B2: Full Experiment Comparison Table
- All 14 experiments with metrics
- Learning rate / augmentation ablation studies

### B3: Additional Grad-CAM Examples
- 8-10 more examples across different cases
- Comparison across architectures

### B4: Threshold Sweep Curves
- Recall vs. Precision curve
- F1-score vs. Threshold plot
- ROC and PR curves

### B5: Calibration Plots
- Reliability diagram
- Confidence histogram
- ECE across different models

### B6: Computational Requirements
- Training time per epoch
- Inference latency
- Memory usage
- Hardware specifications

### B7: Comparison to Literature
- Our results vs. CheXNet
- Our results vs. other Kaggle solutions
- Methodological improvements table

### B8: Data Augmentation Examples
- Before/After augmentation samples
- Impact of augmentation intensity on performance

---

## Design Guidelines

**Color Scheme:**
- **Primary:** Medical blue (#1E3A8A)
- **Accent:** Gold for highlights (#F59E0B)
- **Success:** Green (#10B981) for checkmarks
- **Warning:** Red (#EF4444) for errors/limitations
- **Background:** White or light gray (#F9FAFB)

**Typography:**
- **Headings:** Sans-serif, bold, 32-40pt
- **Body:** Sans-serif, regular, 18-24pt
- **Code/Numbers:** Monospace, 16-20pt
- **Annotations:** Italic, 14-16pt

**Visual Consistency:**
- Use icons from same family (e.g., FontAwesome, Material Icons)
- Consistent spacing (20px margins, 30px padding)
- Align elements to grid (left-align text, center images)
- Limit text per slide (max 6 bullet points)
- One key message per slide

**Image Quality:**
- Use high-resolution X-rays (min 300 DPI)
- Ensure HIPAA compliance (de-identified images only)
- Add scale bars or annotations where helpful
- Use consistent colormap for Grad-CAM (red=high, blue=low)

**Accessibility:**
- High contrast text (WCAG AA standard)
- Avoid red-green color combos (colorblind-friendly)
- Include alt text for images
- Use large fonts (18pt+ body text)

---

## Presentation Checklist

**Before Presentation:**
- [ ] Rehearse 3-4 times (internalize flow, don't memorize)
- [ ] Test on presentation equipment (projector, clicker)
- [ ] Have backup (USB + laptop)
- [ ] Print speaker notes (1 page, bullet points only)
- [ ] Time yourself (aim for 7:00-7:30, leave buffer)
- [ ] Prepare Q&A answers (see script for anticipated questions)
- [ ] Check room 15 min early

**During Presentation:**
- [ ] Start with strong opener (hook audience)
- [ ] Maintain eye contact (look at audience, not slides)
- [ ] Use gestures (emphasize key points)
- [ ] Pause after major points (let info sink in)
- [ ] Monitor time (have watch/timer visible)
- [ ] Speak clearly (130-140 WPM)
- [ ] Show confidence in limitations (honesty builds trust)

**After Presentation:**
- [ ] Thank audience
- [ ] Answer questions thoroughly but concisely
- [ ] Offer to share repo link (QR code on thank you slide)
- [ ] Follow up on unanswered questions (email later)

---

**Version:** 1.0  
**Last Updated:** November 16, 2025  
**Format:** PowerPoint (.pptx) or Google Slides  
**Estimated Prep Time:** 4-6 hours (design + rehearsal)
