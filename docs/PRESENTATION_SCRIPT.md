# Presentation Script: Pneumonia Detection from Chest X-Rays

**Duration:** 7-8 minutes  
**Presenter:** [Your Name]  
**Course:** CSE-4095 Deep Learning  
**Version:** 1.1  
**Last Updated:** November 18, 2025

**Key Updates (v1.1):**
- Updated all metrics to match validation set results (EfficientNet-B2 @ 384px)
- Val Accuracy: 98.30% | Macro Recall: 98.26% | Pneumonia Recall: 98.35%
- Updated threshold optimization with actual validation data
- Corrected confusion matrix and all numerical references
- Added quick reference numbers at the end

---

## Slide 1: Title Slide (15 seconds)

**[Visual: Title + Project Image]**

"Good morning/afternoon, everyone. Today I'll present our deep learning project on pneumonia detection from chest X-rays. This work demonstrates how AI can support medical screening in resource-limited settings, while addressing critical challenges in data quality, model selection, and clinical deployment ethics."

---

## Slide 2: Motivation & Global Context (30 seconds)

**[Visual: WHO statistics, pediatric pneumonia burden chart]**

"Pneumonia is responsible for 15% of deaths in children under 5 globally. Early detection is critical, but many regions lack sufficient radiologists. Our project explores how deep learning can assist in triage—not replacing doctors, but helping prioritize urgent cases.

The key challenge is building a system that's not just accurate, but also reliable, explainable, and aware of its limitations."

---

## Slide 3: Dataset Challenge (45 seconds)

**[Visual: Original dataset problems visualization]**

"We used the Kaggle Chest X-Ray Pneumonia dataset—5,856 pediatric images from a single hospital in China. But this dataset has serious issues:

First, duplicate images inflate performance metrics.
Second, random splits risk data leakage—same patient appearing in both training and validation.
Third, severe class imbalance with a 3-to-1 ratio favoring pneumonia.

These problems are common in medical AI datasets, so addressing them was our first priority."

---

## Slide 4: Optimal Dataset Construction (40 seconds)

**[Visual: Before/After comparison table, patient-level split diagram]**

"We rebuilt the dataset with three key improvements:

One: Removed exact duplicates using perceptual hashing.
Two: Patient-level separation—ensuring images from the same patient stay in one split.
Three: Stratified sampling with 85-10-5 split for train, validation, and test.

This gives us 4,683 training images, 589 for validation, and 296 for final testing. Now our results are trustworthy and reproducible."

---

## Slide 5: Model Architecture Comparison (45 seconds)

**[Visual: Architecture comparison table with 5 models]**

"We compared five CNN architectures, all pre-trained on ImageNet:

ResNet-18 and ResNet-50 for deep residual learning,
DenseNet-121 for dense connectivity,
And EfficientNet-B0 and B2 for efficient scaling.

Our champion is EfficientNet-B2 with 384-pixel resolution. Why? It achieves the best balance—98.30% validation accuracy, 98.35% pneumonia recall, and it converges in just 4 epochs. The higher resolution is crucial for capturing fine details in X-rays that smaller models might miss."

---

## Slide 6: Training Strategy (40 seconds)

**[Visual: Training pipeline flowchart]**

"Our training strategy uses transfer learning from ImageNet, fine-tuning all layers with a learning rate of 5e-4.

For the loss function, we use weighted cross-entropy with a 1.33 weight on pneumonia to address class imbalance.

Data augmentation includes random flips, rotations, and color jitter to prevent overfitting.

We use early stopping with patience of 5 epochs, monitoring macro recall—that's the average of pneumonia and normal recall, ensuring balanced performance."

---

## Slide 7: Evaluation Metrics & Clinical Priority (50 seconds)

**[Visual: Metrics hierarchy diagram]**

"For medical screening, not all metrics are equal. Our top priority is pneumonia recall—also called sensitivity. This measures what percentage of actual pneumonia cases we catch.

Why prioritize recall? Because in medicine, missing a sick patient—a false negative—is far more dangerous than a false positive, which just means one extra review.

We also track ROC-AUC and PR-AUC as threshold-independent metrics, and calibration to ensure the model's confidence scores are reliable.

Our goal: over 95% pneumonia recall while maintaining reasonable specificity."

---

## Slide 8: Experimental Results (50 seconds)

**[Visual: Top 3 models comparison table]**

"We ran 14 controlled experiments across three dimensions: architecture, learning rate, and augmentation intensity.

Here are the top three models:

EfficientNet-B2 leads with 98.26% macro recall and fast convergence at epoch 4.
ResNet-18 achieves the highest pneumonia recall at 99.53%, but sacrifices normal recall at 95.73%.
DenseNet-121 comes in third with balanced 97.60% across both classes.

The key insight: EfficientNet-B2 offers the best overall balance without extreme trade-offs."

---

## Slide 9: Validation Set Performance (45 seconds)

**[Visual: Confusion matrix, key metrics table]**

"On our validation set, EfficientNet-B2 achieves 98.30% accuracy with 98.35% pneumonia recall.

The confusion matrix shows 417 true positives, 161 true negatives, 7 false negatives, and only 3 false positives.

Our ROC-AUC is 99.73% and PR-AUC is 99.89%, indicating near-perfect discrimination between classes.

The Matthew's Correlation Coefficient of 0.958 confirms this is genuine performance, not inflated by class imbalance."

---

## Slide 10: Threshold Optimization (50 seconds)

**[Visual: Three operating point scenarios with metrics]**

"The default 0.5 threshold isn't always optimal for medical use. We identified three operating points:

Screening mode at 0.10 threshold: 98.82% recall with only 5 missed cases out of 424—ideal for emergency triage where we want to catch everything.

Balanced mode at 0.50: 98.35% recall with 7 missed cases and 99.29% precision—suitable for routine outpatient screening.

High precision mode at 0.75: 97.17% recall but 99.52% precision with only 2 false alarms—useful in resource-limited settings where follow-up is expensive.

The key is matching the threshold to the clinical scenario."

---

## Slide 11: Error Analysis - False Negatives (45 seconds)

**[Visual: FN gallery with 4 examples]**

"Let's examine what the model gets wrong. We manually reviewed all 7 false negatives:

Three cases had subtle infiltrates with low-contrast opacities—early-stage pneumonia that even radiologists might miss.

Two had poor image quality from motion blur or underexposure.

One showed atypical presentation in the upper lobe instead of typical lower lobe.

One was a borderline case with model confidence around 0.48—could be a normal variant.

These patterns guide future improvements: better preprocessing, quality checks, and uncertainty flagging."

---

## Slide 12: Error Analysis - False Positives (30 seconds)

**[Visual: FP gallery with 3 examples]**

"For false positives:

One case shows a thymus shadow—a normal pediatric structure that mimics infiltrates.

Another has image artifacts from equipment, like grid lines.

The third shows vascular congestion that the model mistakes for pneumonia.

These errors teach us about pediatric-specific challenges and the importance of understanding normal anatomical variants."

---

## Slide 13: Grad-CAM Visualization (45 seconds)

**[Visual: 4 Grad-CAM examples - TP, TN, FN, FP]**

"Grad-CAM helps us understand where the model looks. For true positives, the heatmap highlights bilateral lower lobes and perihilar regions—exactly where pneumonia typically appears.

For true negatives, activation is evenly distributed, showing the model checks the entire lung but finds no focal abnormality.

For false negatives, we see the model focuses on the heart border instead of the subtle right lower lobe infiltrate—it's distracted by prominent cardiac silhouette.

For false positives, it highlights the thymus, misinterpreting normal anatomy.

This explainability builds trust and helps us understand failure modes."

---

## Slide 14: Calibration Analysis (35 seconds)

**[Visual: Calibration curve and binned table]**

"Calibration measures whether the model's confidence matches reality. Our expected calibration error is 0.025—excellent, below the 0.05 threshold for well-calibrated models.

Ninety percent of predictions have over 0.9 confidence and are 98% accurate. However, the model is slightly overconfident on borderline cases in the 0.4-to-0.5 range.

For deployment, we should flag low-confidence predictions for mandatory human review."

---

## Slide 15: Limitations - Dataset (40 seconds)

**[Visual: Limitations summary table]**

"Now the critical part: our limitations.

Data-wise: Single-center from one Chinese hospital, pediatric-only ages 1-to-5, one imaging protocol. This means zero generalizability to adults, other populations, or different equipment.

We only do binary classification—pneumonia versus normal—missing tuberculosis, lung cancer, effusions.

We have no clinical metadata like age or sex for subgroup analysis.

This is a research prototype, not a deployable system."

---

## Slide 16: Limitations - Model & Ethics (45 seconds)

**[Visual: Prohibited use cases with X marks]**

"Model limitations: We have 7 false negatives—a 3.3% miss rate that's unacceptable for standalone diagnosis. The model is sensitive to image quality and has no uncertainty quantification.

Ethically, this tool is ONLY for educational demos and triage support as a second reader. It is explicitly prohibited for standalone diagnosis, use on adults, deployment without local validation, or legal evidence.

We have population bias from a Chinese cohort and no subgroup performance data. This requires IRB approval and prospective clinical trials before any real-world use."

---

## Slide 17: What We Deliver (30 seconds)

**[Visual: Repository structure diagram]**

"For transparency, we provide:

Full source code for training, evaluation, and analysis.
A detailed model card documenting limitations.
Error analysis with failure modes.
Calibration analysis and Grad-CAM visualizations.
Automated analysis scripts for reproducibility.

This allows anyone to verify our claims, reproduce results, or extend the work."

---

## Slide 18: Future Work (40 seconds)

**[Visual: Roadmap with short/medium/long-term goals]**

"Future directions:

Short-term: External validation on ChestX-ray14 or MIMIC datasets, multi-label classification for multiple diseases, ensemble methods for uncertainty.

Medium-term: Multi-center studies with diverse populations, subgroup analysis by demographics, prospective clinical validation.

Long-term: FDA regulatory approval, integration with hospital PACS systems, generalization to adults and other lung diseases like tuberculosis or COVID-19."

---

## Slide 19: Key Takeaways (35 seconds)

**[Visual: 4 key bullet points]**

"To summarize:

One: Data quality matters more than model complexity—our dataset cleaning improved performance more than architecture tuning.

Two: Medical AI requires different priorities—recall over accuracy, calibration over raw metrics.

Three: Explainability and error analysis build trust and guide improvements.

Four: Ethical deployment requires acknowledging limitations, validating on diverse populations, and never replacing human judgment.

Thank you for your attention. I'm happy to answer questions."

---

## Slide 20: Thank You + Q&A (Open-ended)

**[Visual: Contact info, GitHub link]**

**Backup slides available:**
- Detailed architecture diagrams
- Full experiment comparison table
- Additional Grad-CAM examples
- Threshold sweep curves
- Calibration plots

**Anticipated Questions:**

**Q1: Why EfficientNet-B2 instead of ResNet-18 with 99.53% recall?**

"Great question. ResNet-18 does achieve 99.53% pneumonia recall, but at the cost of 95.73% normal recall—meaning more false alarms. EfficientNet-B2 balances both classes better at 98.35% pneumonia recall and 98.17% normal recall, converges faster at epoch 4, and has fewer parameters. For a general-purpose screening tool, balance is more sustainable than extreme sensitivity."

**Q2: Have you compared against radiologist performance?**

"Not yet—that requires a prospective study with IRB approval. However, the CheXNet paper by Rajpurkar et al. showed radiologists achieve around 93-95% accuracy on similar data. Our validation set performance of 98.30% accuracy is competitive, but we acknowledge our dataset is from a single center. A fair comparison needs multi-reader studies on diverse datasets with proper IRB approval."

**Q3: How would you handle deployment in a real hospital?**

"First, local validation on that hospital's data—our model may not generalize to their equipment or population. Second, integration with radiologist workflow as a second reader, not autonomous decision-maker. Third, continuous monitoring for model drift as imaging protocols or patient demographics change. Fourth, uncertainty thresholding—flag low-confidence cases for mandatory review. Finally, regular audits for fairness and bias."

**Q4: What about computational requirements?**

"EfficientNet-B2 runs inference in about 50 milliseconds per image on a standard GPU, or 300-400ms on CPU. This is fast enough for real-time triage in emergency rooms. Model size is 36 MB—easily deployable on edge devices or mobile X-ray machines. Training takes about 2-3 hours on a single GPU for 15 epochs."

**Q5: Can this work for COVID-19 detection?**

"The architecture and pipeline could be adapted, but you'd need COVID-specific training data. Key challenges: COVID presents differently from bacterial pneumonia, often with ground-glass opacities. You'd also need multi-class output (COVID vs. bacterial pneumonia vs. viral pneumonia vs. normal) and much larger diverse datasets. Our current model would not transfer without retraining."

---

**End of Script**

**Total Time Estimate:** 
- Main presentation: ~7 minutes 30 seconds
- Q&A: 2-5 minutes
- **Total: 9-12 minutes** (adjust pacing as needed)

**Key Numbers to Remember:**
- Best Model: EfficientNet-B2 @ 384px
- Val Accuracy: 98.30%
- Macro Recall: 98.26%
- Pneumonia Recall: 98.35%
- Normal Recall: 98.17%
- ROC-AUC: 99.73%
- PR-AUC: 99.89%
- Best Epoch: 4
- Total Experiments: 14

**Presentation Tips:**
1. **Pace**: Speak clearly at ~130-140 words per minute (conversational speed)
2. **Eye Contact**: Look at audience, not slides (you know the content)
3. **Gestures**: Use hand movements to emphasize key points (e.g., "three improvements")
4. **Pauses**: Brief pause after each major point (e.g., after "98.3% accuracy")
5. **Confidence**: Own the limitations—acknowledging weaknesses shows maturity
6. **Energy**: Start strong, maintain engagement, end with clear takeaway
7. **Backup**: Have laptop + USB backup in case of tech issues
8. **Practice**: Rehearse 3-4 times to internalize flow (don't memorize word-for-word)
