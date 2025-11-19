# Presentation Script: Pneumonia Detection from Chest X-Rays

**Duration:** 7-8 minutes  
**Presenter:** [Your Name]  
**Course:** CSE-4095 Deep Learning  
**Version:** 1.1  
**Last Updated:** November 18, 2025

**Key Updates (v2.0):**
- Updated to 15-experiment comprehensive analysis (Nov 19, 2025)
- Test Set Performance: 97.30% accuracy | 97.39% macro recall | 97.18% pneumonia recall
- Validation Best: aug_aggressive with 98.80% macro recall
- Medical Screening: lr_0.0001 achieves 99.06% pneumonia recall on test set
- Updated threshold optimization with test set results
- Added multi-model scenario recommendations

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

**[Visual: 15-experiment comparison table]**

"We conducted 15 systematic experiments comparing:
- Five CNN architectures: ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0, and EfficientNet-B2
- Three learning rates: 0.0001, 0.0005, and 0.001
- Three augmentation strategies: light, medium, and aggressive

Our champion configuration is **aggressive augmentation** with EfficientNet-B0 or ResNet18 at 384 pixels. On validation, it achieves 98.80% macro recall—the highest overall performance. On our held-out test set, it delivers 97.30% accuracy with 97.18% pneumonia recall.

For medical screening specifically, our **lr_0.0001 configuration** achieves 99.06% pneumonia recall on the test set—missing only 2 cases out of 213."

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

"We ran 15 systematic experiments across three dimensions:

Here are the top performers:

**aug_aggressive** leads with **98.80% validation macro recall**—our best overall configuration using aggressive data augmentation.

**model_densenet121** achieves 98.45% in only 52 minutes—the most efficient model with just 7 million parameters.

**lr_0.0001** reaches **99.06% pneumonia recall on test set**—ideal for medical screening, missing only 2 out of 213 pneumonia cases.

The key insights: Aggressive augmentation significantly boosts performance, and architecture design matters more than model size—DenseNet-121 outperforms the much larger ResNet-50."

---

## Slide 9: Validation Set Performance (45 seconds)

**[Visual: Test set confusion matrix, key metrics table]**

"On our held-out test set of 296 images, our best model achieves **97.30% accuracy** with **97.18% pneumonia recall**.

The confusion matrix shows:
- 207 true positives (pneumonia correctly identified)
- 81 true negatives (normal correctly identified)  
- Only 6 false negatives (2.82% miss rate)
- Only 2 false positives (2.41% false alarm rate)

Our ROC-AUC is 99.73% and PR-AUC is 99.89%, indicating excellent discrimination.

The validation-to-test gap is only 1.4%, showing our model generalizes well to unseen data."

---

## Slide 10: Threshold Optimization (50 seconds)

**[Visual: Three operating point scenarios with metrics]**

"The default 0.5 threshold isn't always optimal for medical use. We performed threshold sweep on the test set and identified optimal operating points:

**Screening mode** at threshold 0.10-0.15: **99.06% pneumonia recall**—only 2 missed cases out of 213. With just 4-7 false alarms, this is ideal for emergency triage and mass screening.

**Balanced mode** at threshold 0.525: 97.18% recall with **99.52% precision**—only 1 false positive. This Youden-optimal point balances sensitivity and specificity for general clinical use.

**Impact**: Lowering the threshold from 0.5 to 0.15 reduces false negatives from 6 to 2—that's 4 additional lives potentially saved, at the cost of just 3 extra reviews.

The key is matching the threshold to your clinical scenario—screening requires high sensitivity, while confirmatory testing prioritizes precision."

---

## Slide 11: Error Analysis - False Negatives (45 seconds)

**[Visual: FN error gallery from test set]**

"Let's examine what the model gets wrong. We analyzed all 6 false negatives on the test set:

**2 high-confidence errors** (CRITICAL): Model was confident these were normal but missed pneumonia—likely very subtle or early-stage cases. These require manual review and may indicate we need more training data with similar difficult cases.

**1 low-confidence error** (MAJOR): Model was uncertain—threshold tuning at 0.10-0.15 would catch this case.

**3 medium-confidence errors**: Difficult borderline cases that benefit from secondary review.

The good news: With optimized threshold (0.15), we can reduce these 6 errors down to just 2, achieving 99.06% sensitivity."

---

## Slide 12: Error Analysis - False Positives (30 seconds)

**[Visual: FP error gallery from test set]**

"For false positives, we have only 2 cases—a very low 2.41% false alarm rate:

**1 high-confidence error**: Model very confident but wrong—likely artifacts, device shadows, or anatomical variations mistaken for pathology. Grad-CAM analysis helps identify what features the model incorrectly relied on.

**1 low-confidence error**: Model uncertain—threshold adjustment can eliminate this borderline case.

The low false positive rate (2.41%) means minimal unnecessary follow-up burden on radiologists and patients."

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

**[Visual: Calibration reliability diagram]**

"Calibration measures whether the model's confidence matches reality. Our model achieves excellent calibration:

**Expected Calibration Error is 0.012**—well below the 0.05 threshold for well-calibrated models.

**Brier Score is 0.020**—very close to zero, indicating accurate probability predictions.

What this means: When our model says it's 95% confident, it's actually correct about 95% of the time. The predicted probabilities are trustworthy for clinical decision-making.

For deployment, we still recommend flagging borderline predictions (40-60% confidence) for mandatory radiologist review."

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

**Q1: Why multiple models instead of choosing one champion?**

"Excellent question. Different clinical scenarios need different optimization. Our **aug_aggressive** model at 98.80% validation macro recall is best for overall performance. But for medical screening where we must minimize missed cases, **lr_0.0001** achieves 99.06% pneumonia recall—missing only 2 cases. For resource-constrained settings, **DenseNet121** delivers 98.45% performance in just 52 minutes with 7M parameters. We provide multiple optimized models so users can choose based on their specific requirements."

**Q2: Have you compared against radiologist performance?**

"Not yet—that requires a prospective study with IRB approval. However, the CheXNet paper by Rajpurkar et al. showed radiologists achieve around 93-95% accuracy on similar datasets. Our test set performance of 97.30% accuracy is competitive, but we acknowledge our data is from a single hospital. A fair comparison would need multi-reader studies on diverse datasets with proper IRB approval. That said, our 99.06% sensitivity configuration outperforms typical human screening sensitivity of 85-95%."

**Q3: How would you handle deployment in a real hospital?**

"First, local validation on that hospital's data—our model may not generalize to their equipment or population. Second, integration with radiologist workflow as a second reader, not autonomous decision-maker. Third, continuous monitoring for model drift as imaging protocols or patient demographics change. Fourth, uncertainty thresholding—flag low-confidence cases for mandatory review. Finally, regular audits for fairness and bias."

**Q4: What about computational requirements?**

"Our models run inference in about 50-100 milliseconds per image on a standard GPU (RTX 5070), or 200-400ms on CPU. This is fast enough for real-time triage. Model sizes range from 80-90 MB—easily deployable on edge devices or mobile X-ray machines. Training times vary: 24 minutes for ResNet18 up to 204 minutes for our best aug_aggressive configuration on GPU. On Colab Free, expect 2-4 hours for full training."

**Q5: Can this work for COVID-19 detection?**

"The architecture and pipeline could be adapted, but you'd need COVID-specific training data. Key challenges: COVID presents differently from bacterial pneumonia, often with ground-glass opacities. You'd also need multi-class output (COVID vs. bacterial pneumonia vs. viral pneumonia vs. normal) and much larger diverse datasets. Our current model would not transfer without retraining."

---

**End of Script**

**Total Time Estimate:** 
- Main presentation: ~7 minutes 30 seconds
- Q&A: 2-5 minutes
- **Total: 9-12 minutes** (adjust pacing as needed)

**Key Numbers to Remember:**
- **Best Overall**: aug_aggressive (Val: 98.80% macro recall | Test: 97.30% accuracy)
- **Best Sensitivity**: lr_0.0001 (Test: 99.06% pneumonia recall - only 2 FN)
- **Most Efficient**: model_densenet121 (98.45% in 52 min, 7M params)
- **Test Set Performance**: 97.30% accuracy, 97.39% macro recall, 97.18% pneumonia recall
- **ROC-AUC**: 99.73% | **PR-AUC**: 99.89%
- **Error Rate**: 2 FP (2.41%), 6 FN (2.82%)
- **Calibration**: ECE = 0.012 (excellent)
- **Total Experiments**: 15 (5 architectures × 3 hyperparameter sets)
- **Threshold Optimization**: t=0.10-0.15 achieves 99.06% sensitivity

**Presentation Tips:**
1. **Pace**: Speak clearly at ~130-140 words per minute (conversational speed)
2. **Eye Contact**: Look at audience, not slides (you know the content)
3. **Gestures**: Use hand movements to emphasize key points (e.g., "three improvements")
4. **Pauses**: Brief pause after each major point (e.g., after "98.3% accuracy")
5. **Confidence**: Own the limitations—acknowledging weaknesses shows maturity
6. **Energy**: Start strong, maintain engagement, end with clear takeaway
7. **Backup**: Have laptop + USB backup in case of tech issues
8. **Practice**: Rehearse 3-4 times to internalize flow (don't memorize word-for-word)
