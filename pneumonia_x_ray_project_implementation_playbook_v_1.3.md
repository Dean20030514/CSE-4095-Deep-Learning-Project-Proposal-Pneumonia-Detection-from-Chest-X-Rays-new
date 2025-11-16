# Pneumonia X‑ray Project — Implementation Playbook (v1.3, Academic Edition)

> School‑friendly revision of the v1.0 playbook with tighter alignment to the proposal, resource‑aware tracks, clearer week‑by‑week plan, and deeper guidance on analysis and collaboration.

---

## Changelog (v1.1 vs v1.0)
- Added **Purpose & Motivation** paragraph linking project to global health context and triage needs.
- Introduced **Two‑Track plan**: Track‑A _MVP_ for limited resources; Track‑B _Advanced_ for stretch goals.
- Marked **advanced items** (Optuna, external validation, ONNX/TensorRT) as optional.
- Expanded **Week‑by‑Week milestones** with concrete student tasks.
- Added **Collaboration** section (Git workflow, experiment logging, data handling).
- Deepened **EDA checklist** with common chest‑X‑ray pitfalls & quick code patterns.
- Clarified **loss choice**: Weighted CE vs Focal (decision guide).
- Strengthened **error analysis** playbook and **calibration/thresholding** steps.
- Bundled **low‑compute configs** (Colab‑friendly) and **report/poster/demo** deliverables template.

---

## 0) Purpose, Motivation & Scope (School Context)
**Problem.** Pneumonia remains a major global health burden; timely triage reduces risk of missed cases. This project builds a **3‑class classifier** (Normal / Bacterial / Viral) for chest X‑rays to support **educational research** and demo‑level decision support.

**Primary KPI.** Maximize **pneumonia recall/sensitivity** (minimize false negatives), while tracking per‑class F1, macro‑F1, PR‑AUC, and calibration.

**Non‑goals.** Clinical deployment or physician replacement. This is **research/education only**.

> Literature touchstones: CheXNet (dense CNNs for pneumonia), CheXpert (labeling & benchmarking), Grad‑CAM (explainability), Focal Loss (imbalanced detection). See **References**.

---

## 1) Two‑Track Execution Plan
**Why two tracks?** Students often face compute/time limits. Pick Track‑A first; upgrade to Track‑B as time allows.

### Track‑A — MVP (Compute‑Light, 1–2 GPUs on Colab Free/Local CPU‑GPU)
- **Backbone**: EfficientNet‑B0 or ResNet18 (ImageNet pretrained).
- **Image size**: 384 (start) → 448 if memory allows.
- **Batch size**: 16 (AMP on GPU; 8 if OOM).
- **Loss**: Weighted Cross‑Entropy; try Focal(γ=1.5) if minority recall stalls.
- **Epochs**: 20–30 with EarlyStopping(patience=5–7) on **pneumonia recall**.
- **Augmentation**: modest (flip/rotate/brightness); keep medical realism.
- **Outputs**: baseline metrics, confusion matrix, Grad‑CAM examples, Streamlit demo.

### Track‑B — Advanced (Stretch)
- **Backbones**: ResNet50 / EfficientNet‑B3; image size 512–640.
- **Search**: Optuna (20–40 trials) over lr/weight_decay/dropout/γ/image size.
- **External validation**: CheXpert/other CXR subset (evaluation‑only).
- **Packaging**: ONNX export; Docker; optional TensorRT.

---

## 2) Resource & Platform Guidance
**Colab Free** (~T4/16GB RAM): keep 384–448 px, AMP on, avoid heavy augs; use gradient accumulation if needed.
**Colab Pro/Local mid‑range GPU**: 512 px feasible; run limited Optuna (≤20 trials).
**CPU‑only**: train at 224–320 px with ResNet18; emphasize thorough evaluation & analysis.

Tips: cache preprocessed tensors; turn off non‑essential visualizations during training; checkpoint every few epochs; reuse frozen early layers for warm‑starts.

---

## 3) Collaboration & Project Hygiene
- **Git workflow**: `main` (stable) / `dev` (integration) / `feat/*` (short‑lived). Require PR + code review checklist.
- **Issues & Kanban**: tag `data`, `training`, `eval`, `ui`, `infra`. Link PRs to issues.
- **Experiment logging**: one row per run (`seed`, `config`, `metrics`, `notes`, `ckpt_path`). Use CSV/Sheets or W&B.
- **Data handling**: store **paths only** in repo; no raw data in Git. Provide `make_data.sh` and `README` for download.
- **Repro**: fix seeds; record `pip freeze`/env YAML; save **best (by pneumonia recall)** and **last** checkpoints.

---

## 4) Week‑by‑Week Plan (8–10 Weeks)
- **W1**: Env setup; data download; quick EDA counts; define patient‑safe split; decide Track.
- **W2**: Implement DataModule & transforms; finalize augmentations; class weights; baseline config file.
- **W3**: Train MVP baseline; log metrics; first confusion matrix; initial Grad‑CAM sanity.
- **W4**: Error buckets (FN/FP galleries); threshold sweep (recall‑first vs balanced); start calibration.
- **W5**: Ablations (image size; backbone swap); decide on Focal vs Weighted CE; finalize MVP.
- **W6**: Streamlit demo; Grad‑CAM overlay; usability polish & disclaimers.
- **W7** *(optional)*: External validation and robustness tests (blur/noise/contrast curves).
- **W8–W9**: Report, poster, and demo video; finalize model card.
- **W10** *(buffer)*: Repro pass; packaging (ONNX/Docker if Track‑B).

---

## 5) EDA & Data Integrity Checklist
1) **Counts** per split/class; ensure tri‑class mapping is correct.
2) **Patient leakage**: verify no patient appears across splits.
3) **Duplicates**: dedupe by image hash (md5/sha1).
4) **Image properties**: dims/aspect ratio, grayscale vs RGB; convert grayscale to 3‑ch.
5) **Artifacts**: device wires/markers/text; note prevalence; flag improbable crops.
6) **Quality**: low dynamic range / overexposure; consider CLAHE cautiously.

_Quick code patterns_ (pseudocode):
```python
# class weights
w = total / (num_classes * counts)
# md5 hash
def md5(p): return hashlib.md5(open(p,'rb').read()).hexdigest()
```

---

## 6) Label Strategy (Tri‑Class)
- Prefer datasets/splits that distinguish **Bacterial** vs **Viral**. If Kaggle base lacks separation, curate via provided subfolders/metadata or treat as **binary pneumonia** for MVP, documenting the limitation and its impact on evaluation.

---

## 7) Handling Class Imbalance — Decision Guide
- Start with **Weighted Cross‑Entropy** using class counts.
- If minority recall plateaus or training unstable → try **Focal Loss (γ=1–2)**.
- Use **WeightedRandomSampler** to balance mini‑batches.
- Prefer stronger **augmentations** over synthetic generation; avoid vanilla SMOTE for images.

---

## 8) Training Recipes (Configs You Can Copy)
**Baseline (Track‑A)**
- Optimizer: AdamW (lr=3e‑4, wd=1e‑4)
- Scheduler: CosineAnnealingLR (T_max=epochs)
- Batch: 16 @ 384–448; AMP on
- Epochs: 25; EarlyStopping on **pneumonia recall**

**When to stop**: if val pneumonia recall hasn’t improved ≥0.5% in 5 epochs and overfit rises (gap >10% between train/val), stop & adjust.

---

## 9) Evaluation, Thresholding & Calibration
- Report: accuracy; per‑class P/R/F1; macro‑F1; PR‑AUC. Always include **pneumonia recall**.
- **Threshold sweep** on validation to produce two modes:
  - **Max‑Recall** (more positives allowed) for triage demos.
  - **Balanced** for overall F1.
- **Confusion matrix reading guide**:
  - **FP (Normal→Pneumonia)**: check device shadows, text markers, poor contrast.
  - **FN (Pneumonia→Normal)**: subtle/early signs, low‑quality scans.
  - **Bacterial↔Viral**: overlaps; document limits and possible radiographic ambiguity.
- **Calibration**: temperature scaling if over‑confident; show reliability diagram.

---

## 10) Explainability & QC
- Integrate Grad‑CAM overlays; verify activations cover lung fields.
- Add simple QC rules: extreme aspect ratio; very low dynamic range; large text overlays ⇒ warn as OOD.

---

## 11) Robustness & External Validation (Optional)
- Keep a **true test** untouched during selection.
- Evaluate on a small **external subset** (e.g., CheXpert) and record domain shift.
- Stress curves: add blur/noise/contrast and plot metric degradation.

---

## 12) Deliverables (School‑Oriented)
1) **Short paper/report (6–10 pages)**: intro/motivation, related work, methods, experiments, analysis, limits/ethics, conclusion.
2) **Poster**: problem → method → key results → CAMs → takeaways.
3) **Demo video (≤3 min)**: upload → prediction → CAM overlay → max‑recall mode.
4) **Model Card** (template provided below).
5) **Repro Pack**: `config.yaml`, `state_dict.pt`, `metrics.json`, seeds, env info, and run script.

**Grading‑friendly acceptance**
- ≥90% accuracy (if tri‑class curated) or strong macro‑F1; **pneumonia recall hits target**.
- Clear error analysis with galleries and at least 5 documented failure modes.
- Sensible CAMs on ≥90% sampled cases.

---

## 13) Ethics & Governance (Student Version)
- No PHI; strip any identifiers.
- Clear **Research‑only** disclaimer in the app.
- Document label provenance, known biases, and non‑generalizability.

---

## 14) Low‑Compute Appendix
**Colab‑friendly config snippet (YAML‑ish)**
```yaml
model: efficientnet_b0
img_size: 384
batch_size: 16
epochs: 25
optimizer: adamw
lr: 3e-4
weight_decay: 1e-4
loss: weighted_ce  # switch to focal(gamma=1.5) if needed
early_stopping: {metric: pneumonia_recall, patience: 5}
aug: {flip: 0.5, rotate: 10, brightness_contrast: 0.2}
```

**Minimal training loop flags**
```bash
python -m src.train --config src/configs/effb0_384.yaml --amp --save-best-by pneumonia_recall
python -m src.eval --ckpt runs/exp_*/best.pt --split val --report reports/val.json
```

**Streamlit quickstart**
```bash
streamlit run src/app/streamlit_app.py
```

---

## 15) Model Card — Student Template
- **Task & Intended Use**: educational research; not a medical device.
- **Data**: sources, splits, curation rules, leakage checks.
- **Training**: backbone, size, loss, optimizer, seed.
- **Metrics**: overall + per‑class; PR‑AUC; calibration; thresholds.
- **Explainability**: Grad‑CAM examples; sanity notes.
- **Robustness**: stress tests; external set (if any).
- **Limitations & Risks**: typical failure modes; domain shift; label noise.
- **Update Plan**: how future iterations will be assessed/documented.

---

## References (Starter List)
- Rajpurkar et al., **CheXNet** (pneumonia detection with deep CNNs)
- Irvin et al., **CheXpert** (labeling and benchmark)
- Selvaraju et al., **Grad‑CAM** (visual explanations)
- Lin et al., **Focal Loss** (dense detection under imbalance)

---

### Notes
- Start simple; iterate with ablations; document everything.
- Prefer patient‑level splitting whenever IDs allow.
- PR‑AUC is often more informative than ROC‑AUC under imbalance.



---

## Changelog (v1.2 vs v1.1)
- Added **Day‑1 Quickstart (30‑min)** path for zero‑friction onboarding.
- Shipped **ready‑to‑use config templates** (`colab_friendly.yaml`, `balanced_training.yaml`, `full_power.yaml`).
- Added **Common Issues Quick Fix** (Colab OOM, NaN loss, non‑converging, mis‑localized CAMs).
- Added **Showcase Guide** for a 3‑min video + poster layout.
- Added **Peer Review Checklist** to ensure quality and reproducibility.
- Added **One‑Page Model Card** template for quick coursework submission.
- Provided **script stubs**: `scripts/verify_environment.py`, `scripts/download_sample_data.py`.

---

## 🚀 Day‑1 Quickstart (30‑minute Environment Check)
```bash
# 1) Clone
git clone <repo_url> && cd chestxray-pneumonia

# 2) Create environment (or open Colab notebook)
conda env create -f environment.yml && conda activate cxr
#  — or —
# Open notebooks/colab_quickstart.ipynb in Google Colab

# 3) Verify CUDA & dependencies
python scripts/verify_environment.py

# 4) Download a tiny sample (10 images) to test the pipeline
python scripts/download_sample_data.py --dest data/sample_10

# 5) Run the demo app in demo_mode
streamlit run src/app/streamlit_app.py -- --demo_mode
```

**What you should see (≤30 min):** Streamlit UI loads, accepts an image from `data/sample_10/`, outputs class probabilities + Grad‑CAM overlay.

---

## 📦 Ready‑to‑Use Config Templates (`src/configs/`)

### `colab_friendly.yaml` — fastest sanity check
```yaml
model: resnet18
img_size: 224
batch_size: 16
epochs: 10
optimizer: adamw
lr: 1e-3
weight_decay: 1e-4
loss: weighted_ce
sampler: weighted_random
amp: true
early_stopping: {metric: pneumonia_recall, patience: 3}
aug: {flip: 0.5, rotate: 10}
```

### `balanced_training.yaml` — recommended baseline (Track‑A)
```yaml
model: efficientnet_b0
img_size: 384
batch_size: 16
epochs: 25
optimizer: adamw
lr: 3e-4
weight_decay: 1e-4
loss: weighted_ce  # switch to focal(gamma: 1.5) if minority recall stalls
scheduler: cosine
amp: true
early_stopping: {metric: pneumonia_recall, patience: 5}
aug: {flip: 0.5, rotate: 10, brightness_contrast: 0.2}
threshold_modes: {max_recall: true, balanced: true}
log: {run_name: balanced_training, csv: runs/metrics.csv}
```

### `full_power.yaml` — stretch (Track‑B)
```yaml
model: resnet50
img_size: 512
batch_size: 16
epochs: 40
optimizer: adamw
lr: 2e-4
weight_decay: 2e-4
loss: focal
focal: {gamma: 1.5}
scheduler: cosine
amp: true
optuna: {trials: 30, search_space: [lr, weight_decay, dropout, img_size]}
early_stopping: {metric: pneumonia_recall, patience: 7}
external_eval: {dataset: chexpert_small, enabled: true}
export: {onnx: true}
```

---

## 🧰 Common Issues — Quick Fix
**Colab OOM** → `batch_size: 8` → `img_size: 224` → `gradient_accumulation: 2` → disable heavy augs.

**Loss becomes NaN** → lower LR (e.g., `3e-4 → 1e-4`) → check normalization pipeline → temporarily disable risky augs.

**Model won’t converge** → freeze early backbone layers (`requires_grad=False`) for 3–5 epochs → simplify to binary (`pneumonia vs normal`) to debug → verify label mapping and class weights.

**CAM heatmaps off‑lung** → ensure **identical** preprocessing in train vs eval → verify model loads the intended weights → sample CAMs from **unseen** images.

---

## 🎬 Showcase Guide — 3‑minute Video & Poster
**Video outline (≤3:00)**
- 0:00–0:30 Problem & societal impact
- 0:30–1:15 Typical cases + CAM overlays
- 1:15–2:00 Thresholding: *Max‑Recall* vs *Balanced* modes
- 2:00–2:30 1–2 tough failures & lessons
- 2:30–3:00 Limits & future work

**Poster hotspots**
- **Top‑left:** clinical motivation + data stats chart
- **Center:** model diagram + CAM comparisons
- **Bottom‑right:** metrics table (highlight recall) + key takeaways

---

## 👥 Peer Review Checklist (Quality & Repro)
- [ ] Dataloader handles corrupted images without crashing (skips & logs)
- [ ] Train/resume/save/load cycle verified; best‑by **pneumonia recall** saved
- [ ] All randomness controlled by `seed` (torch, numpy, python, dataloader workers)
- [ ] Eval and train preprocessing **identical**
- [ ] Grad‑CAM works on held‑out images; lung‑field sanity pass
- [ ] Streamlit demo includes **research‑only** disclaimer
- [ ] `metrics.csv` or W&B run contains config + metrics + ckpt path

---

## 📋 One‑Page Model Card (Course Submission)
**Task & Use**: tri‑class pneumonia classifier for **educational research**; not for clinical use.

**Data**: sources, curation, patient‑level split, dedup & leakage checks.

**Training**: backbone, img_size, loss, optimizer, epochs, seed; config file name.

**Metrics**: overall & per‑class P/R/F1, macro‑F1, PR‑AUC; **thresholded modes** (*max‑recall* & *balanced*); calibration.

**Explainability**: 2–3 Grad‑CAM panels with notes.

**Limits & Risks**: top 3–5 failure modes; domain shift; label noise.

**Contacts/Repo**: team lead name/email; repository URL.

---

## 🧪 Script Stubs

### `scripts/verify_environment.py`
```python
import torch, pkgutil, sys
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
required = ["torch", "torchvision", "numpy", "pandas", "albumentations", "sklearn", "streamlit"]
missing = [m for m in required if pkgutil.find_loader(m) is None]
print("Missing packages:", missing)
if missing:
    sys.exit(1)
print("OK")
```

### `scripts/download_sample_data.py`
```python
import argparse, os, shutil, urllib.request, zipfile
URL = "https://storage.googleapis.com/quickstart-cxr/sample_10.zip"  # replace with real link
parser = argparse.ArgumentParser()
parser.add_argument('--dest', default='data/sample_10')
args = parser.parse_args()
os.makedirs(args.dest, exist_ok=True)
zip_path = os.path.join(args.dest, 'sample_10.zip')
urllib.request.urlretrieve(URL, zip_path)
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(args.dest)
os.remove(zip_path)
print("Sample downloaded to", args.dest)
```

> **Tip**: Put real dataset links in `README_data.md` and keep raw data **out of Git**; store only paths/scripts.



---

## Changelog (v1.3 vs v1.2)
- Added **Project Difficulty Self‑Assessment** (entry/advanced/research tracks) to help students pick scope.
- Shipped **Demo Presentation Script Template** (`scripts/demo_presentation.py`) to auto‑generate key figures.
- Added **Peer Assessment Questionnaire** for intra‑team review.
- Added **Report Section Template** with page/word guidance for 8‑page paper.
- Added **Demo Day Final Checklist** (tech + speaking readiness).
- Added **Grading Rubric Mapping** table aligning course criteria to evidence in the repo.

---

## 📊 Project Difficulty Self‑Assessment
Choose your lane based on time/compute/experience:
- **Entry (Undergrad course project)**: Track‑A MVP **+** binary classification **+** basic CAM **+** short report.
- **Intermediate (Capstone/Master)**: Full Track‑A **+** tri‑class **+** deeper error analysis **+** model card.
- **Advanced (Publication‑oriented)**: Track‑B **+** external validation **+** ablations **+** paper draft.

> Tip: Start Entry, lock your baseline by Week‑3, then promote scope only if velocity and compute allow.

---

## 🖥️ Demo Presentation Script Template (`scripts/demo_presentation.py`)
```python
"""Auto‑generate presentation assets: metrics plots, confusion matrix, CAM grid, threshold curves."""
from pathlib import Path
import json, itertools
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = Path("reports/figs"); FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1) Bar chart: metrics comparison across runs

def plot_metric_bars(metrics_jsons, metric="macro_f1", labels=None, outfile="metric_bars.png"):
    runs = [json.load(open(p)) for p in metrics_jsons]
    vals = [r.get(metric, np.nan) for r in runs]
    labels = labels or [Path(p).stem for p in metrics_jsons]
    plt.figure()
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=15)
    plt.ylabel(metric)
    plt.title(f"Comparison — {metric}")
    plt.tight_layout(); plt.savefig(FIG_DIR/outfile, dpi=200); plt.close()

# 2) Confusion matrix (expects 3x3 or 2x2)

def plot_confusion_matrix(cm, classes, normalize=False, outfile="confusion_matrix.png"):
    cm = np.array(cm).astype(float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix (val)"); plt.colorbar()
    tick = np.arange(len(classes)); plt.xticks(tick, classes); plt.yticks(tick, classes)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout(); plt.savefig(FIG_DIR/outfile, dpi=200); plt.close()

# 3) Threshold sweep (precision‑recall vs threshold)

def plot_threshold_curve(thresholds, recalls, precisions, outfile="threshold_curve.png"):
    plt.figure()
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, precisions, label="Precision")
    plt.xlabel("Threshold"); plt.legend(); plt.title("Threshold Sweep")
    plt.tight_layout(); plt.savefig(FIG_DIR/outfile, dpi=200); plt.close()

# 4) Placeholder for CAM grid (fill with your project’s CAM function)

def save_cam_grid(images, cams, cols=4, outfile="cam_grid.png"):
    rows = int(np.ceil(len(images)/cols))
    plt.figure(figsize=(cols*3, rows*3))
    for i, (img, cam) in enumerate(zip(images, cams)):
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(img, cmap="gray")
        ax.imshow(cam, alpha=0.35)
        ax.axis('off')
    plt.tight_layout(); plt.savefig(FIG_DIR/outfile, dpi=200); plt.close()

if __name__ == "__main__":
    # Example usage (replace with real paths/arrays)
    # plot_metric_bars(["reports/val_run1.json", "reports/val_run2.json"], metric="macro_f1")
    pass
```

---

## 👥 Peer Assessment Questionnaire (Internal)
**Code Quality (1–5)**
- Clear docstrings & comments
- Single‑responsibility functions; modular structure
- Robust error handling; no silent crashes

**Experimental Rigor (1–5)**
- Full reproducibility (fixed seeds)
- Proper validation without leakage
- Justified hyperparameter choices

**Analytical Depth (1–5)**
- Specific & insightful error analysis
- Honest limitations & ethics
- Visualizations that support claims

> Scoring: average each section; flag any item ≤2 for action before submission.

---

## 📝 Report Section Template (8 pages)
1. **Introduction (1 page)** — background, significance, objectives
2. **Related Work (1 page)** — key papers & route rationale
3. **Methods (2 pages)** — data, architecture, training, imbalance handling
4. **Experiments (2 pages)** — setup, results, analysis, visualizations
5. **Discussion (1.5 pages)** — limits, ethics, future work
6. **Conclusion (0.5 page)** — contributions & takeaways

> Guideline: ~800–1000 words per full page (with figures reducing word count).

---

## 🎤 Demo Day — Final Checklist
**Tech**
- Backup weights & code; verify **offline demo** path
- Projector/browser tested; fallback plan ready
- 3 typical cases + 1 edge case queued

**Speaking**
- Timeboxed rehearsal (≤3:00); 1‑minute elevator pitch
- Prepare 3 likely technical Qs with succinct answers
- One‑pager quick reference for the panel

---

## 📋 Grading Rubric Mapping
| Dimension | Playbook Sections | Evidence |
|---|---|---|
| Technical depth | §1, §7, §8 | Model choice rationale; ablation records |
| Experimental rigor | §3, §5, §9 | Data splits; evaluation protocol; reproducibility checklist |
| Analysis quality | §9, §10 | Error galleries; CAM explanations; calibration curves |
| Engineering practice | §3, §14 | Code structure; scripts/configs; CI/checklists |
| Academic integrity | §13, §15 | Ethics statement; citations; model card |

> Place links to concrete artifacts (e.g., `reports/val.json`, `reports/figs/*`, `runs/metrics.csv`) in your README.

