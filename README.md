# CSE-4095-Deep-Learning-Project-Proposal-Pneumonia-Detection-from-Chest-X-Rays-new

This repository provides a minimal, runnable skeleton for the Pneumonia X-ray course project: training/evaluation scripts, config templates, and a Streamlit demo app, aligned with `pneumonia_x_ray_project_implementation_playbook_v_1.3.md`.

## Quickstart

1) Create environment (pick one)

- pip
	```bash
	python -m venv .venv && source .venv/bin/activate
	pip install -U pip
	pip install -r requirements.txt
	```
- conda
	```bash
	conda env create -f environment.yml
	conda activate cxr
	```

2) Verify environment (optional)
```bash
python scripts/verify_environment.py
```

3) Prepare data (ImageFolder layout)
```
data/
	train/
		Normal/
		Bacterial/
		Viral/
	val/
		Normal/
		Bacterial/
		Viral/
	test/  # optional
```
If you do not have data yet, use demo mode in the app (see below).

4) Train (example)
```bash
python -m src.train --config src/configs/colab_friendly.yaml --data_root data
```

5) Evaluate (example)
```bash
python -m src.eval --ckpt runs/best.pt --split val --data_root data --report reports/val.json
```

6) Run the demo app (demo_mode does not require weights)
```bash
streamlit run src/app/streamlit_app.py -- --demo_mode
```

## Project structure
- `src/` training/eval/models/data/utils code
- `src/configs/` ready-to-use configs (Colab-friendly / balanced / full-power)
- `src/app/` Streamlit demo app
- `scripts/` env check, sample download, presentation assets
- `reports/` evaluation reports and figures (git-ignored)
- `runs/` training artifacts (git-ignored)

## Disclaimer
Research/education use only. This is not a medical device and must not be used for clinical diagnosis.