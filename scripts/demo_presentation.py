"""Auto-generate presentation assets: metrics plots, confusion matrix, simple threshold curves."""
from pathlib import Path
import json, itertools
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = Path("reports/figs"); FIG_DIR.mkdir(parents=True, exist_ok=True)


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


def plot_threshold_curve(thresholds, recalls, precisions, outfile="threshold_curve.png"):
    plt.figure()
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, precisions, label="Precision")
    plt.xlabel("Threshold"); plt.legend(); plt.title("Threshold Sweep")
    plt.tight_layout(); plt.savefig(FIG_DIR/outfile, dpi=200); plt.close()


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
    # Example usage:
    # plot_metric_bars(["reports/val_run1.json", "reports/val_run2.json"], metric="macro_f1")
    pass
