"""
Model calibration analysis script.
Evaluates model confidence calibration and generates reliability diagrams.
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.factory import build_model
from src.data.datamodule import build_dataloaders
from src.utils.device import get_device
from src.utils.calibration import (
    compute_calibration_metrics, 
    plot_reliability_diagram,
    TemperatureScaling
)


def collect_predictions(model, loader, device):
    """收集模型预测的 logits 和标签"""
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Collecting predictions"):
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(targets)
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_logits, all_labels


def plot_confidence_histogram(y_probs, y_true, save_path, class_names):
    """
    绘制模型置信度分布直方图(分别显示正确和错误预测)
    
    Args:
        y_probs: 预测概率 (N, C)
        y_true: 真实标签 (N,)
        save_path: 保存路径
        class_names: 类名列表
    """
    y_pred = y_probs.argmax(axis=1)
    confidences = y_probs.max(axis=1)
    
    correct_mask = (y_pred == y_true)
    correct_conf = confidences[correct_mask]
    incorrect_conf = confidences[~correct_mask]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 21)
    ax.hist(correct_conf, bins=bins, alpha=0.6, label=f'Correct ({len(correct_conf)})', 
            color='green', edgecolor='black')
    ax.hist(incorrect_conf, bins=bins, alpha=0.6, label=f'Incorrect ({len(incorrect_conf)})', 
            color='red', edgecolor='black')
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Model Confidence Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = (f"Correct predictions:\n"
                 f"  Mean conf: {correct_conf.mean():.3f}\n"
                 f"  Median conf: {np.median(correct_conf):.3f}\n\n"
                 f"Incorrect predictions:\n"
                 f"  Mean conf: {incorrect_conf.mean():.3f}\n"
                 f"  Median conf: {np.median(incorrect_conf):.3f}")
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {save_path}")


def plot_per_class_calibration(y_probs, y_true, save_path, class_names, n_bins=10):
    """
    绘制每个类别的校准曲线
    
    Args:
        y_probs: 预测概率 (N, C)
        y_true: 真实标签 (N,)
        save_path: 保存路径
        class_names: 类名字典 {idx: name}
        n_bins: 校准bin数量
    """
    num_classes = y_probs.shape[1]
    
    fig, axes = plt.subplots(1, num_classes, figsize=(6 * num_classes, 5))
    if num_classes == 1:
        axes = [axes]
    
    for class_idx in range(num_classes):
        ax = axes[class_idx]
        
        # Get probabilities for this class
        class_probs = y_probs[:, class_idx]
        is_class = (y_true == class_idx).astype(float)
        
        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(class_probs, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_confs = []
        bin_accs = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_confs.append(class_probs[mask].mean())
                bin_accs.append(is_class[mask].mean())
                bin_counts.append(mask.sum())
        
        # Plot
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
        ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, 
              edgecolor='black', label='Model')
        
        # Add counts
        for conf, acc, count in zip(bin_confs, bin_accs, bin_counts):
            ax.text(conf, acc + 0.02, str(count), ha='center', fontsize=8)
        
        class_name = class_names.get(class_idx, f'Class {class_idx}')
        ax.set_xlabel('Confidence', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{class_name} Calibration', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Model calibration analysis")
    parser.add_argument('--ckpt', required=True, help='Path to model checkpoint')
    parser.add_argument('--data_root', default='data', help='Data root directory')
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='Dataset split')
    parser.add_argument('--model', default=None, help='Override model name from checkpoint (e.g., efficientnet_b2)')
    parser.add_argument('--output_dir', default='reports/calibration', help='Output directory')
    parser.add_argument('--fit_temperature', action='store_true', 
                       help='Fit temperature scaling (only use on validation set)')
    parser.add_argument('--n_bins', type=int, default=10, help='Number of bins for calibration')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    class_to_idx = ckpt['classes']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    cfg = ckpt.get('config', {})
    
    model_name = args.model if args.model else cfg.get('model', 'resnet18')
    img_size = int(cfg.get('img_size', 224))
    
    print(f"Model: {model_name} @ {img_size}px")
    if args.model:
        print(f"⚠️  Model name overridden: {cfg.get('model')} -> {model_name}")
    print(f"Classes: {idx_to_class}")
    
    # Build model
    model, _ = build_model(model_name, len(class_to_idx))
    model.load_state_dict(ckpt['model'], strict=False)  # Use strict=False for architecture mismatch
    device = get_device()
    model = model.to(device).eval()
    
    # Load data
    print(f"Loading {args.split} data...")
    loaders, _ = build_dataloaders(args.data_root, img_size=img_size, 
                                   batch_size=16, use_weighted_sampler=False, num_workers=0)
    loader = loaders[args.split]
    
    # Collect predictions
    print("Collecting predictions...")
    logits, labels = collect_predictions(model, loader, device)
    
    # Compute probabilities (before temperature scaling)
    probs_before = F.softmax(logits, dim=1).numpy()
    labels_np = labels.numpy()
    
    # Compute calibration metrics (before)
    print("\nComputing calibration metrics (before temperature scaling)...")
    cal_metrics_before = compute_calibration_metrics(labels_np, probs_before, n_bins=args.n_bins)
    
    print(f"\n{'='*60}")
    print("CALIBRATION METRICS (Before Temperature Scaling)")
    print(f"{'='*60}")
    print(f"Expected Calibration Error (ECE): {cal_metrics_before['ece']:.4f}")
    print(f"Maximum Calibration Error (MCE):  {cal_metrics_before['mce']:.4f}")
    print(f"Brier Score:                       {cal_metrics_before['brier_score']:.4f}")
    print(f"{'='*60}\n")
    
    # Plot reliability diagram (before)
    plot_reliability_diagram(cal_metrics_before, 
                            save_path=output_dir / 'reliability_diagram_before.png')
    
    # Plot confidence histogram
    plot_confidence_histogram(probs_before, labels_np, 
                             output_dir / 'confidence_histogram.png', 
                             idx_to_class)
    
    # Plot per-class calibration
    plot_per_class_calibration(probs_before, labels_np,
                              output_dir / 'per_class_calibration.png',
                              idx_to_class, n_bins=args.n_bins)
    
    # Temperature scaling (optional, only on validation set)
    temp_result = None
    if args.fit_temperature:
        if args.split != 'val':
            print("\n⚠️  Warning: Temperature scaling should only be fit on validation set!")
            print("    Results may be over-optimistic if fit on test set.")
        
        print("\nFitting temperature scaling...")
        temp_scaler = TemperatureScaling()
        optimal_temp = temp_scaler.fit(logits, labels)
        
        print(f"[OK] Optimal temperature: {optimal_temp:.4f}")
        
        # Apply temperature scaling
        scaled_logits = temp_scaler(logits)
        probs_after = F.softmax(scaled_logits, dim=1).detach().numpy()
        
        # Compute calibration metrics (after)
        cal_metrics_after = compute_calibration_metrics(labels_np, probs_after, n_bins=args.n_bins)
        
        print(f"\n{'='*60}")
        print("CALIBRATION METRICS (After Temperature Scaling)")
        print(f"{'='*60}")
        print(f"Expected Calibration Error (ECE): {cal_metrics_after['ece']:.4f}")
        print(f"Maximum Calibration Error (MCE):  {cal_metrics_after['mce']:.4f}")
        print(f"Brier Score:                       {cal_metrics_after['brier_score']:.4f}")
        print(f"{'='*60}\n")
        
        # Plot reliability diagram (after)
        plot_reliability_diagram(cal_metrics_after, 
                               save_path=output_dir / 'reliability_diagram_after.png')
        
        temp_result = {
            'optimal_temperature': float(optimal_temp),
            'ece_before': float(cal_metrics_before['ece']),
            'ece_after': float(cal_metrics_after['ece']),
            'improvement': float(cal_metrics_before['ece'] - cal_metrics_after['ece'])
        }
        
        print(f"[OK] ECE improvement: {temp_result['improvement']:.4f}")
    
    # Save results to JSON
    output_json = {
        'model_config': {'model': model_name, 'img_size': img_size, 'checkpoint': str(args.ckpt)},
        'split': args.split,
        'n_bins': args.n_bins,
        'calibration_before': cal_metrics_before,
        'temperature_scaling': temp_result
    }
    
    json_path = output_dir / 'calibration_report.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Results saved to: {json_path}")
    print(f"\n[SUCCESS] Calibration analysis complete! Check {output_dir} for outputs.")


if __name__ == '__main__':
    main()
