"""
Threshold sweep analysis for pneumonia detection.
Finds optimal thresholds for max-recall and balanced modes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.factory import build_model
from src.data.datamodule import build_dataloaders
from src.utils.device import get_device


def threshold_sweep(y_true, y_probs, target_class_idx, thresholds=None):
    """
    扫描不同阈值,计算每个阈值下的 precision/recall/f1
    
    Args:
        y_true: 真实标签 (N,)
        y_probs: 预测概率 (N, num_classes)
        target_class_idx: 目标类索引(如 PNEUMONIA)
        thresholds: 要扫描的阈值列表
    
    Returns:
        results: 包含每个阈值结果的字典列表
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.025)  # 更细粒度
    
    results = []
    target_probs = y_probs[:, target_class_idx]
    is_target = (y_true == target_class_idx).astype(int)
    
    for thresh in thresholds:
        pred_binary = (target_probs >= thresh).astype(int)
        
        tp = ((pred_binary == 1) & (is_target == 1)).sum()
        fp = ((pred_binary == 1) & (is_target == 0)).sum()
        fn = ((pred_binary == 0) & (is_target == 1)).sum()
        tn = ((pred_binary == 0) & (is_target == 0)).sum()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Youden's Index: sensitivity + specificity - 1
        youden = recall + specificity - 1
        
        results.append({
            'threshold': float(thresh),
            'recall': float(recall),
            'precision': float(precision),
            'specificity': float(specificity),
            'f1': float(f1),
            'youden_index': float(youden),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
        })
    
    return results


def plot_threshold_curves(results, save_dir, class_name='PNEUMONIA'):
    """
    绘制阈值扫描曲线
    
    Args:
        results: threshold_sweep返回的结果列表
        save_dir: 保存目录
        class_name: 目标类名称
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    thresholds = [r['threshold'] for r in results]
    recalls = [r['recall'] for r in results]
    precisions = [r['precision'] for r in results]
    specificities = [r['specificity'] for r in results]
    f1s = [r['f1'] for r in results]
    youdens = [r['youden_index'] for r in results]
    
    # Plot 1: Precision-Recall-F1 curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, recalls, 'o-', label='Recall (Sensitivity)', linewidth=2, markersize=3)
    ax.plot(thresholds, precisions, 's-', label='Precision', linewidth=2, markersize=3)
    ax.plot(thresholds, f1s, '^-', label='F1-Score', linewidth=2, markersize=3)
    ax.plot(thresholds, specificities, 'D-', label='Specificity', linewidth=2, markersize=3)
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{class_name} Detection: Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'threshold_metrics_curve.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_dir / 'threshold_metrics_curve.png'}")
    
    # Plot 2: Precision-Recall curve
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recalls, precisions, 'o-', linewidth=2, markersize=4)
    
    # Mark some key thresholds
    for i in [0, len(results)//4, len(results)//2, 3*len(results)//4, -1]:
        ax.annotate(f"t={results[i]['threshold']:.2f}", 
                   xy=(recalls[i], precisions[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, alpha=0.7,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'{class_name} Detection: Precision-Recall Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'precision_recall_curve.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_dir / 'precision_recall_curve.png'}")
    
    # Plot 3: Youden's Index (optimal balance point)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, youdens, 'o-', linewidth=2, markersize=4, color='purple')
    
    # Mark maximum Youden's index
    max_youden_idx = np.argmax(youdens)
    ax.axvline(thresholds[max_youden_idx], color='red', linestyle='--', 
               label=f"Max Youden at t={thresholds[max_youden_idx]:.3f}")
    ax.plot(thresholds[max_youden_idx], youdens[max_youden_idx], 'r*', markersize=20)
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel("Youden's Index (Sensitivity + Specificity - 1)", fontsize=12)
    ax.set_title(f'{class_name} Detection: Optimal Operating Point', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'youden_index_curve.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_dir / 'youden_index_curve.png'}")


def find_optimal_thresholds(results):
    """
    找到不同模式下的最优阈值
    
    Returns:
        dict with optimal thresholds for different strategies
    """
    # Max Recall mode (prioritize sensitivity, catch all pneumonia cases)
    max_recall_result = max(results, key=lambda x: x['recall'])
    
    # Balanced mode (maximize F1-score)
    max_f1_result = max(results, key=lambda x: x['f1'])
    
    # Max Youden mode (maximize sensitivity + specificity - 1)
    max_youden_result = max(results, key=lambda x: x['youden_index'])
    
    # High Precision mode (minimize false alarms, precision >= 0.95)
    high_prec_candidates = [r for r in results if r['precision'] >= 0.95]
    if high_prec_candidates:
        high_prec_result = max(high_prec_candidates, key=lambda x: x['recall'])
    else:
        high_prec_result = max(results, key=lambda x: x['precision'])
    
    # Min Miss mode (recall >= 0.99, minimize false positives)
    min_miss_candidates = [r for r in results if r['recall'] >= 0.99]
    if min_miss_candidates:
        min_miss_result = max(min_miss_candidates, key=lambda x: x['specificity'])
    else:
        min_miss_result = max_recall_result
    
    return {
        'max_recall': max_recall_result,
        'balanced_f1': max_f1_result,
        'max_youden': max_youden_result,
        'high_precision': high_prec_result,
        'min_miss': min_miss_result
    }


def main():
    parser = argparse.ArgumentParser(description="Threshold sweep analysis for pneumonia detection")
    parser.add_argument('--ckpt', required=True, help='Path to model checkpoint')
    parser.add_argument('--data_root', default='data', help='Data root directory')
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='Dataset split')
    parser.add_argument('--output_dir', default='reports/threshold_analysis', help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    class_to_idx = ckpt['classes']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    cfg = ckpt.get('config', {})
    
    model_name = cfg.get('model', 'resnet18')
    img_size = int(cfg.get('img_size', 224))
    pneumonia_idx = class_to_idx.get('PNEUMONIA', class_to_idx.get('pneumonia', 1))
    
    print(f"Model: {model_name} @ {img_size}px")
    print(f"Classes: {idx_to_class}")
    print(f"Pneumonia class index: {pneumonia_idx}")
    
    # Build model
    model, _ = build_model(model_name, len(class_to_idx))
    model.load_state_dict(ckpt['model'])
    device = get_device()
    model = model.to(device).eval()
    
    # Load data
    print(f"Loading {args.split} data...")
    loaders, _ = build_dataloaders(args.data_root, img_size=img_size, 
                                   batch_size=16, use_weighted_sampler=False)
    loader = loaders[args.split]
    
    # Collect predictions
    print("Collecting predictions...")
    y_true, y_probs = [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=f"Inference on {args.split}"):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            
            y_true.extend(targets.numpy().tolist())
            y_probs.append(probs.cpu().numpy())
    
    y_true = np.array(y_true)
    y_probs = np.concatenate(y_probs, axis=0)
    
    # Perform threshold sweep
    print("\nPerforming threshold sweep...")
    results = threshold_sweep(y_true, y_probs, pneumonia_idx)
    
    # Find optimal thresholds
    optimal_thresholds = find_optimal_thresholds(results)
    
    # Print summary
    print("\n" + "="*70)
    print("THRESHOLD SWEEP RESULTS")
    print("="*70)
    
    for mode_name, result in optimal_thresholds.items():
        print(f"\n【{mode_name.upper().replace('_', ' ')} MODE】")
        print(f"  Threshold: {result['threshold']:.3f}")
        print(f"  Recall (Sensitivity): {result['recall']:.4f}")
        print(f"  Precision (PPV):      {result['precision']:.4f}")
        print(f"  Specificity (TNR):    {result['specificity']:.4f}")
        print(f"  F1-Score:             {result['f1']:.4f}")
        print(f"  Youden's Index:       {result['youden_index']:.4f}")
        print(f"  Confusion: TP={result['tp']}, FP={result['fp']}, FN={result['fn']}, TN={result['tn']}")
    
    print("\n" + "="*70)
    print("\nRECOMMENDATIONS:")
    print("  • Medical Screening (avoid missing cases): Use MIN MISS or MAX RECALL mode")
    print("  • Balanced Clinical Use: Use BALANCED F1 or MAX YOUDEN mode")
    print("  • Confirmatory Testing (minimize false alarms): Use HIGH PRECISION mode")
    print("="*70 + "\n")
    
    # Plot curves
    print("Generating threshold curves...")
    plot_threshold_curves(results, output_dir)
    
    # Save results to JSON
    output_json = {
        'model_config': {'model': model_name, 'img_size': img_size, 'checkpoint': str(args.ckpt)},
        'split': args.split,
        'pneumonia_class_idx': pneumonia_idx,
        'optimal_thresholds': optimal_thresholds,
        'all_results': results
    }
    
    json_path = output_dir / 'threshold_sweep_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to: {json_path}")
    print(f"\n✅ Threshold sweep analysis complete! Check {output_dir} for outputs.")


if __name__ == '__main__':
    main()
