import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.metrics import compute_metrics
from src.data.datamodule import build_dataloaders
from src.models.factory import build_model
from src.utils.device import get_device


def threshold_sweep(y_true, y_probs, target_class_idx, thresholds=None):
    """Sweep classification thresholds to find optimal operating points.
    
    Evaluates model performance across different decision thresholds to identify
    optimal settings for different clinical scenarios (e.g., high sensitivity
    screening vs. high precision confirmation).
    
    Args:
        y_true: Ground truth labels, shape (N,)
        y_probs: Predicted probabilities, shape (N, num_classes)
        target_class_idx: Index of target class (e.g., PNEUMONIA)
        thresholds: List of thresholds to evaluate (default: 0.1 to 0.95 in steps of 0.05)
    
    Returns:
        List of dicts containing metrics for each threshold:
            - threshold: Decision threshold
            - recall: Sensitivity/True Positive Rate
            - precision: Positive Predictive Value
            - f1: F1 score (harmonic mean of precision and recall)
            - tp, fp, fn, tn: Confusion matrix components
    
    Example:
        >>> results = threshold_sweep(y_true, y_probs, pneumonia_idx)
        >>> best = max(results, key=lambda x: x['f1'])
        >>> print(f"Best F1={best['f1']:.3f} at threshold={best['threshold']:.2f}")
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
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
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results.append({
            'threshold': float(thresh),
            'recall': float(recall),
            'precision': float(precision),
            'f1': float(f1),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
        })
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Data root (e.g., data/)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--model', type=str, default=None, help='Override model name from checkpoint (e.g., efficientnet_b2)')
    parser.add_argument('--threshold_sweep', action='store_true', help='Sweep over [0.1..0.9]')
    parser.add_argument('--report', type=str, default=None, help='Save JSON report')
    args = parser.parse_args()

    # 加载 checkpoint
    print(f"Loading checkpoint from: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    class_to_idx = ckpt['classes']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    # 从 checkpoint 中获取配置(如果存在)
    cfg = ckpt.get('config', {})
    model_name = args.model if args.model else cfg.get('model', 'resnet18')
    img_size = int(cfg.get('img_size', 224))
    batch_size = int(cfg.get('batch_size', 16))
    
    print(f"Model config: {model_name} @ {img_size}px")
    if args.model:
        print(f"[WARNING] Model name overridden from checkpoint: {cfg.get('model')} -> {model_name}")
    print(f"Classes: {idx_to_class}")
    
    # 构建模型
    model, _ = build_model(model_name, num_classes)
    model.load_state_dict(ckpt['model'])
    device = get_device()
    model = model.to(device).eval()
    
    # 找到 PNEUMONIA 类索引
    pneumonia_idx = class_to_idx.get('PNEUMONIA', class_to_idx.get('pneumonia', 1))

    # 构建 dataloader (跨平台兼容: Windows使用num_workers=0, 其他平台使用多进程)
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 4
    loaders, _ = build_dataloaders(args.data_root, img_size=img_size, batch_size=batch_size, 
                                    use_weighted_sampler=False, num_workers=num_workers)
    loader = loaders[args.split]

    # 收集预测结果
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc=f"Eval {args.split}"):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            y_pred.extend(preds.tolist())
            y_true.extend(targets.numpy().tolist())
            y_probs.append(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.concatenate(y_probs, axis=0)

    # 计算基本指标
    metrics, cm = compute_metrics(y_true, y_pred, labels=idx_to_class)
    
    # 计算 AUC 指标
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        if num_classes == 2:
            # 二分类: 使用 pneumonia 类概率
            roc_auc_val = roc_auc_score(y_true, y_probs[:, pneumonia_idx])
            pr_auc_val = average_precision_score(y_true, y_probs[:, pneumonia_idx])
        else:
            # 多分类: macro average
            roc_auc_val = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
            pr_auc_val = average_precision_score(y_true, y_probs, average='macro')
        metrics['roc_auc'] = float(roc_auc_val)
        metrics['pr_auc'] = float(pr_auc_val)
    except Exception as e:
        print(f"Warning: Could not compute AUC metrics: {e}")
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None
    
    # 阈值扫描(如果请求)
    threshold_results = None
    if args.threshold_sweep:
        print("\nPerforming threshold sweep for PNEUMONIA class...")
        threshold_results = threshold_sweep(y_true, y_probs, pneumonia_idx)
        
        # 找到 max recall 和 balanced F1 的最佳阈值
        max_recall_result = max(threshold_results, key=lambda x: x['recall'])
        max_f1_result = max(threshold_results, key=lambda x: x['f1'])
        
        print(f"\nMax-Recall mode (threshold={max_recall_result['threshold']:.2f}):")
        print(f"  Recall: {max_recall_result['recall']:.4f}, Precision: {max_recall_result['precision']:.4f}, F1: {max_recall_result['f1']:.4f}")
        
        print(f"\nBalanced mode (threshold={max_f1_result['threshold']:.2f}):")
        print(f"  Recall: {max_f1_result['recall']:.4f}, Precision: {max_f1_result['precision']:.4f}, F1: {max_f1_result['f1']:.4f}")
        
        metrics['threshold_sweep'] = {
            'all_results': threshold_results,
            'max_recall_mode': max_recall_result,
            'balanced_mode': max_f1_result
        }
        
        # 生成 PR 和 ROC 曲线可视化
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import precision_recall_curve, roc_curve, auc
            
            # 获取 PNEUMONIA 类的概率
            pneumonia_probs = y_probs[:, pneumonia_idx]
            y_true_binary = (y_true == pneumonia_idx).astype(int)
            
            # 创建输出目录
            ckpt_path = Path(args.ckpt)
            viz_dir = ckpt_path.parent / 'evaluation_curves'
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Precision-Recall 曲线
            precisions, recalls, _pr_thresholds = precision_recall_curve(y_true_binary, pneumonia_probs)
            pr_auc_score = auc(recalls, precisions)
            
            _fig1, ax = plt.subplots(figsize=(10, 8))
            ax.plot(recalls, precisions, linewidth=2, label=f'PR Curve (AUC={pr_auc_score:.4f})')
            ax.scatter([max_recall_result['recall']], [max_recall_result['precision']], 
                      s=200, c='red', marker='*', label=f"Max Recall (thresh={max_recall_result['threshold']:.2f})", zorder=5)
            ax.scatter([max_f1_result['recall']], [max_f1_result['precision']], 
                      s=200, c='green', marker='D', label=f"Max F1 (thresh={max_f1_result['threshold']:.2f})", zorder=5)
            
            ax.set_xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Precision (PPV)', fontsize=14, fontweight='bold')
            ax.set_title(f'Precision-Recall Curve - {args.split.upper()} Set\nPNEUMONIA Detection', 
                        fontsize=16, fontweight='bold')
            ax.legend(fontsize=12, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0.0, 1.05])
            ax.set_ylim([0.0, 1.05])
            
            pr_curve_path = viz_dir / f'pr_curve_{args.split}.png'
            plt.savefig(pr_curve_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"\n[OK] PR curve saved: {pr_curve_path}")
            
            # 2. ROC 曲线
            fpr, tpr, _roc_thresholds = roc_curve(y_true_binary, pneumonia_probs)
            roc_auc_score_val = auc(fpr, tpr)
            
            _fig2, ax = plt.subplots(figsize=(10, 8))
            ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC={roc_auc_score_val:.4f})')
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            
            ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
            ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
            ax.set_title(f'ROC Curve - {args.split.upper()} Set\nPNEUMONIA Detection', 
                        fontsize=16, fontweight='bold')
            ax.legend(fontsize=12, loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            
            roc_curve_path = viz_dir / f'roc_curve_{args.split}.png'
            plt.savefig(roc_curve_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"[OK] ROC curve saved: {roc_curve_path}")
            
        except ImportError:
            print("\n[WARNING] matplotlib not available, skipping curve visualization")
        except Exception as e:
            print(f"\n[WARNING] Failed to generate curves: {e}")
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"Evaluation Results on {args.split} set:")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Macro Recall: {metrics['overall']['macro_recall']:.4f}")
    if metrics.get('roc_auc'):
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print("\nPer-class metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics.get('precision', 0):.4f}")
        print(f"    Recall: {class_metrics.get('recall', 0):.4f}")
        print(f"    F1-score: {class_metrics.get('f1-score', 0):.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print(f"{'='*60}")
    
    # 保存报告（如果指定了输出路径）
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            'split': args.split,
            'checkpoint': str(args.ckpt),
            'model_config': {'model': model_name, 'img_size': img_size},
            'metrics': metrics,
            'confusion_matrix': cm.tolist()
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[OK] Report saved to: {report_path}")
    else:
        print("\n[INFO] No report path specified (use --report to save JSON)")


if __name__ == '__main__':
    main()
