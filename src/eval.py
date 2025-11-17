import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

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

    # 构建 dataloader (Windows multiprocessing fix: num_workers=0)
    loaders, _ = build_dataloaders(args.data_root, img_size=img_size, batch_size=batch_size, 
                                    use_weighted_sampler=False, num_workers=0)
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
        if num_classes == 2:
            # 二分类: 使用 pneumonia 类概率
            roc_auc = roc_auc_score(y_true, y_probs[:, pneumonia_idx])
            pr_auc = average_precision_score(y_true, y_probs[:, pneumonia_idx])
        else:
            # 多分类: macro average
            roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
            pr_auc = average_precision_score(y_true, y_probs, average='macro')
        metrics['roc_auc'] = float(roc_auc)
        metrics['pr_auc'] = float(pr_auc)
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
    
    # 保存报告
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
    
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
