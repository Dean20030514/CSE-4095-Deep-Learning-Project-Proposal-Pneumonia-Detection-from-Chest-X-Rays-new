"""
模型集成评估脚本

使用多个训练好的模型进行集成预测和评估。

使用方法:
    python scripts/ensemble_evaluation.py --checkpoints model1.pt model2.pt model3.pt --data_root data
    python scripts/ensemble_evaluation.py --runs_dir runs --top_k 3 --strategy weighted
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm


def find_best_checkpoints(runs_dir: Path, top_k: int = 3) -> list:
    """
    从 runs 目录中找到最佳的 K 个模型。
    
    Args:
        runs_dir: 实验目录
        top_k: 选择的模型数量
    
    Returns:
        最佳模型路径列表
    """
    checkpoints = []
    
    for ckpt_path in runs_dir.glob('*/best_model.pt'):
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            metrics = ckpt.get('metrics', {})
            score = metrics.get('macro_f1', 0) + metrics.get('pneumonia_recall', 0)
            checkpoints.append({
                'path': str(ckpt_path),
                'score': score,
                'name': ckpt_path.parent.name
            })
        except Exception as e:
            print(f"Warning: Could not load {ckpt_path}: {e}")
    
    # 按分数排序
    checkpoints.sort(key=lambda x: x['score'], reverse=True)
    
    return [c['path'] for c in checkpoints[:top_k]]


def evaluate_ensemble(
    checkpoint_paths: list,
    data_root: str,
    split: str = 'test',
    strategy: str = 'average',
    weights: list = None,
    threshold: float = 0.5
):
    """
    评估模型集成。
    
    Args:
        checkpoint_paths: 模型检查点路径列表
        data_root: 数据根目录
        split: 数据集划分
        strategy: 集成策略 ('average', 'weighted', 'voting')
        weights: 模型权重
        threshold: 分类阈值
    """
    from src.models.ensemble import ModelEnsemble, load_ensemble_from_checkpoints
    from src.models.factory import build_model
    from src.data.datamodule import build_dataloaders
    from src.utils.metrics import compute_metrics
    from src.utils.device import get_device
    import torch.nn.functional as F
    
    device = get_device()
    print(f"\n[ENSEMBLE] Loading {len(checkpoint_paths)} models...")
    print(f"  Strategy: {strategy}")
    print(f"  Device: {device}")
    
    # 加载集成模型
    ensemble = load_ensemble_from_checkpoints(
        checkpoint_paths,
        model_builder=build_model,
        num_classes=2,
        strategy=strategy,
        weights=weights,
        device=str(device)
    )
    
    # 获取类别映射
    first_ckpt = torch.load(checkpoint_paths[0], map_location='cpu', weights_only=False)
    class_to_idx = first_ckpt['classes']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    pneumonia_idx = class_to_idx.get('PNEUMONIA', class_to_idx.get('pneumonia', 1))
    
    cfg = first_ckpt.get('config', {})
    img_size = int(cfg.get('img_size', 224))
    batch_size = int(cfg.get('batch_size', 16))
    
    # 加载数据
    print(f"\n[DATA] Loading {split} set from {data_root}...")
    loaders, _ = build_dataloaders(
        data_root, 
        img_size=img_size, 
        batch_size=batch_size,
        use_weighted_sampler=False,
        num_workers=0
    )
    loader = loaders[split]
    
    # 评估
    print(f"\n[EVAL] Evaluating on {split} set...")
    y_true, y_pred, y_probs = [], [], []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Ensemble Eval"):
            images = images.to(device)
            probs = ensemble(images)
            
            if threshold != 0.5:
                preds = (probs[:, pneumonia_idx] >= threshold).long()
            else:
                preds = probs.argmax(dim=1)
            
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_probs.append(probs.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.concatenate(y_probs, axis=0)
    
    # 计算指标
    metrics, cm = compute_metrics(y_true, y_pred, labels=idx_to_class)
    
    # 计算 AUC
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc_auc = roc_auc_score(y_true, y_probs[:, pneumonia_idx])
        pr_auc = average_precision_score(y_true, y_probs[:, pneumonia_idx])
        metrics['roc_auc'] = float(roc_auc)
        metrics['pr_auc'] = float(pr_auc)
    except Exception:
        pass
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"Ensemble Evaluation Results ({split} set)")
    print(f"{'='*60}")
    print(f"Models used: {len(checkpoint_paths)}")
    print(f"Strategy: {strategy}")
    print(f"Threshold: {threshold}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {metrics['overall']['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Macro Recall: {metrics['overall']['macro_recall']:.4f}")
    if metrics.get('roc_auc'):
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    
    print(f"\nPer-class Metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {class_metrics.get('precision', 0):.4f}")
        print(f"    Recall: {class_metrics.get('recall', 0):.4f}")
        print(f"    F1: {class_metrics.get('f1-score', 0):.4f}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"{'='*60}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Ensemble model evaluation")
    parser.add_argument('--checkpoints', type=str, nargs='+', default=None,
                       help='Paths to model checkpoints')
    parser.add_argument('--runs_dir', type=str, default='runs',
                       help='Directory containing experiment runs')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top models to use (if --checkpoints not specified)')
    parser.add_argument('--data_root', type=str, default='data',
                       help='Data root directory')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'])
    parser.add_argument('--strategy', type=str, default='average',
                       choices=['average', 'weighted', 'voting'])
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    args = parser.parse_args()
    
    # 获取检查点路径
    if args.checkpoints:
        checkpoint_paths = args.checkpoints
    else:
        runs_dir = Path(args.runs_dir)
        if not runs_dir.exists():
            print(f"Error: Runs directory not found: {runs_dir}")
            return
        
        print(f"Finding top {args.top_k} models from {runs_dir}...")
        checkpoint_paths = find_best_checkpoints(runs_dir, args.top_k)
        
        if not checkpoint_paths:
            print("Error: No valid checkpoints found")
            return
        
        print(f"Selected models:")
        for path in checkpoint_paths:
            print(f"  - {path}")
    
    # 评估
    evaluate_ensemble(
        checkpoint_paths,
        args.data_root,
        args.split,
        args.strategy,
        threshold=args.threshold
    )


if __name__ == '__main__':
    main()

