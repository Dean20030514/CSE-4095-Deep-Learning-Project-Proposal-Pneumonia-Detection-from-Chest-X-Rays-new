"""
不确定性估计脚本

使用 Monte Carlo Dropout 和熵计算评估模型预测的不确定性。

使用方法:
    python scripts/uncertainty_estimation.py --ckpt runs/best/best_model.pt --data_root data
    python scripts/uncertainty_estimation.py --ckpt runs/best/best_model.pt --num_samples 50
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm
import json


def main():
    parser = argparse.ArgumentParser(description="Uncertainty estimation using MC Dropout")
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data_root', type=str, default='data', help='Data root directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--num_samples', type=int, default=30, help='Number of MC samples')
    parser.add_argument('--output', type=str, default='reports/uncertainty_analysis', 
                       help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.7, 
                       help='High uncertainty threshold')
    args = parser.parse_args()
    
    from src.models.factory import build_model
    from src.data.datamodule import build_dataloaders
    from src.utils.device import get_device
    from src.utils.uncertainty import (
        mc_dropout_predict, 
        UncertaintyEstimator, 
        compute_entropy
    )
    
    device = get_device()
    print(f"\n[UNCERTAINTY] Loading model from {args.ckpt}")
    print(f"  Device: {device}")
    print(f"  MC samples: {args.num_samples}")
    
    # 加载模型
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    class_to_idx = ckpt['classes']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    model = build_model(
        cfg.get('model', 'efficientnet_b2'),
        pretrained=False,
        num_classes=len(class_to_idx)
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    # 加载数据
    img_size = int(cfg.get('img_size', 224))
    batch_size = int(cfg.get('batch_size', 16))
    
    loaders, _ = build_dataloaders(
        args.data_root,
        img_size=img_size,
        batch_size=batch_size,
        use_weighted_sampler=False,
        num_workers=0
    )
    loader = loaders[args.split]
    
    # 创建不确定性估计器
    estimator = UncertaintyEstimator(model, num_samples=args.num_samples)
    
    print(f"\n[EVAL] Estimating uncertainty on {args.split} set...")
    
    all_results = []
    high_uncertainty_samples = []
    correct_predictions = []
    incorrect_predictions = []
    
    sample_idx = 0
    for images, targets in tqdm(loader, desc="Uncertainty"):
        images = images.to(device)
        batch_size_cur = images.size(0)
        
        # 使用 MC Dropout 估计不确定性
        result = estimator.estimate(images)
        
        for i in range(batch_size_cur):
            sample_result = {
                'index': sample_idx,
                'true_label': idx_to_class[targets[i].item()],
                'pred_label': idx_to_class[result['predictions'][i].item()],
                'mean_prob': result['mean_probs'][i].cpu().numpy().tolist(),
                'std': result['std'][i].item(),
                'entropy': result['entropy'][i].item(),
                'is_correct': targets[i].item() == result['predictions'][i].item()
            }
            
            all_results.append(sample_result)
            
            # 分类高不确定性样本
            if result['entropy'][i].item() > args.threshold or result['std'][i].item() > 0.2:
                high_uncertainty_samples.append(sample_result)
            
            # 分类正确/错误预测
            if sample_result['is_correct']:
                correct_predictions.append(sample_result)
            else:
                incorrect_predictions.append(sample_result)
            
            sample_idx += 1
    
    # 计算统计信息
    entropies = [r['entropy'] for r in all_results]
    stds = [r['std'] for r in all_results]
    
    correct_entropies = [r['entropy'] for r in correct_predictions]
    incorrect_entropies = [r['entropy'] for r in incorrect_predictions]
    
    print(f"\n{'='*60}")
    print("Uncertainty Analysis Results")
    print(f"{'='*60}")
    print(f"\nOverall Statistics:")
    print(f"  Total samples: {len(all_results)}")
    print(f"  Accuracy: {len(correct_predictions)/len(all_results)*100:.2f}%")
    print(f"\n  Entropy:")
    print(f"    Mean: {np.mean(entropies):.4f}")
    print(f"    Std: {np.std(entropies):.4f}")
    print(f"    Min: {np.min(entropies):.4f}")
    print(f"    Max: {np.max(entropies):.4f}")
    print(f"\n  Prediction Std:")
    print(f"    Mean: {np.mean(stds):.4f}")
    print(f"    Std: {np.std(stds):.4f}")
    
    print(f"\nUncertainty by Prediction Correctness:")
    print(f"  Correct predictions:")
    print(f"    Mean entropy: {np.mean(correct_entropies):.4f}")
    print(f"    Mean std: {np.mean([r['std'] for r in correct_predictions]):.4f}")
    print(f"  Incorrect predictions:")
    if incorrect_predictions:
        print(f"    Mean entropy: {np.mean(incorrect_entropies):.4f}")
        print(f"    Mean std: {np.mean([r['std'] for r in incorrect_predictions]):.4f}")
    else:
        print(f"    (No incorrect predictions)")
    
    print(f"\nHigh Uncertainty Samples: {len(high_uncertainty_samples)}")
    print(f"  (entropy > {args.threshold} or std > 0.2)")
    
    if high_uncertainty_samples:
        print(f"\n  Top 10 highest uncertainty samples:")
        sorted_samples = sorted(high_uncertainty_samples, key=lambda x: x['entropy'], reverse=True)[:10]
        for s in sorted_samples:
            status = "✓" if s['is_correct'] else "✗"
            print(f"    [{status}] idx={s['index']}, true={s['true_label']}, "
                  f"pred={s['pred_label']}, entropy={s['entropy']:.4f}")
    
    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'checkpoint': args.ckpt,
        'split': args.split,
        'num_samples': args.num_samples,
        'total_samples': len(all_results),
        'accuracy': len(correct_predictions) / len(all_results),
        'entropy_stats': {
            'mean': float(np.mean(entropies)),
            'std': float(np.std(entropies)),
            'min': float(np.min(entropies)),
            'max': float(np.max(entropies))
        },
        'std_stats': {
            'mean': float(np.mean(stds)),
            'std': float(np.std(stds))
        },
        'high_uncertainty_count': len(high_uncertainty_samples),
        'correct_entropy_mean': float(np.mean(correct_entropies)),
        'incorrect_entropy_mean': float(np.mean(incorrect_entropies)) if incorrect_entropies else None
    }
    
    with open(output_dir / 'uncertainty_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 保存详细结果
    with open(output_dir / 'uncertainty_details.json', 'w') as f:
        json.dump({
            'all_results': all_results[:100],  # 只保存前100个
            'high_uncertainty_samples': high_uncertainty_samples
        }, f, indent=2)
    
    print(f"\n[SAVE] Results saved to {output_dir}")
    print(f"{'='*60}")
    
    # 可视化（如果matplotlib可用）
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 熵分布
        axes[0].hist(entropies, bins=50, alpha=0.7, color='steelblue')
        axes[0].axvline(args.threshold, color='red', linestyle='--', label=f'Threshold={args.threshold}')
        axes[0].set_xlabel('Entropy')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Entropy Distribution')
        axes[0].legend()
        
        # 正确 vs 错误预测的熵
        if correct_entropies and incorrect_entropies:
            axes[1].hist(correct_entropies, bins=30, alpha=0.7, label='Correct', color='green')
            axes[1].hist(incorrect_entropies, bins=30, alpha=0.7, label='Incorrect', color='red')
            axes[1].set_xlabel('Entropy')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Entropy by Correctness')
            axes[1].legend()
        
        # Std 分布
        axes[2].hist(stds, bins=50, alpha=0.7, color='orange')
        axes[2].set_xlabel('Standard Deviation')
        axes[2].set_ylabel('Count')
        axes[2].set_title('Prediction Std Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'uncertainty_distribution.png', dpi=150)
        print(f"[PLOT] Saved uncertainty_distribution.png")
        plt.close()
        
    except ImportError:
        print("[INFO] matplotlib not available, skipping visualization")


if __name__ == '__main__':
    main()

