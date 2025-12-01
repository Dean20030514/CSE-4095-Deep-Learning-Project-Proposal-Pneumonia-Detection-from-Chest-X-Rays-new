#!/usr/bin/env python3
"""
K-Fold 交叉验证脚本

对数据集进行 k-fold 交叉验证以获得更可靠的性能估计。
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets
import yaml

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_fold_datasets(
    data_root: Path,
    output_dir: Path,
    n_folds: int = 5,
    seed: int = 42
) -> List[Dict[str, Path]]:
    """
    创建 k-fold 交叉验证的数据集分割。
    
    Args:
        data_root: 原始数据根目录
        output_dir: 输出目录
        n_folds: 折数
        seed: 随机种子
    
    Returns:
        每个 fold 的数据路径列表
    """
    # 加载所有图像路径和标签
    train_dir = data_root / 'train'
    
    # 使用 ImageFolder 获取所有样本
    dataset = datasets.ImageFolder(train_dir)
    all_paths = [s[0] for s in dataset.samples]
    all_labels = [s[1] for s in dataset.samples]
    
    # 创建分层 k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    folds = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels)):
        print(f"\nCreating fold {fold_idx + 1}/{n_folds}")
        
        fold_dir = output_dir / f'fold_{fold_idx}'
        fold_train_dir = fold_dir / 'train'
        fold_val_dir = fold_dir / 'val'
        
        # 清理并创建目录
        if fold_dir.exists():
            shutil.rmtree(fold_dir)
        
        for class_name in dataset.classes:
            (fold_train_dir / class_name).mkdir(parents=True, exist_ok=True)
            (fold_val_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # 复制训练集
        for idx in train_idx:
            src_path = Path(all_paths[idx])
            class_name = src_path.parent.name
            dst_path = fold_train_dir / class_name / src_path.name
            shutil.copy(src_path, dst_path)
        
        # 复制验证集
        for idx in val_idx:
            src_path = Path(all_paths[idx])
            class_name = src_path.parent.name
            dst_path = fold_val_dir / class_name / src_path.name
            shutil.copy(src_path, dst_path)
        
        # 如果存在测试集，复制到每个 fold
        test_dir = data_root / 'test'
        if test_dir.exists():
            fold_test_dir = fold_dir / 'test'
            shutil.copytree(test_dir, fold_test_dir)
        
        folds.append({
            'fold_idx': fold_idx,
            'data_dir': fold_dir,
            'train_size': len(train_idx),
            'val_size': len(val_idx)
        })
        
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Val: {len(val_idx)} samples")
    
    return folds


def run_cross_validation(
    config_path: str,
    data_root: str,
    output_dir: str,
    n_folds: int = 5,
    seed: int = 42
) -> Dict:
    """
    运行 k-fold 交叉验证。
    
    Args:
        config_path: 训练配置文件路径
        data_root: 数据根目录
        output_dir: 输出目录
        n_folds: 折数
        seed: 随机种子
    
    Returns:
        交叉验证结果
    """
    import subprocess
    
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建 fold 数据集
    folds_dir = output_dir / 'folds_data'
    folds = create_fold_datasets(data_root, folds_dir, n_folds, seed)
    
    # 加载配置
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # 运行每个 fold
    all_results = []
    
    for fold_info in folds:
        fold_idx = fold_info['fold_idx']
        fold_data_dir = fold_info['data_dir']
        fold_output_dir = output_dir / f'fold_{fold_idx}'
        
        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # 修改配置
        fold_config = base_config.copy()
        fold_config['data_root'] = str(fold_data_dir)
        fold_config['output_dir'] = str(fold_output_dir)
        fold_config['seed'] = seed + fold_idx  # 每个 fold 使用不同种子
        
        # 保存 fold 配置
        fold_config_path = output_dir / f'fold_{fold_idx}_config.yaml'
        with open(fold_config_path, 'w') as f:
            yaml.dump(fold_config, f)
        
        # 运行训练
        train_cmd = [
            'python', 'src/train.py',
            '--config', str(fold_config_path),
            '--data_root', str(fold_data_dir)
        ]
        
        subprocess.run(train_cmd, check=True)
        
        # 运行评估
        best_ckpt = fold_output_dir / 'best_model.pt'
        report_path = fold_output_dir / 'eval_results.json'
        
        eval_cmd = [
            'python', 'src/eval.py',
            '--ckpt', str(best_ckpt),
            '--data_root', str(fold_data_dir),
            '--split', 'val',
            '--report', str(report_path)
        ]
        
        subprocess.run(eval_cmd, check=True)
        
        # 读取结果
        if report_path.exists():
            with open(report_path, 'r') as f:
                fold_results = json.load(f)
                all_results.append(fold_results)
    
    # 汇总结果
    summary = aggregate_results(all_results)
    
    # 保存汇总结果
    summary_path = output_dir / 'cv_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Cross-Validation Summary")
    print(f"{'='*60}")
    print(f"Number of folds: {n_folds}")
    print(f"\nMetrics (mean ± std):")
    print(f"  Accuracy: {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}")
    print(f"  Macro Recall: {summary['macro_recall']['mean']:.4f} ± {summary['macro_recall']['std']:.4f}")
    print(f"  Macro F1: {summary['macro_f1']['mean']:.4f} ± {summary['macro_f1']['std']:.4f}")
    
    if 'pneumonia_recall' in summary:
        print(f"  Pneumonia Recall: {summary['pneumonia_recall']['mean']:.4f} ± {summary['pneumonia_recall']['std']:.4f}")
    
    print(f"\nResults saved to: {summary_path}")
    
    return summary


def aggregate_results(results: List[Dict]) -> Dict:
    """汇总所有 fold 的结果"""
    if not results:
        return {}
    
    # 收集所有指标
    metrics = {}
    
    for result in results:
        result_metrics = result.get('metrics', {})
        overall = result_metrics.get('overall', {})
        
        # 准确率
        if 'accuracy' not in metrics:
            metrics['accuracy'] = []
        metrics['accuracy'].append(overall.get('accuracy', 0))
        
        # 宏平均召回率
        if 'macro_recall' not in metrics:
            metrics['macro_recall'] = []
        metrics['macro_recall'].append(overall.get('macro_recall', 0))
        
        # 宏平均 F1
        if 'macro_f1' not in metrics:
            metrics['macro_f1'] = []
        metrics['macro_f1'].append(result_metrics.get('macro_f1', 0))
        
        # 肺炎召回率
        per_class = result_metrics.get('per_class', {})
        pneumonia = per_class.get('PNEUMONIA', per_class.get('pneumonia', {}))
        if pneumonia and 'recall' in pneumonia:
            if 'pneumonia_recall' not in metrics:
                metrics['pneumonia_recall'] = []
            metrics['pneumonia_recall'].append(pneumonia['recall'])
    
    # 计算均值和标准差
    summary = {}
    for metric_name, values in metrics.items():
        values_array = np.array(values)
        summary[metric_name] = {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'values': [float(v) for v in values]
        }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Run k-fold cross-validation')
    parser.add_argument('--config', required=True, help='Path to training config')
    parser.add_argument('--data_root', default='data', help='Data root directory')
    parser.add_argument('--output_dir', default='runs/cross_validation', help='Output directory')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    run_cross_validation(
        args.config,
        args.data_root,
        args.output_dir,
        args.n_folds,
        args.seed
    )


if __name__ == '__main__':
    main()

