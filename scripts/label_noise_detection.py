"""
Label Noise Detection Script

Identifies potentially mislabeled samples by analyzing high-confidence predictions
that contradict the ground truth labels. This is critical for medical datasets where
label noise can significantly impact model performance.

Usage:
    python scripts/label_noise_detection.py --ckpt runs/best.pt --data_root data --split test
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.models.factory import build_model
from src.data.datamodule import build_dataloaders
from src.utils.device import get_device


def detect_suspicious_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    file_paths: List[str],
    idx_to_class: Dict,
    confidence_threshold: float = 0.95
) -> Tuple[List[Dict], List[Dict]]:
    """Detect potentially mislabeled samples.
    
    Identifies samples where the model has very high confidence in a prediction
    that contradicts the ground truth label. These are candidates for manual review.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities (N, num_classes)
        file_paths: List of image file paths
        idx_to_class: Mapping from class index to class name
        confidence_threshold: Minimum confidence to flag as suspicious (default: 0.95)
    
    Returns:
        Tuple of (suspicious_samples, borderline_samples)
    """
    suspicious = []
    borderline = []
    
    for i in range(len(y_true)):
        true_label = int(y_true[i])
        pred_label = int(y_pred[i])
        
        # Skip correct predictions
        if true_label == pred_label:
            continue
        
        # Get confidence in predicted class
        pred_confidence = float(y_probs[i, pred_label])
        true_confidence = float(y_probs[i, true_label])
        
        sample_info = {
            'file_path': str(file_paths[i]),
            'true_label': idx_to_class[true_label],
            'predicted_label': idx_to_class[pred_label],
            'predicted_confidence': pred_confidence,
            'true_class_confidence': true_confidence,
            'confidence_gap': pred_confidence - true_confidence
        }
        
        # High confidence misclassification - very suspicious
        if pred_confidence >= confidence_threshold:
            sample_info['suspicion_level'] = 'high'
            sample_info['reason'] = f'Model is {pred_confidence:.1%} confident in {idx_to_class[pred_label]}, but label is {idx_to_class[true_label]}'
            suspicious.append(sample_info)
        
        # Moderate confidence with large gap - moderately suspicious
        elif pred_confidence >= 0.80 and (pred_confidence - true_confidence) >= 0.6:
            sample_info['suspicion_level'] = 'medium'
            sample_info['reason'] = f'Large confidence gap ({pred_confidence - true_confidence:.2f}) between predicted and true label'
            suspicious.append(sample_info)
        
        # Borderline cases - worth reviewing
        elif 0.6 <= pred_confidence <= 0.8:
            borderline.append(sample_info)
    
    # Sort by confidence (descending)
    suspicious.sort(key=lambda x: x['predicted_confidence'], reverse=True)
    borderline.sort(key=lambda x: x['predicted_confidence'], reverse=True)
    
    return suspicious, borderline


def analyze_confusion_patterns(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    idx_to_class: Dict
) -> Dict:
    """Analyze systematic confusion patterns between classes.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities
        idx_to_class: Mapping from class index to class name
    
    Returns:
        Dictionary with confusion pattern analysis
    """
    confusion_patterns = {}
    
    num_classes = len(idx_to_class)
    
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            if true_class == pred_class:
                continue
            
            # Find all cases where true_class was predicted as pred_class
            mask = (y_true == true_class) & (y_pred == pred_class)
            count = mask.sum()
            
            if count == 0:
                continue
            
            # Get average confidence for these misclassifications
            avg_confidence = y_probs[mask, pred_class].mean()
            
            pattern_key = f"{idx_to_class[true_class]}_as_{idx_to_class[pred_class]}"
            confusion_patterns[pattern_key] = {
                'count': int(count),
                'avg_confidence': float(avg_confidence),
                'true_class': idx_to_class[true_class],
                'predicted_class': idx_to_class[pred_class]
            }
    
    return confusion_patterns


def main():
    parser = argparse.ArgumentParser(description="Detect potentially mislabeled samples")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Data root directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to analyze')
    parser.add_argument('--confidence_threshold', type=float, default=0.95,
                        help='Confidence threshold for flagging suspicious samples')
    parser.add_argument('--output', type=str, default='reports/label_noise_analysis.json')
    parser.add_argument('--top_k', type=int, default=20,
                        help='Number of top suspicious samples to display')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Label Noise Detection Analysis")
    print("=" * 80)
    
    # Load checkpoint
    print(f"\n[1/4] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    class_to_idx = ckpt['classes']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    cfg = ckpt.get('config', {})
    model_name = cfg.get('model', 'resnet18')
    img_size = int(cfg.get('img_size', 224))
    
    print(f"  Model: {model_name} @ {img_size}px")
    print(f"  Classes: {list(idx_to_class.values())}")
    
    # Build model
    print("\n[2/4] Building model...")
    model, _ = build_model(model_name, num_classes)
    model.load_state_dict(ckpt['model'])
    device = get_device()
    model = model.to(device).eval()
    
    # Get data
    print(f"\n[3/4] Loading {args.split} data...")
    loaders, _ = build_dataloaders(
        args.data_root, img_size=img_size, batch_size=16,
        use_weighted_sampler=False, num_workers=0
    )
    loader = loaders[args.split]
    
    # Collect predictions
    print(f"\n[4/4] Computing predictions...")
    y_true, y_pred, y_probs = [], [], []
    file_paths = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(targets.numpy())
            y_probs.append(probs.cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.concatenate(y_probs, axis=0)
    
    # Get file paths
    dataset = loader.dataset
    file_paths = [dataset.samples[i][0] for i in range(len(dataset))]
    
    # Detect suspicious labels
    print("\n" + "=" * 80)
    print("Analyzing label quality...")
    print("=" * 80)
    
    suspicious, borderline = detect_suspicious_labels(
        y_true, y_pred, y_probs, file_paths, idx_to_class,
        confidence_threshold=args.confidence_threshold
    )
    
    # Analyze confusion patterns
    confusion_patterns = analyze_confusion_patterns(
        y_true, y_pred, y_probs, idx_to_class
    )
    
    # Prepare results
    results = {
        'summary': {
            'total_samples': len(y_true),
            'total_errors': int((y_true != y_pred).sum()),
            'suspicious_samples': len(suspicious),
            'borderline_samples': len(borderline),
            'confidence_threshold': args.confidence_threshold,
            'split': args.split
        },
        'confusion_patterns': confusion_patterns,
        'suspicious_samples': suspicious[:50],  # Save top 50
        'borderline_samples': borderline[:50]
    }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Full results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal samples: {len(y_true)}")
    print(f"Total errors: {(y_true != y_pred).sum()} ({(y_true != y_pred).mean():.1%})")
    print(f"\n🔍 Suspicious samples (high confidence errors): {len(suspicious)}")
    print(f"⚠️  Borderline samples (moderate confidence): {len(borderline)}")
    
    if confusion_patterns:
        print("\n" + "-" * 80)
        print("Confusion Patterns:")
        print("-" * 80)
        for pattern, info in sorted(confusion_patterns.items(), 
                                    key=lambda x: x[1]['count'], reverse=True):
            print(f"  {info['true_class']:12s} → {info['predicted_class']:12s}: "
                  f"{info['count']:3d} cases (avg conf: {info['avg_confidence']:.3f})")
    
    if suspicious:
        print("\n" + "-" * 80)
        print(f"Top {min(args.top_k, len(suspicious))} Most Suspicious Samples:")
        print("-" * 80)
        
        for i, sample in enumerate(suspicious[:args.top_k], 1):
            print(f"\n{i}. {Path(sample['file_path']).name}")
            print(f"   Label: {sample['true_label']} | Predicted: {sample['predicted_label']} "
                  f"(confidence: {sample['predicted_confidence']:.1%})")
            print(f"   {sample['reason']}")
    
    print("\n" + "=" * 80)
    print("💡 Recommendation: Manually review top suspicious samples for potential labeling errors")
    print("=" * 80)


if __name__ == '__main__':
    main()
