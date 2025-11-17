"""
Domain Shift Analysis Script

Analyzes model performance across different image characteristics (brightness, contrast, 
resolution) to detect potential domain shift issues and robustness gaps.

This helps identify scenarios where the model may underperform, which is critical
for deployment in diverse clinical settings.

Usage:
    python scripts/domain_shift_analysis.py --ckpt runs/best.pt --data_root data --split test
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from src.models.factory import build_model
from src.data.datamodule import build_dataloaders
from src.utils.device import get_device


def compute_image_stats(image_path: Path) -> Dict:
    """Compute basic image statistics for domain analysis.
    
    Args:
        image_path: Path to image file
    
    Returns:
        Dictionary with brightness, contrast, resolution metrics
    """
    img = Image.open(image_path).convert('L')  # Grayscale
    img_array = np.array(img, dtype=np.float32)
    
    # Brightness: mean pixel intensity
    brightness = float(img_array.mean())
    
    # Contrast: standard deviation of pixel intensities
    contrast = float(img_array.std())
    
    # Resolution: image dimensions
    width, height = img.size
    resolution = width * height
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'resolution': resolution,
        'width': width,
        'height': height
    }


def categorize_image(stats: Dict) -> Dict[str, str]:
    """Categorize image into brightness/contrast/resolution bins.
    
    Args:
        stats: Dictionary with image statistics
    
    Returns:
        Dictionary with category assignments
    """
    # Brightness categories (0-255 scale)
    brightness = stats['brightness']
    if brightness < 85:
        brightness_cat = 'dark'
    elif brightness < 170:
        brightness_cat = 'normal'
    else:
        brightness_cat = 'bright'
    
    # Contrast categories
    contrast = stats['contrast']
    if contrast < 30:
        contrast_cat = 'low'
    elif contrast < 60:
        contrast_cat = 'medium'
    else:
        contrast_cat = 'high'
    
    # Resolution categories
    resolution = stats['resolution']
    if resolution < 250000:  # < 500x500
        resolution_cat = 'low'
    elif resolution < 1000000:  # < 1000x1000
        resolution_cat = 'medium'
    else:
        resolution_cat = 'high'
    
    return {
        'brightness': brightness_cat,
        'contrast': contrast_cat,
        'resolution': resolution_cat
    }


def analyze_performance_by_category(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    categories: List[Dict],
    category_type: str,
    idx_to_class: Dict
) -> Dict:
    """Analyze model performance within each category.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        categories: List of category assignments for each sample
        category_type: Type of category ('brightness', 'contrast', 'resolution')
        idx_to_class: Mapping from class index to class name
    
    Returns:
        Dictionary with per-category performance metrics
    """
    results = {}
    
    # Get unique categories
    unique_cats = set(cat[category_type] for cat in categories)
    
    for cat in unique_cats:
        # Filter samples in this category
        mask = np.array([c[category_type] == cat for c in categories])
        
        if mask.sum() == 0:
            continue
        
        cat_true = y_true[mask]
        cat_pred = y_pred[mask]
        
        # Compute metrics
        acc = accuracy_score(cat_true, cat_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            cat_true, cat_pred, average=None, zero_division=0
        )
        
        # Per-class metrics
        class_metrics = {}
        for idx, class_name in idx_to_class.items():
            if idx < len(recall):
                class_metrics[class_name] = {
                    'precision': float(precision[idx]),
                    'recall': float(recall[idx]),
                    'f1': float(f1[idx]),
                    'support': int(support[idx])
                }
        
        results[cat] = {
            'accuracy': float(acc),
            'n_samples': int(mask.sum()),
            'class_metrics': class_metrics,
            'macro_recall': float(recall.mean()),
            'macro_f1': float(f1.mean())
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Domain shift analysis for chest X-ray model")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Data root directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output', type=str, default='reports/domain_shift_analysis.json')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Domain Shift Analysis")
    print("=" * 80)
    
    # Load checkpoint
    print(f"\n[1/5] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    class_to_idx = ckpt['classes']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    
    cfg = ckpt.get('config', {})
    model_name = cfg.get('model', 'resnet18')
    img_size = int(cfg.get('img_size', 224))
    batch_size = int(cfg.get('batch_size', 16))
    
    print(f"  Model: {model_name} @ {img_size}px")
    print(f"  Classes: {list(idx_to_class.values())}")
    
    # Build model
    print("\n[2/5] Building model...")
    model, _ = build_model(model_name, num_classes)
    model.load_state_dict(ckpt['model'])
    device = get_device()
    model = model.to(device).eval()
    
    # Get data
    print(f"\n[3/5] Loading {args.split} data...")
    loaders, _ = build_dataloaders(
        args.data_root, img_size=img_size, batch_size=1,  # batch_size=1 for per-image analysis
        use_weighted_sampler=False, num_workers=0
    )
    loader = loaders[args.split]
    
    # Collect predictions and image statistics
    print(f"\n[4/5] Computing predictions and image statistics...")
    y_true, y_pred = [], []
    image_stats = []
    
    dataset = loader.dataset
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Processing"):
            # Get image and label
            image_tensor, label = dataset[idx]
            
            # Get original image path for statistics
            img_path = Path(dataset.samples[idx][0])
            stats = compute_image_stats(img_path)
            image_stats.append(stats)
            
            # Make prediction
            image_tensor = image_tensor.unsqueeze(0).to(device)
            logits = model(image_tensor)
            pred = logits.argmax(dim=1).item()
            
            y_true.append(label)
            y_pred.append(pred)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Categorize images
    print("\n[5/5] Analyzing performance by category...")
    categories = [categorize_image(stats) for stats in image_stats]
    
    # Analyze by each dimension
    results = {
        'overall': {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'n_samples': len(y_true)
        },
        'brightness': analyze_performance_by_category(
            y_true, y_pred, categories, 'brightness', idx_to_class
        ),
        'contrast': analyze_performance_by_category(
            y_true, y_pred, categories, 'contrast', idx_to_class
        ),
        'resolution': analyze_performance_by_category(
            y_true, y_pred, categories, 'resolution', idx_to_class
        )
    }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nOverall Accuracy: {results['overall']['accuracy']:.4f}")
    
    for dimension in ['brightness', 'contrast', 'resolution']:
        print(f"\n{dimension.upper()} Analysis:")
        dim_results = results[dimension]
        
        for category, metrics in dim_results.items():
            print(f"  {category:12s}: Acc={metrics['accuracy']:.4f}, "
                  f"Macro Recall={metrics['macro_recall']:.4f}, "
                  f"n={metrics['n_samples']}")
            
            # Check for concerning drops
            drop = results['overall']['accuracy'] - metrics['accuracy']
            if drop > 0.05:
                print(f"    ⚠️  WARNING: {drop:.2%} accuracy drop in {category} images")
    
    print("\n" + "=" * 80)
    print("✓ Domain shift analysis complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
