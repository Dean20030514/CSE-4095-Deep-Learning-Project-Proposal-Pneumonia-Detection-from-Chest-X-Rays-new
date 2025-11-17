"""
Grad-CAM Quantitative Evaluation Script

Evaluates the quality of Grad-CAM explanations by analyzing whether the highlighted
regions align with clinically relevant areas (lung fields). This provides objective
evidence that the model is "looking at the right places".

Usage:
    python scripts/gradcam_evaluation.py --ckpt runs/best.pt --data_root data --n_samples 50
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.models.factory import build_model
from src.utils.gradcam import GradCAM
from src.utils.device import get_device
import torchvision.transforms as T


def compute_gradcam_statistics(cam: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute quantitative statistics for a Grad-CAM heatmap.
    
    Args:
        cam: Grad-CAM heatmap (H, W)
        threshold: Threshold for considering a region "important"
    
    Returns:
        Dictionary with statistics about the heatmap
    """
    # Normalize cam to [0, 1]
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Compute statistics
    stats = {
        'mean_activation': float(cam_norm.mean()),
        'max_activation': float(cam_norm.max()),
        'std_activation': float(cam_norm.std()),
        'activation_concentration': float((cam_norm > threshold).sum() / cam_norm.size)  # % of pixels above threshold
    }
    
    # Compute spatial distribution (check if focused on center/edges)
    h, w = cam_norm.shape
    center_region = cam_norm[h//4:3*h//4, w//4:3*w//4]
    stats['center_activation_ratio'] = float(center_region.mean() / (cam_norm.mean() + 1e-8))
    
    return stats


def evaluate_lung_region_coverage(
    cam: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """Evaluate whether Grad-CAM focuses on lung regions (heuristic-based).
    
    For chest X-rays, we expect the model to focus on the central lung field area,
    not the edges or corners where no lung tissue is present.
    
    Args:
        cam: Grad-CAM heatmap (H, W), normalized to [0, 1]
        threshold: Activation threshold
    
    Returns:
        Dictionary with coverage metrics
    """
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    h, w = cam_norm.shape
    
    # Define approximate lung region (center 60% horizontally, 70% vertically)
    lung_mask = np.zeros_like(cam_norm)
    h_start, h_end = int(0.15 * h), int(0.85 * h)
    w_start, w_end = int(0.20 * w), int(0.80 * w)
    lung_mask[h_start:h_end, w_start:w_end] = 1
    
    # Compute activation in lung region vs outside
    high_activation_mask = (cam_norm > threshold).astype(float)
    
    lung_activation = (high_activation_mask * lung_mask).sum()
    total_activation = high_activation_mask.sum()
    
    coverage = {
        'lung_region_focus_ratio': float(lung_activation / (total_activation + 1e-8)),
        'lung_region_mean_activation': float((cam_norm * lung_mask).sum() / (lung_mask.sum() + 1e-8)),
        'non_lung_region_mean_activation': float((cam_norm * (1 - lung_mask)).sum() / ((1 - lung_mask).sum() + 1e-8))
    }
    
    return coverage


def visualize_gradcam_sample(
    image: Image.Image,
    cam: np.ndarray,
    pred_class: str,
    true_class: str,
    save_path: Path
):
    """Visualize a Grad-CAM example for report."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')
    
    # Overlay
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap_colored = cm.get_cmap('jet')(cam_norm)[:, :, :3]
    
    img_array = np.array(image.convert('RGB')).astype(float) / 255.0
    overlay = 0.6 * img_array + 0.4 * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\nPred: {pred_class} | True: {true_class}', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Quantitative Grad-CAM evaluation")
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True, help='Data root directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of samples to evaluate (randomly selected)')
    parser.add_argument('--output', type=str, default='reports/gradcam_evaluation.json')
    parser.add_argument('--visualize', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 80)
    print("Grad-CAM Quantitative Evaluation")
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
    
    # Build model
    print("\n[2/4] Building model...")
    model, _ = build_model(model_name, num_classes)
    model.load_state_dict(ckpt['model'])
    device = get_device()
    model = model.to(device).eval()
    
    # Determine target layer for Grad-CAM
    if 'resnet' in model_name.lower():
        target_layer = 'layer4'
    elif 'efficientnet' in model_name.lower():
        target_layer = 'features'
    elif 'densenet' in model_name.lower():
        target_layer = 'features'
    else:
        target_layer = 'layer4'
    
    print(f"  Using Grad-CAM target layer: {target_layer}")
    gradcam = GradCAM(model, target_layer)
    
    # Get data paths
    print(f"\n[3/4] Loading {args.split} data...")
    data_dir = Path(args.data_root) / args.split
    
    # Collect all image paths
    all_samples = []
    for class_name in class_to_idx.keys():
        class_dir = data_dir / class_name
        if class_dir.exists():
            all_samples.extend([
                (str(p), class_to_idx[class_name], class_name) 
                for p in class_dir.glob('*.*') 
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ])
    
    # Randomly sample
    n_samples = min(args.n_samples, len(all_samples))
    sampled_items = random.sample(all_samples, n_samples)
    
    print(f"  Evaluating {n_samples} randomly selected samples")
    
    # Prepare transform
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Evaluate Grad-CAM for each sample
    print(f"\n[4/4] Computing Grad-CAM statistics...")
    
    results_by_class = {class_name: [] for class_name in idx_to_class.values()}
    correct_predictions = []
    incorrect_predictions = []
    
    vis_dir = Path('reports/gradcam_visualizations')
    vis_dir.mkdir(parents=True, exist_ok=True)
    visualized_count = 0
    
    for img_path, true_label, true_class_name in tqdm(sampled_items, desc="Evaluating"):
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Make prediction
        img_tensor.requires_grad = True
        with torch.enable_grad():
            logits = model(img_tensor)
            pred_label = logits.argmax(dim=1).item()
            pred_class_name = idx_to_class[pred_label]
        
        # Generate Grad-CAM
        cam = gradcam(logits, pred_label)
        cam_np = cam.cpu().numpy()
        
        # Compute statistics
        cam_stats = compute_gradcam_statistics(cam_np, threshold=0.5)
        coverage_stats = evaluate_lung_region_coverage(cam_np, threshold=0.5)
        
        result = {
            'image': Path(img_path).name,
            'true_class': true_class_name,
            'predicted_class': pred_class_name,
            'correct': pred_label == true_label,
            'cam_statistics': cam_stats,
            'lung_coverage': coverage_stats
        }
        
        results_by_class[true_class_name].append(result)
        
        if pred_label == true_label:
            correct_predictions.append(result)
        else:
            incorrect_predictions.append(result)
        
        # Visualize a few samples
        if visualized_count < args.visualize:
            save_path = vis_dir / f"sample_{visualized_count:02d}_{true_class_name}_pred_{pred_class_name}.png"
            visualize_gradcam_sample(img, cam_np, pred_class_name, true_class_name, save_path)
            visualized_count += 1
    
    # Aggregate statistics
    print("\n" + "=" * 80)
    print("Computing aggregate statistics...")
    print("=" * 80)
    
    def aggregate_stats(results_list):
        if not results_list:
            return {}
        
        cam_means = [r['cam_statistics']['mean_activation'] for r in results_list]
        lung_focus = [r['lung_coverage']['lung_region_focus_ratio'] for r in results_list]
        center_ratio = [r['cam_statistics']['center_activation_ratio'] for r in results_list]
        
        return {
            'n_samples': len(results_list),
            'mean_cam_activation': float(np.mean(cam_means)),
            'mean_lung_focus_ratio': float(np.mean(lung_focus)),
            'mean_center_activation_ratio': float(np.mean(center_ratio)),
            'std_lung_focus_ratio': float(np.std(lung_focus))
        }
    
    summary = {
        'overall': aggregate_stats(correct_predictions + incorrect_predictions),
        'correct_predictions': aggregate_stats(correct_predictions),
        'incorrect_predictions': aggregate_stats(incorrect_predictions),
        'by_class': {
            class_name: aggregate_stats(results)
            for class_name, results in results_by_class.items()
        }
    }
    
    # Save detailed results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    full_results = {
        'summary': summary,
        'individual_samples': correct_predictions + incorrect_predictions
    }
    
    with open(output_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"✓ Visualizations saved to: {vis_dir}/")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal samples evaluated: {n_samples}")
    print(f"Correct predictions: {len(correct_predictions)}")
    print(f"Incorrect predictions: {len(incorrect_predictions)}")
    
    print("\nOverall Grad-CAM Quality:")
    print(f"  Average lung focus ratio: {summary['overall']['mean_lung_focus_ratio']:.3f}")
    print(f"  Average center activation ratio: {summary['overall']['mean_center_activation_ratio']:.3f}")
    
    print("\nFor Correct Predictions:")
    if correct_predictions:
        print(f"  Lung focus ratio: {summary['correct_predictions']['mean_lung_focus_ratio']:.3f}")
        print(f"  Center activation ratio: {summary['correct_predictions']['mean_center_activation_ratio']:.3f}")
    
    print("\nFor Incorrect Predictions:")
    if incorrect_predictions:
        print(f"  Lung focus ratio: {summary['incorrect_predictions']['mean_lung_focus_ratio']:.3f}")
        print(f"  Center activation ratio: {summary['incorrect_predictions']['mean_center_activation_ratio']:.3f}")
    
    print("\nBy Class:")
    for class_name, stats in summary['by_class'].items():
        if stats:
            print(f"  {class_name}:")
            print(f"    Lung focus ratio: {stats['mean_lung_focus_ratio']:.3f}")
            print(f"    Samples: {stats['n_samples']}")
    
    # Interpretation
    print("\n" + "=" * 80)
    print("💡 INTERPRETATION")
    print("=" * 80)
    
    lung_focus = summary['overall']['mean_lung_focus_ratio']
    if lung_focus > 0.75:
        print("✓ EXCELLENT: Model strongly focuses on lung regions (>75%)")
    elif lung_focus > 0.60:
        print("✓ GOOD: Model predominantly focuses on lung regions (60-75%)")
    elif lung_focus > 0.45:
        print("⚠️  FAIR: Model shows moderate lung region focus (45-60%)")
    else:
        print("⚠️  CONCERN: Model may not be focusing on clinically relevant regions (<45%)")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
