"""
Error analysis script for pneumonia detection model.
Generates error galleries, failure mode analysis, and diagnostic visualizations.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.factory import build_model
from src.data.datamodule import build_dataloaders
from src.utils.device import get_device
from src.utils.gradcam import GradCAM


def collect_errors(model, loader, device, pneumonia_idx):
    """
    Collect all error cases (FP and FN) from the dataset.
    
    Returns:
        errors: dict with 'FP' and 'FN' lists, each containing
                (image_path, true_label, pred_label, confidence)
    """
    model.eval()
    errors = {'FP': [], 'FN': []}
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Collecting errors"):
            images_gpu = images.to(device)
            logits = model(images_gpu)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            
            for i, (pred, target) in enumerate(zip(preds, targets.numpy())):
                if pred != target:
                    confidence = float(probs[i, pred].cpu().item())
                    
                    # False Positive: predicted PNEUMONIA but was NORMAL
                    if pred == pneumonia_idx and target != pneumonia_idx:
                        errors['FP'].append({
                            'image_tensor': images[i],
                            'true_label': int(target),
                            'pred_label': int(pred),
                            'confidence': confidence
                        })
                    
                    # False Negative: predicted NORMAL but was PNEUMONIA
                    elif pred != pneumonia_idx and target == pneumonia_idx:
                        errors['FN'].append({
                            'image_tensor': images[i],
                            'true_label': int(target),
                            'pred_label': int(pred),
                            'confidence': confidence
                        })
    
    return errors


def save_error_gallery(errors, error_type, idx_to_class, save_dir, max_samples=20):
    """
    Save a grid of error images.
    
    Args:
        errors: list of error dicts
        error_type: 'FP' or 'FN'
        idx_to_class: class index to name mapping
        save_dir: directory to save gallery
        max_samples: maximum number of samples to display
    """
    if len(errors) == 0:
        print(f"No {error_type} errors to display")
        return
    
    # Sort by confidence (descending) to show most confident errors
    errors_sorted = sorted(errors, key=lambda x: x['confidence'], reverse=True)
    errors_to_show = errors_sorted[:max_samples]
    
    n_cols = 5
    n_rows = (len(errors_to_show) + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.3, wspace=0.2)
    
    for idx, error in enumerate(errors_to_show):
        ax = fig.add_subplot(gs[idx])
        
        # Denormalize image for display
        img_tensor = error['image_tensor']
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_display = img_tensor * std + mean
        img_display = torch.clamp(img_display, 0, 1)
        img_display = img_display.permute(1, 2, 0).numpy()
        
        ax.imshow(img_display)
        ax.axis('off')
        
        true_name = idx_to_class[error['true_label']]
        pred_name = idx_to_class[error['pred_label']]
        conf = error['confidence']
        
        title = f"True: {true_name}\nPred: {pred_name}\nConf: {conf:.3f}"
        ax.set_title(title, fontsize=9)
    
    # Hide empty subplots
    for idx in range(len(errors_to_show), n_rows * n_cols):
        ax = fig.add_subplot(gs[idx])
        ax.axis('off')
    
    fig.suptitle(f"{error_type} Gallery (Top {len(errors_to_show)} by confidence, "
                 f"Total: {len(errors)})", fontsize=14, fontweight='bold')
    
    save_path = save_dir / f"{error_type}_gallery.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {error_type} gallery to: {save_path}")


def analyze_failure_modes(errors, idx_to_class):
    """
    Categorize and analyze failure modes with detailed descriptions.
    
    Returns:
        failure_modes: dict with categorized failure patterns and recommendations
    """
    fp_errors = errors['FP']
    fn_errors = errors['FN']
    
    # Confidence-based categorization
    fp_high_conf = [e for e in fp_errors if e['confidence'] > 0.8]
    fp_low_conf = [e for e in fp_errors if e['confidence'] < 0.6]
    
    fn_high_conf = [e for e in fn_errors if e['confidence'] > 0.8]
    fn_low_conf = [e for e in fn_errors if e['confidence'] < 0.6]
    
    failure_modes = {
        'summary': {
            'total_FP': len(fp_errors),
            'total_FN': len(fn_errors),
            'FP_high_confidence': len(fp_high_conf),
            'FP_low_confidence': len(fp_low_conf),
            'FN_high_confidence': len(fn_high_conf),
            'FN_low_confidence': len(fn_low_conf)
        },
        'failure_categories': [
            {
                'category': 'FP-1: High Confidence False Positives',
                'count': len(fp_high_conf),
                'description': 'Model very confident but wrong - likely subtle artifacts, device shadows, or anatomical variations mistaken for pathology',
                'severity': 'MAJOR',
                'recommendation': 'Review with Grad-CAM to identify spurious features. Consider data augmentation to reduce reliance on artifacts.'
            },
            {
                'category': 'FP-2: Low Confidence False Positives',
                'count': len(fp_low_conf),
                'description': 'Model uncertain predictions - borderline cases with ambiguous features',
                'severity': 'MINOR',
                'recommendation': 'Threshold adjustment can eliminate these. Acceptable trade-off for higher recall.'
            },
            {
                'category': 'FN-1: High Confidence False Negatives',
                'count': len(fn_high_conf),
                'description': 'Model confident NORMAL but missed pneumonia - critical errors indicating early/subtle signs or poor image quality',
                'severity': 'CRITICAL',
                'recommendation': '⚠️ PRIORITY: Manually review these cases. May need additional training data with subtle pneumonia cases.'
            },
            {
                'category': 'FN-2: Low Confidence False Negatives',
                'count': len(fn_low_conf),
                'description': 'Model uncertain about pneumonia - lowering threshold can catch these cases',
                'severity': 'MAJOR',
                'recommendation': 'Threshold tuning recommended. Consider ensemble or secondary review for low-confidence predictions.'
            }
        ],
        'medical_implications': {
            'FP_clinical_impact': 'False positives lead to unnecessary follow-up tests (X-ray confirmation, CT, labs). Increases healthcare costs but low patient harm.',
            'FN_clinical_impact': '⚠️ CRITICAL: False negatives mean missed pneumonia cases. Can delay treatment and worsen patient outcomes. Must be minimized.',
            'threshold_strategy': 'For medical screening: Lower threshold to maximize recall (sensitivity ≥99%), accept higher false positive rate. Use as triage tool, not diagnostic.'
        },
        'next_steps': [
            '1. Generate Grad-CAM visualizations for top high-confidence errors',
            '2. Perform threshold sweep to find optimal operating point (Max Recall mode)',
            '3. Consider ensemble methods or secondary classifier for borderline cases',
            '4. Collect additional training data for failure modes (if available)',
            '5. Document error patterns in Model Card for transparency'
        ]
    }
    
    return failure_modes


def main():
    parser = argparse.ArgumentParser(description="Error analysis for pneumonia detection")
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint')
    parser.add_argument('--data_root', default='data', help='Data root directory')
    parser.add_argument('--split', default='val', choices=['val', 'test'], help='Dataset split')
    parser.add_argument('--model', default=None, help='Override model name from checkpoint (e.g., efficientnet_b2)')
    parser.add_argument('--output_dir', default='reports/error_analysis', help='Output directory')
    parser.add_argument('--max_samples', type=int, default=20, help='Max samples per gallery')
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
    pneumonia_idx = class_to_idx.get('PNEUMONIA', class_to_idx.get('pneumonia', 1))
    
    print(f"Model: {model_name} @ {img_size}px")
    if args.model:
        print(f"⚠️  Model name overridden: {cfg.get('model')} -> {model_name}")
    print(f"Classes: {idx_to_class}")
    print(f"Pneumonia index: {pneumonia_idx}")
    
    # Build model
    model, _ = build_model(model_name, len(class_to_idx))
    model.load_state_dict(ckpt['model'], strict=False)  # Use strict=False for architecture mismatch
    device = get_device()
    model = model.to(device).eval()
    
    # Load data
    print(f"Loading {args.split} data...")
    loaders, _ = build_dataloaders(args.data_root, img_size, batch_size=16, use_weighted_sampler=False, num_workers=0)
    loader = loaders[args.split]
    
    # Collect errors
    print("\nCollecting error cases...")
    errors = collect_errors(model, loader, device, pneumonia_idx)
    
    print(f"\nError Summary:")
    print(f"  False Positives (Normal→Pneumonia): {len(errors['FP'])}")
    print(f"  False Negatives (Pneumonia→Normal): {len(errors['FN'])}")
    
    # Analyze failure modes
    print("\nAnalyzing failure modes...")
    failure_modes = analyze_failure_modes(errors, idx_to_class)
    
    print(f"\n{'='*70}")
    print("FAILURE MODE ANALYSIS")
    print(f"{'='*70}")
    
    for category in failure_modes['failure_categories']:
        severity_marker = "⚠️" if category['severity'] == 'CRITICAL' else "🔴" if category['severity'] == 'MAJOR' else "🟡"
        print(f"\n{severity_marker} {category['category']}")
        print(f"  Count: {category['count']}")
        print(f"  Severity: {category['severity']}")
        print(f"  Description: {category['description']}")
        print(f"  Recommendation: {category['recommendation']}")
    
    print(f"\n{'='*70}")
    print("MEDICAL IMPLICATIONS")
    print(f"{'='*70}")
    for key, value in failure_modes['medical_implications'].items():
        print(f"\n{key.replace('_', ' ').title()}:")
        print(f"  {value}")
    
    print(f"\n{'='*70}")
    print("RECOMMENDED NEXT STEPS")
    print(f"{'='*70}")
    for step in failure_modes['next_steps']:
        print(f"  {step}")
    print(f"{'='*70}\n")
    
    # Save failure mode report
    report_path = output_dir / 'failure_modes.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(failure_modes, f, indent=2, ensure_ascii=False)
    print(f"\nFailure mode report saved to: {report_path}")
    
    # Generate galleries
    print("\nGenerating error galleries...")
    save_error_gallery(errors['FP'], 'FP', idx_to_class, output_dir, args.max_samples)
    save_error_gallery(errors['FN'], 'FN', idx_to_class, output_dir, args.max_samples)
    
    # Generate summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # FP breakdown
    fp_data = [
        failure_modes['summary']['FP_high_confidence'],
        failure_modes['summary']['FP_low_confidence'],
        len(errors['FP']) - failure_modes['summary']['FP_high_confidence'] - failure_modes['summary']['FP_low_confidence']
    ]
    if len(errors['FP']) > 0:
        ax1.pie(fp_data, labels=['High Conf', 'Low Conf', 'Medium Conf'], autopct='%1.1f%%', startangle=90)
        ax1.set_title(f"False Positives (Total: {len(errors['FP'])})")
    else:
        ax1.text(0.5, 0.5, 'No False Positives', ha='center', va='center', fontsize=12)
        ax1.set_title("False Positives (Total: 0)")
        ax1.axis('off')
    
    # FN breakdown
    fn_data = [
        failure_modes['summary']['FN_high_confidence'],
        failure_modes['summary']['FN_low_confidence'],
        len(errors['FN']) - failure_modes['summary']['FN_high_confidence'] - failure_modes['summary']['FN_low_confidence']
    ]
    if len(errors['FN']) > 0:
        ax2.pie(fn_data, labels=['High Conf', 'Low Conf', 'Medium Conf'], autopct='%1.1f%%', startangle=90)
        ax2.set_title(f"False Negatives (Total: {len(errors['FN'])})")
    else:
        ax2.text(0.5, 0.5, 'No False Negatives', ha='center', va='center', fontsize=12)
        ax2.set_title("False Negatives (Total: 0)")
        ax2.axis('off')
    
    plt.tight_layout()
    summary_path = output_dir / 'error_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error summary plot saved to: {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"Error analysis complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
