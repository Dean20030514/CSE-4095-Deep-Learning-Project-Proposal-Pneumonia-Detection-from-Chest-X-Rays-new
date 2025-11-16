"""
Quick visualization script for training metrics.
Generates plots from metrics.csv for easy analysis.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(csv_path, output_dir):
    """Plot training curves from metrics CSV."""
    df = pd.read_csv(csv_path)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    ax.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150)
    plt.close()
    
    # Plot 2: Pneumonia metrics
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.plot(df['epoch'], df['pneumonia_recall'], label='Recall', marker='o', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Pneumonia Recall')
    ax1.set_title('Pneumonia Recall (Primary KPI)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    ax2.plot(df['epoch'], df['pneumonia_precision'], label='Precision', marker='s', color='blue')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Pneumonia Precision')
    ax2.set_title('Pneumonia Precision')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    ax3.plot(df['epoch'], df['pneumonia_f1'], label='F1', marker='^', color='green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Pneumonia F1')
    ax3.set_title('Pneumonia F1-Score')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pneumonia_metrics.png', dpi=150)
    plt.close()
    
    # Plot 3: Overall metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['val_acc'], label='Accuracy', marker='o')
    ax.plot(df['epoch'], df['macro_recall'], label='Macro Recall', marker='s')
    ax.plot(df['epoch'], df['macro_f1'], label='Macro F1', marker='^')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Overall Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_metrics.png', dpi=150)
    plt.close()
    
    # Plot 4: Learning rate
    if 'lr' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['epoch'], df['lr'], marker='o', color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'learning_rate.png', dpi=150)
        plt.close()
    
    print(f"✓ Plots saved to: {output_dir}")
    print(f"  - loss_curves.png")
    print(f"  - pneumonia_metrics.png")
    print(f"  - overall_metrics.png")
    if 'lr' in df.columns:
        print(f"  - learning_rate.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument('--csv', required=True, help='Path to metrics.csv')
    parser.add_argument('--output', default='reports/plots', help='Output directory')
    args = parser.parse_args()
    
    plot_metrics(args.csv, args.output)


if __name__ == '__main__':
    main()
